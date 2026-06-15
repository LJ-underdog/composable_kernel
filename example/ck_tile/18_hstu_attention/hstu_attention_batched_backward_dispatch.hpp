// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <stdexcept>

#include <hip/hip_runtime.h>

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/host/hip_check_error.hpp"
#include "ck_tile/host/stream_config.hpp"

#include "hstu_attention_bool_switch.hpp"
#include "hstu_attention_fwd_type_config.hpp"
#include "hstu_attention_bwd_params.hpp"
#include "hstu_attention_no_softmax_bwd_pipeline.hpp"
#include "hstu_attention_with_softmax_bwd_pipeline.hpp"
#include "hstu_attention_bwd_kernel.hpp"
#include "hstu_attention_bwd_shape.hpp"
#include "hstu_attention_bwd_perf.hpp"

// HSTU attention backward — batched NoGroup dispatch (M1: SiLU + no-mask + atomic).
//
// Template signature mirrors fwd / DESIGN §4.3:
//   run_batched_backward_dispatch<InOutDataType, kUseCausal, kUseSoftmax,
//                                 kHasBias, kIsDeterministic, MaxK>(param, stream)
//
// M1 implements ONLY the (SiLU, no-mask, hdim64, atomic) path:
//   zero dq_acc -> MAIN (5 GEMM + dsilu, atomic into float dq_acc) -> POST (convert dq_acc->dq).
// causal -> M2, softmax -> M5, deterministic -> M6: each compiles to a runtime throw
// (gated by `if constexpr`, so no real instantiation of those paths yet).

template <typename InOutDataType,
          bool kUseCausal,
          bool kUseSoftmax,
          bool kHasBias,
          bool kIsDeterministic,
          ck_tile::index_t MaxK>
struct batched_backward_dispatch
{
    // M7b: tile shape is now selected by MaxK (HstuBwdShape<MaxK>). HstuBwdShape<64>
    // is byte-identical to the pre-M7b hardcoded hd64 preset (FMHA bwd '64' preset;
    // warp_tile0 16x16x32 = CDNA4 K-doubled MFMA), so the hd64 path is unchanged.
    using FmhaBwdShape = typename HstuBwdShape<MaxK>::Type;

    using TC = HstuAttentionFwdTypeConfig<InOutDataType>;

    // Shared tail for both SiLU and softmax MAIN: zero dq_acc, launch MAIN, then POST.
    // Atomic (kIsDeterministic=false): single dq_acc slot -> convert. Deterministic: dq_acc
    // is num_splits stacked slots (each KV-block wrote its own via set) -> reduce-then-convert.
    template <typename Pipeline, typename Kernel, typename Kargs>
    static void
    launch_main_and_post(HstuAttentionNoGroupBwdParams& param, hipStream_t stream, Kargs& kargs)
    {
        // single-slot element count (== atomic dq_acc size; == split_stride in determ)
        const size_t single =
            param.is_jagged
                ? static_cast<size_t>(param.batch_stride_dq_acc)
                : static_cast<size_t>(param.num_batch) *
                      static_cast<size_t>(param.batch_stride_dq_acc);

        // grid.x covers the largest seqlen_kv; split_idx = i_tile_n -> num_splits = grid.x.
        // cross: jagged uses max_seqlen_kv (self path aliases it to max_seqlen_q -> unchanged);
        // batched already carries the seqlen_kv scalar (correct in both directions).
        const ck_tile::index_t grid_seqlen_kv =
            param.is_jagged ? param.max_seqlen_kv : param.seqlen_kv;
        const int num_splits =
            kIsDeterministic
                ? static_cast<int>(ck_tile::integer_divide_ceil(grid_seqlen_kv, Pipeline::kN0))
                : 1;

        // M8 MI: time the ZERO_dq_acc memset separately (kept out of MAIN TFLOPS).
        param.perf_memset_ms = hstu_bwd_perf::time_op(param.measure_perf, stream, [&] {
            HIP_CHECK_ERROR(hipMemsetAsync(param.dq_acc_ptr,
                                           0,
                                           single * static_cast<size_t>(num_splits) *
                                               sizeof(typename TC::GemmAccDataType),
                                           stream));
        });

        dim3 grid  = Kernel::GridSize(param.num_batch, param.num_head, grid_seqlen_kv);
        dim3 block = Kernel::BlockSize();
        constexpr ck_tile::index_t kBlockPerCu = Kernel::kBlockPerCu;
        // M8 MI: MAIN dqdkdv — the bottleneck (B2/B3 target). Inner launch keeps
        // stream_config{stream,false}; the warmup+repeat is done by time_op.
        param.perf_main_ms = hstu_bwd_perf::time_op(param.measure_perf, stream, [&] {
            (void)ck_tile::launch_kernel(
                ck_tile::stream_config{stream, false},
                ck_tile::make_kernel<kBlockPerCu>(Kernel{}, grid, block, 0, kargs));
        });

        // POST: dq_acc(float) -> dq. Determ reduces over splits (fixed order -> reproducible).
        const ck_tile::long_index_t n = static_cast<ck_tile::long_index_t>(single);
        constexpr int kPostThreads = 256;
        const int post_blocks      = static_cast<int>((n + kPostThreads - 1) / kPostThreads);
        // M8 MI: POST convert (atomic) / reduce_convert (deterministic) dq_acc->dq.
        param.perf_post_ms = hstu_bwd_perf::time_op(param.measure_perf, stream, [&] {
            if constexpr(kIsDeterministic)
            {
                hipLaunchKernelGGL(
                    (ck_tile::hstu_bwd_reduce_convert_dq_kernel<InOutDataType,
                                                                typename TC::GemmAccDataType>),
                    dim3(post_blocks),
                    dim3(kPostThreads),
                    0,
                    stream,
                    reinterpret_cast<const typename TC::GemmAccDataType*>(param.dq_acc_ptr),
                    reinterpret_cast<InOutDataType*>(param.dq_ptr),
                    n,
                    num_splits,
                    static_cast<ck_tile::long_index_t>(single));
            }
            else
            {
                hipLaunchKernelGGL(
                    (ck_tile::hstu_bwd_convert_dq_kernel<InOutDataType,
                                                         typename TC::GemmAccDataType>),
                    dim3(post_blocks),
                    dim3(kPostThreads),
                    0,
                    stream,
                    reinterpret_cast<const typename TC::GemmAccDataType*>(param.dq_acc_ptr),
                    reinterpret_cast<InOutDataType*>(param.dq_ptr),
                    n);
            }
        });
    }

    template <typename Mask, bool kPadHeadDimQ, bool kPadHeadDimV>
    static void RunSilu(HstuAttentionNoGroupBwdParams& param, hipStream_t stream)
    {
        // M7c: kPadHeadDimQ/V are now NTTPs from the Run() BOOL_SWITCH_2 (modulo-derived).
        // Canonical hdim==MaxK -> both false -> byte-identical to M7b. Q-group flag pads
        // Q/K/dQ/dK/dQ_acc (kQKHeaddim); V-group flag pads V/dO/dV/O (kVHeaddim) — never crossed.
        constexpr ck_tile::index_t occupancy = 1;

        // ck_hstu TileFmhaBwdTraits<kPadHeadDimQ, kPadHeadDimV, BiasEnum, kHasBiasGrad, kBlockPerCu>
        using Traits = ck_tile::TileFmhaBwdTraits<kPadHeadDimQ,
                                                  kPadHeadDimV,
                                                  ck_tile::BlockAttentionBiasEnum::NO_BIAS, // P1-A
                                                  false, // kHasBiasGrad
                                                  occupancy>;

        using Dropout = ck_tile::BlockDropoutBwd<false, true, false>;  // no-dropout

        using Problem = ck_tile::BlockFmhaBwdPipelineProblem<
            typename TC::ODataType,         // QDataType (== InOutDataType)
            typename TC::ODataType,         // KDataType
            typename TC::ODataType,         // VDataType
            typename TC::ODataType,         // GemmDataType
            typename TC::CompDataType,      // LSEDataType (dummy on SiLU path)
            typename TC::GemmAccDataType,   // AccDataType (float)
            typename TC::CompDataType,      // DDataType (dummy on SiLU path)
            typename TC::BiasDataType,      // BiasDataType (dummy, P1-A)
            uint8_t,                        // RandValOutputDataType (dummy)
            typename TC::ODataType,         // ODataType
            typename TC::ODataType,         // OGradDataType
            typename TC::ODataType,         // QGradDataType
            typename TC::ODataType,         // KGradDataType
            typename TC::ODataType,         // VGradDataType
            typename TC::BiasDataType,      // BiasGradDataType (dummy)
            FmhaBwdShape,
            false, // kIsGroupMode
            kIsDeterministic, // M6: deterministic (set+split) vs atomic (false)
            Mask,
            Dropout,
            false, // kUseTrLoad (M1 non-trload; trload perf is M8)
            Traits>;

        using Pipeline = ck_tile::HstuAttentionBwdDQDKDVPipelineKRKTRVR<Problem>;

        using DKEpilogue = ck_tile::Default2DEpilogue<ck_tile::Default2DEpilogueProblem<
            typename TC::GemmAccDataType,
            typename TC::ODataType,
            false,
            (kPadHeadDimQ > 0)>>;
        using DVEpilogue = ck_tile::Default2DEpilogue<ck_tile::Default2DEpilogueProblem<
            typename TC::GemmAccDataType,
            typename TC::ODataType,
            false,
            (kPadHeadDimV > 0)>>;

        using Kernel = ck_tile::HstuAttentionBwdDQDKDVKernel<Pipeline, DKEpilogue, DVEpilogue>;

        const float scale_p =
            (param.attn_scale != 0.f) ? param.attn_scale
                                      : 1.0f / static_cast<float>(param.max_seqlen_q);

        auto kargs = Kernel::MakeKargs(param.q_ptr,
                                       param.k_ptr,
                                       param.v_ptr,
                                       param.do_ptr,
                                       param.dk_ptr,
                                       param.dv_ptr,
                                       param.dq_acc_ptr,
                                       param.is_jagged,
                                       param.seq_q_offsets_ptr,
                                       // self-attention: kv offsets == q offsets (mirrors fwd)
                                       param.is_cross_attention ? param.seq_kv_offsets_ptr
                                                                : param.seq_q_offsets_ptr,
                                       param.seqlen_q,
                                       param.seqlen_kv,
                                       param.hdim_qk,
                                       param.hdim_v,
                                       param.nhead_ratio_qk,
                                       param.alpha,
                                       scale_p,
                                       param.num_targets_ptr,
                                       param.contextual_seqlen,
                                       param.window_size, // max_attn_len
                                       param.min_full_attn_seqlen,
                                       param.seq_stride_q,
                                       param.seq_stride_k,
                                       param.seq_stride_v,
                                       param.seq_stride_do,
                                       param.seq_stride_dk,
                                       param.seq_stride_dv,
                                       param.stride_dq_acc,
                                       param.nhead_stride_q,
                                       param.nhead_stride_k,
                                       param.nhead_stride_v,
                                       param.nhead_stride_do,
                                       param.nhead_stride_dk,
                                       param.nhead_stride_dv,
                                       param.nhead_stride_dq_acc,
                                       param.batch_stride_q,
                                       param.batch_stride_k,
                                       param.batch_stride_v,
                                       param.batch_stride_do,
                                       param.batch_stride_dk,
                                       param.batch_stride_dv,
                                       param.batch_stride_dq_acc,
                                       param.split_stride_dq_acc);

        launch_main_and_post<Pipeline, Kernel>(param, stream, kargs);
    }

    // M5 softmax path (no_group = batched + jagged). PRE (D=rowsum(O*dO)) -> memset
    // dq_acc -> MAIN (softmax pipeline, reads LSE+D) -> POST (convert dq_acc->dq).
    template <typename Mask, bool kPadHeadDimQ, bool kPadHeadDimV>
    static void RunSoftmax(HstuAttentionNoGroupBwdParams& param, hipStream_t stream)
    {
        // M7c: kPadHeadDimQ/V NTTPs from Run() BOOL_SWITCH_2 (see RunSilu note).
        constexpr ck_tile::index_t occupancy = 1;

        using Traits = ck_tile::TileFmhaBwdTraits<kPadHeadDimQ,
                                                  kPadHeadDimV,
                                                  ck_tile::BlockAttentionBiasEnum::NO_BIAS,
                                                  false,
                                                  occupancy>;
        using Dropout = ck_tile::BlockDropoutBwd<false, true, false>;

        using Problem = ck_tile::BlockFmhaBwdPipelineProblem<
            typename TC::ODataType,
            typename TC::ODataType,
            typename TC::ODataType,
            typename TC::ODataType,
            typename TC::CompDataType,    // LSEDataType (real on softmax path)
            typename TC::GemmAccDataType,
            typename TC::CompDataType,    // DDataType (real on softmax path)
            typename TC::BiasDataType,
            uint8_t,
            typename TC::ODataType,
            typename TC::ODataType,
            typename TC::ODataType,
            typename TC::ODataType,
            typename TC::ODataType,
            typename TC::BiasDataType,
            FmhaBwdShape,
            false, // kIsGroupMode
            kIsDeterministic, // M6: deterministic (set+split) vs atomic (false)
            Mask,
            Dropout,
            false, // kUseTrLoad
            Traits>;

        using Pipeline = ck_tile::HstuAttentionWithSoftmaxBwdDQDKDVPipelineKRKTRVR<Problem>;

        using DKEpilogue = ck_tile::Default2DEpilogue<ck_tile::Default2DEpilogueProblem<
            typename TC::GemmAccDataType, typename TC::ODataType, false, (kPadHeadDimQ > 0)>>;
        using DVEpilogue = ck_tile::Default2DEpilogue<ck_tile::Default2DEpilogueProblem<
            typename TC::GemmAccDataType, typename TC::ODataType, false, (kPadHeadDimV > 0)>>;

        using Kernel =
            ck_tile::HstuAttentionBwdDQDKDVSoftmaxKernel<Pipeline, DKEpilogue, DVEpilogue>;

        // ---- PRE: D = rowsum(O .* dO) -> param.d_ptr ([batch,head,seq] layout) ----
        const ck_tile::index_t grid_seqlen =
            param.is_jagged ? param.max_seqlen_q : param.seqlen_q;
        {
            const ck_tile::long_index_t total =
                static_cast<ck_tile::long_index_t>(param.num_batch) * param.num_head * grid_seqlen;
            constexpr int kPreThreads = 256;
            const int pre_blocks = static_cast<int>((total + kPreThreads - 1) / kPreThreads);
            // M8 MI: time PRE D=rowsum(O*dO) (softmax path only).
            param.perf_pre_ms = hstu_bwd_perf::time_op(param.measure_perf, stream, [&] {
            hipLaunchKernelGGL(
                (ck_tile::hstu_bwd_dot_do_o_kernel<InOutDataType, typename TC::CompDataType>),
                dim3(pre_blocks),
                dim3(kPreThreads),
                0,
                stream,
                reinterpret_cast<const InOutDataType*>(param.o_ptr),
                reinterpret_cast<const InOutDataType*>(param.do_ptr),
                reinterpret_cast<typename TC::CompDataType*>(param.d_ptr),
                param.is_jagged,
                reinterpret_cast<const int32_t*>(
                    param.is_jagged ? param.seq_q_offsets_ptr : nullptr),
                param.num_batch,
                param.num_head,
                grid_seqlen,
                param.hdim_v,
                // PRE reads O and dO with the SAME (O) strides: the harness allocates dO
                // with a layout identical to O (both [batch, sq, head, hdim_v]), so this is
                // always valid here. If dO ever gets a distinct layout (e.g. cross-attn /
                // external dO), pass param.{seq,nhead,batch}_stride_do separately for dO.
                static_cast<ck_tile::long_index_t>(param.seq_stride_o),
                static_cast<ck_tile::long_index_t>(param.nhead_stride_o),
                static_cast<ck_tile::long_index_t>(param.batch_stride_o),
                static_cast<ck_tile::long_index_t>(param.nhead_stride_lsed),
                static_cast<ck_tile::long_index_t>(param.batch_stride_lsed));
            }); // time_op (PRE)
        }

        auto kargs = Kernel::MakeKargs(param.q_ptr,
                                       param.k_ptr,
                                       param.v_ptr,
                                       param.do_ptr,
                                       param.lse_ptr,
                                       param.d_ptr,
                                       param.dk_ptr,
                                       param.dv_ptr,
                                       param.dq_acc_ptr,
                                       param.is_jagged,
                                       param.seq_q_offsets_ptr,
                                       param.is_cross_attention ? param.seq_kv_offsets_ptr
                                                                : param.seq_q_offsets_ptr,
                                       param.seqlen_q,
                                       param.seqlen_kv,
                                       param.hdim_qk,
                                       param.hdim_v,
                                       param.nhead_ratio_qk,
                                       param.alpha,
                                       param.num_targets_ptr,
                                       param.contextual_seqlen,
                                       param.window_size,
                                       param.min_full_attn_seqlen,
                                       param.seq_stride_q,
                                       param.seq_stride_k,
                                       param.seq_stride_v,
                                       param.seq_stride_do,
                                       param.seq_stride_dk,
                                       param.seq_stride_dv,
                                       param.stride_dq_acc,
                                       param.nhead_stride_q,
                                       param.nhead_stride_k,
                                       param.nhead_stride_v,
                                       param.nhead_stride_do,
                                       param.nhead_stride_dk,
                                       param.nhead_stride_dv,
                                       param.nhead_stride_dq_acc,
                                       param.nhead_stride_lsed,
                                       param.batch_stride_q,
                                       param.batch_stride_k,
                                       param.batch_stride_v,
                                       param.batch_stride_do,
                                       param.batch_stride_dk,
                                       param.batch_stride_dv,
                                       param.batch_stride_dq_acc,
                                       param.batch_stride_lsed,
                                       param.split_stride_dq_acc);

        launch_main_and_post<Pipeline, Kernel>(param, stream, kargs);
    }

    static void Run(HstuAttentionNoGroupBwdParams& param, hipStream_t stream)
    {
        // M3: jagged handled by the same MAIN kernel (runtime param.is_jagged). M5 adds the
        // softmax branch (PRE D + LSE/D). M6: kIsDeterministic is a template axis threaded
        // into the pipeline/kernel (set+split dq_acc + POST reduce) — no separate branch here;
        // both SiLU and softmax paths support atomic & deterministic.

        // M7c: accept asymmetric hdim_qk!=hdim_v and non-canonical hdim via head-dim padding.
        // HDIM_SWITCH already guarantees MaxK >= max(hdim_qk,hdim_v); the only real reject is
        // hdim>256 (kept in hstu_attention_hdim_switch.hpp). pad flags are modulo-derived
        // (fwd-style: !(hdim % tile == 0)) so canonical hdim==MaxK -> false -> byte-identical
        // to M7b. Q-group flag (kQKHeaddim) and V-group flag (kVHeaddim) are tracked SEPARATELY.
        if(param.hdim_qk <= 0 || param.hdim_v <= 0 || param.hdim_qk > MaxK || param.hdim_v > MaxK)
            throw std::runtime_error("HSTU bwd: hdim_qk/hdim_v must be in (0, MaxK]; hdim>256 unsupported");

        constexpr ck_tile::index_t kQKHeaddim = HstuBwdShape<MaxK>::kQKHeaddim;
        constexpr ck_tile::index_t kVHeaddim  = HstuBwdShape<MaxK>::kVHeaddim;
        const bool pad_qk = !(param.hdim_qk % kQKHeaddim == 0);
        const bool pad_v  = !(param.hdim_v % kVHeaddim == 0);
        const bool use_local = (param.window_size > 0);

        // Hoist the pad BOOL_SWITCH_2 once (fwd style) so SiLU+softmax share it.
        BOOL_SWITCH_2(pad_qk, kPadHeadDimQ, pad_v, kPadHeadDimV, [&] {
            BOOL_SWITCH(use_local, kUseLocal, [&] {
                // cross-attention is a RUNTIME switch (mirrors fwd dispatch :97), not an
                // instance axis: both legs compile in the same .cpp. Wrap ONLY the mask
                // typedef + downstream Pipeline/Kernel; the false leg resolves to the same
                // HstuBlockMasking<false,...>::Type as M7c -> byte-identical (see draft §4).
                BOOL_SWITCH(param.is_cross_attention, kIsCrossAttention, [&] {
                    using Mask = typename ck_tile::
                        HstuBlockMasking<kIsCrossAttention, kUseCausal, kUseLocal>::Type;
                    if constexpr(kUseSoftmax)
                        RunSoftmax<Mask, kPadHeadDimQ, kPadHeadDimV>(param, stream);
                    else
                        RunSilu<Mask, kPadHeadDimQ, kPadHeadDimV>(param, stream);
                });
            });
        });
    }
};

template <typename InOutDataType,
          bool kUseCausal,
          bool kUseSoftmax,
          bool kHasBias,
          bool kIsDeterministic,
          ck_tile::index_t MaxK>
void run_batched_backward_dispatch(HstuAttentionNoGroupBwdParams& param, hipStream_t stream)
{
    batched_backward_dispatch<InOutDataType,
                              kUseCausal,
                              kUseSoftmax,
                              kHasBias,
                              kIsDeterministic,
                              MaxK>::Run(param, stream);
}
