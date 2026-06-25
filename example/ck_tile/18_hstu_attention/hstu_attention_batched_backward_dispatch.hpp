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
#include "hstu_attention_bwd_kernel.hpp"

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
    // --- M1 hd64 tile (FMHA bwd '64' preset; warp_tile0 16x16x32 = CDNA4 K-doubled MFMA) ---
    using FmhaBlockTile = ck_tile::sequence<32, 128, 64, 32, 64, 32, 32, 64, 64>;
    using BlockWarps0   = ck_tile::sequence<1, 4, 1>;
    using BlockWarps1   = ck_tile::sequence<4, 1, 1>;
    using BlockWarps2   = ck_tile::sequence<1, 4, 1>;
    using WarpTile0     = ck_tile::sequence<16, 16, 32>;
    using WarpTile1     = ck_tile::sequence<16, 16, 16>;

    // Gemm4WarpTile = sequence<wm0,wn0,min(wk0,bk4)> = <16,16,32> = WarpTile0 (hd64)
    using FmhaBwdShape = ck_tile::TileFmhaBwdShape<FmhaBlockTile,
                                                   BlockWarps0,
                                                   WarpTile0,
                                                   BlockWarps1,
                                                   WarpTile1,
                                                   BlockWarps0,
                                                   WarpTile0,
                                                   BlockWarps1,
                                                   WarpTile1,
                                                   BlockWarps2,
                                                   WarpTile0,
                                                   0 /* kMaxSeqLenQ: 0 = unlimited */>;

    using TC = HstuAttentionFwdTypeConfig<InOutDataType>;

    template <typename Mask>
    static void RunSilu(HstuAttentionNoGroupBwdParams& param, hipStream_t stream)
    {
        // hdim64: head-dim padding never needed in M1 (pad value 0)
        constexpr ck_tile::index_t kPadHeadDimQ = 0;
        constexpr ck_tile::index_t kPadHeadDimV = 0;
        constexpr ck_tile::index_t occupancy    = 1;

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
            false, // kIsDeterministic (M1 atomic)
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
                                       param.batch_stride_dq_acc);

        // zero the float dq_acc workspace before atomic accumulation. Batched layout
        // is [num_batch, sq, h, hdim]; dim0 stride (batch_stride_dq_acc) is the
        // per-batch element count.
        const size_t dq_acc_elems = static_cast<size_t>(param.num_batch) *
                                    static_cast<size_t>(param.batch_stride_dq_acc);
        HIP_CHECK_ERROR(hipMemsetAsync(
            param.dq_acc_ptr, 0, dq_acc_elems * sizeof(typename TC::GemmAccDataType), stream));

        dim3 grid = Kernel::GridSize(param.num_batch, param.num_head, param.seqlen_kv);
        dim3 block = Kernel::BlockSize();
        constexpr ck_tile::index_t kBlockPerCu = Kernel::kBlockPerCu;

        (void)ck_tile::launch_kernel(
            ck_tile::stream_config{stream, false},
            ck_tile::make_kernel<kBlockPerCu>(Kernel{}, grid, block, 0, kargs));

        // POST: convert dq_acc(float) -> dq(InOutDataType) elementwise (atomic path, nsplits=1)
        const ck_tile::long_index_t n = static_cast<ck_tile::long_index_t>(dq_acc_elems);
        constexpr int kPostThreads = 256;
        const int post_blocks      = static_cast<int>((n + kPostThreads - 1) / kPostThreads);
        hipLaunchKernelGGL(
            (ck_tile::hstu_bwd_convert_dq_kernel<InOutDataType, typename TC::GemmAccDataType>),
            dim3(post_blocks),
            dim3(kPostThreads),
            0,
            stream,
            reinterpret_cast<const typename TC::GemmAccDataType*>(param.dq_acc_ptr),
            reinterpret_cast<InOutDataType*>(param.dq_ptr),
            n);
    }

    static void Run(HstuAttentionNoGroupBwdParams& param, hipStream_t stream)
    {
        if constexpr(kIsDeterministic)
        {
            throw std::runtime_error("HSTU bwd: deterministic path not implemented yet (M6)");
        }
        else if constexpr(kUseSoftmax)
        {
            throw std::runtime_error("HSTU bwd: softmax path not implemented yet (M5)");
        }
        else
        {
            // M3: jagged (variable-length) not implemented yet.
            if(param.is_jagged)
                throw std::runtime_error("HSTU bwd: jagged path not implemented yet (M3)");

            // M1/M2: SiLU. hdim64 only. (seqlen never padded — OOB via buffer_load)
            // M2: HSTU 5-factor mask (causal/window/contextual/min_full/num_target).
            // kUseLocal selected at runtime by window_size>0 (mirrors fwd). kUseCausal=0 &
            // window=0 -> NoLocal<false> with IsMasking=false (== M1 no-mask).
            if(param.hdim_qk != 64 || param.hdim_v != 64)
                throw std::runtime_error("HSTU bwd M1/M2 supports hdim_qk=hdim_v=64 only (M7)");

            const bool use_local = (param.window_size > 0);
            BOOL_SWITCH(use_local, kUseLocal, [&] {
                using Mask = typename ck_tile::
                    HstuBlockMasking<false /*cross*/, kUseCausal, kUseLocal>::Type;
                RunSilu<Mask>(param, stream);
            });
        }
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
