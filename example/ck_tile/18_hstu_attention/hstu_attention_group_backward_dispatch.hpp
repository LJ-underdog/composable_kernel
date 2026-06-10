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

// HSTU attention backward — GROUP dispatch (M4: SiLU + atomic).
//
// group = jagged superset (DESIGN §3.4 / §4.7 D6): same packed + cu_seqlens
// indexing as M3, plus per-group hyper-params indexed by i_group inside the
// kernel. `alpha` is a global scalar; scale_p + the 4 mask hyper-params are
// per-group device pointers. Because the per-group window cannot be resolved to
// a compile-time kUseLocal, the GROUP kernel holds BOTH pipelines (with-local +
// without-local, same kUseCausal) and branches at runtime (mirrors fwd).
//
// causal -> compile-time axis (uniform across groups). softmax -> M5,
// deterministic -> M6: gated by `if constexpr` (no real instantiation yet).

template <typename InOutDataType,
          bool kUseCausal,
          bool kUseSoftmax,
          bool kHasBias,
          bool kIsDeterministic,
          ck_tile::index_t MaxK>
struct group_backward_dispatch
{
    // hd64 tile (identical to the no_group/batched preset)
    using FmhaBlockTile = ck_tile::sequence<32, 128, 64, 32, 64, 32, 32, 64, 64>;
    using BlockWarps0   = ck_tile::sequence<1, 4, 1>;
    using BlockWarps1   = ck_tile::sequence<4, 1, 1>;
    using BlockWarps2   = ck_tile::sequence<1, 4, 1>;
    using WarpTile0     = ck_tile::sequence<16, 16, 32>;
    using WarpTile1     = ck_tile::sequence<16, 16, 16>;

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
                                                   0 /* kMaxSeqLenQ */>;

    using TC = HstuAttentionFwdTypeConfig<InOutDataType>;

    template <typename Mask>
    using ProblemFor = ck_tile::BlockFmhaBwdPipelineProblem<
        typename TC::ODataType,
        typename TC::ODataType,
        typename TC::ODataType,
        typename TC::ODataType,
        typename TC::CompDataType,
        typename TC::GemmAccDataType,
        typename TC::CompDataType,
        typename TC::BiasDataType,
        uint8_t,
        typename TC::ODataType,
        typename TC::ODataType,
        typename TC::ODataType,
        typename TC::ODataType,
        typename TC::ODataType,
        typename TC::BiasDataType,
        FmhaBwdShape,
        false, // kIsGroupMode (FMHA group-mode flag; HSTU group uses jagged indexing in-kernel)
        kIsDeterministic, // M6b: deterministic (set+split) vs atomic (false)
        Mask,
        ck_tile::BlockDropoutBwd<false, true, false>,
        false, // kUseTrLoad
        ck_tile::TileFmhaBwdTraits<0, 0,
                                   ck_tile::BlockAttentionBiasEnum::NO_BIAS,
                                   false, 1>>;

    // Shared tail for group SiLU/softmax MAIN: zero dq_acc, launch MAIN, POST. Atomic:
    // single packed slot -> convert. Deterministic (M6b): num_splits stacked slots (each
    // KV-block wrote its own via set) -> fixed-order reduce-then-convert (bit-reproducible).
    template <typename Pipeline, typename Kernel, typename Kargs>
    static void
    launch_main_and_post(HstuAttentionGroupBwdParams& param, hipStream_t stream, Kargs& kargs)
    {
        const size_t single = static_cast<size_t>(param.total_dq_acc_elems); // one packed slot
        const int num_splits =
            kIsDeterministic
                ? static_cast<int>(
                      ck_tile::integer_divide_ceil(param.max_seqlen_q, Pipeline::kN0))
                : 1;

        HIP_CHECK_ERROR(hipMemsetAsync(param.dq_acc_ptr,
                                       0,
                                       single * static_cast<size_t>(num_splits) *
                                           sizeof(typename TC::GemmAccDataType),
                                       stream));

        dim3 grid  = Kernel::GridSize(param.num_batch, param.num_head, param.max_seqlen_q);
        dim3 block = Kernel::BlockSize();
        constexpr ck_tile::index_t kBlockPerCu = Kernel::kBlockPerCu;
        (void)ck_tile::launch_kernel(
            ck_tile::stream_config{stream, false},
            ck_tile::make_kernel<kBlockPerCu>(Kernel{}, grid, block, 0, kargs));

        const ck_tile::long_index_t n = static_cast<ck_tile::long_index_t>(single);
        constexpr int kPostThreads = 256;
        const int post_blocks      = static_cast<int>((n + kPostThreads - 1) / kPostThreads);
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
                (ck_tile::hstu_bwd_convert_dq_kernel<InOutDataType, typename TC::GemmAccDataType>),
                dim3(post_blocks),
                dim3(kPostThreads),
                0,
                stream,
                reinterpret_cast<const typename TC::GemmAccDataType*>(param.dq_acc_ptr),
                reinterpret_cast<InOutDataType*>(param.dq_ptr),
                n);
        }
    }

    static void RunSilu(HstuAttentionGroupBwdParams& param, hipStream_t stream)
    {
        constexpr ck_tile::index_t kPadHeadDimQ = 0;
        constexpr ck_tile::index_t kPadHeadDimV = 0;

        // both mask types share kUseCausal; window resolved at runtime in-kernel
        using LocalMask =
            typename ck_tile::HstuBlockMasking<false /*cross*/, kUseCausal, true>::Type;
        using NoLocalMask =
            typename ck_tile::HstuBlockMasking<false /*cross*/, kUseCausal, false>::Type;

        using PipelineLocal =
            ck_tile::HstuAttentionBwdDQDKDVPipelineKRKTRVR<ProblemFor<LocalMask>>;
        using PipelineNoLocal =
            ck_tile::HstuAttentionBwdDQDKDVPipelineKRKTRVR<ProblemFor<NoLocalMask>>;

        using DKEpilogue = ck_tile::Default2DEpilogue<ck_tile::Default2DEpilogueProblem<
            typename TC::GemmAccDataType, typename TC::ODataType, false, (kPadHeadDimQ > 0)>>;
        using DVEpilogue = ck_tile::Default2DEpilogue<ck_tile::Default2DEpilogueProblem<
            typename TC::GemmAccDataType, typename TC::ODataType, false, (kPadHeadDimV > 0)>>;

        using Kernel = ck_tile::HstuAttentionBwdDQDKDVGroupKernel<PipelineLocal,
                                                                  PipelineNoLocal,
                                                                  DKEpilogue,
                                                                  DVEpilogue>;

        auto kargs = Kernel::MakeKargs(param.q_ptr,
                                       param.k_ptr,
                                       param.v_ptr,
                                       param.do_ptr,
                                       param.dk_ptr,
                                       param.dv_ptr,
                                       param.dq_acc_ptr,
                                       param.seq_q_offsets_ptr,
                                       // self-attention: kv offsets == q offsets
                                       param.is_cross_attention ? param.seq_kv_offsets_ptr
                                                                : param.seq_q_offsets_ptr,
                                       param.group_attn_scale_ptr,
                                       param.group_max_seqlen_q_ptr,
                                       param.group_window_size_ptr,
                                       param.group_contextual_seqlen_ptr,
                                       param.group_min_full_attn_seqlen_ptr,
                                       param.num_batch / param.num_group,
                                       param.num_targets_ptr,
                                       param.hdim_qk,
                                       param.hdim_v,
                                       param.nhead_ratio_qk,
                                       param.alpha,
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
                                       param.split_stride_dq_acc);

        launch_main_and_post<PipelineLocal, Kernel>(param, stream, kargs);
    }

    // M5b group softmax. PRE (D=rowsum(O*dO)) -> memset dq_acc -> MAIN (group softmax
    // kernel: per-group hyper-params + double pipeline + LSE/D) -> POST (convert dq).
    static void RunSoftmax(HstuAttentionGroupBwdParams& param, hipStream_t stream)
    {
        constexpr ck_tile::index_t kPadHeadDimQ = 0;
        constexpr ck_tile::index_t kPadHeadDimV = 0;

        using LocalMask =
            typename ck_tile::HstuBlockMasking<false /*cross*/, kUseCausal, true>::Type;
        using NoLocalMask =
            typename ck_tile::HstuBlockMasking<false /*cross*/, kUseCausal, false>::Type;

        using PipelineLocal =
            ck_tile::HstuAttentionWithSoftmaxBwdDQDKDVPipelineKRKTRVR<ProblemFor<LocalMask>>;
        using PipelineNoLocal =
            ck_tile::HstuAttentionWithSoftmaxBwdDQDKDVPipelineKRKTRVR<ProblemFor<NoLocalMask>>;

        using DKEpilogue = ck_tile::Default2DEpilogue<ck_tile::Default2DEpilogueProblem<
            typename TC::GemmAccDataType, typename TC::ODataType, false, (kPadHeadDimQ > 0)>>;
        using DVEpilogue = ck_tile::Default2DEpilogue<ck_tile::Default2DEpilogueProblem<
            typename TC::GemmAccDataType, typename TC::ODataType, false, (kPadHeadDimV > 0)>>;

        using Kernel = ck_tile::HstuAttentionBwdDQDKDVGroupSoftmaxKernel<PipelineLocal,
                                                                         PipelineNoLocal,
                                                                         DKEpilogue,
                                                                         DVEpilogue>;

        // ---- PRE: D = rowsum(O .* dO). group is packed (jagged) -> token base via offsets.
        {
            const ck_tile::long_index_t total =
                static_cast<ck_tile::long_index_t>(param.num_batch) * param.num_head *
                param.max_seqlen_q;
            constexpr int kPreThreads = 256;
            const int pre_blocks = static_cast<int>((total + kPreThreads - 1) / kPreThreads);
            hipLaunchKernelGGL(
                (ck_tile::hstu_bwd_dot_do_o_kernel<InOutDataType, typename TC::CompDataType>),
                dim3(pre_blocks),
                dim3(kPreThreads),
                0,
                stream,
                reinterpret_cast<const InOutDataType*>(param.o_ptr),
                reinterpret_cast<const InOutDataType*>(param.do_ptr),
                reinterpret_cast<typename TC::CompDataType*>(param.d_ptr),
                /*is_jagged=*/true,
                reinterpret_cast<const int32_t*>(param.seq_q_offsets_ptr),
                param.num_batch,
                param.num_head,
                param.max_seqlen_q,
                param.hdim_v,
                static_cast<ck_tile::long_index_t>(param.seq_stride_o),
                static_cast<ck_tile::long_index_t>(param.nhead_stride_o),
                static_cast<ck_tile::long_index_t>(0), // packed: no batch stride
                static_cast<ck_tile::long_index_t>(param.nhead_stride_lsed),
                static_cast<ck_tile::long_index_t>(0));
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
                                       param.seq_q_offsets_ptr,
                                       param.is_cross_attention ? param.seq_kv_offsets_ptr
                                                                : param.seq_q_offsets_ptr,
                                       param.group_attn_scale_ptr,
                                       param.group_max_seqlen_q_ptr,
                                       param.group_window_size_ptr,
                                       param.group_contextual_seqlen_ptr,
                                       param.group_min_full_attn_seqlen_ptr,
                                       param.num_batch / param.num_group,
                                       param.num_targets_ptr,
                                       param.hdim_qk,
                                       param.hdim_v,
                                       param.nhead_ratio_qk,
                                       param.alpha,
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
                                       param.split_stride_dq_acc);

        launch_main_and_post<PipelineLocal, Kernel>(param, stream, kargs);
    }

    static void Run(HstuAttentionGroupBwdParams& param, hipStream_t stream)
    {
        if(param.hdim_qk != 64 || param.hdim_v != 64)
            throw std::runtime_error("HSTU bwd group supports hdim_qk=hdim_v=64 only (M7)");

        // M6b: kIsDeterministic threaded into the Problem/kernel (set+split dq_acc + reduce
        // POST). Both SiLU and softmax group paths support atomic & deterministic.
        if constexpr(kUseSoftmax)
            RunSoftmax(param, stream); // M5b / M6b
        else
            RunSilu(param, stream);
    }
};

template <typename InOutDataType,
          bool kUseCausal,
          bool kUseSoftmax,
          bool kHasBias,
          bool kIsDeterministic,
          ck_tile::index_t MaxK>
void run_group_backward_dispatch(HstuAttentionGroupBwdParams& param, hipStream_t stream)
{
    group_backward_dispatch<InOutDataType,
                            kUseCausal,
                            kUseSoftmax,
                            kHasBias,
                            kIsDeterministic,
                            MaxK>::Run(param, stream);
}
