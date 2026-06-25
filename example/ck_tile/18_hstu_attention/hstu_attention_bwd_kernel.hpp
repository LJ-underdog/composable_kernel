// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <hip/hip_runtime.h>
#include "ck_tile/core.hpp"
#include "hstu_block_masking.hpp"

// HSTU attention backward — kernel layer (DESIGN §1.1).
//
// 3-kernel pipeline (sequence-conditional launch):
//   [PRE]  D[sq]=rowsum(O.*dO)            -- softmax path only        (M5)
//   [MAIN] HstuAttentionBwdDQDKDVKernel   -- 5 GEMM/7 stage, dV/dK + float dq_acc  (this file)
//   [POST] hstu_bwd_convert_dq_kernel     -- dq_acc(float) -> dQ(bf16/fp16)         (this file)
//
// launch order (SiLU): MAIN -> POST.
//
// M1: MAIN is a thin HSTU wrapper around the FMHA bwd kernel body (batched, no
// bias/dropout/group), differing from FMHA in that it carries TWO scalars
// (alpha, scale_p) to the SiLU pipeline instead of FMHA's single raw_scale/scale.
// POST is the atomic-path convert-only (single dq_acc, nsplits=1); the
// reduce+convert deterministic path (BlockFmhaBwdConvertQGrad) is M6.
//
//   TODO(M2): mask geometry (GenericAttentionMask<false> -> HSTU 5-factor mask).
//   TODO(M3/M4): jagged / group indexing.
//   TODO(M5): PRE (HstuAttentionBwdOGradDotO) + LSE read for softmax path.
//   TODO(M6): deterministic POST (reduce+convert over dq_acc splits).

namespace ck_tile {

// HSTU bwd MAIN kernel: batched, SiLU, no bias/dropout/group (M1 scope).
// Mirrors FMHA FmhaBwdDQDKDVKernel::operator() window setup, but passes
// (alpha, scale_p) to the HSTU SiLU pipeline and uses HSTU plain kargs.
template <typename HstuPipeline_, typename KGradEpiloguePipeline_, typename VGradEpiloguePipeline_>
struct HstuAttentionBwdDQDKDVKernel
{
    using HstuPipeline          = remove_cvref_t<HstuPipeline_>;
    using KGradEpiloguePipeline = remove_cvref_t<KGradEpiloguePipeline_>;
    using VGradEpiloguePipeline = remove_cvref_t<VGradEpiloguePipeline_>;

    static constexpr index_t kBlockSize  = HstuPipeline::kBlockSize;
    static constexpr index_t kBlockPerCu = HstuPipeline::kBlockPerCu;

    using QDataType     = remove_cvref_t<typename HstuPipeline::QDataType>;
    using KDataType     = remove_cvref_t<typename HstuPipeline::KDataType>;
    using VDataType     = remove_cvref_t<typename HstuPipeline::VDataType>;
    using AccDataType   = remove_cvref_t<typename HstuPipeline::AccDataType>;
    using OGradDataType = remove_cvref_t<typename HstuPipeline::OGradDataType>;
    using KGradDataType = remove_cvref_t<typename HstuPipeline::KGradDataType>;
    using VGradDataType = remove_cvref_t<typename HstuPipeline::VGradDataType>;
    using FmhaMask      = remove_cvref_t<typename HstuPipeline::FmhaMask>;

    // ck_tile bwd: seqlen never padded (OOB via buffer_load); headdim pad is index_t (0/8/1)
    static constexpr index_t kPadHeadDimQ  = HstuPipeline::kPadHeadDimQ;
    static constexpr index_t kPadHeadDimV  = HstuPipeline::kPadHeadDimV;
    static constexpr bool kIsDeterministic = HstuPipeline::kIsDeterministic;

    struct Kargs
    {
        const void* q_ptr;
        const void* k_ptr;
        const void* v_ptr;
        const void* do_ptr;
        void* dk_ptr;
        void* dv_ptr;
        void* dq_acc_ptr; // float

        index_t seqlen_q;
        index_t seqlen_kv;
        index_t hdim_qk;
        index_t hdim_v;
        index_t nhead_ratio_qk;

        float alpha;
        float scale_p;

        // HSTU mask params (M2). num_targets_ptr is per-batch (int32); null => num_target=0.
        const void* num_targets_ptr;
        index_t contextual_seqlen;
        index_t max_attn_len; // == window_size (local_len)
        index_t min_full_attn_seqlen;

        index_t stride_q;
        index_t stride_k;
        index_t stride_v;
        index_t stride_do;
        index_t stride_dk;
        index_t stride_dv;
        index_t stride_dq_acc;

        index_t nhead_stride_q;
        index_t nhead_stride_k;
        index_t nhead_stride_v;
        index_t nhead_stride_do;
        index_t nhead_stride_dk;
        index_t nhead_stride_dv;
        index_t nhead_stride_dq_acc;

        index_t batch_stride_q;
        index_t batch_stride_k;
        index_t batch_stride_v;
        index_t batch_stride_do;
        index_t batch_stride_dk;
        index_t batch_stride_dv;
        index_t batch_stride_dq_acc;
    };

    CK_TILE_HOST static constexpr Kargs MakeKargs(const void* q_ptr,
                                                  const void* k_ptr,
                                                  const void* v_ptr,
                                                  const void* do_ptr,
                                                  void* dk_ptr,
                                                  void* dv_ptr,
                                                  void* dq_acc_ptr,
                                                  index_t seqlen_q,
                                                  index_t seqlen_kv,
                                                  index_t hdim_qk,
                                                  index_t hdim_v,
                                                  index_t nhead_ratio_qk,
                                                  float alpha,
                                                  float scale_p,
                                                  const void* num_targets_ptr,
                                                  index_t contextual_seqlen,
                                                  index_t max_attn_len,
                                                  index_t min_full_attn_seqlen,
                                                  index_t stride_q,
                                                  index_t stride_k,
                                                  index_t stride_v,
                                                  index_t stride_do,
                                                  index_t stride_dk,
                                                  index_t stride_dv,
                                                  index_t stride_dq_acc,
                                                  index_t nhead_stride_q,
                                                  index_t nhead_stride_k,
                                                  index_t nhead_stride_v,
                                                  index_t nhead_stride_do,
                                                  index_t nhead_stride_dk,
                                                  index_t nhead_stride_dv,
                                                  index_t nhead_stride_dq_acc,
                                                  index_t batch_stride_q,
                                                  index_t batch_stride_k,
                                                  index_t batch_stride_v,
                                                  index_t batch_stride_do,
                                                  index_t batch_stride_dk,
                                                  index_t batch_stride_dv,
                                                  index_t batch_stride_dq_acc)
    {
        Kargs k;
        k.q_ptr               = q_ptr;
        k.k_ptr               = k_ptr;
        k.v_ptr               = v_ptr;
        k.do_ptr              = do_ptr;
        k.dk_ptr              = dk_ptr;
        k.dv_ptr              = dv_ptr;
        k.dq_acc_ptr          = dq_acc_ptr;
        k.seqlen_q            = seqlen_q;
        k.seqlen_kv           = seqlen_kv;
        k.hdim_qk             = hdim_qk;
        k.hdim_v              = hdim_v;
        k.nhead_ratio_qk      = nhead_ratio_qk;
        k.alpha               = alpha;
        k.scale_p             = scale_p;
        k.num_targets_ptr     = num_targets_ptr;
        k.contextual_seqlen   = contextual_seqlen;
        k.max_attn_len        = max_attn_len;
        k.min_full_attn_seqlen = min_full_attn_seqlen;
        k.stride_q            = stride_q;
        k.stride_k            = stride_k;
        k.stride_v            = stride_v;
        k.stride_do           = stride_do;
        k.stride_dk           = stride_dk;
        k.stride_dv           = stride_dv;
        k.stride_dq_acc       = stride_dq_acc;
        k.nhead_stride_q      = nhead_stride_q;
        k.nhead_stride_k      = nhead_stride_k;
        k.nhead_stride_v      = nhead_stride_v;
        k.nhead_stride_do     = nhead_stride_do;
        k.nhead_stride_dk     = nhead_stride_dk;
        k.nhead_stride_dv     = nhead_stride_dv;
        k.nhead_stride_dq_acc = nhead_stride_dq_acc;
        k.batch_stride_q      = batch_stride_q;
        k.batch_stride_k      = batch_stride_k;
        k.batch_stride_v      = batch_stride_v;
        k.batch_stride_do     = batch_stride_do;
        k.batch_stride_dk     = batch_stride_dk;
        k.batch_stride_dv     = batch_stride_dv;
        k.batch_stride_dq_acc = batch_stride_dq_acc;
        return k;
    }

    CK_TILE_HOST static constexpr auto
    GridSize(index_t batch_size, index_t nhead, index_t seqlen_kv)
    {
        return dim3(integer_divide_ceil(seqlen_kv, HstuPipeline::kN0), nhead, batch_size);
    }

    CK_TILE_DEVICE static constexpr auto GetTileIndex()
    {
        return make_tuple(static_cast<index_t>(blockIdx.x),
                          static_cast<index_t>(blockIdx.y),
                          static_cast<index_t>(blockIdx.z));
    }

    CK_TILE_HOST static constexpr auto BlockSize() { return dim3(kBlockSize); }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return max(HstuPipeline::GetSmemSize(),
                   KGradEpiloguePipeline::GetSmemSize(),
                   VGradEpiloguePipeline::GetSmemSize());
    }

    CK_TILE_DEVICE void operator()(Kargs kargs) const
    {
        __shared__ char smem_ptr[GetSmemSize()];

        const auto [i_tile_n, i_nhead, i_batch] = GetTileIndex();
        const index_t i_n0 = __builtin_amdgcn_readfirstlane(i_tile_n * HstuPipeline::kN0);

        // Per-(batch) base offsets: i_batch*batch_stride.
        const long_index_t batch_offset_q  = static_cast<long_index_t>(i_batch) * kargs.batch_stride_q;
        const long_index_t batch_offset_k  = static_cast<long_index_t>(i_batch) * kargs.batch_stride_k;
        const long_index_t batch_offset_v  = static_cast<long_index_t>(i_batch) * kargs.batch_stride_v;
        const long_index_t batch_offset_do = static_cast<long_index_t>(i_batch) * kargs.batch_stride_do;
        const long_index_t batch_offset_dk = static_cast<long_index_t>(i_batch) * kargs.batch_stride_dk;
        const long_index_t batch_offset_dv = static_cast<long_index_t>(i_batch) * kargs.batch_stride_dv;
        const long_index_t batch_offset_dq_acc =
            static_cast<long_index_t>(i_batch) * kargs.batch_stride_dq_acc;

        const QDataType* q_ptr = reinterpret_cast<const QDataType*>(kargs.q_ptr) +
                                 static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_q +
                                 batch_offset_q;
        const KDataType* k_ptr =
            reinterpret_cast<const KDataType*>(kargs.k_ptr) +
            static_cast<long_index_t>(i_nhead / kargs.nhead_ratio_qk) * kargs.nhead_stride_k +
            batch_offset_k;
        const VDataType* v_ptr =
            reinterpret_cast<const VDataType*>(kargs.v_ptr) +
            static_cast<long_index_t>(i_nhead / kargs.nhead_ratio_qk) * kargs.nhead_stride_v +
            batch_offset_v;
        const OGradDataType* do_ptr = reinterpret_cast<const OGradDataType*>(kargs.do_ptr) +
                                      static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_do +
                                      batch_offset_do;
        KGradDataType* dk_ptr = reinterpret_cast<KGradDataType*>(kargs.dk_ptr) +
                                static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_dk +
                                batch_offset_dk;
        VGradDataType* dv_ptr = reinterpret_cast<VGradDataType*>(kargs.dv_ptr) +
                                static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_dv +
                                batch_offset_dv;

        // Q/K/V/dO DRAM views + windows
        const auto q_dram = pad_tensor_view(
            make_naive_tensor_view<address_space_enum::global>(
                q_ptr,
                make_tuple(kargs.seqlen_q, kargs.hdim_qk),
                make_tuple(kargs.stride_q, 1),
                number<HstuPipeline::kAlignmentQ>{},
                number<1>{}),
            make_tuple(number<HstuPipeline::kM0>{}, number<HstuPipeline::kQKHeaddim>{}),
            sequence<false, (kPadHeadDimQ > 0)>{});

        const auto k_dram = pad_tensor_view(
            make_naive_tensor_view<address_space_enum::global>(
                k_ptr,
                make_tuple(kargs.seqlen_kv, kargs.hdim_qk),
                make_tuple(kargs.stride_k, 1),
                number<HstuPipeline::kAlignmentK>{},
                number<1>{}),
            make_tuple(number<HstuPipeline::kN0>{}, number<HstuPipeline::kQKHeaddim>{}),
            sequence<false, (kPadHeadDimQ > 0)>{});

        const auto v_dram = pad_tensor_view(
            make_naive_tensor_view<address_space_enum::global>(
                v_ptr,
                make_tuple(kargs.seqlen_kv, kargs.hdim_v),
                make_tuple(kargs.stride_v, 1),
                number<HstuPipeline::kAlignmentV>{},
                number<1>{}),
            make_tuple(number<HstuPipeline::kN0>{}, number<HstuPipeline::kVHeaddim>{}),
            sequence<false, (kPadHeadDimV > 0)>{});

        const auto do_dram = pad_tensor_view(
            make_naive_tensor_view<address_space_enum::global>(
                do_ptr,
                make_tuple(kargs.seqlen_q, kargs.hdim_v),
                make_tuple(kargs.stride_do, 1),
                number<HstuPipeline::kAlignmentOGrad>{},
                number<1>{}),
            make_tuple(number<HstuPipeline::kM0>{}, number<HstuPipeline::kVHeaddim>{}),
            sequence<false, (kPadHeadDimV > 0)>{});

        auto q_dram_window = make_tile_window(
            q_dram, make_tuple(number<HstuPipeline::kM0>{}, number<HstuPipeline::kQKHeaddim>{}),
            {0, 0});
        auto k_dram_window = make_tile_window(
            k_dram, make_tuple(number<HstuPipeline::kN0>{}, number<HstuPipeline::kQKHeaddim>{}),
            {i_n0, 0});
        auto v_dram_window = make_tile_window(
            v_dram, make_tuple(number<HstuPipeline::kN0>{}, number<HstuPipeline::kVHeaddim>{}),
            {i_n0, 0});
        auto do_dram_window = make_tile_window(
            do_dram, make_tuple(number<HstuPipeline::kM0>{}, number<HstuPipeline::kVHeaddim>{}),
            {0, 0});

        // dQ_acc atomic_add window (M1: atomic path, nsplits=1)
        AccDataType* dq_acc_ptr =
            reinterpret_cast<AccDataType*>(kargs.dq_acc_ptr) +
            static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_dq_acc + batch_offset_dq_acc;

        auto dq_acc_dram = pad_tensor_view(
            make_naive_tensor_view<address_space_enum::global, memory_operation_enum::atomic_add>(
                dq_acc_ptr,
                make_tuple(kargs.seqlen_q, kargs.hdim_qk),
                make_tuple(kargs.stride_dq_acc, 1),
                number<HstuPipeline::kAlignmentQGrad>{},
                number<1>{}),
            make_tuple(number<HstuPipeline::kM0>{}, number<HstuPipeline::kQKHeaddim>{}),
            sequence<false, (kPadHeadDimQ > 0)>{});

        auto dq_dram_window = make_tile_window(
            dq_acc_dram, make_tuple(number<HstuPipeline::kM0>{}, number<HstuPipeline::kQKHeaddim>{}),
            {0, 0});

        // Build the HSTU mask identically to fwd/reference (self-attention; M2 batched).
        // is_tile_in_first_split=true (conservative: disables the IsFullTileInsideMask
        // fast-path so every edge tile is per-pixel checked; the tile-level first-split
        // optimization is a later perf item, and IsTokenPairInsideMask is self-contained).
        const int num_target =
            (kargs.num_targets_ptr != nullptr)
                ? reinterpret_cast<const int32_t*>(kargs.num_targets_ptr)[i_batch]
                : 0;
        auto mask = [&]() {
            if constexpr(FmhaMask::kUseLocal)
            {
                // clamp min_full like reference (reference_hstu_attention_bwd.hpp:177/198)
                const int eff_min_full =
                    (kargs.seqlen_q - num_target > kargs.min_full_attn_seqlen)
                        ? kargs.min_full_attn_seqlen
                        : (kargs.seqlen_q - num_target);
                return make_hstu_self_attention_block_mask_with_local<FmhaMask>(
                    /*is_tile_in_first_split=*/true,
                    kargs.seqlen_q,
                    kargs.contextual_seqlen,
                    num_target,
                    kargs.max_attn_len,
                    eff_min_full);
            }
            else
            {
                return make_hstu_self_attention_block_mask_without_local<FmhaMask>(
                    kargs.seqlen_q, kargs.contextual_seqlen, num_target);
            }
        }();

        auto [dk_acc_tile, dv_acc_tile] = HstuPipeline{}(q_dram_window,
                                                         k_dram_window,
                                                         v_dram_window,
                                                         do_dram_window,
                                                         dq_dram_window,
                                                         mask,
                                                         kargs.alpha,
                                                         kargs.scale_p,
                                                         smem_ptr);

        auto dk_dram = pad_tensor_view(
            make_naive_tensor_view<address_space_enum::global>(
                dk_ptr,
                make_tuple(kargs.seqlen_kv, kargs.hdim_qk),
                make_tuple(kargs.stride_dk, 1),
                number<HstuPipeline::kAlignmentKGrad>{},
                number<1>{}),
            make_tuple(number<HstuPipeline::kN0>{}, number<HstuPipeline::kQKHeaddim>{}),
            sequence<false, (kPadHeadDimQ > 0)>{});

        auto dv_dram = pad_tensor_view(
            make_naive_tensor_view<address_space_enum::global>(
                dv_ptr,
                make_tuple(kargs.seqlen_kv, kargs.hdim_v),
                make_tuple(kargs.stride_dv, 1),
                number<HstuPipeline::kAlignmentVGrad>{},
                number<1>{}),
            make_tuple(number<HstuPipeline::kN0>{}, number<HstuPipeline::kVHeaddim>{}),
            sequence<false, (kPadHeadDimV > 0)>{});

        auto dk_dram_window = make_tile_window(
            dk_dram, make_tuple(number<HstuPipeline::kN0>{}, number<HstuPipeline::kQKHeaddim>{}),
            {i_n0, 0});
        auto dv_dram_window = make_tile_window(
            dv_dram, make_tuple(number<HstuPipeline::kN0>{}, number<HstuPipeline::kVHeaddim>{}),
            {i_n0, 0});

        KGradEpiloguePipeline{}(dk_dram_window, dk_acc_tile, nullptr);
        VGradEpiloguePipeline{}(dv_dram_window, dv_acc_tile, nullptr);
    }
};

// POST (atomic path): convert-only dq_acc(float) -> dQ(bf16/fp16).
// M1 atomic path: nsplits=1 and dq_acc shares dQ's layout, so the convert is a
// pure elementwise cast over the full contiguous buffer. Templated so it has
// vague linkage (safe to include in multiple TUs). The deterministic
// reduce+convert path (BlockFmhaBwdConvertQGrad) is M6.
template <typename QGradDataType, typename AccDataType>
__global__ void hstu_bwd_convert_dq_kernel(const AccDataType* __restrict__ dq_acc,
                                           QGradDataType* __restrict__ dq,
                                           long_index_t n)
{
    const long_index_t i =
        static_cast<long_index_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(i < n)
        dq[i] = type_convert<QGradDataType>(dq_acc[i]);
}

} // namespace ck_tile
