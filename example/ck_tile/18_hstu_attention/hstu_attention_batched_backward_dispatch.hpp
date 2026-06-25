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

    static void RunSilu(HstuAttentionNoGroupBwdParams& param, hipStream_t stream)
    {
        // M0 scaffold: end-to-end plumbing only — zero dQ/dK/dV so the dispatch +
        // harness wiring can be exercised before the real SiLU MAIN kernel lands (M1).
        // Batched layout [num_batch, sq, h, hdim]; batch_stride_* is the per-batch
        // element count, so total elements = num_batch * batch_stride_*.
        const size_t nb = static_cast<size_t>(param.num_batch);

        const size_t dq_elems = nb * static_cast<size_t>(param.batch_stride_q);
        const size_t dk_elems = nb * static_cast<size_t>(param.batch_stride_k);
        const size_t dv_elems = nb * static_cast<size_t>(param.batch_stride_v);

        HIP_CHECK_ERROR(hipMemsetAsync(
            param.dq_ptr, 0, dq_elems * sizeof(InOutDataType), stream));
        HIP_CHECK_ERROR(hipMemsetAsync(
            param.dk_ptr, 0, dk_elems * sizeof(InOutDataType), stream));
        HIP_CHECK_ERROR(hipMemsetAsync(
            param.dv_ptr, 0, dv_elems * sizeof(InOutDataType), stream));
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
        else if constexpr(kUseCausal)
        {
            throw std::runtime_error("HSTU bwd: causal/mask path not implemented yet (M2)");
        }
        else
        {
            // M3: jagged (variable-length) not implemented yet.
            if(param.is_jagged)
                throw std::runtime_error("HSTU bwd: jagged path not implemented yet (M3)");

            // M1: SiLU + no-mask. hdim64 only. (seqlen never padded — OOB via buffer_load)
            if(param.hdim_qk != 64 || param.hdim_v != 64)
                throw std::runtime_error("HSTU bwd M1 supports hdim_qk=hdim_v=64 only (M7)");

            RunSilu(param, stream);
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
