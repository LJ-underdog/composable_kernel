// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#include <ck_tile/core.hpp>
#include <stdexcept>

#include "hstu_attention_bool_switch.hpp"
#include "hstu_attention_bwd_params.hpp"
#include "hstu_attention_batched_backward_dispatch.hpp"

#include "instances/hstu_attention_batched_backward_bf16_instances_ref.hpp"

// HSTU attention backward — NoGroup (batched + jagged) bf16 entry point.
//
// M0: only the batched + SiLU path is wired (single instance). The full
// BOOL_SWITCH / HDIM_SWITCH fan-out (causal/softmax/deterministic/maxk) and the
// jagged branch are added alongside the real kernels in M1+ (mirrors the fwd
// hstu_attention_no_group_forward_bf16.cpp structure).
void hstu_attention_no_group_backward_bf16(HstuAttentionNoGroupBwdParams& param, hipStream_t stream)
{
    const bool use_causal = param.use_causal;

    // M6: kIsDeterministic is now a runtime-selected template axis (causal × softmax × determ).
    BOOL_SWITCH_3(use_causal, kUseCausal, param.use_softmax, kUseSoftmax,
                  param.kIsDeterministic, kIsDeterministic, [&] {
        run_batched_backward_dispatch<ck_tile::bf16_t,
                                      kUseCausal,
                                      kUseSoftmax,
                                      false, // kHasBias
                                      kIsDeterministic,
                                      64>    // MaxK
            (param, stream);
    });
}
