// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#include <ck_tile/core.hpp>
#include <stdexcept>

#include "hstu_attention_bool_switch.hpp"
#include "hstu_attention_bwd_params.hpp"
#include "hstu_attention_batched_backward_dispatch.hpp"

#include "instances/hstu_attention_batched_backward_fp16_instances_ref.hpp"

// HSTU attention backward — NoGroup (batched + jagged) fp16 entry point (M7a).
//
// Mirror of hstu_attention_no_group_backward_bf16.cpp with ck_tile::bf16_t ->
// ck_tile::fp16_t. The dispatch/kernel/pipeline are templated on InOutDataType,
// so fp16 reuses the exact bf16 code path (same causal/softmax/determ fan-out,
// hd64). M7a only widens the dtype axis; hdim variants remain M7b.
void hstu_attention_no_group_backward_fp16(HstuAttentionNoGroupBwdParams& param, hipStream_t stream)
{
    const bool use_causal = param.use_causal;

    // M6: kIsDeterministic is now a runtime-selected template axis (causal × softmax × determ).
    BOOL_SWITCH_3(use_causal, kUseCausal, param.use_softmax, kUseSoftmax,
                  param.kIsDeterministic, kIsDeterministic, [&] {
        run_batched_backward_dispatch<ck_tile::fp16_t,
                                      kUseCausal,
                                      kUseSoftmax,
                                      false, // kHasBias
                                      kIsDeterministic,
                                      64>    // MaxK
            (param, stream);
    });
}
