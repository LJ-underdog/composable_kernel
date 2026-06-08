// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#include <ck_tile/core.hpp>

#include "hstu_attention_bool_switch.hpp"
#include "hstu_attention_bwd_params.hpp"
#include "hstu_attention_group_backward_dispatch.hpp"

// HSTU attention backward — GROUP bf16 entry point (M4).
//
// Instantiates the group dispatch directly (no extern-template instance files):
// only the SiLU causal x {with-local, without-local} pipelines are compiled (the
// window branch is runtime). softmax/deterministic compile to a runtime throw.
void hstu_attention_group_backward_bf16(HstuAttentionGroupBwdParams& param, hipStream_t stream)
{
    BOOL_SWITCH_2(param.use_causal, kUseCausal, param.use_softmax, kUseSoftmax, [&] {
        run_group_backward_dispatch<ck_tile::bf16_t,
                                    kUseCausal,
                                    kUseSoftmax,
                                    false, // kHasBias
                                    false, // kIsDeterministic
                                    64>    // MaxK
            (param, stream);
    });
}
