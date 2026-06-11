// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#include <ck_tile/core.hpp>

#include "hstu_attention_bool_switch.hpp"
#include "hstu_attention_hdim_switch.hpp"
#include "hstu_attention_bwd_params.hpp"
#include "hstu_attention_group_backward_dispatch.hpp"

// HSTU attention backward — GROUP fp16 entry point (M7a).
//
// Mirror of hstu_attention_group_backward_bf16.cpp with ck_tile::bf16_t ->
// ck_tile::fp16_t. Group instantiates the dispatch directly (no extern-template
// instance files), so widening the dtype here is sufficient — the M5b softmax /
// M6b determ template axes are unchanged.
void hstu_attention_group_backward_fp16(HstuAttentionGroupBwdParams& param, hipStream_t stream)
{
    // M7b: MaxK selected at runtime from hdim via HDIM_SWITCH (symmetric {64,96,128,256}).
    BOOL_SWITCH_3(param.use_causal, kUseCausal, param.use_softmax, kUseSoftmax,
                  param.kIsDeterministic, kIsDeterministic, [&] {
        HDIM_SWITCH(param.hdim_qk, param.hdim_v, MaxK, [&] {
            run_group_backward_dispatch<ck_tile::fp16_t,
                                        kUseCausal,
                                        kUseSoftmax,
                                        false, // kHasBias
                                        kIsDeterministic,
                                        MaxK>
                (param, stream);
        });
    });
}
