
// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

// The file is automatically generated, don't modify!
// See the generator script
// `ck_hstu/example/ck_tile/18_hstu_attention/generate_instances.py`

#include <ck_tile/core/numeric/half.hpp>
#include "hstu_attention_batched_backward_dispatch.hpp"

template void run_batched_backward_dispatch<
    ck_tile::fp16_t,
    false,
    true,
    false,
    false,
    256>(HstuAttentionNoGroupBwdParams& param, hipStream_t stream);
