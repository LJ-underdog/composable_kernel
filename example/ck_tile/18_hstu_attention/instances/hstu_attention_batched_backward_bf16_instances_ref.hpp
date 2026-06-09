
// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

// The file is automatically generated, don't modify!
// See the generator script
// `ck_hstu/example/ck_tile/18_hstu_attention/generate_instances.py`

#include <ck_tile/core/numeric/bfloat16.hpp>
#include "hstu_attention_batched_backward_dispatch.hpp"

extern template void run_batched_backward_dispatch<
    ck_tile::bf16_t,
    true,
    true,
    false,
    false,
    64>(HstuAttentionNoGroupBwdParams& param, hipStream_t stream);

extern template void run_batched_backward_dispatch<
    ck_tile::bf16_t,
    true,
    false,
    false,
    false,
    64>(HstuAttentionNoGroupBwdParams& param, hipStream_t stream);

extern template void run_batched_backward_dispatch<
    ck_tile::bf16_t,
    false,
    true,
    false,
    false,
    64>(HstuAttentionNoGroupBwdParams& param, hipStream_t stream);

extern template void run_batched_backward_dispatch<
    ck_tile::bf16_t,
    false,
    false,
    false,
    false,
    64>(HstuAttentionNoGroupBwdParams& param, hipStream_t stream);

extern template void run_batched_backward_dispatch<
    ck_tile::bf16_t,
    true,
    true,
    false,
    true,
    64>(HstuAttentionNoGroupBwdParams& param, hipStream_t stream);

extern template void run_batched_backward_dispatch<
    ck_tile::bf16_t,
    true,
    false,
    false,
    true,
    64>(HstuAttentionNoGroupBwdParams& param, hipStream_t stream);

extern template void run_batched_backward_dispatch<
    ck_tile::bf16_t,
    false,
    true,
    false,
    true,
    64>(HstuAttentionNoGroupBwdParams& param, hipStream_t stream);

extern template void run_batched_backward_dispatch<
    ck_tile::bf16_t,
    false,
    false,
    false,
    true,
    64>(HstuAttentionNoGroupBwdParams& param, hipStream_t stream);
