// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <hip/hip_runtime.h>
#include "ck_tile/core.hpp"

// HSTU attention backward — kernel layer (DESIGN §1.1).
//
// M0 scaffold: this header declares the 3-kernel pipeline structure only. The
// real kernel bodies are wired in later milestones (the M0 dispatch is a
// hipMemsetAsync stub that zeroes dQ/dK/dV — it does NOT launch a MAIN kernel).
//
// 3-kernel pipeline (sequence-conditional launch):
//   [PRE]  D[sq]=rowsum(O.*dO)            -- softmax path only        (M5)
//   [MAIN] HstuAttentionBwdDQDKDVKernel   -- 5 GEMM/7 stage, dV/dK + float dq_acc  (M1)
//   [POST] hstu_bwd_convert_dq_kernel     -- dq_acc(float) -> dQ(bf16/fp16)         (M1)
//
// launch order (SiLU): MAIN -> POST.
//
//   TODO(M1): MAIN (thin HSTU wrapper around FMHA bwd body, carrying alpha+scale_p)
//             + POST (atomic-path convert-only dq_acc(float) -> dQ).
//   TODO(M5): PRE (HstuAttentionBwdOGradDotO) + LSE read for softmax path.
//   TODO(M6): deterministic POST (reduce+convert over dq_acc splits).

namespace ck_tile {

// TODO(M1): struct HstuAttentionBwdDQDKDVKernel<HstuPipeline, KGradEpi, VGradEpi>
//           (batched, SiLU, no bias/dropout/group). plain kargs carry TWO scalars
//           (alpha, scale_p) to the HSTU SiLU pipeline.

// TODO(M1): __global__ void hstu_bwd_convert_dq_kernel(dq_acc(float) -> dQ).

} // namespace ck_tile
