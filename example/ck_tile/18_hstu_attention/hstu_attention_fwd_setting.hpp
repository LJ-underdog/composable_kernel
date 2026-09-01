// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <ck_tile/core/numeric/integer.hpp>

#include "hstu_attention_fwd_setting_gfx94.hpp"
#include "hstu_attention_fwd_setting_gfx95.hpp"
#include "hstu_attention_fwd_setting_gfx125.hpp"

// Which forward pipeline a dispatch instance selects. Replaces the previous pair of
// use_trload_pipeline / use_tdm_pipeline static members: those two bools encoded three
// states, and nothing stopped them from drifting apart between dispatch files.
enum class HstuFwdPipelineKind
{
    Default, // HstuAttention{With,No}SoftmaxFwdPipelineQRKSVS
    TrLoad,  // HstuAttention{With,No}SoftmaxFwdPipelineQRKSVSTrLoad
    Tdm,     // HstuAttentionNoSoftmaxFwdPipelineQRKSVSTdm (gfx1250 only)
};

// Single source of truth for the forward pipeline choice, so that a dispatch site cannot
// pick up part of the gating and silently miss the rest.
//
// The gfx1250 TDM pipeline is preferred over trload where it applies. Everything it does
// not cover falls back to trload: softmax, MaxK != 128, and padded head dims.
// Head-dim padding is excluded because TDM clamps out-of-bound reads against the tensor
// view lengths, and the HSTU kernel wraps K/V in a pad_tensor_view whose lengths are
// already rounded up to a full tile - so the clamp lets the DMA walk past the end of the
// real K/V buffers.
//
// Keep this gating in sync with the static_asserts inside the tdm pipeline.
template <bool kUseSoftmax,
          bool kHasDropout,
          bool kPadHeadDimQK,
          bool kPadHeadDimV,
          ck_tile::index_t MaxK>
constexpr HstuFwdPipelineKind get_hstu_fwd_pipeline_kind()
{
#if defined(BUILD_HSTU_FOR_GFX125)
    if constexpr(!kUseSoftmax && MaxK == 128 && !kPadHeadDimQK && !kPadHeadDimV)
        return HstuFwdPipelineKind::Tdm;
    else
        return HstuFwdPipelineKind::TrLoad;
#elif defined(BUILD_HSTU_FOR_GFX95)
    return HstuFwdPipelineKind::TrLoad;
#else
    return HstuFwdPipelineKind::Default;
#endif
}
