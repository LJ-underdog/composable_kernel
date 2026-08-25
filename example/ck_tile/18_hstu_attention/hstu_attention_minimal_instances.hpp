// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <cstdio>

// Fast-iteration build support. Enabled by the CMake option HSTU_FWD_MINIMAL_INSTANCES,
// which also narrows instances/*_forward_*.cpp to the softmax=false + dropout=false subset
// (48 of 288). Without the compile-time filter below the fwd interface TUs would still
// instantiate — and therefore reference — the softmax/dropout dispatch symbols, and the
// link would fail with undefined references (BOOL_SWITCH_* is a runtime `if`, so both
// branches are instantiated; see hstu_attention_bool_switch.hpp:8,:22,:36).
//
// Semantics: this only *removes* instantiations. The branch that is selected for a given
// input is byte-for-byte the same template instance as in a full build. When the option is
// OFF (default) hstu_fwd_keep_instance() is constant true and the preprocessed result is
// identical to the original code.
//
// HSTU_FWD_NO_SOFTMAX_INSTANCES is the wider variant of the same mechanism: it keeps every
// softmax=false instance including the dropout ones (96 of 288), and only drops the softmax
// branch. Use it to validate the full no-softmax (SiLU) surface while the softmax instances
// are still broken on gfx1250. The two options are mutually exclusive; MINIMAL wins.

template <bool kUseSoftmax, bool kHasDropout>
constexpr bool hstu_fwd_keep_instance()
{
#if defined(HSTU_FWD_MINIMAL_INSTANCES)
    return (!kUseSoftmax) && (!kHasDropout);
#elif defined(HSTU_FWD_NO_SOFTMAX_INSTANCES)
    (void)kHasDropout;
    return !kUseSoftmax;
#else
    return true;
#endif
}

// Runtime guard: in minimal mode the softmax / dropout kernels are not compiled in, so such
// a request must fail loudly instead of silently falling through to the SiLU / no-dropout
// path. Returns true when the request cannot be served.
static inline bool hstu_fwd_minimal_reject(bool use_softmax, float p_drop, const char* who)
{
#if defined(HSTU_FWD_MINIMAL_INSTANCES)
    if(use_softmax || p_drop > 0.0f)
    {
        std::fprintf(stderr,
                     "[%s] this binary was built with HSTU_FWD_MINIMAL_INSTANCES=ON: the "
                     "softmax and dropout kernels are NOT compiled in (requested "
                     "use_softmax=%d p_drop=%g). Rebuild with "
                     "-DHSTU_FWD_MINIMAL_INSTANCES=OFF to use them.\n",
                     who,
                     static_cast<int>(use_softmax),
                     static_cast<double>(p_drop));
        return true;
    }
    return false;
#elif defined(HSTU_FWD_NO_SOFTMAX_INSTANCES)
    (void)p_drop;
    if(use_softmax)
    {
        std::fprintf(stderr,
                     "[%s] this binary was built with HSTU_FWD_NO_SOFTMAX_INSTANCES=ON: the "
                     "softmax kernels are NOT compiled in (requested use_softmax=%d). Rebuild "
                     "with -DHSTU_FWD_NO_SOFTMAX_INSTANCES=OFF to use them.\n",
                     who,
                     static_cast<int>(use_softmax));
        return true;
    }
    return false;
#else
    (void)use_softmax;
    (void)p_drop;
    (void)who;
    return false;
#endif
}
