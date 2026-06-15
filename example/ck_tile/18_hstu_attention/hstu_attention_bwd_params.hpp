// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <ck_tile/core.hpp>

// HSTU attention backward params (canonical field set, DESIGN §4.6).
// Two structs: NoGroup (batch + jagged, distinguished by is_jagged) and Group.
// Mirrors the fwd params layout (hstu_attention_params.hpp), reusing the full fwd
// input field set and adding the bwd-specific inputs / outputs / workspace.
//
// M0: only the NoGroup (batched, SiLU, bf16) path is exercised. The Group struct
// is declared but left to be filled in M4. All field names are fixed now so M1+
// only adds device-side wiring, never a params refactor.

struct HstuAttentionNoGroupBwdParams
{
    // ---- reused fwd inputs ---------------------------------------------------
    // for self-attention (is_cross_attention = false), we require
    // 1) either seqlen_kv == 0 or seqlen_kv == seqlen_q
    // 2) either seq_kv_offsets_ptr == nullptr, or seq_kv_offsets_ptr == seq_q_offsets_ptr
    bool is_cross_attention;
    bool is_jagged;

    ck_tile::index_t num_batch;
    ck_tile::index_t seqlen_q;      // batched mode only
    ck_tile::index_t seqlen_kv;     // batched mode only
    const void* seq_q_offsets_ptr;  // jagged mode only
    const void* seq_kv_offsets_ptr; // jagged mode only
    ck_tile::index_t max_seqlen_q;  // jagged mode only
    // cross-attention KV grid sizing (bwd is KV-block-parallel). self path: caller sets
    // max_seqlen_kv == max_seqlen_q -> grid/num_splits unchanged -> byte-identical.
    ck_tile::index_t max_seqlen_kv; // jagged mode only

    const void* q_ptr;
    const void* k_ptr;
    const void* v_ptr;
    const void* o_ptr; // softmax path only (PRE computes D = rowsum(O .* dO))

    ck_tile::index_t hdim_qk;
    ck_tile::index_t hdim_v;
    ck_tile::index_t num_head;
    ck_tile::index_t nhead_ratio_qk; // HSTU is MHA for now (=1); GQA/MQA placeholder (U4)

    // ---- scales (DESIGN §4.7) ------------------------------------------------
    float alpha;      // QK scaling (== fwd scale_s); FMHA raw_scale slot
    float attn_scale; // source of scale_p (SiLU output scaling); softmax path unused

    // ---- mask hyper-params (batch/jagged: scalars) ---------------------------
    bool use_causal;
    bool use_softmax;
    ck_tile::index_t window_size;
    ck_tile::index_t contextual_seqlen;
    ck_tile::index_t min_full_attn_seqlen;
    const void* num_targets_ptr;

    // ---- input strides (Q/K/V/O reuse fwd layout) ---------------------------
    ck_tile::index_t seq_stride_q;
    ck_tile::index_t seq_stride_k;
    ck_tile::index_t seq_stride_v;
    ck_tile::index_t seq_stride_o;
    ck_tile::index_t nhead_stride_q;
    ck_tile::index_t nhead_stride_k;
    ck_tile::index_t nhead_stride_v;
    ck_tile::index_t nhead_stride_o;
    ck_tile::index_t batch_stride_q;
    ck_tile::index_t batch_stride_k;
    ck_tile::index_t batch_stride_v;
    ck_tile::index_t batch_stride_o;

    // ---- bwd input: dO + (softmax) LSE --------------------------------------
    const void* do_ptr;
    ck_tile::index_t seq_stride_do;
    ck_tile::index_t nhead_stride_do;
    ck_tile::index_t batch_stride_do;

    const void* lse_ptr; // softmax path only; SiLU = nullptr

    // ---- bwd outputs: dQ/dK/dV ----------------------------------------------
    // dq/dk are hdim_qk-wide, dv is hdim_v-wide
    void* dq_ptr;
    void* dk_ptr;
    void* dv_ptr;
    ck_tile::index_t seq_stride_dq;
    ck_tile::index_t seq_stride_dk;
    ck_tile::index_t seq_stride_dv;
    ck_tile::index_t nhead_stride_dq;
    ck_tile::index_t nhead_stride_dk;
    ck_tile::index_t nhead_stride_dv;
    ck_tile::index_t batch_stride_dq;
    ck_tile::index_t batch_stride_dk;
    ck_tile::index_t batch_stride_dv;

    // ---- PRE products (softmax path only): D, and LSE/D share a layout ------
    void* d_ptr;
    ck_tile::index_t nhead_stride_lsed;
    ck_tile::index_t batch_stride_lsed;

    // ---- dQ workspace (float dq_acc, both atomic and deterministic) ----------
    void* dq_acc_ptr;
    ck_tile::index_t stride_dq_acc;
    ck_tile::index_t nhead_stride_dq_acc;
    ck_tile::index_t batch_stride_dq_acc;
    ck_tile::index_t split_stride_dq_acc;
    int num_splits; // atomic path nsplits=1; deterministic path ceil(max_seqlen_k/kN0)

    // ---- switch (template axis; kept as bool so host can pick instance) ------
    bool kIsDeterministic;

    // ---- M8 MI: per-kernel perf (HOST-ONLY; NOT passed to MakeKargs) ---------
    // measure_perf gates a hipEvent timing path in the dispatch. When false the
    // dispatch runs each kernel exactly once (byte-identical device code + host
    // behavior). When true the dispatch fills the perf_*_ms outputs (mean per-launch
    // ms over warmup+repeat). These never reach the device, so device symbols are
    // byte-identical to a build without MI.
    bool measure_perf      = false;
    float perf_pre_ms      = 0.f; // PRE dot_do_o (softmax only; 0 on SiLU)
    float perf_memset_ms   = 0.f; // ZERO_dq_acc memset
    float perf_main_ms     = 0.f; // MAIN dqdkdv (bottleneck; B2/B3 target)
    float perf_post_ms     = 0.f; // POST convert_dq / reduce_convert_dq
};

// Group bwd params (DESIGN §4.6 group row, §4.7 D6). M4.
//
// group = jagged superset: same dim0=1 token-major packed tensors + cu_seqlens
// (offset indexing reused from M3), PLUS per-group hyper-params indexed by
// i_group = i_batch / num_batch_per_group. `alpha` stays a GLOBAL scalar (D6);
// scale_p + the 4 mask hyper-params become per-group device pointers; num_target
// stays per-batch. Field naming mirrors HstuAttentionGroupFwdParams.
struct HstuAttentionGroupBwdParams
{
    bool is_cross_attention; // M4: self-attention only (kv offsets == q offsets)

    ck_tile::index_t num_group;
    ck_tile::index_t num_batch;
    const void* seq_q_offsets_ptr;  // int32, size num_batch+1 (token-major packed)
    const void* seq_kv_offsets_ptr; // int32, size num_batch+1
    ck_tile::index_t max_seqlen_q;  // max over all groups' max_seqlen_q (grid sizing)
    // cross-attention KV grid sizing (bwd is KV-block-parallel; max over all groups'
    // max_seqlen_kv). self path: caller sets == max_seqlen_q -> byte-identical.
    ck_tile::index_t max_seqlen_kv;

    const void* q_ptr;
    const void* k_ptr;
    const void* v_ptr;
    const void* o_ptr; // softmax path only

    ck_tile::index_t hdim_qk;
    ck_tile::index_t hdim_v;
    ck_tile::index_t num_head;
    ck_tile::index_t nhead_ratio_qk; // =1 (HSTU MHA; GQA/MQA placeholder, U4)

    // ---- global scale (alpha is single-valued in all three modes, D6) --------
    float alpha;

    // ---- mask switches / per-group hyper-params (device pointers) ------------
    bool use_causal;  // uniform across groups (compile-time axis in dispatch)
    bool use_softmax; // M5
    const void* num_targets_ptr; // per-batch int32 (== num_targets_ptr[i_batch])
    const void* group_attn_scale_ptr;          // float[num_group]; scale_p source
    const void* group_max_seqlen_q_ptr;        // int32[num_group]; scale_p fallback
    const void* group_window_size_ptr;         // int32[num_group]
    const void* group_contextual_seqlen_ptr;   // int32[num_group]
    const void* group_min_full_attn_seqlen_ptr;// int32[num_group]

    // ---- input strides (packed token-major; batch_stride = dim0 stride) ------
    ck_tile::index_t seq_stride_q;
    ck_tile::index_t seq_stride_k;
    ck_tile::index_t seq_stride_v;
    ck_tile::index_t seq_stride_o;
    ck_tile::index_t nhead_stride_q;
    ck_tile::index_t nhead_stride_k;
    ck_tile::index_t nhead_stride_v;
    ck_tile::index_t nhead_stride_o;

    // ---- bwd input: dO -------------------------------------------------------
    const void* do_ptr;
    ck_tile::index_t seq_stride_do;
    ck_tile::index_t nhead_stride_do;

    const void* lse_ptr; // softmax path only (M5b); [head, ΣL] seq-continuous packed

    // ---- softmax PRE output (M5b): D = rowsum(O*dO), same [head,ΣL] layout as LSE ----
    void* d_ptr;                          // softmax path only
    ck_tile::index_t nhead_stride_lsed;   // == ΣL (group packed; LSE/D seq stride is 1)

    // ---- bwd outputs: dQ/dK/dV ----------------------------------------------
    void* dq_ptr;
    void* dk_ptr;
    void* dv_ptr;
    ck_tile::index_t seq_stride_dq;
    ck_tile::index_t seq_stride_dk;
    ck_tile::index_t seq_stride_dv;
    ck_tile::index_t nhead_stride_dq;
    ck_tile::index_t nhead_stride_dk;
    ck_tile::index_t nhead_stride_dv;

    // ---- dQ workspace (float dq_acc) ----------------------------------------
    void* dq_acc_ptr;
    ck_tile::index_t stride_dq_acc;       // token stride (== H*hdim_qk)
    ck_tile::index_t nhead_stride_dq_acc; // == hdim_qk
    ck_tile::index_t total_dq_acc_elems;  // single packed slot = ΣL*H*hdim_qk
    // M6b deterministic: dq_acc has num_splits stacked single-slots; split_stride = one slot.
    ck_tile::index_t split_stride_dq_acc; // == total_dq_acc_elems (determ); 0 (atomic)
    int num_splits;                       // determ: ceil(max_seqlen_q/kN0); atomic: 1

    bool kIsDeterministic; // M6

    // ---- M8 MI: per-kernel perf (HOST-ONLY; NOT passed to MakeKargs; see NoGroup) ----
    bool measure_perf      = false;
    float perf_pre_ms      = 0.f;
    float perf_memset_ms   = 0.f;
    float perf_main_ms     = 0.f;
    float perf_post_ms     = 0.f;
};
