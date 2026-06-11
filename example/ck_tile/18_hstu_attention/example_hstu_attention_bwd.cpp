// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

// HSTU attention BACKWARD example / correctness harness.
//
// Mirrors example_hstu_attention_fwd.cpp. Flow (DESIGN §5.1):
//   random seed -> Q/K/V/dO
//     GPU: fwd produces O (+LSE if softmax) -> bwd (PRE->MAIN->POST) produces dQ/dK/dV
//     CPU: reference bwd produces dQ*/dK*/dV*
//   compare (dQ,dK,dV) vs (dQ*,dK*,dV*) per-tensor via ck_tile::check_err.
//
// M0 SCOPE: batched, NoGroup, bf16, SiLU. The GPU bwd is a zeroing scaffold, so a
// large numerical error is EXPECTED here — M0 acceptance is "pipeline runs end to
// end + prints the three gradient errors", not numerical match. The harness still
// computes and prints PASS/FAIL per tensor for use from M1 onward, but exits 0 as
// long as the dispatch -> launch -> reference path completes (see main()).
//
// TODO(M1): flip the exit code to be driven by numerical PASS once MAIN is wired.
// TODO(M3/M4): jagged / group harness paths.
// TODO(M5): softmax path (needs GPU fwd to emit O + LSE).

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include <ck_tile/host/host_tensor.hpp>
#include <ck_tile/host/fill.hpp>
#include <ck_tile/host/device_memory.hpp>
#include <ck_tile/host/stream_config.hpp>
#include <ck_tile/host/arg_parser.hpp>
#include <ck_tile/host/hip_check_error.hpp>
#include <ck_tile/host/check_err.hpp>

#include "hstu_attention_fwd_type_config.hpp"
#include "hstu_attention_bool_switch.hpp"
#include "hstu_attention_params.hpp"
#include "hstu_attention_bwd_params.hpp"
#include "reference_hstu_attention_bwd.hpp"

#include "hstu_attention_host_util.hpp"
#include "hstu_attention_api.hpp"

static std::vector<int> get_integers_from_string(std::string srcStr)
{
    std::vector<int> integers;
    std::size_t pos = 0;
    std::size_t new_pos;

    new_pos = srcStr.find(',', pos);
    while(new_pos != std::string::npos)
    {
        std::string sliceStr = srcStr.substr(pos, new_pos - pos);
        integers.push_back(std::stoi(sliceStr));
        pos     = new_pos + 1;
        new_pos = srcStr.find(',', pos);
    };

    std::string sliceStr = srcStr.substr(pos);
    if(!sliceStr.empty())
        integers.push_back(std::stoi(sliceStr));

    return integers;
}

static std::vector<float> get_floats_from_string(std::string srcStr)
{
    std::vector<float> values;
    std::size_t pos = 0;
    std::size_t new_pos;

    new_pos = srcStr.find(',', pos);
    while(new_pos != std::string::npos)
    {
        std::string sliceStr = srcStr.substr(pos, new_pos - pos);
        values.push_back(std::stof(sliceStr));
        pos     = new_pos + 1;
        new_pos = srcStr.find(',', pos);
    };

    std::string sliceStr = srcStr.substr(pos);
    if(!sliceStr.empty())
        values.push_back(std::stof(sliceStr));

    return values;
}

template <typename T>
static void supplement_array_by_last_element(std::vector<T>& arr, int target_num_elements)
{
    if(static_cast<int>(arr.size()) < target_num_elements)
    {
        T last_val = arr.back();
        for(int i = arr.size(); i < target_num_elements; i++)
            arr.push_back(last_val);
    }
}

auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;

    // clang-format off
    arg_parser.insert("v", "1", "whether do CPU validation (reference bwd) or not")
        .insert("bwd_v", "1", "alias of -v for the bwd harness; either enables validation")
        .insert("prec", "bf16", "data type. bf16 or fp16 (M7a added fp16 backward)")
        .insert("b", "2", "number of batches")
        .insert("nhead", "2", "number of heads")
        .insert("hdim_qk", "64", "headdim size of Q/K")
        .insert("hdim_v", "64", "headdim size of V/O")
        .insert("seqlens", "128", "uih seqlen for query tensor (batched: single value; jagged: comma list, supplemented to num_batch)")
        .insert("jagged", "0", "variable-length (jagged) packed mode: q/k/v/dO/dQ/dK/dV are [1, sum(seqlen), h, d] with cu_seqlens offsets")
        .insert("softmax", "0", "use softmax activation (M0: SiLU only, 0)")
        .insert("causal", "0", "enable causal mask (M0: no-mask, 0)")
        .insert("local_len", "0", "diagonal window length; 0 disables")
        .insert("context_len", "0", "contextual seqlen at the begin of the query sequence")
        .insert("minfull_len", "0", "min full-attn seqlen at the end of the query sequence")
        .insert("targets", "", "num_targets per batch (comma list); empty disables")
        .insert("seed", "13579", "seed for the distribution generator")
        .insert("norm_dist", "0", "normal (1) vs uniform (0) initialization")
        .insert("alpha", "0", "scale of S=Q@K. 0 means 1/sqrt(hdim_qk)")
        .insert("attn_scale", "0", "scale of SiLU(Q@K). 0 means 1/max_seqlen_q")
        .insert("deterministic", "0", "deterministic dQ path (M0/M1: 0)")
        .insert("dump_grad", "0", "dump device and reference gradients to files")
        // group mode (M4): num_group>1 enables group HSTU (packed + per-group hyper-params)
        .insert("g", "1", "num attention groups; >1 enables group HSTU (num_batch must be a multiple)")
        .insert("g_max_seqlens", "0", "per-group max uih seqlen (comma list); 0 = derive from seqlens")
        .insert("g_local_lens", "0", "per-group diagonal window length (comma list); 0 disables")
        .insert("g_context_lens", "0", "per-group contextual seqlen (comma list)")
        .insert("g_minfull_lens", "0", "per-group min full-attn seqlen (comma list)")
        .insert("g_attn_scales", "1.0", "per-group SiLU output scale (comma list); 0 = 1/group_max_seqlen_q");
    // clang-format on

    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}

// bwd tolerances (DESIGN §5.2). M0 prints err regardless of pass/fail.
template <typename DataType>
auto get_bwd_elimit()
{
    return ck_tile::make_tuple(/*rtol*/ 2e-2, /*atol*/ 5e-2);
}
template <>
auto get_bwd_elimit<ck_tile::fp16_t>()
{
    return ck_tile::make_tuple(/*rtol*/ 5e-3, /*atol*/ 1e-2);
}

template <typename InOutDataType>
bool run_no_group_hstu_bwd(const ck_tile::ArgParser& arg_parser)
{
    const bool do_validation =
        static_cast<bool>(arg_parser.get_int("v")) || static_cast<bool>(arg_parser.get_int("bwd_v"));
    const int num_batch    = arg_parser.get_int("b");
    const int num_head     = arg_parser.get_int("nhead");
    const int hdim_qk      = arg_parser.get_int("hdim_qk");
    const int hdim_v       = arg_parser.get_int("hdim_v");
    const bool use_softmax = static_cast<bool>(arg_parser.get_int("softmax"));
    const bool use_causal  = static_cast<bool>(arg_parser.get_int("causal"));
    const bool is_jagged   = static_cast<bool>(arg_parser.get_int("jagged"));

    const float in_alpha      = arg_parser.get_float("alpha");
    const float attn_scale    = arg_parser.get_float("attn_scale");
    const int seed            = arg_parser.get_int("seed");
    const bool use_normal_dist = static_cast<bool>(arg_parser.get_int("norm_dist"));
    const bool dump_grad      = static_cast<bool>(arg_parser.get_int("dump_grad"));

    const int window_size          = arg_parser.get_int("local_len");
    const int contextual_seqlen    = arg_parser.get_int("context_len");
    const int min_full_attn_seqlen = arg_parser.get_int("minfull_len");

    std::vector<int> num_targets = get_integers_from_string(arg_parser.get_str("targets"));
    std::vector<int> seq_lengths_q = get_integers_from_string(arg_parser.get_str("seqlens"));

    HSTU_CHECK(!seq_lengths_q.empty(), "sequence lengths of q should be defined!");
    if(!is_jagged)
        HSTU_CHECK(seq_lengths_q.size() == 1, "batched harness expects a single seqlen value!");

    const bool is_cross_attention = false; // M3: self-attention only

    // jagged accepts a per-batch comma list; supplement to num_batch (mirrors fwd harness)
    if(is_jagged)
        supplement_array_by_last_element(seq_lengths_q, num_batch);

    int max_target = 0;
    if(!num_targets.empty())
    {
        // supplement to num_batch (kernel + reference index num_targets[i_batch])
        supplement_array_by_last_element(num_targets, num_batch);
        for(int i = 0; i < num_batch; i++)
            max_target = std::max(max_target, num_targets[i]);
    }

    // max uih seqlen over batches (== seq_lengths_q[0] in batched mode)
    int max_uih_seqlen_q = 0;
    for(int i = 0; i < (is_jagged ? num_batch : 1); i++)
        max_uih_seqlen_q = std::max(max_uih_seqlen_q, seq_lengths_q[i]);

    // max_seqlen_q drives scale_p (=1/max_seqlen_q) identically on GPU and reference.
    const int max_seqlen_q  = max_uih_seqlen_q + max_target + contextual_seqlen;
    const int max_seqlen_kv = max_seqlen_q; // self-attention

    // Build jagged cu_seqlens offsets (token-major packed). Self-attention: kv == q.
    // Per-batch physical seqlen includes num_target + contextual (mirrors fwd harness).
    std::vector<int> seq_offsets_q;
    int phy_seqlen_q  = 0;
    int phy_seqlen_kv = 0;
    if(is_jagged)
    {
        seq_offsets_q.push_back(0);
        for(int i = 0; i < num_batch; i++)
        {
            const int batch_seqlen = seq_lengths_q[i] +
                                     (num_targets.empty() ? 0 : num_targets[i]) + contextual_seqlen;
            phy_seqlen_q += batch_seqlen;
            seq_offsets_q.push_back(phy_seqlen_q);
        }
        phy_seqlen_kv = phy_seqlen_q; // self-attention
    }
    else
    {
        phy_seqlen_q  = max_seqlen_q;
        phy_seqlen_kv = max_seqlen_kv;
    }
    const std::vector<int>& seq_offsets_kv = seq_offsets_q; // self-attention

    // dim0 of the packed tensors: 1 for jagged (ΣL along dim1), num_batch otherwise.
    const int batches_for_alloc = is_jagged ? 1 : num_batch;

    using CompDataType = typename HstuAttentionFwdTypeConfig<InOutDataType>::CompDataType;

    // ---- host tensors (dim0 = 1 packed when jagged) --------------------------
    ck_tile::HostTensor<InOutDataType> q_host(
        std::array<ck_tile::index_t, 4>{batches_for_alloc, phy_seqlen_q, num_head, hdim_qk});
    ck_tile::HostTensor<InOutDataType> k_host(
        std::array<ck_tile::index_t, 4>{batches_for_alloc, phy_seqlen_kv, num_head, hdim_qk});
    ck_tile::HostTensor<InOutDataType> v_host(
        std::array<ck_tile::index_t, 4>{batches_for_alloc, phy_seqlen_kv, num_head, hdim_v});
    ck_tile::HostTensor<InOutDataType> o_host(
        std::array<ck_tile::index_t, 4>{batches_for_alloc, phy_seqlen_q, num_head, hdim_v});
    ck_tile::HostTensor<InOutDataType> do_host(
        std::array<ck_tile::index_t, 4>{batches_for_alloc, phy_seqlen_q, num_head, hdim_v});
    // LSE (softmax path only); allocated to satisfy the reference signature.
    ck_tile::HostTensor<CompDataType> lse_host(
        std::array<ck_tile::index_t, 3>{batches_for_alloc, phy_seqlen_q, num_head});

    // gradient outputs (GPU) and references (CPU)
    ck_tile::HostTensor<InOutDataType> dq_host(
        std::array<ck_tile::index_t, 4>{batches_for_alloc, phy_seqlen_q, num_head, hdim_qk});
    ck_tile::HostTensor<InOutDataType> dk_host(
        std::array<ck_tile::index_t, 4>{batches_for_alloc, phy_seqlen_kv, num_head, hdim_qk});
    ck_tile::HostTensor<InOutDataType> dv_host(
        std::array<ck_tile::index_t, 4>{batches_for_alloc, phy_seqlen_kv, num_head, hdim_v});
    ck_tile::HostTensor<InOutDataType> dq_host_ref(
        std::array<ck_tile::index_t, 4>{batches_for_alloc, phy_seqlen_q, num_head, hdim_qk});
    ck_tile::HostTensor<InOutDataType> dk_host_ref(
        std::array<ck_tile::index_t, 4>{batches_for_alloc, phy_seqlen_kv, num_head, hdim_qk});
    ck_tile::HostTensor<InOutDataType> dv_host_ref(
        std::array<ck_tile::index_t, 4>{batches_for_alloc, phy_seqlen_kv, num_head, hdim_v});

    if(use_normal_dist)
    {
        ck_tile::FillNormalDistribution<InOutDataType>{0.f, 1.f, seed}(q_host);
        ck_tile::FillNormalDistribution<InOutDataType>{0.f, 1.f, seed + 1}(k_host);
        ck_tile::FillNormalDistribution<InOutDataType>{0.f, 1.f, seed + 2}(v_host);
        ck_tile::FillNormalDistribution<InOutDataType>{0.f, 1.f, seed + 3}(do_host);
    }
    else
    {
        ck_tile::FillUniformDistribution<InOutDataType>{-1.f, 1.f, seed}(q_host);
        ck_tile::FillUniformDistribution<InOutDataType>{-1.f, 1.f, seed + 1}(k_host);
        ck_tile::FillUniformDistribution<InOutDataType>{-1.f, 1.f, seed + 2}(v_host);
        ck_tile::FillUniformDistribution<InOutDataType>{-1.f, 1.f, seed + 3}(do_host);
    }

    // ---- device buffers ------------------------------------------------------
    ck_tile::DeviceMem q_dev(q_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem k_dev(k_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem v_dev(v_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem o_dev(o_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem do_dev(do_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem dq_dev(dq_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem dk_dev(dk_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem dv_dev(dv_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem num_targets_dev(std::max<size_t>(num_targets.size(), 1) * sizeof(int));

    // jagged cu_seqlens offsets (size num_batch+1); allocate >=1 elem so batched is benign
    ck_tile::DeviceMem seq_offsets_q_dev(std::max<size_t>(seq_offsets_q.size(), 1) * sizeof(int));

    // float dQ accumulation workspace. atomic: 1 slot (same packed layout as dQ). M6
    // deterministic: num_splits stacked slots (one per KV-block) -> POST reduces over them.
    // num_splits = ceil(grid_seqlen_kv / kN0); kN0 = the bwd tile's bn0 (k-seqlen block),
    // which MUST match the dispatch's Pipeline::kN0 or the determ reduce overruns. hd256's
    // preset uses bn0=64 (all other hdims use 128) — see HstuBwdShape<256> (M7b).
    const bool is_deterministic = static_cast<bool>(arg_parser.get_int("deterministic"));
    const int kN0_bwd           = (hdim_qk == 256) ? 64 : 128;
    const int grid_seqlen_kv_h  = is_jagged ? max_seqlen_q : phy_seqlen_kv;
    const int num_splits =
        is_deterministic ? ((grid_seqlen_kv_h + kN0_bwd - 1) / kN0_bwd) : 1;
    const size_t single_dq_acc_elems =
        static_cast<size_t>(batches_for_alloc) * phy_seqlen_q * num_head * hdim_qk;
    const size_t dq_acc_elems = single_dq_acc_elems * static_cast<size_t>(num_splits);
    ck_tile::DeviceMem dq_acc_dev(dq_acc_elems * sizeof(CompDataType));

    // LSE (fwd output) + D (PRE output), softmax path only. Layout [batch,head,seq]
    // (seq-continuous; jagged: [head,ΣL]). nhead_stride_lsed=phy_seqlen_q,
    // batch_stride_lsed=num_head*phy_seqlen_q. Allocated unconditionally (cheap).
    const size_t lsed_elems =
        static_cast<size_t>(batches_for_alloc) * num_head * phy_seqlen_q;
    const ck_tile::index_t nhead_stride_lsed = phy_seqlen_q;
    const ck_tile::index_t batch_stride_lsed = num_head * phy_seqlen_q;
    ck_tile::DeviceMem lse_dev(lsed_elems * sizeof(CompDataType));
    ck_tile::DeviceMem d_dev(lsed_elems * sizeof(CompDataType));

    q_dev.ToDevice(q_host.data());
    k_dev.ToDevice(k_host.data());
    v_dev.ToDevice(v_host.data());
    do_dev.ToDevice(do_host.data());
    if(!num_targets.empty())
        num_targets_dev.ToDevice(num_targets.data());
    if(is_jagged)
        seq_offsets_q_dev.ToDevice(seq_offsets_q.data());

    const float scale_s = (in_alpha != 0.f) ? in_alpha : 1.0f / std::sqrt(static_cast<float>(hdim_qk));

    hipStream_t stream;
    HIP_CHECK_ERROR(hipStreamCreate(&stream));

    // ---- (1) GPU forward to produce O ---------------------------------------
    {
        HstuAttentionNoGroupFwdParams fp;
        fp.is_cross_attention = is_cross_attention;
        fp.is_jagged          = is_jagged;
        fp.num_batch          = num_batch;
        fp.seqlen_q           = phy_seqlen_q;  // jagged: ignored (per-batch via offsets)
        fp.seqlen_kv          = phy_seqlen_kv; // jagged: ignored
        fp.seq_q_offsets_ptr  = is_jagged ? seq_offsets_q_dev.GetDeviceBuffer() : nullptr;
        fp.seq_kv_offsets_ptr = is_jagged ? seq_offsets_q_dev.GetDeviceBuffer() : nullptr;
        fp.max_seqlen_q       = max_seqlen_q;
        fp.q_ptr              = q_dev.GetDeviceBuffer();
        fp.k_ptr              = k_dev.GetDeviceBuffer();
        fp.v_ptr              = v_dev.GetDeviceBuffer();
        fp.bias_ptr           = nullptr;
        fp.o_ptr              = o_dev.GetDeviceBuffer();
        fp.hdim_qk            = hdim_qk;
        fp.hdim_v             = hdim_v;
        fp.num_head           = num_head;
        fp.scale_s            = scale_s;
        fp.attn_scale         = attn_scale;
        fp.seq_stride_q       = q_host.get_strides()[1];
        fp.seq_stride_k       = k_host.get_strides()[1];
        fp.seq_stride_v       = v_host.get_strides()[1];
        fp.seq_stride_bias    = 0;
        fp.seq_stride_o       = o_host.get_strides()[1];
        fp.nhead_stride_q     = q_host.get_strides()[2];
        fp.nhead_stride_k     = k_host.get_strides()[2];
        fp.nhead_stride_v     = v_host.get_strides()[2];
        fp.nhead_stride_bias  = 0;
        fp.nhead_stride_o     = o_host.get_strides()[2];
        fp.batch_stride_q     = q_host.get_strides()[0];
        fp.batch_stride_k     = k_host.get_strides()[0];
        fp.batch_stride_v     = v_host.get_strides()[0];
        fp.batch_stride_bias  = 0;
        fp.batch_stride_o     = o_host.get_strides()[0];
        fp.num_targets_ptr = num_targets.empty() ? nullptr : num_targets_dev.GetDeviceBuffer();
        fp.use_softmax        = use_softmax;
        // M5: softmax needs fwd to store LSE (natural log) in [batch,head,seq]
        // seq-continuous layout so bwd reads it directly (jagged: token base via offsets).
        fp.is_training        = use_softmax;
        fp.lse_ptr            = use_softmax ? lse_dev.GetDeviceBuffer() : nullptr;
        fp.seq_stride_lse     = 1;
        fp.nhead_stride_lse   = nhead_stride_lsed;
        fp.batch_stride_lse   = batch_stride_lsed;
        fp.use_causal         = use_causal;
        fp.window_size        = window_size;
        fp.contextual_seqlen  = contextual_seqlen;
        fp.min_full_attn_seqlen = min_full_attn_seqlen;
        fp.p_drop             = 0.0f;
        fp.philox_seed        = 0UL;
        fp.philox_offset      = 0UL;

        if constexpr(std::is_same<InOutDataType, ck_tile::bf16_t>::value)
            hstu_attention_no_group_forward_bf16(fp, stream);
        else if constexpr(std::is_same<InOutDataType, ck_tile::fp16_t>::value)
            hstu_attention_no_group_forward_fp16(fp, stream);
        else
            throw std::runtime_error("bwd harness only wires bf16/fp16 forward paths");

        HIP_CHECK_ERROR(hipStreamSynchronize(stream));
        o_dev.FromDevice(o_host.data());

        // softmax: pull the GPU LSE ([batch,head,seq]) and transpose into the reference's
        // [batch,seq,head] layout (lse_host) so CPU bwd consumes the SAME LSE the GPU bwd
        // reads. (Critical: mismatched layout would be silently wrong.)
        if(use_softmax)
        {
            std::vector<CompDataType> lse_flat(lsed_elems);
            lse_dev.FromDevice(lse_flat.data());
            for(int b = 0; b < batches_for_alloc; b++)
                for(int h = 0; h < num_head; h++)
                    for(int s = 0; s < phy_seqlen_q; s++)
                        lse_host(b, s, h) =
                            lse_flat[(static_cast<size_t>(b) * num_head + h) * phy_seqlen_q + s];
        }
    }

    // ---- (2) GPU backward (M0 scaffold: zeroes dQ/dK/dV) ---------------------
    {
        HstuAttentionNoGroupBwdParams bp{};
        bp.is_cross_attention = is_cross_attention;
        bp.is_jagged          = is_jagged;
        bp.num_batch          = num_batch;
        bp.seqlen_q           = phy_seqlen_q;  // jagged: ignored (per-batch via offsets)
        bp.seqlen_kv          = phy_seqlen_kv; // jagged: ignored
        bp.seq_q_offsets_ptr  = is_jagged ? seq_offsets_q_dev.GetDeviceBuffer() : nullptr;
        bp.seq_kv_offsets_ptr = is_jagged ? seq_offsets_q_dev.GetDeviceBuffer() : nullptr;
        bp.max_seqlen_q       = max_seqlen_q; // scale_p = attn_scale ? attn_scale : 1/max_seqlen_q
        bp.q_ptr              = q_dev.GetDeviceBuffer();
        bp.k_ptr              = k_dev.GetDeviceBuffer();
        bp.v_ptr              = v_dev.GetDeviceBuffer();
        bp.o_ptr              = o_dev.GetDeviceBuffer();
        bp.hdim_qk            = hdim_qk;
        bp.hdim_v             = hdim_v;
        bp.num_head           = num_head;
        bp.nhead_ratio_qk     = 1;
        bp.alpha              = scale_s;
        bp.attn_scale         = attn_scale;
        bp.use_causal         = use_causal;
        bp.use_softmax        = use_softmax;
        bp.window_size        = window_size;
        bp.contextual_seqlen  = contextual_seqlen;
        bp.min_full_attn_seqlen = min_full_attn_seqlen;
        bp.num_targets_ptr = num_targets.empty() ? nullptr : num_targets_dev.GetDeviceBuffer();

        bp.seq_stride_q   = q_host.get_strides()[1];
        bp.seq_stride_k   = k_host.get_strides()[1];
        bp.seq_stride_v   = v_host.get_strides()[1];
        bp.seq_stride_o   = o_host.get_strides()[1];
        bp.nhead_stride_q = q_host.get_strides()[2];
        bp.nhead_stride_k = k_host.get_strides()[2];
        bp.nhead_stride_v = v_host.get_strides()[2];
        bp.nhead_stride_o = o_host.get_strides()[2];
        bp.batch_stride_q = q_host.get_strides()[0];
        bp.batch_stride_k = k_host.get_strides()[0];
        bp.batch_stride_v = v_host.get_strides()[0];
        bp.batch_stride_o = o_host.get_strides()[0];

        bp.do_ptr           = do_dev.GetDeviceBuffer();
        bp.seq_stride_do    = do_host.get_strides()[1];
        bp.nhead_stride_do  = do_host.get_strides()[2];
        bp.batch_stride_do  = do_host.get_strides()[0];
        // softmax (M5): LSE (fwd-stored) + D (PRE) in [batch,head,seq] layout. SiLU: null.
        bp.lse_ptr          = use_softmax ? lse_dev.GetDeviceBuffer() : nullptr;

        bp.dq_ptr         = dq_dev.GetDeviceBuffer();
        bp.dk_ptr         = dk_dev.GetDeviceBuffer();
        bp.dv_ptr         = dv_dev.GetDeviceBuffer();
        bp.seq_stride_dq  = dq_host.get_strides()[1];
        bp.seq_stride_dk  = dk_host.get_strides()[1];
        bp.seq_stride_dv  = dv_host.get_strides()[1];
        bp.nhead_stride_dq = dq_host.get_strides()[2];
        bp.nhead_stride_dk = dk_host.get_strides()[2];
        bp.nhead_stride_dv = dv_host.get_strides()[2];
        bp.batch_stride_dq = dq_host.get_strides()[0];
        bp.batch_stride_dk = dk_host.get_strides()[0];
        bp.batch_stride_dv = dv_host.get_strides()[0];

        bp.d_ptr               = use_softmax ? d_dev.GetDeviceBuffer() : nullptr;
        bp.nhead_stride_lsed   = nhead_stride_lsed;
        bp.batch_stride_lsed   = batch_stride_lsed;
        bp.dq_acc_ptr          = dq_acc_dev.GetDeviceBuffer();
        bp.stride_dq_acc       = dq_host.get_strides()[1]; // same layout as dQ
        bp.nhead_stride_dq_acc = dq_host.get_strides()[2];
        bp.batch_stride_dq_acc = dq_host.get_strides()[0];
        // M6 deterministic: split slot stride = single-slot element count; num_splits stacks.
        bp.split_stride_dq_acc = is_deterministic ? static_cast<ck_tile::index_t>(single_dq_acc_elems) : 0;
        bp.num_splits          = num_splits;
        bp.kIsDeterministic    = is_deterministic;

        if constexpr(std::is_same<InOutDataType, ck_tile::bf16_t>::value)
            hstu_attention_no_group_backward_bf16(bp, stream);
        else if constexpr(std::is_same<InOutDataType, ck_tile::fp16_t>::value)
            hstu_attention_no_group_backward_fp16(bp, stream);
        else
            throw std::runtime_error("bwd harness only wires bf16/fp16 backward paths");

        HIP_CHECK_ERROR(hipStreamSynchronize(stream));
        dq_dev.FromDevice(dq_host.data());
        dk_dev.FromDevice(dk_host.data());
        dv_dev.FromDevice(dv_host.data());
    }

    bool numeric_pass = true;

    if(do_validation)
    {
        using GemmAccDataType = typename HstuAttentionFwdTypeConfig<InOutDataType>::GemmAccDataType;

        const std::vector<int> empty_offsets; // batched mode: no jagged offsets

        // jagged: reference takes dim0=1 packed tensors + cu_seqlens (kv == q for self-attn).
        const std::vector<int>& ref_q_offsets  = is_jagged ? seq_offsets_q : empty_offsets;
        const std::vector<int>& ref_kv_offsets = is_jagged ? seq_offsets_kv : empty_offsets;

        BOOL_SWITCH_3(is_jagged, kIsJagged, use_softmax, kUseSoftmax, use_causal, kUseCausal, [&] {
            ck_tile::reference_no_group_hstu_attention_bwd<InOutDataType,
                                                           GemmAccDataType,
                                                           CompDataType,
                                                           kIsJagged,
                                                           kUseSoftmax,
                                                           kUseCausal>::Run(is_cross_attention,
                                                                            q_host,
                                                                            k_host,
                                                                            v_host,
                                                                            lse_host,
                                                                            o_host,
                                                                            do_host,
                                                                            dq_host_ref,
                                                                            dk_host_ref,
                                                                            dv_host_ref,
                                                                            num_batch,
                                                                            scale_s,
                                                                            attn_scale,
                                                                            max_seqlen_q,
                                                                            max_seqlen_kv,
                                                                            ref_q_offsets,
                                                                            ref_kv_offsets,
                                                                            num_targets,
                                                                            contextual_seqlen,
                                                                            window_size,
                                                                            min_full_attn_seqlen);
        });

        auto [rtol, atol] = get_bwd_elimit<InOutDataType>();

        // Explicit per-tensor error magnitudes (max/mean abs |dev - ref|), printed
        // unconditionally so the M0 evidence shows real numbers even though the
        // scaffold output is zero.
        auto report = [](const char* name,
                         const ck_tile::HostTensor<InOutDataType>& dev,
                         const ck_tile::HostTensor<InOutDataType>& ref) {
            const size_t n = dev.get_element_space_size();
            double max_abs = 0.0, sum_abs = 0.0, max_ref = 0.0;
            for(size_t i = 0; i < n; ++i)
            {
                const double a = ck_tile::type_convert<float>(dev.data()[i]);
                const double b = ck_tile::type_convert<float>(ref.data()[i]);
                const double e = std::abs(a - b);
                max_abs        = std::max(max_abs, e);
                sum_abs += e;
                max_ref = std::max(max_ref, std::abs(b));
            }
            std::cout << "  " << name << ": max_abs_err=" << max_abs
                      << " mean_abs_err=" << (n ? sum_abs / n : 0.0)
                      << " (max|ref|=" << max_ref << ")" << std::endl;
        };
        report("dQ", dq_host, dq_host_ref);
        report("dK", dk_host, dk_host_ref);
        report("dV", dv_host, dv_host_ref);

        const bool dq_ok =
            ck_tile::check_err(dq_host, dq_host_ref, std::string("dQ error"), rtol, atol);
        const bool dk_ok =
            ck_tile::check_err(dk_host, dk_host_ref, std::string("dK error"), rtol, atol);
        const bool dv_ok =
            ck_tile::check_err(dv_host, dv_host_ref, std::string("dV error"), rtol, atol);

        numeric_pass = dq_ok && dk_ok && dv_ok;

        std::cout << "[" << (dq_ok ? "PASS" : "FAIL") << "] dQ   "
                  << "[" << (dk_ok ? "PASS" : "FAIL") << "] dK   "
                  << "[" << (dv_ok ? "PASS" : "FAIL") << "] dV" << std::endl;

        if(dump_grad)
        {
            auto dump = [](const char* fn, const ck_tile::HostTensor<InOutDataType>& t) {
                std::ofstream f(fn, std::ios::binary);
                f.write(reinterpret_cast<const char*>(t.data()),
                        t.get_element_space_size() * sizeof(InOutDataType));
            };
            dump("dq_dev.dat", dq_host);
            dump("dq_ref.dat", dq_host_ref);
            dump("dk_dev.dat", dk_host);
            dump("dk_ref.dat", dk_host_ref);
            dump("dv_dev.dat", dv_host);
            dump("dv_ref.dat", dv_host_ref);
        }
    }

    return numeric_pass;
}

// ---------------------------------------------------------------------------
// GROUP mode harness (M4). group = jagged superset: packed [1, ΣL, h, d] +
// cu_seqlens, with per-group hyper-params (window/contextual/min_full/max_seqlen/
// attn_scale) indexed by i_group = i_batch / num_batch_per_group. alpha global.
// SiLU only (softmax is M5) -> the GPU forward (O) is unused, so we skip it.
// ---------------------------------------------------------------------------
template <typename InOutDataType>
bool run_group_hstu_bwd(const ck_tile::ArgParser& arg_parser, int num_group)
{
    const bool do_validation =
        static_cast<bool>(arg_parser.get_int("v")) || static_cast<bool>(arg_parser.get_int("bwd_v"));
    const int num_batch    = arg_parser.get_int("b");
    const int num_head     = arg_parser.get_int("nhead");
    const int hdim_qk      = arg_parser.get_int("hdim_qk");
    const int hdim_v       = arg_parser.get_int("hdim_v");
    const bool use_softmax = static_cast<bool>(arg_parser.get_int("softmax"));
    const bool use_causal  = static_cast<bool>(arg_parser.get_int("causal"));

    const float in_alpha       = arg_parser.get_float("alpha");
    const int seed             = arg_parser.get_int("seed");
    const bool use_normal_dist = static_cast<bool>(arg_parser.get_int("norm_dist"));
    const bool dump_grad       = static_cast<bool>(arg_parser.get_int("dump_grad"));

    HSTU_CHECK(num_group > 1, "run_group_hstu_bwd should only be called when num_group > 1!");
    HSTU_CHECK(num_batch > 0 && num_batch % num_group == 0,
               "number of batches should be a multiple of num_group!");
    const int num_batch_per_group = num_batch / num_group;
    const bool is_cross_attention = false; // M4: self-attention only

    std::vector<int> num_targets   = get_integers_from_string(arg_parser.get_str("targets"));
    std::vector<int> seq_lengths_q = get_integers_from_string(arg_parser.get_str("seqlens"));
    HSTU_CHECK(!seq_lengths_q.empty(), "sequence lengths of q should be defined!");
    supplement_array_by_last_element(seq_lengths_q, num_batch);
    if(!num_targets.empty())
        supplement_array_by_last_element(num_targets, num_batch);

    // per-group hyper-params (comma lists), supplemented to num_group
    std::vector<int> group_input_max_uih_seqlens_q =
        get_integers_from_string(arg_parser.get_str("g_max_seqlens"));
    std::vector<int> group_window_sizes = get_integers_from_string(arg_parser.get_str("g_local_lens"));
    std::vector<int> group_contextual_seqlens =
        get_integers_from_string(arg_parser.get_str("g_context_lens"));
    std::vector<int> group_min_full_attn_seqlens =
        get_integers_from_string(arg_parser.get_str("g_minfull_lens"));
    std::vector<float> group_attn_scales = get_floats_from_string(arg_parser.get_str("g_attn_scales"));
    supplement_array_by_last_element(group_input_max_uih_seqlens_q, num_group);
    supplement_array_by_last_element(group_window_sizes, num_group);
    supplement_array_by_last_element(group_contextual_seqlens, num_group);
    supplement_array_by_last_element(group_min_full_attn_seqlens, num_group);
    supplement_array_by_last_element(group_attn_scales, num_group);

    // group_max_seqlens_q = MAX packed seqlen over the group's batches, consistent with the
    // cu_seqlens offset formula (batch_seqlen = seq_lengths_q[b] + num_targets[b] + ctx_grp).
    // BUGFIX (M6b): the old form used num_targets[i_grp] (group-index into the per-batch
    // num_targets array), so when a group's LONGEST-packed batch was not batch[i_grp] the
    // value UNDER-estimated -> max_max_seqlen_q < some batch's real seqlen -> the PRE
    // hstu_bwd_dot_do_o_kernel (grid bounded by max_seqlen_q) skipped that batch's tail tokens
    // -> garbage D -> wrong dQ (target rows). Must be a true per-batch max over the group.
    // (This also drives scale_p fallback for SiLU group when attn_scale==0.)
    std::vector<int> group_max_seqlens_q(num_group, 0);
    int max_max_seqlen_q = 0;
    for(int i_grp = 0; i_grp < num_group; i_grp++)
    {
        int grp_max_packed_uih_plus_tgt = 0; // max over batches of (uih_b + target_b)
        for(int i = 0; i < num_batch_per_group; i++)
        {
            const int b = i_grp * num_batch_per_group + i;
            const int packed_uih_tgt =
                seq_lengths_q[b] + (num_targets.empty() ? 0 : num_targets[b]);
            grp_max_packed_uih_plus_tgt = std::max(grp_max_packed_uih_plus_tgt, packed_uih_tgt);
        }
        // honor user override of the group's uih cap (adds its own representative target)
        if(group_input_max_uih_seqlens_q[i_grp] > 0)
            grp_max_packed_uih_plus_tgt =
                std::max(grp_max_packed_uih_plus_tgt,
                         group_input_max_uih_seqlens_q[i_grp] +
                             (num_targets.empty() ? 0 : num_targets[i_grp * num_batch_per_group]));
        group_max_seqlens_q[i_grp] =
            grp_max_packed_uih_plus_tgt + group_contextual_seqlens[i_grp];
        max_max_seqlen_q = std::max(max_max_seqlen_q, group_max_seqlens_q[i_grp]);
    }

    // cu_seqlens (token-major packed); self-attention -> kv == q
    std::vector<int> seq_offsets_q;
    seq_offsets_q.push_back(0);
    int phy_seqlen_q = 0;
    for(int i = 0; i < num_batch; i++)
    {
        const int i_group = i / num_batch_per_group;
        const int batch_seqlen = seq_lengths_q[i] + (num_targets.empty() ? 0 : num_targets[i]) +
                                 group_contextual_seqlens[i_group];
        phy_seqlen_q += batch_seqlen;
        seq_offsets_q.push_back(phy_seqlen_q);
        // Precondition (M6b): max_seqlen_q must cover EVERY batch's packed seqlen, else the
        // PRE D kernel (grid bounded by max_seqlen_q) and the fwd grid skip this batch's tail
        // tokens -> garbage D / missing O,LSE -> wrong dQ. Loudly fail a bad group_max_seqlens_q
        // rather than silently producing wrong gradients.
        HSTU_CHECK(max_max_seqlen_q >= batch_seqlen,
                   "group max_seqlen_q under-covers a batch's packed seqlen (group_max_seqlens_q "
                   "computed too small)");
    }
    const int phy_seqlen_kv                = phy_seqlen_q;
    const std::vector<int>& seq_offsets_kv = seq_offsets_q;
    const int batches_for_alloc            = 1;

    using CompDataType = typename HstuAttentionFwdTypeConfig<InOutDataType>::CompDataType;

    // ---- host tensors (packed dim0=1) ----------------------------------------
    ck_tile::HostTensor<InOutDataType> q_host(
        std::array<ck_tile::index_t, 4>{batches_for_alloc, phy_seqlen_q, num_head, hdim_qk});
    ck_tile::HostTensor<InOutDataType> k_host(
        std::array<ck_tile::index_t, 4>{batches_for_alloc, phy_seqlen_kv, num_head, hdim_qk});
    ck_tile::HostTensor<InOutDataType> v_host(
        std::array<ck_tile::index_t, 4>{batches_for_alloc, phy_seqlen_kv, num_head, hdim_v});
    ck_tile::HostTensor<InOutDataType> o_host( // SiLU: unused, kept for ref signature
        std::array<ck_tile::index_t, 4>{batches_for_alloc, phy_seqlen_q, num_head, hdim_v});
    ck_tile::HostTensor<InOutDataType> do_host(
        std::array<ck_tile::index_t, 4>{batches_for_alloc, phy_seqlen_q, num_head, hdim_v});
    ck_tile::HostTensor<CompDataType> lse_host(
        std::array<ck_tile::index_t, 3>{batches_for_alloc, phy_seqlen_q, num_head});

    ck_tile::HostTensor<InOutDataType> dq_host(
        std::array<ck_tile::index_t, 4>{batches_for_alloc, phy_seqlen_q, num_head, hdim_qk});
    ck_tile::HostTensor<InOutDataType> dk_host(
        std::array<ck_tile::index_t, 4>{batches_for_alloc, phy_seqlen_kv, num_head, hdim_qk});
    ck_tile::HostTensor<InOutDataType> dv_host(
        std::array<ck_tile::index_t, 4>{batches_for_alloc, phy_seqlen_kv, num_head, hdim_v});
    ck_tile::HostTensor<InOutDataType> dq_host_ref(
        std::array<ck_tile::index_t, 4>{batches_for_alloc, phy_seqlen_q, num_head, hdim_qk});
    ck_tile::HostTensor<InOutDataType> dk_host_ref(
        std::array<ck_tile::index_t, 4>{batches_for_alloc, phy_seqlen_kv, num_head, hdim_qk});
    ck_tile::HostTensor<InOutDataType> dv_host_ref(
        std::array<ck_tile::index_t, 4>{batches_for_alloc, phy_seqlen_kv, num_head, hdim_v});

    if(use_normal_dist)
    {
        ck_tile::FillNormalDistribution<InOutDataType>{0.f, 1.f, seed}(q_host);
        ck_tile::FillNormalDistribution<InOutDataType>{0.f, 1.f, seed + 1}(k_host);
        ck_tile::FillNormalDistribution<InOutDataType>{0.f, 1.f, seed + 2}(v_host);
        ck_tile::FillNormalDistribution<InOutDataType>{0.f, 1.f, seed + 3}(do_host);
    }
    else
    {
        ck_tile::FillUniformDistribution<InOutDataType>{-1.f, 1.f, seed}(q_host);
        ck_tile::FillUniformDistribution<InOutDataType>{-1.f, 1.f, seed + 1}(k_host);
        ck_tile::FillUniformDistribution<InOutDataType>{-1.f, 1.f, seed + 2}(v_host);
        ck_tile::FillUniformDistribution<InOutDataType>{-1.f, 1.f, seed + 3}(do_host);
    }

    // ---- device buffers ------------------------------------------------------
    ck_tile::DeviceMem q_dev(q_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem k_dev(k_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem v_dev(v_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem do_dev(do_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem dq_dev(dq_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem dk_dev(dk_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem dv_dev(dv_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem num_targets_dev(std::max<size_t>(num_targets.size(), 1) * sizeof(int));
    ck_tile::DeviceMem seq_offsets_q_dev(seq_offsets_q.size() * sizeof(int));

    ck_tile::DeviceMem group_attn_scales_dev(group_attn_scales.size() * sizeof(float));
    ck_tile::DeviceMem group_max_seqlens_q_dev(group_max_seqlens_q.size() * sizeof(int));
    ck_tile::DeviceMem group_window_sizes_dev(group_window_sizes.size() * sizeof(int));
    ck_tile::DeviceMem group_contextual_seqlens_dev(group_contextual_seqlens.size() * sizeof(int));
    ck_tile::DeviceMem group_min_full_attn_seqlens_dev(group_min_full_attn_seqlens.size() *
                                                       sizeof(int));

    // M6b group deterministic: dq_acc workspace = single packed slot × num_splits (one slot
    // per KV-block, no atomic) -> POST reduces over splits. atomic: 1 slot. kN0 = tile bn0
    // (hd256: 64, else 128) — must match the dispatch's Pipeline::kN0 (M7b).
    const bool is_deterministic    = static_cast<bool>(arg_parser.get_int("deterministic"));
    const int kN0_bwd              = (hdim_qk == 256) ? 64 : 128;
    const int num_splits =
        is_deterministic ? ((max_max_seqlen_q + kN0_bwd - 1) / kN0_bwd) : 1;
    const size_t single_dq_acc_elems =
        static_cast<size_t>(batches_for_alloc) * phy_seqlen_q * num_head * hdim_qk;
    const size_t dq_acc_elems = single_dq_acc_elems * static_cast<size_t>(num_splits);
    ck_tile::DeviceMem dq_acc_dev(dq_acc_elems * sizeof(CompDataType));

    // softmax (M5b): O (fwd output, used by PRE for D), LSE (fwd), D (PRE). group is
    // packed -> LSE/D layout [head, ΣL] (seq-continuous), nhead_stride_lsed = ΣL.
    const size_t lsed_elems = static_cast<size_t>(num_head) * phy_seqlen_q;
    const ck_tile::index_t nhead_stride_lsed = phy_seqlen_q;
    ck_tile::DeviceMem o_dev(o_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem lse_dev(lsed_elems * sizeof(CompDataType));
    ck_tile::DeviceMem d_dev(lsed_elems * sizeof(CompDataType));

    q_dev.ToDevice(q_host.data());
    k_dev.ToDevice(k_host.data());
    v_dev.ToDevice(v_host.data());
    do_dev.ToDevice(do_host.data());
    if(!num_targets.empty())
        num_targets_dev.ToDevice(num_targets.data());
    seq_offsets_q_dev.ToDevice(seq_offsets_q.data());
    group_attn_scales_dev.ToDevice(group_attn_scales.data());
    group_max_seqlens_q_dev.ToDevice(group_max_seqlens_q.data());
    group_window_sizes_dev.ToDevice(group_window_sizes.data());
    group_contextual_seqlens_dev.ToDevice(group_contextual_seqlens.data());
    group_min_full_attn_seqlens_dev.ToDevice(group_min_full_attn_seqlens.data());

    const float scale_s =
        (in_alpha != 0.f) ? in_alpha : 1.0f / std::sqrt(static_cast<float>(hdim_qk));

    hipStream_t stream;
    HIP_CHECK_ERROR(hipStreamCreate(&stream));

    // ---- GPU group forward (softmax only): produce O + LSE. SiLU skips this (O unused).
    //      LSE stored [head, ΣL] seq-continuous (packed) so group bwd reads it directly;
    //      transposed into lse_host [1, ΣL, head] for the reference.
    if(use_softmax)
    {
        HstuAttentionGroupFwdParams fp{};
        fp.is_cross_attention = is_cross_attention;
        fp.use_softmax        = true;
        fp.is_training        = true;
        fp.num_group          = num_group;
        fp.num_batch          = num_batch;
        fp.seq_q_offsets_ptr  = seq_offsets_q_dev.GetDeviceBuffer();
        fp.seq_kv_offsets_ptr = seq_offsets_q_dev.GetDeviceBuffer();
        fp.max_seqlen_q       = max_max_seqlen_q;
        fp.q_ptr              = q_dev.GetDeviceBuffer();
        fp.k_ptr              = k_dev.GetDeviceBuffer();
        fp.v_ptr              = v_dev.GetDeviceBuffer();
        fp.bias_ptr           = nullptr;
        fp.o_ptr              = o_dev.GetDeviceBuffer();
        fp.lse_ptr            = lse_dev.GetDeviceBuffer();
        fp.hdim_qk            = hdim_qk;
        fp.hdim_v             = hdim_v;
        fp.num_head           = num_head;
        fp.scale_s            = scale_s;
        fp.seq_stride_q       = q_host.get_strides()[1];
        fp.seq_stride_k       = k_host.get_strides()[1];
        fp.seq_stride_v       = v_host.get_strides()[1];
        fp.seq_stride_bias    = 0;
        fp.seq_stride_o       = o_host.get_strides()[1];
        fp.seq_stride_lse     = 1;
        fp.nhead_stride_q     = q_host.get_strides()[2];
        fp.nhead_stride_k     = k_host.get_strides()[2];
        fp.nhead_stride_v     = v_host.get_strides()[2];
        fp.nhead_stride_bias  = 0;
        fp.nhead_stride_o     = o_host.get_strides()[2];
        fp.nhead_stride_lse   = nhead_stride_lsed;
        fp.num_targets_ptr = num_targets.empty() ? nullptr : num_targets_dev.GetDeviceBuffer();
        fp.use_causal         = use_causal;
        fp.group_attn_scale_ptr           = group_attn_scales_dev.GetDeviceBuffer();
        fp.group_max_seqlen_q_ptr         = group_max_seqlens_q_dev.GetDeviceBuffer();
        fp.group_window_size_ptr          = group_window_sizes_dev.GetDeviceBuffer();
        fp.group_contextual_seqlen_ptr    = group_contextual_seqlens_dev.GetDeviceBuffer();
        fp.group_min_full_attn_seqlen_ptr = group_min_full_attn_seqlens_dev.GetDeviceBuffer();
        fp.p_drop             = 0.0f;
        fp.philox_seed        = 0UL;
        fp.philox_offset      = 0UL;

        if constexpr(std::is_same<InOutDataType, ck_tile::bf16_t>::value)
            hstu_attention_group_forward_bf16(fp, stream);
        else if constexpr(std::is_same<InOutDataType, ck_tile::fp16_t>::value)
            hstu_attention_group_forward_fp16(fp, stream);
        else
            throw std::runtime_error("group bwd harness only wires bf16/fp16 forward paths");

        HIP_CHECK_ERROR(hipStreamSynchronize(stream));
        o_dev.FromDevice(o_host.data());

        // transpose GPU LSE ([head, ΣL]) -> reference layout lse_host ([1, ΣL, head])
        std::vector<CompDataType> lse_flat(lsed_elems);
        lse_dev.FromDevice(lse_flat.data());
        for(int h = 0; h < num_head; h++)
            for(int s = 0; s < phy_seqlen_q; s++)
                lse_host(0, s, h) = lse_flat[static_cast<size_t>(h) * phy_seqlen_q + s];
    }

    // ---- GPU backward (group) ------------------------------------------------
    {
        HstuAttentionGroupBwdParams bp{};
        bp.is_cross_attention = is_cross_attention;
        bp.num_group          = num_group;
        bp.num_batch          = num_batch;
        bp.seq_q_offsets_ptr  = seq_offsets_q_dev.GetDeviceBuffer();
        bp.seq_kv_offsets_ptr = seq_offsets_q_dev.GetDeviceBuffer();
        bp.max_seqlen_q       = max_max_seqlen_q;
        bp.q_ptr              = q_dev.GetDeviceBuffer();
        bp.k_ptr              = k_dev.GetDeviceBuffer();
        bp.v_ptr              = v_dev.GetDeviceBuffer();
        // softmax (M5b): O fed to PRE for D. SiLU: O unused.
        bp.o_ptr              = use_softmax ? o_dev.GetDeviceBuffer() : nullptr;
        bp.hdim_qk            = hdim_qk;
        bp.hdim_v             = hdim_v;
        bp.num_head           = num_head;
        bp.nhead_ratio_qk     = 1;
        bp.alpha              = scale_s;
        bp.use_causal         = use_causal;
        bp.use_softmax        = use_softmax;
        bp.num_targets_ptr = num_targets.empty() ? nullptr : num_targets_dev.GetDeviceBuffer();
        bp.group_attn_scale_ptr           = group_attn_scales_dev.GetDeviceBuffer();
        bp.group_max_seqlen_q_ptr         = group_max_seqlens_q_dev.GetDeviceBuffer();
        bp.group_window_size_ptr          = group_window_sizes_dev.GetDeviceBuffer();
        bp.group_contextual_seqlen_ptr    = group_contextual_seqlens_dev.GetDeviceBuffer();
        bp.group_min_full_attn_seqlen_ptr = group_min_full_attn_seqlens_dev.GetDeviceBuffer();

        bp.seq_stride_q   = q_host.get_strides()[1];
        bp.seq_stride_k   = k_host.get_strides()[1];
        bp.seq_stride_v   = v_host.get_strides()[1];
        bp.seq_stride_o   = o_host.get_strides()[1];
        bp.nhead_stride_q = q_host.get_strides()[2];
        bp.nhead_stride_k = k_host.get_strides()[2];
        bp.nhead_stride_v = v_host.get_strides()[2];
        bp.nhead_stride_o = o_host.get_strides()[2];

        bp.do_ptr          = do_dev.GetDeviceBuffer();
        bp.seq_stride_do   = do_host.get_strides()[1];
        bp.nhead_stride_do = do_host.get_strides()[2];
        // softmax (M5b): LSE (fwd) + D (PRE), [head, ΣL] packed. SiLU: null.
        bp.lse_ptr           = use_softmax ? lse_dev.GetDeviceBuffer() : nullptr;
        bp.d_ptr             = use_softmax ? d_dev.GetDeviceBuffer() : nullptr;
        bp.nhead_stride_lsed = nhead_stride_lsed;

        bp.dq_ptr          = dq_dev.GetDeviceBuffer();
        bp.dk_ptr          = dk_dev.GetDeviceBuffer();
        bp.dv_ptr          = dv_dev.GetDeviceBuffer();
        bp.seq_stride_dq   = dq_host.get_strides()[1];
        bp.seq_stride_dk   = dk_host.get_strides()[1];
        bp.seq_stride_dv   = dv_host.get_strides()[1];
        bp.nhead_stride_dq = dq_host.get_strides()[2];
        bp.nhead_stride_dk = dk_host.get_strides()[2];
        bp.nhead_stride_dv = dv_host.get_strides()[2];

        bp.dq_acc_ptr           = dq_acc_dev.GetDeviceBuffer();
        bp.stride_dq_acc        = dq_host.get_strides()[1];
        bp.nhead_stride_dq_acc  = dq_host.get_strides()[2];
        // total_dq_acc_elems keeps SINGLE-slot semantics (dispatch memsets single*num_splits).
        bp.total_dq_acc_elems   = static_cast<int>(single_dq_acc_elems);
        // M6b determ: split slot stride = single-slot elems; num_splits stacks.
        bp.split_stride_dq_acc  = is_deterministic ? static_cast<ck_tile::index_t>(single_dq_acc_elems) : 0;
        bp.num_splits           = num_splits;
        bp.kIsDeterministic     = is_deterministic;

        if constexpr(std::is_same<InOutDataType, ck_tile::bf16_t>::value)
            hstu_attention_group_backward_bf16(bp, stream);
        else if constexpr(std::is_same<InOutDataType, ck_tile::fp16_t>::value)
            hstu_attention_group_backward_fp16(bp, stream);
        else
            throw std::runtime_error("group bwd harness only wires bf16/fp16 backward paths");

        HIP_CHECK_ERROR(hipStreamSynchronize(stream));
        dq_dev.FromDevice(dq_host.data());
        dk_dev.FromDevice(dk_host.data());
        dv_dev.FromDevice(dv_host.data());
    }

    bool numeric_pass = true;
    if(do_validation)
    {
        using GemmAccDataType = typename HstuAttentionFwdTypeConfig<InOutDataType>::GemmAccDataType;

        BOOL_SWITCH_2(use_softmax, kUseSoftmax, use_causal, kUseCausal, [&] {
            ck_tile::reference_group_hstu_attention_bwd<InOutDataType,
                                                        GemmAccDataType,
                                                        CompDataType,
                                                        kUseSoftmax,
                                                        kUseCausal>::Run(is_cross_attention,
                                                                         q_host,
                                                                         k_host,
                                                                         v_host,
                                                                         lse_host,
                                                                         o_host,
                                                                         do_host,
                                                                         dq_host_ref,
                                                                         dk_host_ref,
                                                                         dv_host_ref,
                                                                         num_batch,
                                                                         num_batch_per_group,
                                                                         scale_s,
                                                                         seq_offsets_q,
                                                                         seq_offsets_kv,
                                                                         num_targets,
                                                                         group_max_seqlens_q,
                                                                         group_contextual_seqlens,
                                                                         group_window_sizes,
                                                                         group_min_full_attn_seqlens,
                                                                         group_attn_scales);
        });

        auto [rtol, atol] = get_bwd_elimit<InOutDataType>();

        auto report = [](const char* name,
                         const ck_tile::HostTensor<InOutDataType>& dev,
                         const ck_tile::HostTensor<InOutDataType>& ref) {
            const size_t n = dev.get_element_space_size();
            double max_abs = 0.0, sum_abs = 0.0, max_ref = 0.0;
            for(size_t i = 0; i < n; ++i)
            {
                const double a = ck_tile::type_convert<float>(dev.data()[i]);
                const double b = ck_tile::type_convert<float>(ref.data()[i]);
                const double e = std::abs(a - b);
                max_abs        = std::max(max_abs, e);
                sum_abs += e;
                max_ref = std::max(max_ref, std::abs(b));
            }
            std::cout << "  " << name << ": max_abs_err=" << max_abs
                      << " mean_abs_err=" << (n ? sum_abs / n : 0.0)
                      << " (max|ref|=" << max_ref << ")" << std::endl;
        };
        report("dQ", dq_host, dq_host_ref);
        report("dK", dk_host, dk_host_ref);
        report("dV", dv_host, dv_host_ref);

        const bool dq_ok =
            ck_tile::check_err(dq_host, dq_host_ref, std::string("dQ error"), rtol, atol);
        const bool dk_ok =
            ck_tile::check_err(dk_host, dk_host_ref, std::string("dK error"), rtol, atol);
        const bool dv_ok =
            ck_tile::check_err(dv_host, dv_host_ref, std::string("dV error"), rtol, atol);
        numeric_pass = dq_ok && dk_ok && dv_ok;

        std::cout << "[" << (dq_ok ? "PASS" : "FAIL") << "] dQ   "
                  << "[" << (dk_ok ? "PASS" : "FAIL") << "] dK   "
                  << "[" << (dv_ok ? "PASS" : "FAIL") << "] dV" << std::endl;

        if(dump_grad)
        {
            auto dump = [](const char* fn, const ck_tile::HostTensor<InOutDataType>& t) {
                std::ofstream f(fn, std::ios::binary);
                f.write(reinterpret_cast<const char*>(t.data()),
                        t.get_element_space_size() * sizeof(InOutDataType));
            };
            dump("dq_dev.dat", dq_host);
            dump("dq_ref.dat", dq_host_ref);
            dump("dk_dev.dat", dk_host);
            dump("dk_ref.dat", dk_host_ref);
            dump("dv_dev.dat", dv_host);
            dump("dv_ref.dat", dv_host_ref);
        }
    }

    return numeric_pass;
}

int main(int argc, char* argv[])
{
    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
    {
        std::cerr << "Invalid arguments, Failed to parse!" << std::endl;
        return -1;
    }

    const std::string data_type = arg_parser.get_str("prec");
    const int num_group         = arg_parser.get_int("g");

    if(data_type != "bf16" && data_type != "fp16")
    {
        std::cerr << "bwd harness only supports -prec=bf16 or -prec=fp16" << std::endl;
        return -3;
    }

    // num_group>1 routes to the group HSTU path (M4); otherwise no_group (batched/jagged).
    // M7a: dtype is selected at runtime from -prec (fp16 reuses the bf16 code path).
    bool numeric_pass;
    if(data_type == "fp16")
        numeric_pass = (num_group > 1)
                           ? run_group_hstu_bwd<ck_tile::fp16_t>(arg_parser, num_group)
                           : run_no_group_hstu_bwd<ck_tile::fp16_t>(arg_parser);
    else
        numeric_pass = (num_group > 1)
                           ? run_group_hstu_bwd<ck_tile::bf16_t>(arg_parser, num_group)
                           : run_no_group_hstu_bwd<ck_tile::bf16_t>(arg_parser);

    // M1: real SiLU MAIN — exit code is driven by numerical correctness.
    std::cout << "numeric_pass=" << (numeric_pass ? "true" : "false") << std::endl;
    return numeric_pass ? 0 : -2;
}
