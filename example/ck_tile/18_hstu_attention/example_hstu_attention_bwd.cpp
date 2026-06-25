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

#include "hstu_attention_util.hpp"
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
        .insert("prec", "bf16", "data type. bf16 (fp16 backward not in M0)")
        .insert("b", "2", "number of batches")
        .insert("nhead", "2", "number of heads")
        .insert("hdim_qk", "64", "headdim size of Q/K")
        .insert("hdim_v", "64", "headdim size of V/O")
        .insert("seqlens", "128", "uih seqlen for query tensor (batched: single value)")
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
        .insert("dump_grad", "0", "dump device and reference gradients to files");
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
    HSTU_CHECK(seq_lengths_q.size() == 1, "batched harness expects a single seqlen value!");

    const bool is_cross_attention = false; // M2: self-attention only

    int max_target = 0;
    if(!num_targets.empty())
    {
        for(int i = 0; i < num_batch; i++)
            max_target = std::max(max_target, num_targets[i]);
    }

    // max uih seqlen over batches (== seq_lengths_q[0] in batched mode)
    int max_uih_seqlen_q = 0;
    for(int i = 0; i < 1; i++)
        max_uih_seqlen_q = std::max(max_uih_seqlen_q, seq_lengths_q[i]);

    // max_seqlen_q drives scale_p (=1/max_seqlen_q) identically on GPU and reference.
    const int max_seqlen_q  = max_uih_seqlen_q + max_target + contextual_seqlen;
    const int max_seqlen_kv = max_seqlen_q; // self-attention

    const int phy_seqlen_q  = max_seqlen_q;
    const int phy_seqlen_kv = max_seqlen_kv;

    // dim0 of the packed tensors: num_batch (batched mode).
    const int batches_for_alloc = num_batch;

    using CompDataType = typename HstuAttentionFwdTypeConfig<InOutDataType>::CompDataType;

    // ---- host tensors --------------------------------------------------------
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

    // float dQ accumulation workspace (atomic path, nsplits=1; same packed layout as dQ)
    const size_t dq_acc_elems =
        static_cast<size_t>(batches_for_alloc) * phy_seqlen_q * num_head * hdim_qk;
    ck_tile::DeviceMem dq_acc_dev(dq_acc_elems * sizeof(CompDataType));

    q_dev.ToDevice(q_host.data());
    k_dev.ToDevice(k_host.data());
    v_dev.ToDevice(v_host.data());
    do_dev.ToDevice(do_host.data());
    if(!num_targets.empty())
        num_targets_dev.ToDevice(num_targets.data());

    const float scale_s = (in_alpha != 0.f) ? in_alpha : 1.0f / std::sqrt(static_cast<float>(hdim_qk));

    hipStream_t stream;
    HIP_CHECK_ERROR(hipStreamCreate(&stream));

    // ---- (1) GPU forward to produce O ---------------------------------------
    {
        HstuAttentionNoGroupFwdParams fp;
        fp.is_cross_attention = is_cross_attention;
        fp.is_jagged          = false;
        fp.num_batch          = num_batch;
        fp.seqlen_q           = phy_seqlen_q;
        fp.seqlen_kv          = phy_seqlen_kv;
        fp.seq_q_offsets_ptr  = nullptr;
        fp.seq_kv_offsets_ptr = nullptr;
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
        fp.use_causal         = use_causal;
        fp.window_size        = window_size;
        fp.contextual_seqlen  = contextual_seqlen;
        fp.min_full_attn_seqlen = min_full_attn_seqlen;
        fp.p_drop             = 0.0f;
        fp.philox_seed        = 0UL;
        fp.philox_offset      = 0UL;

        if constexpr(std::is_same<InOutDataType, ck_tile::bf16_t>::value)
            hstu_attention_no_group_forward_bf16(fp, stream);
        else
            throw std::runtime_error("M0 bwd harness only wires the bf16 forward path");

        HIP_CHECK_ERROR(hipStreamSynchronize(stream));
        o_dev.FromDevice(o_host.data());
    }

    // ---- (2) GPU backward (M0 scaffold: zeroes dQ/dK/dV) ---------------------
    {
        HstuAttentionNoGroupBwdParams bp{};
        bp.is_cross_attention = is_cross_attention;
        bp.is_jagged          = false;
        bp.num_batch          = num_batch;
        bp.seqlen_q           = phy_seqlen_q;
        bp.seqlen_kv          = phy_seqlen_kv;
        bp.seq_q_offsets_ptr  = nullptr;
        bp.seq_kv_offsets_ptr = nullptr;
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
        bp.lse_ptr          = nullptr; // SiLU path

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

        bp.d_ptr               = nullptr;
        bp.nhead_stride_lsed   = 0;
        bp.batch_stride_lsed   = 0;
        bp.dq_acc_ptr          = dq_acc_dev.GetDeviceBuffer();
        bp.stride_dq_acc       = dq_host.get_strides()[1]; // same layout as dQ
        bp.nhead_stride_dq_acc = dq_host.get_strides()[2];
        bp.batch_stride_dq_acc = dq_host.get_strides()[0];
        bp.split_stride_dq_acc = 0;
        bp.num_splits          = 1;
        bp.kIsDeterministic    = static_cast<bool>(arg_parser.get_int("deterministic"));

        if constexpr(std::is_same<InOutDataType, ck_tile::bf16_t>::value)
            hstu_attention_no_group_backward_bf16(bp, stream);
        else
            throw std::runtime_error("M0 bwd harness only wires the bf16 backward path");

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

        const std::vector<int>& ref_q_offsets  = empty_offsets;
        const std::vector<int>& ref_kv_offsets = empty_offsets;

        BOOL_SWITCH_2(use_softmax, kUseSoftmax, use_causal, kUseCausal, [&] {
            ck_tile::reference_no_group_hstu_attention_bwd<InOutDataType,
                                                           GemmAccDataType,
                                                           CompDataType,
                                                           false,
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

int main(int argc, char* argv[])
{
    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
    {
        std::cerr << "Invalid arguments, Failed to parse!" << std::endl;
        return -1;
    }

    const std::string data_type = arg_parser.get_str("prec");

    if(data_type != "bf16")
    {
        std::cerr << "M0 bwd harness only supports -prec=bf16" << std::endl;
        return -3;
    }

    bool numeric_pass = run_no_group_hstu_bwd<ck_tile::bf16_t>(arg_parser);

    // M1: real SiLU MAIN — exit code is driven by numerical correctness.
    std::cout << "numeric_pass=" << (numeric_pass ? "true" : "false") << std::endl;
    return numeric_pass ? 0 : -2;
}
