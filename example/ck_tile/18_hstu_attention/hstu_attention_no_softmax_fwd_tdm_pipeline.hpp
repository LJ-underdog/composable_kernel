// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <ck_tile/core.hpp>
#include <ck_tile/ops/fmha/block/block_dropout.hpp>

#include "hstu_attention_fwd_pipeline_policy.hpp"
#include "hstu_attention_kernel_util.hpp"

namespace ck_tile {

// TDM (Tensor Data Mover) variant of the no-softmax forward pipeline, targeting gfx1250.
//
// The LDS layout is the plain row-major one that TDM's box-major DMA requires (no XOR
// swizzle), and K/V are moved by load_tile_tdm: the hardware DMA writes global -> LDS
// directly, so no register staging (and no store_tile) is involved.
//
// Q is deliberately left untouched: HSTU loads Q once straight into registers and has no
// Q LDS descriptor at all, so only K/V can benefit from TDM here.
template <typename Problem_,
          typename Traits_,
          typename Policy_ = HstuAttentionFwdPipelineQRKSVSPolicy>
struct HstuAttentionNoSoftmaxFwdPipelineQRKSVSTdm
{
    using Problem         = remove_cvref_t<Problem_>;
    using Traits          = remove_cvref_t<Traits_>;
    using Policy          = remove_cvref_t<Policy_>;
    using QKVDataType     = remove_cvref_t<typename Problem::InOutDataType>;
    using GemmAccDataType = remove_cvref_t<typename Problem::GemmAccDataType>;
    using CompDataType    = remove_cvref_t<typename Problem::CompDataType>;
    using BiasDataType    = remove_cvref_t<typename Problem::BiasDataType>;
    using PDataType       = remove_cvref_t<typename Problem::InOutDataType>;
    using ODataType       = remove_cvref_t<typename Problem::InOutDataType>;

    using HstuAttentionTileSetting = remove_cvref_t<typename Problem::HstuAttentionTileSetting>;

    static constexpr index_t kBlockSize = Problem::kBlockSize;

    static constexpr index_t kM0        = HstuAttentionTileSetting::kM0;
    static constexpr index_t kN0        = HstuAttentionTileSetting::kN0;
    static constexpr index_t kN0Sub     = HstuAttentionTileSetting::kN0Sub;
    static constexpr index_t kN1        = HstuAttentionTileSetting::kN1;
    static constexpr index_t kK1        = HstuAttentionTileSetting::kK1;
    static constexpr index_t kQKHeaddim = HstuAttentionTileSetting::kQKHeaddim;

    static_assert(kQKHeaddim <= 256, "hdim bigger than 256 is not suitable for this pipeline!");

    static_assert(Problem::kUseSoftmax == false, "This pipeline only works with not-using softmax");

    static constexpr bool kIsJagged   = Problem::kIsJagged;
    static constexpr auto kHasBias    = Problem::kHasBias;
    static constexpr bool kHasDropout = Problem::kHasDropout;
    static constexpr bool kHasCausal  = Problem::kHasCausal;

    // gemm_1 still reads V through ds_load_tr, so the trload block-gemm stays selected.
    static constexpr bool kUseTrLoad = true;
    // ... but the LDS descriptors must be the un-swizzled ones (TDM box-major write).
    static constexpr bool kUsePlainLds = true;
    // ... and the K/V dram windows must carry a trivial tile-major distribution: TDM's
    // box_dim is reverse-derived from the dram window's ys_to_d lengths, so one wave must
    // own one contiguous (MN/warpNum x K) rectangle.
    static constexpr bool kTrivialTileMajorDram = true;

    // Guard-rails matching the dispatch gating. If codegen/dispatch ever drifts, this
    // fails to compile instead of silently taking a wrong path.
    static_assert(!kHasDropout, "hstu no-softmax tdm pipeline does not support dropout yet");

    static constexpr bool kPadSeqLenQ   = Traits::kPadSeqLenQ;
    static constexpr bool kPadSeqLenK   = Traits::kPadSeqLenK;
    static constexpr bool kPadHeadDimQK = Traits::kPadHeadDimQK;
    static constexpr bool kPadHeadDimV  = Traits::kPadHeadDimV;

    // last dimension vector length used to create tensor view(and decide buffer_load vector length)
    // ... together with tensor distribution. tensor dist should able to overwrite this
    static constexpr index_t kAlignmentQ =
        kPadHeadDimQK ? 1 : Policy::template GetAlignmentQ<Problem>();
    static constexpr index_t kAlignmentK =
        kPadHeadDimQK ? 1 : Policy::template GetAlignmentK<Problem>();
    static constexpr index_t kAlignmentV =
        Traits::kPadHeadDimV ? 1 : Policy::template GetAlignmentV<Problem, true /*kUseTrLoad*/>();

    static constexpr index_t kAlignmentO =
        kPadHeadDimV ? 1 : Policy::template GetAlignmentO<Problem, true /*kUseTrLoad */>();
    static constexpr index_t kAlignmentBias =
        kPadSeqLenK ? 1 : Policy::template GetAlignmentBias<Problem>();

    // used by NRepetitions2DEpilogue
    static constexpr index_t kGemm1SingleRepN =
        Policy::template GetPVTBlockGemmSingleRepN<Problem>();

    static constexpr index_t kBlockPerCu = []() {
        if constexpr(Traits::kBlockPerCu != -1)
            return Traits::kBlockPerCu;
        else
        {
            if constexpr(kQKHeaddim == 32)
            {
                return 2;
            }
            else if constexpr(kQKHeaddim == 64)
            {
                return 2;
            }
            else if constexpr(kQKHeaddim == 96 || kQKHeaddim == 128)
            {
                if constexpr(kHasBias)
                    return 2;
                else
                    return 2;
            }
            else if constexpr(kQKHeaddim == 256)
            {
                return 1;
            }
            else
            {
                return 1;
            };
        }
    }();

    using DropoutType = std::conditional_t<kHasDropout, BlockDropout, NullBlockDropout>;

    CK_TILE_DEVICE static constexpr ck_tile::index_t GetSmemSize()
    {
        return Policy::template GetSmemSize<Problem, true /*kPipelineUseTrLoad*/, kUsePlainLds>();
    }

    template <typename QDramBlockWindowTmp,
              typename KDramBlockWindowTmp,
              typename VDramBlockWindowTmp,
              typename BiasDramBlockWindowTmp,
              typename NullRandValDramWindowTmp,
              typename HstuMask>
    CK_TILE_DEVICE auto
    operator()(const QDramBlockWindowTmp& q_dram_block_window_tmp,       // M0*kQKHeaddim tile
               const KDramBlockWindowTmp& k_dram_block_window_tmp,       // N0*kQKHeaddim tile
               const VDramBlockWindowTmp& v_dram_block_window_tmp,       // N1*K1 tile
               const BiasDramBlockWindowTmp& bias_dram_block_window_tmp, // M0*N0 tile
               NullRandValDramWindowTmp& null_randval_window_tmp,        // M0*N0 tile
               index_t seqlen_k_start,
               index_t seqlen_k_end,
               HstuMask& mask,
               float scale_s, // scaling value exerted on the immediate Q@K result
               float scale_p, // scaling value exerted on the SiLu result
               void* smem_ptr,
               DropoutType& dropout) const
    {
        static_assert(
            std::is_same_v<QKVDataType, remove_cvref_t<typename QDramBlockWindowTmp::DataType>> &&
                std::is_same_v<QKVDataType,
                               remove_cvref_t<typename KDramBlockWindowTmp::DataType>> &&
                std::is_same_v<QKVDataType, remove_cvref_t<typename VDramBlockWindowTmp::DataType>>,
            "wrong!");

        static_assert(kM0 == QDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kN0 == KDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kQKHeaddim == KDramBlockWindowTmp{}.get_window_lengths()[number<1>{}] &&
                          kN1 == VDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kK1 == VDramBlockWindowTmp{}.get_window_lengths()[number<1>{}] &&
                          kM0 == BiasDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kN0 == BiasDramBlockWindowTmp{}.get_window_lengths()[number<1>{}],
                      "wrong!");

        constexpr index_t n0_loops = kN0 / kN0Sub;
        constexpr index_t k1_loops = kN0 / kK1;

        static_assert(n0_loops == k1_loops, "n0_loops == k1_loops required by this pipeline");

        constexpr auto NumKVLdsBuffers = Policy::template GetNumKVLdsBuffers<Problem>();

        // Block GEMM
        constexpr auto gemm_0 = Policy::template GetQKBlockGemm<Problem>();
        constexpr auto gemm_1 = Policy::template GetPVTBlockGemm<Problem, true /*kUseTrLoad*/>();

        using Gemm0Combined = decltype(Policy::template GetQKCombinedBlockGemm<Problem>());

        // SaccBlockTile size is [kM0, kN0Sub]
        // PcompBlockTile size is [kM0, kN0]
        using SaccBlockTileType        = decltype(gemm_0.template MakeCBlockTile<kM0, kN0Sub>());
        using CombineSaccBlockTileType = decltype(gemm_0.template MakeCBlockTile<kM0, kN0>());
        using PcompBlockTileType = decltype(cast_tile<CompDataType>(CombineSaccBlockTileType{}));

        SaccBlockTileType sacc_tile;
        PcompBlockTileType pcomp_tile;

        using OaccBlockTileType = decltype(gemm_1.MakeCBlockTile());
        OaccBlockTileType o_acc;

        if(seqlen_k_end <= seqlen_k_start)
        {
            clear_tile(o_acc);

            return o_acc;
        };

        auto q_dram_window = make_tile_window(q_dram_block_window_tmp.get_bottom_tensor_view(),
                                              make_tuple(number<kM0>{}, number<kQKHeaddim>{}),
                                              q_dram_block_window_tmp.get_window_origin(),
                                              Policy::template MakeQRegTileDistribution<Problem>());

        auto q_tile = load_tile(q_dram_window);

        const auto q_origin = q_dram_window.get_window_origin();

        auto k_dram_window =
            make_tile_window(k_dram_block_window_tmp.get_bottom_tensor_view(),
                             make_tuple(number<kN0Sub>{}, number<kQKHeaddim>{}),
                             {seqlen_k_start, 0},
                             Policy::template MakeKDramTileDistribution<Problem,
                                                                        kTrivialTileMajorDram>());

        // TDM descriptor config for the K/V global->LDS DMA. The hardware LDS padding is
        // enabled so that the plain row-major LDS row stride stops being a multiple of the
        // 256 B LDS bank wrap-around; the reader-side descriptors carry the very same
        // padded stride (Policy::GetPlainLdsPadElements). K and V share one config, which
        // is why the padding is only enabled when their row lengths match.
        TDMConfig tdm_config_kv{};
        if constexpr(Policy::template GetPlainLdsPadElements<Problem>() != 0)
        {
            tdm_config_kv.pad_enable            = true;
            tdm_config_kv.pad_config.pad_amount = Policy::kTdmLdsPadAmount;
            tdm_config_kv.pad_config.pad_interval =
                Policy::template GetPlainLdsPadInterval<Problem>();
        }

        // provide partition_index for LDS tile window so that warp_id is in vgpr
        array<index_t, 2> partition_index{get_warp_id<false>(), get_lane_id()};

        // K tile in LDS
        QKVDataType* k_lds_ptr = static_cast<QKVDataType*>(smem_ptr);
        auto k_lds             = make_tensor_view<address_space_enum::lds>(
            k_lds_ptr,
            Policy::template MakeKLdsBlockDescriptor<Problem, true /*trload*/, kUsePlainLds>());
        auto k_lds_monolithic_window = make_tile_window(
            k_lds,
            Policy::template MakeKLdsBlockDescriptor<Problem, true /*trload*/, kUsePlainLds>()
                .get_lengths(),
            {0, 0});

        static_assert(
            Policy::template MakeKLdsBlockDescriptor<Problem>().get_lengths()[number<0>{}] ==
                NumKVLdsBuffers * kN0Sub,
            "Check failed!");
        static_assert(
            Policy::template MakeKLdsBlockDescriptor<Problem>().get_lengths()[number<1>{}] ==
                kQKHeaddim,
            "Check failed!");

        using k_lds_window_type = decltype(get_slice_tile(
            k_lds_monolithic_window, sequence<0, 0>{}, sequence<kN0Sub, kQKHeaddim>{}));

        statically_indexed_array<k_lds_window_type, NumKVLdsBuffers> k_lds_windows;

        static_for<0, NumKVLdsBuffers, 1>{}([&](auto i_buf) {
            k_lds_windows[i_buf] = get_slice_tile(k_lds_monolithic_window,
                                                  sequence<i_buf * kN0Sub, 0>{},
                                                  sequence<(i_buf + 1) * kN0Sub, kQKHeaddim>{});
        });

        // V tile in LDS
        auto v_lds = make_tensor_view<address_space_enum::lds>(
            reinterpret_cast<QKVDataType*>(smem_ptr),
            Policy::template MakeVLdsBlockDescriptor<Problem, true /*trload*/, kUsePlainLds>());
        auto v_lds_monolithic_window = make_tile_window(
            v_lds,
            Policy::template MakeVLdsBlockDescriptor<Problem, true /*trload*/, kUsePlainLds>().get_lengths(),
            {0, 0});

        static_assert(Policy::template MakeVLdsBlockDescriptor<Problem, true /*trload*/, kUsePlainLds>()
                              .get_lengths()[number<0>{}] == NumKVLdsBuffers * kK1,
                      "Check failed!");
        static_assert(Policy::template MakeVLdsBlockDescriptor<Problem, true /*trload*/, kUsePlainLds>()
                              .get_lengths()[number<1>{}] == kN1,
                      "Check failed!");

        using v_lds_window_type = decltype(get_slice_tile(
            v_lds_monolithic_window, sequence<0, 0>{}, sequence<kK1, kN1>{}));

        statically_indexed_array<v_lds_window_type, NumKVLdsBuffers> v_lds_windows;

        static_for<0, NumKVLdsBuffers, 1>{}([&](auto i_buf) {
            v_lds_windows[i_buf] = get_slice_tile(v_lds_monolithic_window,
                                                  sequence<i_buf * kK1, 0>{},
                                                  sequence<(i_buf + 1) * kK1, kN1>{});
        });

        auto v_dram_window = make_tile_window(
            v_dram_block_window_tmp.get_bottom_tensor_view(),
            v_dram_block_window_tmp.get_window_lengths(),
            {seqlen_k_start, 0},
            Policy::template MakeVDramTileDistribution<Problem,
                                                       true /*kUseTrLoad*/,
                                                       kTrivialTileMajorDram>());

        // reduction function for softmax
        const auto f_silu = [&](CompDataType& x) {
            const auto one = ck_tile::type_convert<CompDataType>(1.0f);

            if constexpr(std::is_same_v<CompDataType, float>)
            {
                x = x * __builtin_amdgcn_rcpf(one + __expf(-x));
            }
            else
            {
                x = x / (one + exp(-x));
            }
        };

        const auto bias_origin = bias_dram_block_window_tmp.get_window_origin();
        auto bias_dram_window =
            make_tile_window(bias_dram_block_window_tmp.get_bottom_tensor_view(),
                             make_tuple(number<kM0>{}, number<kK1>{}),
                             {bias_origin.at(number<0>{}), seqlen_k_start}, // M/N
                             Policy::template MakeBiasDramTileDistribution<Problem>());

        auto null_randval_window = dropout.template MakeRandvalDramWindow<Gemm0Combined>(
            null_randval_window_tmp, seqlen_k_start);

        clear_tile(o_acc);

        auto seqlen_k_curr = seqlen_k_start;

        // Running K one stage ahead (issued during the previous iteration's gemm_1) only
        // pays off without a causal mask. Measured on gfx1250, b=512/64, hdim 128: the
        // K prefetch gains 7.9% (seqlen 128) / 10.7% (seqlen 1024) without a mask but
        // costs 4.9% / 7.6% with one, so it is gated on the mask mode. The V prefetch
        // below is unconditional -- it wins in every measured configuration.
        constexpr bool kPrefetchK = !kHasCausal;

        // With n0_loops == k1_loops == 1 (hdim 128) K only ever lands in buffer 0 and V in
        // buffer 2, leaving buffers 1 and 3 unused. Ping-pong across kv-tiles instead: issue
        // K[t+1] and V[t+1] into the spare half right after the single barrier of iteration
        // t, so both descriptors stay in flight for the whole iteration. Measured on a
        // standalone TDM micro-benchmark, going from 1 to 2 outstanding descriptors nearly
        // doubles the raw transfer rate (+79% cache-resident / +99% DRAM-bound).
        constexpr bool kDeepPipeline = (n0_loops == 1 && NumKVLdsBuffers >= 4);

        [[maybe_unused]] auto k_lds_cur  = k_lds_windows[number<0>{}];
        [[maybe_unused]] auto k_lds_next = k_lds_windows[number<1 % NumKVLdsBuffers>{}];
        [[maybe_unused]] auto v_lds_cur  = v_lds_windows[number<2 % NumKVLdsBuffers>{}];
        [[maybe_unused]] auto v_lds_next = v_lds_windows[number<3 % NumKVLdsBuffers>{}];

        if constexpr(kDeepPipeline)
        {
            load_tile_tdm(tdm_config_kv, k_lds_cur, k_dram_window);
            move_tile_window(k_dram_window, {kN0Sub, 0});
            load_tile_tdm(tdm_config_kv, v_lds_cur, v_dram_window);
            move_tile_window(v_dram_window, {kK1, 0});
        }
        else if constexpr(kPrefetchK)
        {
            load_tile_tdm(tdm_config_kv, k_lds_windows[number<0>{}], k_dram_window);
            move_tile_window(k_dram_window, {kN0Sub, 0});
        }

        do
        {
            // STAGE 1, Gemm_0 ( S = Q@K )
            static_for<0, n0_loops, 1>{}([&](auto i_n0) {
                if constexpr(kDeepPipeline)
                {
                    // K[t]/V[t] were issued one kv-tile ago. This is the only barrier of the
                    // iteration: it retires them (RAW for both gemms) and, because every
                    // reader of the spare half is a gemm of iteration t-1, it doubles as the
                    // WAR fence for the K[t+1]/V[t+1] writes issued right below.
                    s_wait_tensorcnt_barrier<0>();

                    load_tile_tdm(tdm_config_kv, k_lds_next, k_dram_window);
                    move_tile_window(k_dram_window, {kN0Sub, 0});
                    load_tile_tdm(tdm_config_kv, v_lds_next, v_dram_window);
                    move_tile_window(v_dram_window, {kK1, 0});

                    __builtin_amdgcn_sched_barrier(0x00000001);

                    gemm_0(sacc_tile, q_tile, k_lds_cur);
                }
                else
                {
                // K goes global -> LDS through the TDM engine, no register staging.
                if constexpr(!kPrefetchK)
                {
                    load_tile_tdm(tdm_config_kv,
                                  k_lds_windows[number<i_n0 % NumKVLdsBuffers>{}],
                                  k_dram_window);
                    move_tile_window(k_dram_window, {kN0Sub, 0});

                    __builtin_amdgcn_sched_barrier(0x00000001);
                }

                // TDM writes are tracked by tensorcnt, not vmcnt. When K is prefetched,
                // the only descriptors still in flight ahead of K[i_n0] are the i_n0 V
                // transfers issued below, and TDM retires them in order within a wave.
                s_wait_tensorcnt_barrier<kPrefetchK ? index_t{i_n0} : 0>();

                if constexpr(kPrefetchK && i_n0 + 1 < n0_loops)
                {
                    load_tile_tdm(tdm_config_kv,
                                  k_lds_windows[number<(i_n0 + 1) % NumKVLdsBuffers>{}],
                                  k_dram_window);
                    move_tile_window(k_dram_window, {kN0Sub, 0});
                }

                // Issue the matching V DMA right here instead of in STAGE 3: its LDS
                // buffer ((i+2)%4) is disjoint from the K buffers, and the barrier just
                // above is the WAR fence against the previous iteration's gemm_1 reads.
                // The transfer then overlaps gemm_0 + the whole of STAGE 2.
                load_tile_tdm(tdm_config_kv,
                              v_lds_windows[number<(i_n0 + 2) % NumKVLdsBuffers>{}],
                              v_dram_window);
                move_tile_window(v_dram_window, {kK1, 0});

                __builtin_amdgcn_sched_barrier(0x00000001);

                // execute current unroll of gemm_0
                gemm_0(sacc_tile, q_tile, k_lds_windows[number<i_n0 % NumKVLdsBuffers>{}]);
                }

                auto tmp_tile = cast_tile<CompDataType>(sacc_tile);

                set_slice_tile(pcomp_tile,
                               tmp_tile,
                               sequence<0, i_n0 * kN0Sub>{},
                               sequence<kM0, (i_n0 + 1) * kN0Sub>{});
            });

            __builtin_amdgcn_sched_barrier(0x00000001);

            // STAGE 2, scale_s, add bias, mask, siLU
            if constexpr(kHasBias)
            {
                const auto bias_tile = load_tile(bias_dram_window);

                tile_elementwise_inout(
                    [&scale_s](auto& x, const auto& y) {
                        x = x * scale_s + type_convert<CompDataType>(y);
                    },
                    pcomp_tile,
                    bias_tile);

                move_tile_window(bias_dram_window, {0, kN0});
            }
            else
            {
                tile_elementwise_inout([&scale_s](auto& x) { x = x * scale_s; }, pcomp_tile);
            }

            if(!mask.IsFullTileInsideMask(
                   q_origin.at(number<0>{}), seqlen_k_curr, number<kN0>{}, number<kM0>{}))
            {
                constexpr auto p_spans = PcompBlockTileType::get_distributed_spans();
                sweep_tile_span(p_spans[number<0>{}], [&](auto idx0) {
                    sweep_tile_span(p_spans[number<1>{}], [&](auto idx1) {
                        const auto tile_idx = get_x_indices_from_distributed_indices(
                            pcomp_tile.get_tile_distribution(),
                            make_tuple(idx0, idx1),
                            partition_index);

                        const auto row = q_origin.at(number<0>{}) + tile_idx.at(number<0>{});
                        const auto col = seqlen_k_curr + tile_idx.at(number<1>{});
                        constexpr auto i_j_idx = make_tuple(idx0, idx1);

                        if(!mask.IsTokenPairInsideMask(row, col))
                        {
                            pcomp_tile(i_j_idx) = type_convert<CompDataType>(0.0f);
                        };
                    });
                });
            }

            tile_elementwise_inout(f_silu, pcomp_tile);

            detail::scale_tile_in_pack(pcomp_tile, scale_p);

            if constexpr(kHasDropout)
            {
                __builtin_amdgcn_sched_barrier(0);

                auto randval_lds_ptr =
                    reinterpret_cast<char*>(smem_ptr) +
                    Policy::template GetSmemSizeKV<Problem,
                                                   true /*kPipelineUseTrLoad*/,
                                                   kUsePlainLds>();

                dropout.template Run<Gemm0Combined, CompDataType, uint8_t>(
                    randval_lds_ptr, seqlen_k_curr, pcomp_tile, null_randval_window);

                __builtin_amdgcn_sched_barrier(0);
            }

            seqlen_k_curr += kN0;

            auto p = cast_tile<PDataType>(pcomp_tile);

            // check whether first V-LdsBufer overlap with last K-LdsBuffer,
            // this does not occur when k1_loops == 2 and NumKVLdsBuffers == 4
            if constexpr((k1_loops - 1) % NumKVLdsBuffers == 2 % NumKVLdsBuffers)
            {
                __builtin_amdgcn_s_barrier();
            };

            // STAGE 3, Gemm_1 ( O = P@V )
            static_for<0, k1_loops, 1>{}([&](auto i_k1) {
                if constexpr(kDeepPipeline)
                {
                    // V[t] was already retired by the single barrier in STAGE 1, and the
                    // in-flight K[t+1]/V[t+1] target the other half of the buffers, so
                    // gemm_1 needs neither a wait nor a barrier of its own.
                    gemm_1(o_acc,
                           get_slice_tile(
                               p, sequence<0, i_k1 * kK1>{}, sequence<kM0, (i_k1 + 1) * kK1>{}),
                           v_lds_cur);
                }
                else
                {
                // The V DMAs were all issued back in STAGE 1, in ascending i_k1 order.
                // TDM descriptors retire in order within a wave, so V[i_k1] is done once
                // at most the (k1_loops-1-i_k1) later ones are still outstanding.
                s_wait_tensorcnt_barrier<k1_loops - 1 - i_k1>();

                // On the last unroll, prefetch K[0] of the next kv-tile so that its
                // transfer overlaps this gemm_1. Issuing it behind the barrier above --
                // rather than adding a barrier of its own -- is what makes that barrier
                // double as the WAR fence against gemm_0's reads of the same LDS buffer.
                if constexpr(kPrefetchK && i_k1 + 1 == k1_loops)
                {
                    load_tile_tdm(tdm_config_kv, k_lds_windows[number<0>{}], k_dram_window);
                    move_tile_window(k_dram_window, {kN0Sub, 0});
                }

                __builtin_amdgcn_sched_barrier(0x00000001);

                gemm_1(
                    o_acc,
                    get_slice_tile(p, sequence<0, i_k1 * kK1>{}, sequence<kM0, (i_k1 + 1) * kK1>{}),
                    v_lds_windows[number<(i_k1 + 2) % NumKVLdsBuffers>{}]);
                }
            });

            if constexpr(kDeepPipeline)
            {
                // swap the halves: what was just consumed becomes the DMA target next round
                auto k_lds_tmp = k_lds_cur;
                k_lds_cur      = k_lds_next;
                k_lds_next     = k_lds_tmp;

                auto v_lds_tmp = v_lds_cur;
                v_lds_cur      = v_lds_next;
                v_lds_next     = v_lds_tmp;
            }
        } while(seqlen_k_curr < seqlen_k_end);

        return o_acc;
    }
};

} // namespace ck_tile
