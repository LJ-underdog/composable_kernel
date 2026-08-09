// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#include "hstu_attention_bool_switch.hpp"
#include "hstu_attention_hdim_switch.hpp"

#include "instances/hstu_attention_batched_backward_bf16_instances_ref.hpp"
#include "instances/hstu_attention_jagged_backward_bf16_instances_ref.hpp"

#if defined(HSTU_BWD_SINGLE_KERNEL)
// flag=ON 时引入 single dispatch 的 extern template 声明，使本 TU 复用
// instances_single/ 下预编译的显式实例化（对应下方 run_batched_backward_single_dispatch）。
#include "instances_single/hstu_attention_batched_backward_single_bf16_instances_ref.hpp"
#endif

void hstu_attention_no_group_backward_bf16(HstuAttentionNoGroupBwdParams& param, hipStream_t stream)
{
    bool has_dropout = (param.p_drop > 0.0f);

    constexpr bool kHasBias = false;
    BOOL_SWITCH_3(param.use_causal,
                  kUseCausal,
                  param.use_softmax,
                  kUseSoftmax,
                  has_dropout,
                  kHasDropout,
                  [&] {
                      HDIM_SWITCH(param.hdim_qk, param.hdim_v, MaxK, [&] {
#if defined(HSTU_BWD_SINGLE_KERNEL)
                          // single kernel 现已支持 dropout（第 6 模板轴 kHasDropout，
                          // pipeline 侧用 BlockDropoutBwd 的 M-major 变体重算 mask）
                          // 与 jagged（运行期 param.is_jagged）⇒ no-group bwd 无条件走
                          // single；下面的 base 原路只在 flag=OFF 时编译。
                          // 第 5 轴 kIsDeterministic 是 OURS 专有轴（qf/base 没有）。
                          BOOL_SWITCH(param.kIsDeterministic, kIsDeterministic, [&] {
                              run_batched_backward_single_dispatch<ck_tile::bf16_t,
                                                                   kUseCausal,
                                                                   kUseSoftmax,
                                                                   kHasBias,
                                                                   kIsDeterministic,
                                                                   kHasDropout,
                                                                   MaxK>(param, stream);
                          });
                          return;
#endif
                          if(param.is_jagged)
                              run_jagged_backward_dispatch<ck_tile::bf16_t,
                                                           kUseCausal,
                                                           kUseSoftmax,
                                                           kHasBias,
                                                           kHasDropout,
                                                           MaxK>(param, stream);
                          else
                              run_batched_backward_dispatch<ck_tile::bf16_t,
                                                            kUseCausal,
                                                            kUseSoftmax,
                                                            kHasBias,
                                                            kHasDropout,
                                                            MaxK>(param, stream);
                      });
                  });
}
