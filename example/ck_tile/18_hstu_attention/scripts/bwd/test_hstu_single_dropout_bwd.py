#!/usr/bin/env python3
"""HSTU bwd dropout — 补充覆盖 case(卡 DROP-T4)。

为什么有这个脚本
----------------
既有 `test_hstu_attention_with_dropout_bwd.py` 的 28 条(14 形状 × fp16/bf16)
覆盖了 causal / local_len / context_len / minfull_len / targets / jagged,
但有几个维度**一条都没有**(实测自 runs/V6-F2REDO/V6-ledger-FINAL-166.csv):

  * `-softmax`     28 条全部未给 => 默认 0 => **全是 SiLU 路**,softmax 路零覆盖
  * `-deterministic` 28 条全部未给 => 默认 0 => 只测 atomic dQ
  * `-p_drop`      只有 0.2 一个取值
  * `-hdim_qk/-hdim_v` 只有 128/128
  * jagged 的 `-b=10` 只配 5 个 seqlens 元素,被 example_helper.hpp:111-120 的
    supplement_array_by_last_element 静默补齐成等长尾巴,且首两元素相同
    => "名字叫 jagged 其实大部分定长"

本脚本只补这些空档,不重复既有 28 条已覆盖的形状。

约定(必须遵守,否则 run_v6.py 收割不到)
--------------------------------------
  * 模块级 EXE 变量 + cmd 首 token 是 exe(run_v6.py:353 整体替换首 token)
  * 只用 `import subprocess` + `subprocess.run(...)`,不用 os.system / Popen
  * main() 不依赖位置参数 —— run_v6.py:375 用 `sys.argv = [script_path]` 调用,
    **任何位置参数都拿不到默认值以外的东西**。所以形状差异一律写进 CASES。

参数一律显式给全,不吃默认值(默认值坑:softmax=0 / causal=1 / local_len=5 /
context_len=6 / minfull_len=6,见 example_hstu_attention_bwd.cpp:71-79)。
"""

import subprocess
import sys

BUILD = "build"
EXE = f"{BUILD}/bin/tile_example_hstu_attention_bwd"

# jagged 用:元素数 == -b 且两两不等,避开静默补齐 + 避开重复长度
JAGGED_6 = "300,291,277,256,312,264"

# 每条 case 显式写全下列参数;`why` 只是注释,不进命令行。
CASES = [
    # ===================== P0-A: softmax=1(28 条零覆盖的主路)=====================
    # 【T5 裁定②】原为 seqlens_kv="" 的自注意力最平形态。问题:harness 用同一 seed 填
    # Q/K/V/dO(example_hstu_attention_bwd.cpp:381-384)⇒ Q=K=V=dO,无 mask 时 S=Q·Qᵀ 对称
    # ⇒ dQ≡dK 数学恒等(docs/DETERM-DQDK.md §4.2),会掩盖「dK 被写成 dQ」这类缺陷。
    # 解法:给 KV 一个不同长度(-seqlens_kv)=> cross-attention,S 不再是方阵,
    # dQ(256×d) 与 dK(320×d) 形状都不同,对称退化被结构性打破。
    # 选它而不选 causal=1,是因为 causal=1 会让本条与下面第 2 条完全重复。
    dict(why="P0 softmax路 最平 mask 形态(cross:seqlens_kv≠seqlens,打破 dQ≡dK 对称退化)",
         softmax=1, prec="fp16", b=10, jagged=0, seqlens="256", seqlens_kv="320", nhead=4,
         hdim_qk=128, hdim_v=128, causal=0, local_len=0, context_len=0,
         minfull_len=0, targets=0, p_drop=0.2, deterministic=0),
    dict(why="P0 softmax路 + causal",
         softmax=1, prec="fp16", b=10, jagged=0, seqlens="256", nhead=4,
         hdim_qk=128, hdim_v=128, causal=1, local_len=0, context_len=0,
         minfull_len=0, targets=0, p_drop=0.2, deterministic=0),
    dict(why="P0 softmax路 + causal+local+context+minfull(minfull_len>0 首次与 softmax 同现)",
         softmax=1, prec="fp16", b=10, jagged=0, seqlens="256", nhead=4,
         hdim_qk=128, hdim_v=128, causal=1, local_len=5, context_len=8,
         minfull_len=7, targets=0, p_drop=0.2, deterministic=0),
    dict(why="P0 softmax路 + 全 mask 开 + targets",
         softmax=1, prec="fp16", b=10, jagged=0, seqlens="256", nhead=4,
         hdim_qk=128, hdim_v=128, causal=1, local_len=5, context_len=8,
         minfull_len=7, targets=8, p_drop=0.2, deterministic=0),
    dict(why="P0 softmax路 bf16(另一套实例)",
         softmax=1, prec="bf16", b=10, jagged=0, seqlens="256", nhead=4,
         hdim_qk=128, hdim_v=128, causal=1, local_len=5, context_len=8,
         minfull_len=7, targets=8, p_drop=0.2, deterministic=0),

    # ========== P0-B: minfull_len > max_uih(踩 kernel 层 split × dropout 的 4 处分支)==========
    # bwd_kernel_1.hpp:683/838/867/1057 在 kHasDropout=true 时关掉 split 优化。
    # SiLU 路已有 6 条 minfull_len=290;softmax 路一条都没有。
    dict(why="P0 softmax路 + minfull_len=290 > uih(split 优化被 dropout 关掉的分支)",
         softmax=1, prec="fp16", b=6, jagged=1, seqlens=JAGGED_6, nhead=4,
         hdim_qk=128, hdim_v=128, causal=1, local_len=5, context_len=0,
         minfull_len=290, targets=8, p_drop=0.2, deterministic=0),
    dict(why="P0 softmax路 + minfull_len=290 + context(同上,加 context 改变物理长度)",
         softmax=1, prec="bf16", b=6, jagged=1, seqlens=JAGGED_6, nhead=4,
         hdim_qk=128, hdim_v=128, causal=1, local_len=5, context_len=8,
         minfull_len=290, targets=8, p_drop=0.2, deterministic=0),
    dict(why="P0 SiLU路 + minfull_len=290 + 真变长 jagged(基线那 6 条被静默补齐,这条不是)",
         softmax=0, prec="fp16", b=6, jagged=1, seqlens=JAGGED_6, nhead=4,
         hdim_qk=128, hdim_v=128, causal=1, local_len=5, context_len=0,
         minfull_len=290, targets=8, p_drop=0.2, deterministic=0),

    # ===================== P1: deterministic=1(第5轴 × 第6轴同时 true)=====================
    # 【T5 裁定①】seqlens 原为 256。hd128 的 kN0=128(hstu_attention_bwd_shape.hpp:108)
    # ⇒ num_splits = ceil(256/128) = 2(example_hstu_attention_bwd.cpp:412-413)。
    # num_splits ≤ 2 时 determ=0 与 =1 **必然**逐字节相同:两项相加,IEEE-754 加法可交换
    # (a+b == b+a 逐位相等),只有 ≥3 项才因不可结合而顺序敏感。
    # ⇒ 256 对「determ 开关是否生效」零区分力,只会复制 docs/DETERM-FACTCHECK.md:37-40 的盲区。
    # 改 400 ⇒ ceil(400/128) = 4,进入顺序敏感区。
    dict(why="P1 determ+dropout, softmax路 —— 两个模板轴同时为 true 的实例从未被跑过(seqlens=400 使 num_splits=4)",
         softmax=1, prec="fp16", b=10, jagged=0, seqlens="400", nhead=4,
         hdim_qk=128, hdim_v=128, causal=1, local_len=5, context_len=8,
         minfull_len=7, targets=8, p_drop=0.2, deterministic=1),
    dict(why="P1 determ+dropout, SiLU路(seqlens=400 使 num_splits=4,同上)",
         softmax=0, prec="fp16", b=10, jagged=0, seqlens="400", nhead=4,
         hdim_qk=128, hdim_v=128, causal=1, local_len=5, context_len=8,
         minfull_len=7, targets=8, p_drop=0.2, deterministic=1),
    dict(why="P1 determ+dropout+minfull>uih:determ 的 split_stride 与 dropout 关 split 同时生效",
         softmax=1, prec="fp16", b=6, jagged=1, seqlens=JAGGED_6, nhead=4,
         hdim_qk=128, hdim_v=128, causal=1, local_len=5, context_len=0,
         minfull_len=290, targets=8, p_drop=0.2, deterministic=1),

    # ===================== P1: 真变长 jagged × dropout =====================
    dict(why="P1 真变长 jagged(b==len(seqlens),两两不等)+ softmax路",
         softmax=1, prec="fp16", b=6, jagged=1, seqlens=JAGGED_6, nhead=4,
         hdim_qk=128, hdim_v=128, causal=1, local_len=5, context_len=8,
         minfull_len=7, targets=8, p_drop=0.2, deterministic=0),
    dict(why="P1 真变长 jagged + SiLU路 + 无 mask(与基线第2条同形但不被补齐)",
         softmax=0, prec="fp16", b=6, jagged=1, seqlens=JAGGED_6, nhead=4,
         hdim_qk=128, hdim_v=128, causal=0, local_len=0, context_len=0,
         minfull_len=0, targets=0, p_drop=0.2, deterministic=0),

    # ===================== P2: p_drop 取值(uint8 阈值粒度 1/255)=====================
    # p_undrop_in_uint8_t = floor((1-p)*255),见 bwd_kernel_2.hpp:264-265
    dict(why="P2 p_drop=0.1(低丢弃率,keep 阈值 229)",
         softmax=1, prec="fp16", b=10, jagged=0, seqlens="256", nhead=4,
         hdim_qk=128, hdim_v=128, causal=1, local_len=5, context_len=8,
         minfull_len=7, targets=8, p_drop=0.1, deterministic=0),
    dict(why="P2 p_drop=0.5(半数丢弃)",
         softmax=1, prec="fp16", b=10, jagged=0, seqlens="256", nhead=4,
         hdim_qk=128, hdim_v=128, causal=1, local_len=5, context_len=8,
         minfull_len=7, targets=8, p_drop=0.5, deterministic=0),
    dict(why="P2 p_drop=0.9(高丢弃率,rp_undrop=10 放大误差,最容易暴露 scale 错)",
         softmax=1, prec="fp16", b=10, jagged=0, seqlens="256", nhead=4,
         hdim_qk=128, hdim_v=128, causal=1, local_len=5, context_len=8,
         minfull_len=7, targets=8, p_drop=0.9, deterministic=0),
    dict(why="P2 p_drop=0.9 SiLU路(SiLU 的 drop_scale 同时乘 P 和 dS,放大更敏感)",
         softmax=0, prec="fp16", b=10, jagged=0, seqlens="256", nhead=4,
         hdim_qk=128, hdim_v=128, causal=1, local_len=5, context_len=8,
         minfull_len=7, targets=8, p_drop=0.9, deterministic=0),

    # ===================== P2: hdim ≠ 128(另一套 MaxK 实例)=====================
    # 注意:hdim96 基线本就全 FAIL(阳性对照,JAG-ACCEPT §2.3 ②),故这里只取 64,
    # 不引入已知红,以免污染新功能信号。
    dict(why="P2 hdim=64(MaxK=64 实例,32x32x16 路)+ softmax路",
         softmax=1, prec="fp16", b=10, jagged=0, seqlens="256", nhead=4,
         hdim_qk=64, hdim_v=64, causal=1, local_len=5, context_len=8,
         minfull_len=7, targets=8, p_drop=0.2, deterministic=0),
    dict(why="P2 hdim=64 + SiLU路",
         softmax=0, prec="fp16", b=10, jagged=0, seqlens="256", nhead=4,
         hdim_qk=64, hdim_v=64, causal=1, local_len=5, context_len=8,
         minfull_len=7, targets=8, p_drop=0.2, deterministic=0),

    # ===================== P2: 规模轴(b / nhead 基线单一)=====================
    dict(why="P2 b=1 单 batch(边界:i_batch 恒 0,philox 的 batch 维退化)",
         softmax=1, prec="fp16", b=1, jagged=0, seqlens="256", nhead=4,
         hdim_qk=128, hdim_v=128, causal=1, local_len=5, context_len=8,
         minfull_len=7, targets=8, p_drop=0.2, deterministic=0),
    dict(why="P2 nhead=1(边界:philox 的 head 维退化,ph_head_offset 只剩 batch 项)",
         softmax=1, prec="fp16", b=10, jagged=0, seqlens="256", nhead=1,
         hdim_qk=128, hdim_v=128, causal=1, local_len=5, context_len=8,
         minfull_len=7, targets=8, p_drop=0.2, deterministic=0),

    # ===================== 对照:p_drop=0(必须与上面某条只差 p_drop)=====================
    # 供 pane2 的「判别力对照」判据使用:同形状下 p_drop=0.2 与 0 的输出必须不同。
    # 若两者输出相同 => dropout 根本没生效,全部绿色结论作废。
    dict(why="对照 p_drop=0(与 P0-A 第3条只差 p_drop,判别力对照的 B 侧)",
         softmax=1, prec="fp16", b=10, jagged=0, seqlens="256", nhead=4,
         hdim_qk=128, hdim_v=128, causal=1, local_len=5, context_len=8,
         minfull_len=7, targets=0, p_drop=0, deterministic=0),
    dict(why="对照 p_drop=0 SiLU路(与 P2 hdim=64 SiLU 条只差 p_drop)",
         softmax=0, prec="fp16", b=10, jagged=0, seqlens="256", nhead=4,
         hdim_qk=64, hdim_v=64, causal=1, local_len=5, context_len=8,
         minfull_len=7, targets=8, p_drop=0, deterministic=0),
]


def build_cmd(case):
    """把一条 case 展开成完整命令行。所有参数显式给出,不依赖任何默认值。

    `-seqlens_kv` 是唯一的可选项:给了就是 cross-attention
    (example_hstu_attention_bwd.cpp:186-190 仅凭该串非空判定,与 jagged 无关),
    不给则 harness 令 seqlens_kv = seqlens(自注意力)。默认不发这个 flag,
    以免把其余 22 条自注意力 case 悄悄变成 cross。
    """
    cmd = [
        EXE,
        "-v=1",                                   # 必须 =1,否则校验块整块跳过、RC=0 但什么都没验
        f"-prec={case['prec']}",
        f"-b={case['b']}",
        "-g=1",                                   # 显式 no-group(>1 会走 group 并忽略 -jagged)
        f"-jagged={case['jagged']}",
        f"-nhead={case['nhead']}",
        f"-hdim_qk={case['hdim_qk']}",
        f"-hdim_v={case['hdim_v']}",
        f"-seqlens={case['seqlens']}",
        f"-softmax={case['softmax']}",
        f"-p_drop={case['p_drop']}",
        f"-causal={case['causal']}",
        f"-local_len={case['local_len']}",
        f"-context_len={case['context_len']}",
        f"-minfull_len={case['minfull_len']}",
        f"-targets={case['targets']}",
        f"-deterministic={case['deterministic']}",
        "-attn_scale=0",
        "-norm_dist=0",
    ]
    if case.get("seqlens_kv"):
        cmd.insert(cmd.index(f"-seqlens={case['seqlens']}") + 1,
                   f"-seqlens_kv={case['seqlens_kv']}")
    return cmd


def _self_check():
    """静态自检:jagged 的 -b 必须等于 seqlens 元素数且两两不等;dense 必须单元素。

    这两条都是本项目咬过的坑 —— jagged 元素不足会被静默补齐成定长(假覆盖),
    dense 给多元素会直接 abort(RC=134)。宁可在这里失败,也不要跑出假绿。
    """
    for i, c in enumerate(CASES):
        for key in ("seqlens", "seqlens_kv"):
            if not c.get(key):
                continue
            ps = [p for p in str(c[key]).split(",") if p != ""]
            if c["jagged"]:
                assert len(ps) == c["b"], (
                    f"case[{i}] jagged: -b={c['b']} 与 {key} 元素数 {len(ps)} 不等,会被静默补齐")
                assert len(set(ps)) == len(ps), (
                    f"case[{i}] jagged: {key} 元素有重复 {ps},变长覆盖被削弱")
            else:
                assert len(ps) == 1, (
                    f"case[{i}] dense: -{key} 必须单元素,给了 {ps} 会 abort(RC=134)")
        # cross-attention 必须真的不等长,否则退化回自注意力、对称退化仍在
        if c.get("seqlens_kv"):
            assert str(c["seqlens_kv"]) != str(c["seqlens"]), (
                f"case[{i}] seqlens_kv == seqlens,cross 退化为自注意力,dQ≡dK 对称退化未打破")
        parts = [p for p in str(c["seqlens"]).split(",") if p != ""]
        if c["jagged"]:
            assert len(parts) == c["b"], (
                f"case[{i}] jagged: -b={c['b']} 与 seqlens 元素数 {len(parts)} 不等,"
                f"会被 supplement_array_by_last_element 静默补齐")
            assert len(set(parts)) == len(parts), (
                f"case[{i}] jagged: seqlens 元素有重复 {parts},变长覆盖被削弱")
        else:
            assert len(parts) == 1, (
                f"case[{i}] dense: -seqlens 必须单元素,给了 {parts} 会 abort(RC=134)")


def main():
    _self_check()
    rc = 0
    for case in CASES:
        cmd = build_cmd(case)
        print("+ " + " ".join(cmd), flush=True)
        result = subprocess.run(cmd)
        if result.returncode != 0:
            rc = result.returncode
    return rc


if __name__ == "__main__":
    sys.exit(main())
