# noqa: C801
# Copyright (c) 2023-2024, Advanced Micro Devices, Inc. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#

import os
from pathlib import Path
from typing import List

HSTU_COPYRIGHT_HEADER = """
// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

// The file is automatically generated, don't modify!
// See the generator script
// `{file}`
""".format(
    file=os.path.relpath(os.path.realpath(__file__), start=Path(__file__).parents[4])
)

HSTU_FORWARD_INSTANCE_TEMPLATE_INC = """
#include <ck_tile/core/numeric/{dtype_file}.hpp>
#include \"hstu_attention_{mode}_forward_dispatch.hpp\"
#include \"hstu_attention_params.hpp\"
"""

HSTU_FORWARD_INSTANCE_TEMPLATE = """
{extern}template void run_{mode}_forward_causal_softmax_bias_dropout_dispatch<
    {dtype},
    {has_causal},
    {use_softmax},
    {store_lse},
    {has_bias},
    {has_dropout},
    {max_k}>(HstuAttention{group_or_not}FwdParams& param, hipStream_t stream);
"""

HSTU_FORWARD_INSTANCE_FNAME = (
    "hstu_attention_{mode}_forward_{dtype_str}_{has_or_no_causal_str}_{use_softmax_or_not_str}_"
    "{store_lse_or_not_str}_{has_or_no_bias_str}_{has_or_no_dropout_str}_{max_k_str}.cpp"
)

HSTU_INSTANCE_REF_FNAME = "hstu_attention_{mode}_{function}_{dtype}_instances_ref.hpp"

BOOL_MAP = {True: "true", False: "false"}

BOOL_MAP_CAUSAL = {
    True: "has_causal",
    False: "no_causal",
}

BOOL_MAP_SOFTMAX = {
    True: "softmax_true",
    False: "softmax_false",
}

BOOL_MAP_LSE = {
    True: "lse_true",
    False: "lse_false",
}

BOOL_MAP_BIAS = {
    True: "has_bias",
    False: "no_bias",
}

BOOL_MAP_DROPOUT = {
    True: "has_dropout",
    False: "no_dropout",
}

INT_MAP_MAX_K = {hd: f"maxk_{hd}" for hd in [64, 96, 128, 256]}

TYPE_CTYPE_MAP = {
    "fp16": "ck_tile::fp16_t",
    "bf16": "ck_tile::bf16_t",
}

TYPE_FNAME_MAP = {
    "fp16": "half",
    "bf16": "bfloat16",
}

MODE_GROUP_OR_NOT_MAP = {
    "batched": "NoGroup",
    "jagged": "NoGroup",
    "group": "Group",
}


def create_forward_instances(instance_dir: Path, headdims: List) -> None:
    for mode in ["batched", "jagged", "group"]:
        for dtype in ["fp16", "bf16"]:
            for has_causal in [True, False]:
                for use_softmax, store_lse in [
                    (True, False),
                    (True, True),
                    (False, False),
                ]:
                    for has_bias in [True, False]:
                        for has_dropout in [False]:
                            for max_k in headdims:
                                fname = HSTU_FORWARD_INSTANCE_FNAME.format(
                                    mode=mode,
                                    dtype_str=dtype,
                                    has_or_no_causal_str=BOOL_MAP_CAUSAL[has_causal],
                                    use_softmax_or_not_str=BOOL_MAP_SOFTMAX[
                                        use_softmax
                                    ],
                                    store_lse_or_not_str=BOOL_MAP_LSE[store_lse],
                                    has_or_no_bias_str=BOOL_MAP_BIAS[has_bias],
                                    has_or_no_dropout_str=BOOL_MAP_DROPOUT[has_dropout],
                                    max_k_str=INT_MAP_MAX_K[max_k],
                                )
                                forward_instance_inc = (
                                    HSTU_FORWARD_INSTANCE_TEMPLATE_INC.format(
                                        mode=mode,
                                        dtype_file=TYPE_FNAME_MAP[dtype],
                                    )
                                )
                                forward_instance = (
                                    HSTU_FORWARD_INSTANCE_TEMPLATE.format(
                                        extern="",
                                        mode=mode,
                                        dtype=TYPE_CTYPE_MAP[dtype],
                                        has_causal=BOOL_MAP[has_causal],
                                        use_softmax=BOOL_MAP[use_softmax],
                                        store_lse=BOOL_MAP[store_lse],
                                        has_bias=BOOL_MAP[has_bias],
                                        has_dropout=BOOL_MAP[has_dropout],
                                        max_k=max_k,
                                        group_or_not=MODE_GROUP_OR_NOT_MAP[mode],
                                    )
                                )
                                (instance_dir / fname).write_text(
                                    HSTU_COPYRIGHT_HEADER
                                    + forward_instance_inc
                                    + forward_instance
                                )


HSTU_BACKWARD_INSTANCE_TEMPLATE_INC = """
#include <ck_tile/core/numeric/{dtype_file}.hpp>
#include \"hstu_attention_{mode}_backward_dispatch.hpp\"
"""

HSTU_BACKWARD_INSTANCE_TEMPLATE = """
{extern}template void run_{mode}_backward_dispatch<
    {dtype},
    {has_causal},
    {use_softmax},
    {has_bias},
    {is_deterministic},
    {max_k}>(HstuAttention{group_or_not}BwdParams& param, hipStream_t stream);
"""

HSTU_BACKWARD_INSTANCE_FNAME = (
    "hstu_attention_{mode}_backward_{dtype_str}_{has_or_no_causal_str}_{use_softmax_or_not_str}_"
    "{has_or_no_bias_str}_{deterministic_str}_{max_k_str}.cpp"
)

BOOL_MAP_DETERMINISTIC = {
    True: "deterministic",
    False: "atomic",
}

# M0 backward axes (DESIGN §4.5 MVP, scaffold subset):
#   mode=batched, dtype=bf16, bias=false, deterministic=false, maxk=64,
#   causal x softmax = the 4 combos the NoGroup bf16 entry switches over.
BWD_MODES_M0 = ["batched"]
BWD_DTYPES_M0 = ["bf16"]
BWD_HEADDIMS_M0 = [64]


def create_backward_instances(instance_dir: Path) -> None:
    for mode in BWD_MODES_M0:
        for dtype in BWD_DTYPES_M0:
            for has_causal in [True, False]:
                for use_softmax in [True, False]:
                    for has_bias in [False]:
                        for is_deterministic in [False]:
                            for max_k in BWD_HEADDIMS_M0:
                                fname = HSTU_BACKWARD_INSTANCE_FNAME.format(
                                    mode=mode,
                                    dtype_str=dtype,
                                    has_or_no_causal_str=BOOL_MAP_CAUSAL[has_causal],
                                    use_softmax_or_not_str=BOOL_MAP_SOFTMAX[use_softmax],
                                    has_or_no_bias_str=BOOL_MAP_BIAS[has_bias],
                                    deterministic_str=BOOL_MAP_DETERMINISTIC[
                                        is_deterministic
                                    ],
                                    max_k_str=INT_MAP_MAX_K[max_k],
                                )
                                inc = HSTU_BACKWARD_INSTANCE_TEMPLATE_INC.format(
                                    mode=mode,
                                    dtype_file=TYPE_FNAME_MAP[dtype],
                                )
                                body = HSTU_BACKWARD_INSTANCE_TEMPLATE.format(
                                    extern="",
                                    mode=mode,
                                    dtype=TYPE_CTYPE_MAP[dtype],
                                    has_causal=BOOL_MAP[has_causal],
                                    use_softmax=BOOL_MAP[use_softmax],
                                    has_bias=BOOL_MAP[has_bias],
                                    is_deterministic=BOOL_MAP[is_deterministic],
                                    max_k=max_k,
                                    group_or_not=MODE_GROUP_OR_NOT_MAP[mode],
                                )
                                (instance_dir / fname).write_text(
                                    HSTU_COPYRIGHT_HEADER + inc + body
                                )


def create_backward_instances_ref(instance_dir: Path) -> None:
    for mode in BWD_MODES_M0:
        for dtype in BWD_DTYPES_M0:
            ref_fname = HSTU_INSTANCE_REF_FNAME.format(
                mode=mode,
                function="backward",
                dtype=dtype,
            )
            ref_fname_path = instance_dir / ref_fname
            inc = HSTU_BACKWARD_INSTANCE_TEMPLATE_INC.format(
                mode=mode,
                dtype_file=TYPE_FNAME_MAP[dtype],
            )
            with open(ref_fname_path, "a") as file:
                file.write(HSTU_COPYRIGHT_HEADER)
                file.write(inc)
                for max_k in BWD_HEADDIMS_M0:
                    for has_bias in [False]:
                        for is_deterministic in [False]:
                            for has_causal in [True, False]:
                                for use_softmax in [True, False]:
                                    file.write(
                                        HSTU_BACKWARD_INSTANCE_TEMPLATE.format(
                                            extern="extern ",
                                            mode=mode,
                                            dtype=TYPE_CTYPE_MAP[dtype],
                                            has_causal=BOOL_MAP[has_causal],
                                            use_softmax=BOOL_MAP[use_softmax],
                                            has_bias=BOOL_MAP[has_bias],
                                            is_deterministic=BOOL_MAP[is_deterministic],
                                            max_k=max_k,
                                            group_or_not=MODE_GROUP_OR_NOT_MAP[mode],
                                        )
                                    )


def create_forward_instances_ref(instance_dir: Path, headdims: List) -> None:
    for mode in ["batched", "jagged", "group"]:
        for dtype in ["fp16", "bf16"]:
            ref_fname = HSTU_INSTANCE_REF_FNAME.format(
                mode=mode,
                function="forward",
                dtype=dtype,
            )
            ref_fname_path = instance_dir / ref_fname
            forward_instance_inc = HSTU_FORWARD_INSTANCE_TEMPLATE_INC.format(
                mode=mode,
                dtype_file=TYPE_FNAME_MAP[dtype],
            )
            with open(ref_fname_path, "a") as file:
                file.write(HSTU_COPYRIGHT_HEADER)
                file.write(forward_instance_inc)
                for max_k in headdims:
                    for has_bias in [True, False]:
                        for has_dropout in [False]:
                            for has_causal in [True, False]:
                                for use_softmax, store_lse in [
                                    (True, False),
                                    (True, True),
                                    (False, False),
                                ]:
                                    forward_instance = (
                                        HSTU_FORWARD_INSTANCE_TEMPLATE.format(
                                            extern="extern ",
                                            mode=mode,
                                            dtype=TYPE_CTYPE_MAP[dtype],
                                            has_causal=BOOL_MAP[has_causal],
                                            use_softmax=BOOL_MAP[use_softmax],
                                            store_lse=BOOL_MAP[store_lse],
                                            has_bias=BOOL_MAP[has_bias],
                                            has_dropout=BOOL_MAP[has_dropout],
                                            max_k=max_k,
                                            group_or_not=MODE_GROUP_OR_NOT_MAP[mode],
                                        )
                                    )
                                    file.write(forward_instance)


if __name__ == "__main__":
    headdims_fwd = [64, 96, 128, 256]

    this_dir = os.path.dirname(__file__)
    output_dir = Path(this_dir) / "instances"
    output_dir.mkdir(parents=True, exist_ok=True)

    # remove existing files in the directory
    files = os.listdir(output_dir)
    for ff in files:
        file_path = os.path.join(output_dir, ff)
        os.remove(file_path)

    create_forward_instances(output_dir, headdims_fwd)
    create_forward_instances_ref(output_dir, headdims_fwd)

    # HSTU backward instances (M0 scaffold subset)
    create_backward_instances(output_dir)
    create_backward_instances_ref(output_dir)
