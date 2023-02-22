# Copyright (c) 2021-2023, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from __future__ import annotations

import math
import string
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class ConfigStringFormatter(string.Formatter):
    def get_field(self, field_name, args, kwargs):
        return self.get_value(field_name, args, kwargs), field_name


def np_img_to_tensor(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(x).moveaxis(-1, -3).to(torch.float32) / 255


def tensor_to_np_img(x: torch.Tensor) -> np.ndarray:
    return (x * 255).clip(0, 255).to(torch.uint8).moveaxis(-3, -1).cpu().numpy()


def compute_padding(in_h: int, in_w: int, *, out_h=None, out_w=None, min_div=1):
    """Returns tuples for padding and unpadding.

    NOTE: This is also available in ``compressai.ops`` as of v1.2.4.

    Args:
        in_h: Input height.
        in_w: Input width.
        out_h: Output height.
        out_w: Output width.
        min_div: Length that output dimensions should be divisible by.
    """
    if out_h is None:
        out_h = (in_h + min_div - 1) // min_div * min_div
    if out_w is None:
        out_w = (in_w + min_div - 1) // min_div * min_div

    if out_h % min_div != 0 or out_w % min_div != 0:
        raise ValueError(
            f"Padded output height and width are not divisible by min_div={min_div}."
        )

    left = (out_w - in_w) // 2
    right = out_w - in_w - left
    top = (out_h - in_h) // 2
    bottom = out_h - in_h - top

    pad = (left, right, top, bottom)
    unpad = (-left, -right, -top, -bottom)

    return pad, unpad


def num_parameters(net: nn.Module, predicate=lambda x: x.requires_grad) -> int:
    unique = {x.data_ptr(): x for x in net.parameters() if predicate(x)}.values()
    return sum(x.numel() for x in unique)


def arg_pareto_optimal_set(xs, objectives):
    """Returns pareto-optimal set indexes."""
    if len(xs) != 2:
        raise NotImplementedError

    if np.isnan(xs).any():
        raise ValueError("xs contains NaN values.")

    x, y = xs
    xo, yo = [-1 if o == "min" else 1 for o in objectives]
    perm = x.argsort()
    if xo == 1:
        perm = perm[::-1]
    yp = y[perm]
    best_y = -yo * np.inf
    idxs = []

    for curr_p, curr_y in zip(perm, yp):
        if yo * (curr_y - best_y) <= 0:
            continue
        best_y = curr_y
        idxs.append(curr_p)

    return idxs


def format_dataframe(
    df: pd.DataFrame,
    x: str,
    y: str,
    curves: list[dict[str, Any]],
    skip_nan: bool = True,
) -> pd.DataFrame:
    """Returns dataframe prepared for plotting multiple metrics.

    Args:
        df: Dataframe.
        x: Destination x series.
        y: Destination y series.
        curves:
            Source y series. Useful for plotting multiple curves of the
            same unit scale (e.g. dB) on the same plot.
        skip_nan: Skip accumulating NaN values into x, y series.

    Examples:

    .. code-block:: python

        # Basic, single curve.
        [{"name": "{experiment}", "x": "bpp", "y": "psnr"}]

        # Multiple series with different suffixes.
        [
            {"name": "{experiment} (RGB-PSNR)", "x": "bpp", "y": "psnr_rgb"},
            {"name": "{experiment} (YUV-PSNR)", "x": "bpp", "y": "psnr_yuv"},
        ]

        # Flatten multiple bpps/psnrs onto a single curve.
        [
            {
                "name": "{experiment}",
                "x": ["bpp_0", "bpp_1", "bpp_2"],
                "y": ["psnr_0", "psnr_1", "psnr_2"],
            }
        ]
    """
    formatter = ConfigStringFormatter()
    records = []
    for record in df.to_dict("records"):
        for curve in curves:
            for x_src, y_src in zip(_coerce_list(curve["x"]), _coerce_list(curve["y"])):
                r = dict(record)
                fmt = curve.get("name", "{name}")
                r["name"] = formatter.vformat(fmt, [], record)
                r[x] = record[x_src]
                r[y] = record[y_src]
                if skip_nan and (_is_nan(r[x]) or _is_nan(r[y])):
                    continue
                records.append(r)
    return pd.DataFrame.from_records(records)


def _is_nan(x):
    return x is None or math.isnan(x)


def _coerce_list(x):
    if isinstance(x, list):
        return x
    return [x]
