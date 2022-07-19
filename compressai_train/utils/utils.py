# Copyright (c) 2021-2022, InterDigital Communications, Inc
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

import json
from typing import Any

import compressai
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


def preprocess_img(img: np.ndarray) -> torch.Tensor:
    x = (img.transpose(2, 0, 1) / 255).astype(np.float32)
    x = torch.from_numpy(x).unsqueeze(0)
    return x


def postprocess_img(x_hat: torch.Tensor) -> np.ndarray:
    x = x_hat.squeeze(0).numpy()
    x = (x.transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
    return x


@torch.no_grad()
def inference_single_image_uint8(
    model, img: np.ndarray, device=None
) -> tuple[np.ndarray, list[bytes]]:
    """Run inference on a single HWC uint8 RGB image."""
    x = preprocess_img(img)
    x = x.to(device=device)
    result = inference(model, x, skip_decompress=False)
    x_hat = result["out_dec"]["x_hat"].cpu()
    img_rec = postprocess_img(x_hat)
    encoded = [s[0] for s in result["out_enc"]["strings"]]
    return img_rec, encoded


@torch.no_grad()
def inference(model, x: torch.Tensor, skip_decompress: bool = False) -> dict[str, Any]:
    """Run compression model on image batch."""
    n, _, h, w = x.shape
    pad, unpad = _get_pad(h, w)

    x_padded = F.pad(x, pad, mode="constant", value=0)
    out_net = model(x_padded)
    out_net["x_hat"] = F.pad(out_net["x_hat"], unpad)
    out_enc = model.compress(x_padded)
    if skip_decompress:
        out_dec = dict(out_net)
        del out_dec["likelihoods"]
    else:
        out_dec = model.decompress(out_enc["strings"], out_enc["shape"])
    out_dec["x_hat"] = F.pad(out_dec["x_hat"], unpad)

    num_pixels = n * h * w
    num_bits = sum(sum(map(len, s)) for s in out_enc["strings"]) * 8.0
    bpp = num_bits / num_pixels

    return {
        "out_net": out_net,
        "out_enc": out_enc,
        "out_dec": out_dec,
        "bpp": bpp,
    }


def _get_pad(h, w):
    p = 64  # maximum 6 strides of 2

    new_h = (h + p - 1) // p * p
    new_w = (w + p - 1) // p * p

    left = (new_w - w) // 2
    right = new_w - w - left
    top = (new_h - h) // 2
    bottom = new_h - h - top

    pad = (left, right, top, bottom)
    unpad = (-left, -right, -top, -bottom)

    return pad, unpad


def compressai_dataframe(model_name: str, **kwargs):
    d = compressai_result(model_name, **kwargs)
    df = pd.DataFrame.from_dict(d["results"])
    df["name"] = d["name"]
    df["description"] = d["description"]
    return df


def compressai_result(
    model_name: str,
    dataset: str = "kodak",
    opt_metric: str = "mse",
    device: str = "cuda",
) -> dict[str, Any]:
    path = (
        f"{compressai.__path__[0]}/../results/{dataset}/"
        f"compressai-{model_name}_{opt_metric}_{device}.json"
    )

    with open(path) as f:
        return json.load(f)


def arg_pareto_optimal_set(xs, objectives):
    """Returns pareto-optimal set indexes."""
    if len(xs) != 2:
        raise NotImplementedError

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
    df: pd.DataFrame, y: str, y_metrics: list[dict[str, Any]]
) -> pd.DataFrame:
    """Returns dataframe prepared for plotting multiple metrics.

    Args:
        df: Dataframe.
        y: Destination y series.
        y_metrics:
            Source y series. Useful for plotting multiple curves of the
            same unit scale (e.g. dB) on the same plot.

    Examples:

    .. code-block:: python

        # Basic, single curve.
        [{"suffix": "", "y": "psnr"}]

        # Multiple series with different suffixes.
        [
            {"suffix": " (psnr_x)", "y": "psnr_x"},
            {"suffix": " (psnr_s)", "y": "psnr_s"},
        ]

        # Flatten multiple psnrs onto a single curve.
        [{"suffix": "", "y": ["psnr_0", "psnr_1", "psnr_2"]}]
    """
    records = []
    for record in df.to_dict("records"):
        for y_metric in y_metrics:
            base_record = dict(record)
            base_record["name"] = record["name"] + y_metric["suffix"]
            y_srcs = y_metric["y"]
            if isinstance(y_srcs, str):
                y_srcs = [y_srcs]
            for y_src in y_srcs:
                r = dict(base_record)
                r[y] = record[y_src]
                records.append(r)
    return pd.DataFrame.from_records(records)
