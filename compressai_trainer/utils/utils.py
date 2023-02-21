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

import json
import math
import string
from typing import Any

import compressai
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConfigStringFormatter(string.Formatter):
    def get_field(self, field_name, args, kwargs):
        return self.get_value(field_name, args, kwargs), field_name


def np_img_to_tensor(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(x).moveaxis(-1, -3).to(torch.float32) / 255


def tensor_to_np_img(x: torch.Tensor) -> np.ndarray:
    return (x * 255).clip(0, 255).to(torch.uint8).moveaxis(-3, -1).cpu().numpy()


@torch.no_grad()
def inference_single_image_uint8(
    model, img: np.ndarray, device=None
) -> tuple[np.ndarray, list[bytes]]:
    """Run inference on a single HWC uint8 RGB image."""
    x = np_img_to_tensor(img[None, ...])
    x = x.to(device=device)
    result = inference(model, x, skip_decompress=False)
    x_hat = result["out_dec"]["x_hat"].cpu()
    img_rec = tensor_to_np_img(x_hat).squeeze(0)
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


def num_parameters(net: nn.Module, predicate=lambda x: x.requires_grad) -> int:
    unique = {x.data_ptr(): x for x in net.parameters() if predicate(x)}.values()
    return sum(x.numel() for x in unique)


def compressai_dataframe(
    codec_name: str,
    dataset: str = "image/kodak",
    opt_metric: str = "mse",
    device: str = "cuda",
):
    generic_codecs = ["av1", "hm", "jpeg", "jpeg2000", "vtm", "webp"]

    if codec_name in generic_codecs:
        d = generic_codec_result(codec_name, dataset=dataset)
    else:
        d = compressai_result(
            codec_name, dataset=dataset, opt_metric=opt_metric, device=device
        )

    d["results"] = _process_results(_rename_results(d["results"]))
    df = pd.DataFrame.from_dict(d["results"])
    df["name"] = d["name"]
    df["model.name"] = d["name"]
    df["description"] = d["description"]
    return df


def compressai_result(
    model_name: str,
    dataset: str = "image/kodak",
    opt_metric: str = "mse",
    device: str = "cuda",
) -> dict[str, Any]:
    path = (
        f"{compressai.__path__[0]}/../results/{dataset}/"
        f"compressai-{model_name}_{opt_metric}_{device}.json"
    )

    with open(path) as f:
        return json.load(f)


def generic_codec_result(
    codec_name: str,
    dataset: str = "image/kodak",
) -> dict[str, Any]:
    path = f"{compressai.__path__[0]}/../results/{dataset}/" f"{codec_name}.json"

    with open(path) as f:
        return json.load(f)


def _rename_results(results):
    """Adapter for different versions of CompressAI.

    Renames METRIC-rgb -> METRIC.

    https://github.com/InterDigitalInc/CompressAI/commit/3d3c9bbd92989b1cf19e122281161f7aac8ee769
    """
    for metric in ["psnr", "ms-ssim"]:
        if f"{metric}-rgb" not in results:
            continue
        results[f"{metric}"] = results[f"{metric}-rgb"]
        del results[f"{metric}-rgb"]
    return results


def _process_results(results):
    results["ms-ssim-db"] = (-10 * np.log10(1 - np.array(results["ms-ssim"]))).tolist()
    return results


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
