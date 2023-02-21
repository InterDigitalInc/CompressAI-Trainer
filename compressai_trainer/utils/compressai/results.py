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
from typing import Any

import compressai
import numpy as np
import pandas as pd


def compressai_dataframe(
    codec_name: str,
    dataset: str = "image/kodak",
    opt_metric: str = "mse",
    device: str = "cuda",
) -> pd.DataFrame:
    """Returns a dataframe containing the results of a given codec."""
    generic_codecs = ["av1", "hm", "jpeg", "jpeg2000", "vtm", "webp"]

    if codec_name in generic_codecs:
        d = generic_codec_result(codec_name, dataset=dataset)
    else:
        d = deep_codec_result(
            codec_name, dataset=dataset, opt_metric=opt_metric, device=device
        )

    d["results"] = _process_results(_rename_results(d["results"]))
    df = pd.DataFrame.from_dict(d["results"])
    df["name"] = d["name"]
    df["model.name"] = d["name"]
    df["description"] = d["description"]
    return df


def deep_codec_result(
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
