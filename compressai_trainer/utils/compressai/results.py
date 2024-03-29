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
import os
from typing import Any, Optional

import compressai
import numpy as np
import pandas as pd

DEFAULT_RESULTS_ROOT = f"{compressai.__path__[0]}/../results"


def compressai_results_dataframe(
    filename: str,
    base_path: Optional[str] = None,
) -> pd.DataFrame:
    """Returns a dataframe containing the results from the given path."""
    if base_path is None:
        base_path = DEFAULT_RESULTS_ROOT
    with open(os.path.join(base_path, filename)) as f:
        d = json.load(f)
    df = _compressai_results_json_to_dataframe(d)
    return df


def _compressai_results_json_to_dataframe(d: dict[str, Any]) -> pd.DataFrame:
    d["results"] = _process_results(_rename_results(d["results"]))
    df = pd.DataFrame.from_dict(d["results"])
    df["name"] = d.get("name")
    df["model.name"] = d.get("meta", {}).get("model.name")
    df["description"] = d.get("description")
    return df


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
    if "ms-ssim" in results and "ms-ssim-db" not in results:
        # NOTE: The dB of the mean of MS-SSIM samples
        # is not the same as the mean of MS-SSIM dB samples.
        results["ms-ssim-db"] = (
            -10 * np.log10(1 - np.array(results["ms-ssim"]))
        ).tolist()
    return results
