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

from typing import Any

import pandas as pd
import torch
from compressai.entropy_models import EntropyBottleneck

_PLOT_DISTRIBUTIONS_EB_LINE_SETTINGS_COMMON = dict(
    x="dy",
    y="pdf",
    color="name",
    hover_data=["y", "dy", "pdf", "cdf", "median"],
    line_shape="hvh",
)

_PLOT_DISTRIBUTIONS_EB_LAYOUT_SETTINGS_COMMON = dict(
    xaxis_title="Δy = y - median",
    yaxis_title="p(Δy)",
)


def plot_entropy_bottleneck_distributions(
    entropy_bottleneck: EntropyBottleneck,
    scatter_kwargs: dict[str, Any] = {},
    layout_kwargs: dict[str, Any] = {},
):
    """Plots EntropyBottleneck distributions."""
    import plotly.express as px

    df = _get_entropy_bottleneck_distributions_dataframe(entropy_bottleneck)
    line_kwargs = {**_PLOT_DISTRIBUTIONS_EB_LINE_SETTINGS_COMMON, **scatter_kwargs}
    layout_kwargs = {**_PLOT_DISTRIBUTIONS_EB_LAYOUT_SETTINGS_COMMON, **layout_kwargs}
    fig = px.line(df, **line_kwargs)
    fig.update_layout(**layout_kwargs)

    return fig


def _get_entropy_bottleneck_distributions_dataframe(
    entropy_bottleneck: EntropyBottleneck,
) -> pd.DataFrame:
    c = entropy_bottleneck.channels
    q = entropy_bottleneck.quantiles[:, 0, :].detach()

    left = (q[:, 1] - q[:, 0]).ceil().int()
    right = (q[:, 2] - q[:, 1]).ceil().int()
    sizes = left + 1 + right
    max_size = sizes.max().item()
    num_samples = max_size

    t = torch.linspace(0, max_size - 1, num_samples, device=q.device)
    dy = t[None, :] - left[:, None]
    y = dy + q[:, 1, None]

    with torch.no_grad():
        _, y_likelihoods = entropy_bottleneck(y.unsqueeze(0), training=False)

    pdfs = y_likelihoods.squeeze(0)
    cdfs = y_likelihoods.squeeze(0).cumsum(-1)
    medians = q[:, 1, None].repeat(1, num_samples)
    name = [f"{i}" for i in range(c) for _ in range(sizes[i].item())]

    def trim(y):
        y = y.cpu().numpy()
        xss = [y[i, :l] for i, l in enumerate(sizes.cpu().tolist())]
        return [x for xs in xss for x in xs]

    d = {
        "name": name,
        "y": trim(y),
        "dy": trim(dy),
        "pdf": trim(pdfs),
        "cdf": trim(cdfs),
        "median": trim(medians),
    }

    df = pd.DataFrame.from_dict(d)

    return df
