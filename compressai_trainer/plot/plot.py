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

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd


_PLOT_RD_SCATTER_SETTINGS_COMMON = dict(
    x="bpp",
    color="name",
    hover_data=["psnr", "ms-ssim", "epoch"],
)

_PLOT_RD_LAYOUT_SETTINGS_COMMON = dict(
    xaxis_title="Bit-rate [bpp]",
    xaxis=dict(range=[0.0, 2.25], tick0=0.0, dtick=0.25),
)

PLOT_RD_SETTINGS = {
    "psnr": {
        "scatter_kwargs": dict(
            **_PLOT_RD_SCATTER_SETTINGS_COMMON,
            y="psnr",
        ),
        "layout_kwargs": dict(
            **_PLOT_RD_LAYOUT_SETTINGS_COMMON,
            yaxis_title="PSNR [dB]",
            yaxis=dict(range=[26, 41], tick0=26, dtick=1),
        ),
    },
    "ms-ssim": {
        "scatter_kwargs": dict(
            **_PLOT_RD_SCATTER_SETTINGS_COMMON,
            y="ms-ssim",
        ),
        "layout_kwargs": dict(
            **_PLOT_RD_LAYOUT_SETTINGS_COMMON,
            yaxis_title="MS-SSIM",
            yaxis=dict(range=[0.9, 1.0], tick0=0.9, dtick=0.01),
        ),
    },
    "ms-ssim-db": {
        "scatter_kwargs": dict(
            **_PLOT_RD_SCATTER_SETTINGS_COMMON,
            y="ms-ssim-db",
        ),
        "layout_kwargs": dict(
            **_PLOT_RD_LAYOUT_SETTINGS_COMMON,
            yaxis_title="MS-SSIM [dB]",
            yaxis=dict(range=[10, 24], tick0=10, dtick=1),
        ),
    },
}


def plot_rd(
    df: pd.DataFrame,
    metric: str = "psnr",
    scatter_kwargs: dict[str, Any] = {},
    layout_kwargs: dict[str, Any] = {},
):
    """Plots RD curve."""
    import plotly.express as px
    from plotly.subplots import make_subplots

    scatter_kwargs = {**PLOT_RD_SETTINGS[metric]["scatter_kwargs"], **scatter_kwargs}
    layout_kwargs = {**PLOT_RD_SETTINGS[metric]["layout_kwargs"], **layout_kwargs}
    fig = make_subplots()
    fig = px.line(df, **scatter_kwargs, markers=True)
    fig.update_layout(**layout_kwargs)
    return fig
