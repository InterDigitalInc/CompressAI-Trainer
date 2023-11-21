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

from typing import Optional

import numpy as np

from .utils import latex_matplotlib_rcparams, np_image_to_base64, plt_hide_axes


def plot_pdf_signatures(
    x: list[np.ndarray],
    p: list[np.ndarray],
    p_hat: list[np.ndarray],
    *,
    backend: str = "matplotlib",
    **kwargs,
):
    """Plots PDF signatures of per-sample measured and encoding distributions.

    Probability distribution functions (pdf) can be plotted as a 2D image.

    * The x-axis represents the channel index.
    * The y-axis represents the bin index.
    * Color intensity represents the negative log-likelihood (in bits) of
      the probability of the bin.

    This is useful for visualizing the efficacy of probability
    distribution reconstruction/correction models.

    Args:
        x: List of input images.
        p: List of measured probability distributions (targets).
        p_hat: List of encoding probability distributions.
        backend: Backend to use for plotting ("matplotlib" or "plotly").
    """

    # WARNING: matplotlib interface may not be compatible with plotly interface.
    if backend == "matplotlib":
        return plot_pdf_signatures_matplotlib(x, p, p_hat, **kwargs)
    elif backend == "plotly":
        return plot_pdf_signatures_plotly(x, p, p_hat, **kwargs)

    raise ValueError(f"Unknown backend: {backend}")


def plot_pdf_signatures_matplotlib(
    x: list[np.ndarray],
    p: list[np.ndarray],
    p_hat: list[np.ndarray],
    p_default: list[np.ndarray],
    # stats: dict, #  e.g. KL div/bpp saved, etc.
    ax_kwargs: dict = {"cmap": "BuPu_r"},
    use_latex: bool = True,
    boldsymbol: Optional[str] = None,
    **fig_kwargs,
):
    import matplotlib.pyplot as plt

    if boldsymbol is None:
        boldsymbol = r"\boldsymbol" if use_latex else ""

    if use_latex:
        # WARNING: This mutates global state!
        plt.rcParams.update(latex_matplotlib_rcparams())

    num_samples = len(x)
    # x = [(x_ * 255).round().astype(np.uint8).transpose(1, 2, 0) for x_ in x]
    p = [_preprocess_pdf(p_) for p_ in p]
    p_hat = [_preprocess_pdf(p_) for p_ in p_hat]
    p_default = [_preprocess_pdf(p_) for p_ in p_default]

    fig, axs = plt.subplots(num_samples + 1, 3, squeeze=False, **fig_kwargs)

    ax_kwargs = dict(
        interpolation="none",
        vmin=p.min(),
        vmax=p.max(),
        **ax_kwargs,
    )

    axs[0, 0].set_title("Input image", pad=16)
    axs[0, 1].set_title(r"$-\log_2 {" f"{boldsymbol}" r"{{p}}}$", pad=16)
    axs[0, 2].set_title(r"$-\log_2 {" f"{boldsymbol}" r"{\hat{p}}}$", pad=16)

    ax = axs[0, 0]
    ax.set(xlim=[0, 1], ylim=[0, 1])
    ax.text(0.5, 0.5, "(Default)", fontsize=10, va="center", ha="center")
    plt_hide_axes(ax)

    ax = axs[0, 1]
    im = ax.matshow(p_default, **ax_kwargs)
    plt_hide_axes(ax)

    ax = axs[0, 2]
    im = ax.matshow(p_default, **ax_kwargs)
    plt_hide_axes(ax)

    for i in range(num_samples):
        i_ax = i + 1

        ax = axs[i_ax, 0]
        im = ax.matshow(x[i], interpolation="bicubic")
        plt_hide_axes(ax)

        ax = axs[i_ax, 1]
        im = ax.matshow(p[i], **ax_kwargs)
        plt_hide_axes(ax)

        ax = axs[i_ax, 2]
        im = ax.matshow(p_hat[i], **ax_kwargs)
        plt_hide_axes(ax)

    fig.tight_layout()
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.90, 0.25, 0.02, 0.5])
    cbar = fig.colorbar(im, cax=cbar_ax)
    yticklabels = cbar.ax.get_yticklabels()
    yticklabels[-1] = f"{yticklabels[-1].get_text()}+"
    cbar.ax.set_yticklabels(yticklabels)

    return fig


def plot_pdf_signatures_plotly(
    x: list[np.ndarray],
    p: list[np.ndarray],
    p_hat: list[np.ndarray],
    p_kwargs: dict = {
        "colorscale": "BuPu_r",
        "zmin": 0,
        "zmax": 10,
    },
    horizontal_spacing=0.01,
    vertical_spacing=0.001,
):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    num_samples = len(x)
    x = [(x_ * 255).round().astype(np.uint8).transpose(1, 2, 0) for x_ in x]
    p = [_preprocess_pdf(p_) for p_ in p]
    p_hat = [_preprocess_pdf(p_) for p_ in p_hat]

    fig = make_subplots(
        rows=num_samples,
        cols=3,
        horizontal_spacing=horizontal_spacing,
        vertical_spacing=vertical_spacing,
        subplot_titles=[
            "Input image",
            r"-log_2(p)",
            r"-log_2(p_hat)",
            # NOTE: LaTeX renders a bit slowly... Disable.
            # r"$-\log_2 {" f"{boldsymbol}" r"{{p}}}$",
            # r"$-\log_2 {" f"{boldsymbol}" r"{\hat{p}}}$",
        ],
    )

    for i in range(num_samples):
        row = i + 1

        fig.add_traces(
            [
                go.Image(source=np_image_to_base64(x[i]), name=f"x_{i}"),
                go.Heatmap(z=p[i], name=f"p_{i}", **p_kwargs),
                go.Heatmap(z=p_hat[i], name=f"p_hat_{i}", **p_kwargs),
            ],
            rows=[row, row, row],
            cols=[1, 2, 3],
        )

        for col in [1, 2, 3]:
            fig.update_xaxes(showticklabels=False, row=row, col=col)
            fig.update_yaxes(showticklabels=False, row=row, col=col)

    return fig


def _preprocess_pdf(p, max_bits=10, max_bins=128):
    offset = max(0, (p.shape[-1] - max_bins) // 2)
    p = p[..., offset : offset + max_bins]
    return (-np.log2(p + 2**-max_bits)).clip(min=0, max=max_bits).swapaxes(-1, -2)
