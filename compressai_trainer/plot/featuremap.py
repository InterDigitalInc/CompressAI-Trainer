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

from math import ceil, sqrt
from typing import Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

DEFAULT_COLORMAP = "plasma"


def featuremap_matplotlib_looptiled(
    arr: np.ndarray,
    *,
    nrows: Optional[int] = None,
    ncols: Optional[int] = None,
    clim: Optional[Tuple[float, float]] = None,
    cmap: str = DEFAULT_COLORMAP,
    cbar: bool = True,
    **fig_kw,
) -> Tuple[plt.Figure, np.ndarray]:
    """Plots 3D tensor as a 2D featuremap of tiled channels.

    .. note:: This is slow due to the nested loop. For a faster
       alternative with slightly lower publication quality, try
       ``featuremap_matplotlib`` or ``featuremap_image``.

    Args:
        arr: chw tensor
        nrows: number of tiled rows
        ncols: number of tiled columns
        clim: colorbar limits
        cmap: colormap
        cbar: whether to show colorbar
        fig_kw: keyword arguments to pass to matplotlib
    """
    c, _, _ = arr.shape

    if clim is None:
        clim = (arr.min(), arr.max())
    if nrows is None:
        nrows = ceil(sqrt(c))
    if ncols is None:
        ncols = ceil(c / nrows)

    fig, axs = plt.subplots(nrows, ncols, squeeze=False, **fig_kw)
    im = None

    for i in range(nrows):
        for j in range(ncols):
            ax = axs[i, j]
            idx = i * ncols + j
            if idx >= c:
                ax.axis("off")
                continue
            img = arr[idx]
            im = ax.matshow(img, cmap=cmap)
            im.set_clim(*clim)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
            ax.tick_params(axis="y", direction="in", pad=0)
            ax.tick_params(axis="x", direction="in", pad=0)

    fig.subplots_adjust(wspace=0, hspace=0)

    if cbar:
        cbar = fig.colorbar(im, ax=axs)

    return fig, axs


def featuremap_matplotlib(
    arr: np.ndarray,
    title: str,
    *,
    nrows: Optional[int] = None,
    ncols: Optional[int] = None,
    padding: int = 2,
    fill_value: Optional[float] = None,
    clim: Optional[Tuple[float, float]] = None,
    cmap: str = DEFAULT_COLORMAP,
    cbar: bool = True,
    **fig_kw,
) -> plt.Figure:
    """Plots 3D tensor as a 2D featuremap of tiled channels.

    Args:
        arr: chw tensor
        nrows: number of tiled rows
        ncols: number of tiled columns
        padding: padding between tiles
        fill_value: value to set remaining area to
        clim: colorbar limits
        cmap: colormap
        cbar: whether to show colorbar
        fig_kw: keyword arguments to pass to matplotlib
    """
    img = featuremap_image(
        arr,
        nrows=nrows,
        ncols=ncols,
        padding=padding,
        fill_value=fill_value,
        clim=clim,
    )
    fig, ax = plt.subplots(tight_layout=True, **fig_kw)
    im = ax.matshow(img, cmap=cmap)
    ax.set_title(title, fontsize="xx-small")
    ax.set_xticks([])
    ax.set_yticks([])
    if clim is not None:
        im.set_clim(*clim)
    if cbar:
        fig.colorbar(im)
    return fig


def featuremap_image(
    arr: np.ndarray,
    nrows: Optional[int] = None,
    ncols: Optional[int] = None,
    padding: Optional[int] = None,
    fill_value: Optional[float] = None,
    clim: Optional[Tuple[float, float]] = None,
    cmap: Optional[str] = None,
) -> np.ndarray:
    """Returns 2D featuremap image of tiled channels for the given tensor.

    Args:
        arr: tensor of shape (c, ...)
        nrows: number of tiled rows
        ncols: number of tiled columns
        padding: padding between tiles (default is 2 for arr.ndim > 2)
        fill_value: value to set remaining area to
        clim: colorbar limits
        cmap: colormap; if None, no colormap is applied
    """
    if clim is None:
        clim = (arr.min(), arr.max())
    if fill_value is None:
        fill_value, _ = clim

    arr = tile_featuremap(arr, nrows, ncols, padding, fill_value)

    if cmap is not None:
        arr = ((arr - clim[0]) / (clim[1] - clim[0])).clip(0, 1)
        arr = (matplotlib.colormaps[cmap](arr)[..., :3] * 255).astype(np.uint8)

    return arr


def tile_featuremap(
    arr: np.ndarray,
    nrows: Optional[int] = None,
    ncols: Optional[int] = None,
    padding: Optional[int] = None,
    fill_value: Optional[float] = None,
) -> np.ndarray:
    """Tiles arr into a 2D image featuremap.

    Args:
        arr: tensor of shape (c, ...)
        nrows: number of tiled rows
        ncols: number of tiled columns
        padding: padding between tiles (default is 2 for arr.ndim > 2)
        fill_value: value to set remaining area to

    Returns:
        np.ndarray: tiled array
    """
    if arr.ndim == 0:
        arr = arr.reshape(1)

    if arr.ndim <= 2:
        c, *tail = arr.shape
        arr = arr.reshape(c, *tail, *([1] * (2 - len(tail))))
        assert nrows is None and ncols is None and padding is None
        nrows = 1
        ncols = c
        padding = 0
        return _tile_featuremap_3d(arr, nrows, ncols, padding, fill_value)

    if arr.ndim > 3:
        *_, h, w = arr.shape
        arr = arr.reshape(-1, h, w)

    if arr.ndim == 3:
        if padding is None:
            padding = 2
        return _tile_featuremap_3d(arr, nrows, ncols, padding, fill_value)

    raise NotImplementedError(f"Unsupported number of dimensions: {arr.ndim}.")


def _tile_featuremap_3d(
    arr: np.ndarray,
    nrows: Optional[int] = None,
    ncols: Optional[int] = None,
    padding: Optional[int] = None,
    fill_value: Optional[float] = None,
) -> np.ndarray:
    if fill_value is None:
        fill_value = arr.min()

    pad = ((0, 0), (padding, padding), (padding, padding))
    arr = np.pad(arr, pad, "constant", constant_values=fill_value)
    c, h, w = arr.shape

    if nrows is None:
        nrows = ceil(sqrt(c))
    if ncols is None:
        ncols = ceil(c / nrows)

    assert c <= nrows * ncols

    # Ensure nrows * ncols channels by creating empty channels if needed.
    if c < nrows * ncols:
        arr = arr.reshape(-1).copy()
        prev_size = arr.size
        arr.resize(nrows * ncols * h * w)
        arr[prev_size:] = fill_value

    return arr.reshape(nrows, ncols, h, w).swapaxes(1, 2).reshape(nrows * h, ncols * w)
