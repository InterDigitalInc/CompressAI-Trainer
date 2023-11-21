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
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    import matplotlib.pyplot as plt

DEFAULT_COLORMAP = "plasma"


def featuremap_matplotlib(
    arr: np.ndarray,
    *,
    nrows: Optional[int] = None,
    ncols: Optional[int] = None,
    padding: Optional[int] = None,
    fill_value: Optional[float] = None,
    clim: Optional[Tuple[float, float]] = None,
    cmap: str = DEFAULT_COLORMAP,
    cbar: bool = True,
    ax: Optional[plt.Axes] = None,
    tile_method: str = "reshape",
    **fig_kw,
) -> plt.Figure:
    """Plots 3D tensor as a 2D featuremap of tiled channels.

    .. note:: ``tile_method="loop"`` is slow due to the nested loop.
       For a faster alternative with slightly lower publication quality,
       try ``tile_method="reshape"``.

    Args:
        arr: chw tensor
        nrows: number of tiled rows
        ncols: number of tiled columns
        padding: padding between tiles
        fill_value: value to set remaining area to
        clim: colorbar limits
        cmap: colormap
        cbar: whether to show colorbar
        tile_method: "reshape" (default, fast) or "loop" (slow)
        fig_kw: keyword arguments to pass to matplotlib
    """
    import matplotlib.pyplot as plt

    if tile_method == "loop":
        assert padding is None
        assert fill_value is None
        assert ax is None

        c, *_ = arr.shape

        if clim is None:
            clim = (arr.min(), arr.max())
        nrows, ncols = _compute_tiling(c, nrows, ncols)

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

        if cbar:
            cbar = fig.colorbar(im, ax=axs)

        return fig

    elif tile_method == "reshape":
        img = featuremap_image(
            arr,
            nrows=nrows,
            ncols=ncols,
            padding=padding,
            fill_value=fill_value,
            clim=clim,
        )
        if ax is None:
            fig, ax = plt.subplots(**fig_kw)
        else:
            fig = ax.get_figure()
        im = ax.matshow(img, cmap=cmap)
        if clim is not None:
            im.set_clim(*clim)
        ax.set_xticks([])
        ax.set_yticks([])
        if cbar:
            fig.colorbar(im, ax=ax)
        return fig

    else:
        raise ValueError(f"Unknown tile_method: {tile_method}")


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

    if arr.ndim == 0:
        arr = arr.reshape(1)

    if arr.ndim > 3:
        *_, h, w = arr.shape
        arr = arr.reshape(-1, h, w)

    if arr.ndim <= 2:
        c, *tail = arr.shape
        arr = arr.reshape(c, *tail, *([1] * (2 - len(tail))))
        if nrows is None and ncols is None:
            nrows, ncols = 1, c
        if padding is None:
            padding = 0

    if arr.ndim == 3:
        if padding is None:
            padding = 2

    arr = _tile_featuremap_3d(arr, nrows, ncols, padding, fill_value)

    if cmap is not None:
        import matplotlib

        arr = ((arr - clim[0]) / (clim[1] - clim[0])).clip(0, 1)
        arr = (matplotlib.colormaps[cmap](arr)[..., :3] * 255).astype(np.uint8)

    return arr


def _tile_featuremap_3d(
    arr: np.ndarray,
    nrows: Optional[int] = None,
    ncols: Optional[int] = None,
    padding: int = 0,
    fill_value: Optional[float] = None,
) -> np.ndarray:
    if fill_value is None:
        fill_value = arr.min()

    pad = ((0, 0), (padding, padding), (padding, padding))
    arr = np.pad(arr, pad, "constant", constant_values=fill_value)
    c, h, w = arr.shape

    nrows, ncols = _compute_tiling(c, nrows, ncols)

    # Ensure nrows * ncols channels by creating empty channels if needed.
    if c < nrows * ncols:
        arr = arr.reshape(-1).copy()
        prev_size = arr.size
        arr.resize(nrows * ncols * h * w)
        arr[prev_size:] = fill_value

    return arr.reshape(nrows, ncols, h, w).swapaxes(1, 2).reshape(nrows * h, ncols * w)


def _compute_tiling(c, nrows, ncols):
    if nrows is None and ncols is None:
        nrows = ceil(sqrt(c))
    if nrows is None:
        nrows = ceil(c / ncols)
    if ncols is None:
        ncols = ceil(c / nrows)
    assert c <= nrows * ncols
    return nrows, ncols
