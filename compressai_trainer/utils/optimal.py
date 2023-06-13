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
import pandas as pd


def optimal_dataframe(
    df: pd.DataFrame,
    *,
    x: str = "bpp",
    y: str = "psnr",
    x_objective: str = "min",
    y_objective: str = "max",
    keep: Optional[str] = None,
    method: str = "pareto",
    groupby: Optional[str] = None,
) -> pd.DataFrame:
    """Returns dataframe of best models at frontier.

    Keeps only optimal data points in that dataframe based
    on given x and y and their corresponding objectives (min, max).
    Optionally, also keeps data points that are marked by the keep key.

    Valid methods are "none", "pareto", and "convex".
    """
    if groupby is not None:
        return (
            df.sort_values(groupby)
            .groupby(groupby)
            .apply(
                lambda df_group: optimal_dataframe(
                    df=df_group,
                    x=x,
                    y=y,
                    x_objective=x_objective,
                    y_objective=y_objective,
                    keep=keep,
                    method=method,
                )
            )
            .reset_index(drop=True)
        )
    points = df[[x, y]].values.T
    idxs = arg_optimal_set(points, [x_objective, y_objective], method)
    if keep is not None:
        idxs = {*idxs, *df.index[df[keep] == True]}
    df = df.iloc[sorted(idxs)]
    df.sort_values([x, y], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def arg_optimal_set(points, objectives, method="pareto"):
    """Returns optimal set indexes.

    Valid methods are "none", "pareto", and "convex".
    """
    if method == "none":
        return np.arange(len(points[0]))
    if method == "pareto":
        return _arg_pareto_optimal_set(points, objectives)
    if method == "convex":
        return _arg_convex_optimal_set(points, objectives)
    raise ValueError(f"Unknown method: {method}")


def _arg_optimal_canonicalize(points, objectives):
    if np.isnan(points).any():
        raise ValueError("points contains NaN values.")

    # Force objectives to be ("max", "max").
    signs = [-1 if o == "min" else 1 for o in objectives]
    signs = np.array(signs)
    points = np.array(points)
    points *= signs[:, None]

    return points


def _arg_pareto_optimal_set(points, objectives):
    """Returns pareto-optimal set indexes."""
    if len(points) != 2:
        raise NotImplementedError

    xs, ys = _arg_optimal_canonicalize(points, objectives)

    # Sort by x (descending) and y (descending).
    idxs = np.lexsort((-ys, -xs))
    # xs = xs[idxs]  # Not needed.
    ys = ys[idxs]

    # Find the indexes where ys improves (strictly monotonically).
    ys_mono = np.maximum.accumulate(ys)
    d_ys_mono = np.diff(ys_mono, prepend=[ys_mono[0] - 1])
    [y_idxs] = (d_ys_mono > 0).nonzero()
    idxs = idxs[y_idxs]

    return idxs


def _arg_convex_optimal_set(points, objectives):
    """Returns convex-optimal set indexes."""
    points = _arg_optimal_canonicalize(points, objectives)

    # perf: convex optimal set is a subset of pareto optimal set.
    idxs_pareto = _arg_pareto_optimal_set(points, ["max", "max"])
    points = points[:, idxs_pareto]

    idxs_convex = [0]
    i = 0

    while i < points.shape[1] - 1:
        # Compute dy/dx for each point to the left of current convex point.
        d = points[:, i + 1 :] - points[:, i, None]
        dy_dx = d[1] / d[0]

        # Choose the point with most negative slope as a convex point.
        i += dy_dx.argmin() + 1
        idxs_convex.append(i)

    idxs = idxs_pareto[idxs_convex]
    return idxs
