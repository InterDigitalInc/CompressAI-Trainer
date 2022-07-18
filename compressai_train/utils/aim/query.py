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

from typing import Any, Callable, Optional, Sequence, TypeVar

import aim
import numpy as np
import pandas as pd
from aim.storage.context import Context

from compressai_train.utils.utils import arg_pareto_optimal_set

T = TypeVar("T")


def get_runs_dataframe(
    repo: aim.Repo,
    conf: dict[str, Any],
    *,
    min_metric: str = "loss",
    metrics: list[str] = ["bpp", "psnr", "ms-ssim"],
    x: str = "bpp",
    y: str = "psnr",
    identifiers: list[str] = ["model.name"],
    to_df: Callable[[dict[str, Any]], pd.DataFrame] = (
        lambda d: pd.DataFrame.from_dict([d])
    ),
):
    """Returns dataframe of best model metrics for filtered runs.

    Filters runs based on given identifiers.
    For each run, determines epoch at which a min_metric is minimum,
    and accumulates infer metric values at that epoch into a dataframe.
    Keeps only pareto-optimal data points in that dataframe based
    on given x and y and their corresponding objectives (min, max).
    Optionally, also keeps data points that are part of a given run.
    """
    metrics = list(set(metrics + [x, y]))
    runs = runs_by_identifiers(conf, repo, identifiers=identifiers)
    dfs = [
        to_df(metrics_at_index(r, metrics, best_metric_index(r, min_metric)))
        for r in runs
    ]
    df = pd.concat(dfs)
    df.sort_values(["name", x], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def pareto_optimal_dataframe(
    df: pd.DataFrame,
    *,
    x: str = "bpp",
    y: str = "psnr",
    x_objective: str = "min",
    y_objective: str = "max",
    keep_run_hash: Optional[str] = None,
) -> pd.DataFrame:
    """Returns dataframe of best models at pareto frontier.

    Keeps only pareto-optimal data points in that dataframe based
    on given x and y and their corresponding objectives (min, max).
    Optionally, also keeps data points that are part of a given run.
    """
    df = df.copy()
    df.sort_values(["name", x], inplace=True)
    df.reset_index(drop=True, inplace=True)
    idxs = arg_pareto_optimal_set([df[x], df[y]], [x_objective, y_objective])
    if keep_run_hash is not None:
        idxs.extend(df.index[df["run_hash"] == keep_run_hash].tolist())
        idxs = sorted(set(idxs))
    df = df.iloc[idxs]
    df.reset_index(drop=True, inplace=True)
    return df


def runs_by_identifiers(
    conf: dict[str, Any], repo: aim.Repo, identifiers: list[str]
) -> list[aim.Run]:
    """Returns runs that match the same identifiers present in conf."""
    if identifiers == []:
        return list(repo.iter_runs())
    targets = tuple(_get_path(conf, path.split(".")) for path in identifiers)
    query = " and ".join(
        f"run.{key} == {repr(value)}" for key, value in zip(identifiers, targets)
    )
    runs = [x.run for x in repo.query_runs(query).iter_runs()]  # type: ignore
    return runs


def metrics_at_index(
    run: aim.Run,
    metrics: list[str],
    index: np.intp,
    loader: str = "infer",
    scope: str = "epoch",
) -> dict[str, Any]:
    """Returns metrics logged at a particular step index."""
    context = Context({"loader": loader, "scope": scope})
    results = {
        metric: _map_none(
            lambda x: x.values.sparse_numpy()[1][index],
            run.get_metric(metric, context),
        )
        for metric in metrics
    }
    info = {
        "name": run["model", "name"],
        "run_hash": run.hash,
        "experiment": run.experiment,
        "model.name": run["model", "name"],
        "epoch": index + 1,  # TODO not always true
    }
    return {**info, **results}


def best_metric_index(
    run: aim.Run,
    min_metric: str = "loss",
    loader: str = "valid",
    scope: str = "epoch",
) -> np.intp:
    """Returns step index at which a given metric is minimized."""
    context = Context({"loader": loader, "scope": scope})
    metric = run.get_metric(min_metric, context)
    assert metric is not None
    _, metric_values = metric.values.sparse_numpy()
    return metric_values.argmin()


def _map_none(f: Callable[[T], T], x: Optional[T]) -> Optional[T]:
    """Applies fmap (functor map) to nullable types."""
    return None if x is None else f(x)


def _get_path(x, path: Sequence[str]):
    for key in path:
        x = x[key]
    return x
