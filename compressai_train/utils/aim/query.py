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

from typing import Any, Callable, Literal, Optional, Sequence, TypeVar

import aim
import pandas as pd
from aim.storage.context import Context

from compressai_train.utils.utils import arg_pareto_optimal_set

T = TypeVar("T")


def runs_by_query(repo: aim.Repo, query: str) -> list[aim.Run]:
    """Returns runs that match given query."""
    if query == "":
        return list(repo.iter_runs())
    runs = [x.run for x in repo.query_runs(query).iter_runs()]  # type: ignore
    return runs


def get_runs_dataframe(
    runs: list[aim.Run],
    *,
    min_metric: str = "loss",
    metrics: list[str] = ["bpp", "psnr", "ms-ssim"],
    hparams: list[str] = [],
    epoch: int | Literal["best"] | Literal["last"] = "best",
):
    """Returns dataframe of best model metrics for runs.

    For each run, accumulates infer metric values at a particular epoch
    into a dataframe.
    If epoch == "best", the epoch minimizing the min_metric is chosen.
    """
    idxs = [_find_index(run, epoch, min_metric) for run in runs]
    records = [
        metrics_at_index(run, metrics, hparams, idx)
        for run, idx in zip(runs, idxs)
        if idx is not None
    ]
    df = pd.DataFrame.from_records(records)
    df.sort_values(["name"], inplace=True)
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
    df.sort_values(["name", x, y], inplace=True)
    df.reset_index(drop=True, inplace=True)
    idxs = arg_pareto_optimal_set([df[x], df[y]], [x_objective, y_objective])
    if keep_run_hash is not None:
        idxs.extend(df.index[df["run_hash"] == keep_run_hash].tolist())
        idxs = sorted(set(idxs))
    df = df.iloc[idxs]
    df.reset_index(drop=True, inplace=True)
    return df


def metrics_at_index(
    run: aim.Run,
    metrics: list[str],
    hparams: list[str],
    index: int,
    loader: str = "infer",
    scope: str = "epoch",
) -> dict[str, Any]:
    """Returns metrics logged at a particular step index."""
    context = Context({"loader": loader, "scope": scope})
    metric_results = {
        metric: _metric_at_index(run, metric, context, index) for metric in metrics
    }
    hparam_results = {hparam: _get_path(run, hparam.split(".")) for hparam in hparams}
    context = Context({"loader": "_epoch_", "scope": "epoch"})
    epoch = _metric_at_index(run, "epoch", context, index)
    info = {
        "name": run.experiment,
        "run_hash": run.hash,
        "experiment": run.experiment,
        "model.name": run["model", "name"],
        "epoch": epoch,
    }
    return {**info, **metric_results, **hparam_results}


def best_metric_index(
    run: aim.Run,
    min_metric: str = "loss",
    loader: str = "valid",
    scope: str = "epoch",
) -> Optional[int]:
    """Returns step index at which a given metric is minimized."""
    context = Context({"loader": loader, "scope": scope})
    values = _metric_series(run, min_metric, context)
    if values is None:
        return None
    return int(values.argmin())


def _find_index(
    run: aim.Run,
    epoch: int | Literal["best"] | Literal["last"] = "best",
    min_metric: str = "loss",
    loader: str = "valid",
    scope: str = "epoch",
):
    if epoch == "best":
        return best_metric_index(run, min_metric, loader=loader, scope=scope)
    if epoch == "last":
        return -1
    if isinstance(epoch, int):
        return epoch
    raise ValueError(f"Unknown epoch={epoch}.")


def _metric_series(run: aim.Run, metric: str, context: Context):
    m = run.get_metric(metric, context)
    if m is None:
        return None
    _, values = m.values.sparse_numpy()
    return values


def _metric_at_index(run: aim.Run, metric: str, context: Context, index: int):
    m = run.get_metric(metric, context)
    if m is None:
        return None
    _, values = m.values.sparse_numpy()
    return values[index]


def _map_none(f: Callable[[T], T], x: Optional[T]) -> Optional[T]:
    """Applies fmap (functor map) to nullable types."""
    return None if x is None else f(x)


def _get_path(x, path: Sequence[str]):
    for key in path:
        x = x[key]
    return x
