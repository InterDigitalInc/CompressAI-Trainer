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

import argparse
import sys
from typing import Iterable, cast

import aim
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from plotly.offline import plot

from compressai_train.plot import plot_rd
from compressai_train.utils.aim.query import (
    get_runs_dataframe,
    pareto_optimal_dataframe,
)
from compressai_train.utils.utils import compressai_dataframe, format_dataframe

TITLE = "Performance evaluation on Kodak - PSNR (RGB)"

COMPRESSAI_CODECS = [
    "bmshj2018-factorized",
    "bmshj2018-hyperprior",
    "mbt2018",
    "cheng2020-anchor",
]

REFERENCE_DF = pd.concat(
    [compressai_dataframe(name, dataset="kodak") for name in COMPRESSAI_CODECS]
)

HOVER_DATA = [
    "run_hash",
    "experiment",
    "epoch",
]


def style_trace_by_idx(trace, idx, key, default, replacement):
    n = len(trace.customdata)
    v = np.full(n, default, dtype=object)
    v[idx] = replacement
    trace[key] = v.tolist()


def create_dataframe(repo, conf, args, run_hash, identifiers):
    metrics = sorted(set(_needed_metrics(args.y_metrics, "y")) | {args.x, args.y})
    df = get_runs_dataframe(
        repo=repo,
        conf=conf,
        identifiers=identifiers,
        metrics=metrics,
    )
    df = format_dataframe(df, args.y, args.y_metrics)
    df = pareto_optimal_dataframe(df, x=args.x, y=args.y)
    if run_hash:
        df_run = df[df["run_hash"] == run_hash].copy()
        df_run["name"] = df_run["name"].apply(lambda x: x + " (current)")
        df = pd.concat([df, df_run])
    current_df = df
    return pd.concat([REFERENCE_DF, current_df])


def _needed_metrics(xs, key) -> Iterable[str]:
    for x in xs:
        xk = x[key]
        if isinstance(xk, str):
            yield xk
        yield from xk


def plot_dataframe(df, run_hash, args):
    run_hash_col = next(i for i, x in enumerate(HOVER_DATA) if x == "run_hash")
    scatter_kwargs = dict(
        x=args.x,
        y=args.y,
        hover_data=HOVER_DATA,
    )
    fig = plot_rd(df, scatter_kwargs=scatter_kwargs, title=TITLE)
    traces = list(fig.select_traces())

    for trace in traces:
        (idx,) = (trace.customdata[:, run_hash_col] == run_hash).nonzero()
        style_trace_by_idx(trace, idx, "marker.symbol", trace.marker.symbol, "x")

    print(fig)

    if args.show:
        fig.show()

    if args.out_file:
        plot(fig, auto_open=False, filename=args.out_file)


def build_args(argv):
    parser = argparse.ArgumentParser(description="Plot.")
    parser.add_argument("--conf", type=str)
    parser.add_argument("--identifiers", type=str, default="model.name")
    parser.add_argument("--out_file", type=str, default="plot_result.html")
    parser.add_argument("--show", action="store_true", help="Show figure in browser.")
    parser.add_argument("--x", type=str, default="bpp")
    parser.add_argument("--y", type=str, default="psnr")
    parser.add_argument(
        "--y_metrics", type=str, default='[{"suffix": "", "y": "psnr"}]'
    )
    args = parser.parse_args(argv)
    return args


def main(argv):
    args = build_args(argv)
    conf = cast(DictConfig, OmegaConf.load(args.conf))
    repo = aim.Repo(conf.paths.aim)
    run_hash = conf.env.aim.run_hash
    args.y_metrics = eval(args.y_metrics)  # WARNING: unsafe!
    df = create_dataframe(
        repo, conf, args, run_hash, identifiers=args.identifiers.split(",")
    )
    plot_dataframe(df, run_hash, args)


if __name__ == "__main__":
    main(sys.argv[1:])
