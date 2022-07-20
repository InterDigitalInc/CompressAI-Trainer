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
from typing import Iterable

import aim
import pandas as pd
from plotly.offline import plot

from compressai_train.plot import plot_rd
from compressai_train.utils.aim.query import (
    get_runs_dataframe,
    pareto_optimal_dataframe,
    runs_by_query,
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


def create_dataframe(repo, args):
    metrics = sorted(set(_needed_metrics(args.y_metrics, "y")) | {args.x, args.y})
    name, query = args.name, args.query
    runs = runs_by_query(repo, query)
    df = get_runs_dataframe(
        runs=runs,
        metrics=metrics,
        choose_metric="best",
    )
    if name:
        df["name"] = name
    df = format_dataframe(df, args.y, args.y_metrics)
    df = pareto_optimal_dataframe(df, x=args.x, y=args.y)
    if args.run_hash:
        df_run = df[df["run_hash"] == args.run_hash].copy()
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


def plot_dataframe(df, args):
    scatter_kwargs = dict(
        x=args.x,
        y=args.y,
        hover_data=HOVER_DATA,
    )

    fig = plot_rd(df, scatter_kwargs=scatter_kwargs, title=TITLE)
    print(fig)

    if args.show:
        fig.show()

    if args.out_file:
        plot(fig, auto_open=False, filename=args.out_file)


def build_args(argv):
    parser = argparse.ArgumentParser(description="Plot.")
    parser.add_argument("--aim_repo", type=str, required=True)
    parser.add_argument("--run_hash", type=str, default=None)
    parser.add_argument("--out_file", type=str, default="plot_result.html")
    parser.add_argument("--show", action="store_true", help="Show figure in browser.")
    parser.add_argument("--x", type=str, default="bpp")
    parser.add_argument("--y", type=str, default="psnr")
    parser.add_argument("--query", "-q", type=str, default="")
    parser.add_argument("--name", "-n", type=str, default="")
    parser.add_argument(
        "--y_metrics", "-ym", type=str, default='[{"suffix": "", "y": "psnr"}]'
    )
    args = parser.parse_args(argv)
    return args


def main(argv):
    args = build_args(argv)
    repo = aim.Repo(args.aim_repo)
    args.y_metrics = eval(args.y_metrics)  # WARNING: unsafe!
    df = create_dataframe(repo, args)
    plot_dataframe(df, args)


if __name__ == "__main__":
    main(sys.argv[1:])
