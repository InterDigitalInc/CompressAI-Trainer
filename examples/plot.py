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

HOVER_HPARAMS = [
    "criterion.lmbda",
]

HOVER_METRICS = [
    #
]

HOVER_DATA = [
    "run_hash",
    "name",
    "model.name",
    "experiment",
    "epoch",
]

HOVER_DATA += HOVER_HPARAMS + HOVER_METRICS


def create_dataframe(repo, args):
    dfs = [
        _create_dataframe(repo, args.x, args.y, name, query, curves, pareto)
        for name, query, curves, pareto in zip(
            args.name, args.query, args.curves, args.pareto
        )
    ]
    df = pd.concat([REFERENCE_DF, *dfs])
    df = _reorder_dataframe_columns(df)
    return df


def _create_dataframe(repo, x, y, name, query, curves, pareto):
    runs = runs_by_query(repo, query)
    metrics = sorted(
        {x, y, *HOVER_METRICS}
        | set(_needed_metrics(curves, "x"))
        | set(_needed_metrics(curves, "y"))
    )
    hparams = HOVER_HPARAMS
    df = get_runs_dataframe(
        runs=runs,
        metrics=metrics,
        hparams=hparams,
        choose_metric="best",
    )
    if name:
        df["name"] = name
    df = format_dataframe(df, x, y, curves, skip_nan=True)
    if pareto:
        df = pareto_optimal_dataframe(df, x=x, y=y)
    return df


def _needed_metrics(xs, key) -> Iterable[str]:
    for x in xs:
        xk = x[key]
        if isinstance(xk, str):
            yield xk
            continue
        yield from xk


def _reorder_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    head = HOVER_DATA
    head_set = set(head)
    columns = head + [x for x in df.columns if x not in head_set]
    return df[columns]


def plot_dataframe(df: pd.DataFrame, args):
    scatter_kwargs = dict(
        x=args.x,
        y=args.y,
        hover_data=HOVER_DATA,
    )

    print(df)

    if args.out_csv:
        df.to_csv(args.out_csv, index=False)

    fig = plot_rd(df, scatter_kwargs=scatter_kwargs, title=TITLE)

    if args.out_html:
        plot(fig, auto_open=False, filename=args.out_html)

    if args.show:
        fig.show()


def build_args(argv):
    help = {
        "show": "Show figure in browser.",
        "query": 'Example: \'run.model.name == "bmshj2018-factorized" and run.exp.name == "example-experiment"\'',
        "curves": 'Default: [{"name": "{name}", "suffix": "", "x": "bpp", "y": "psnr"}]',
    }

    parser = argparse.ArgumentParser(description="Plot.")
    parser.add_argument("--aim_repo", type=str, required=True)
    parser.add_argument("--out_html", type=str, default="plot_result.html")
    parser.add_argument("--out_csv", type=str, default="plot_result.csv")
    parser.add_argument("--show", action="store_true", help=help["show"])
    parser.add_argument("--x", "-x", type=str, default="bpp")
    parser.add_argument("--y", "-y", type=str, default="psnr")
    parser.add_argument("--name", "-n", action="append", default=[])
    parser.add_argument(
        "--query", "-q", action="append", default=[], help=help["query"]
    )
    parser.add_argument(
        "--curves", "-c", action="append", default=[], help=help["curves"]
    )
    parser.add_argument("--pareto", action="append", default=[])
    args = parser.parse_args(argv)

    if len(args.query) == 0:
        args.query = [""]
    if len(args.name) == 0:
        args.name = [""]
    if len(args.query) != len(args.name):
        raise RuntimeError("--query and --name should appear the same number of times.")
    curves_default = [{"name": "{name}", "suffix": "", "x": "bpp", "y": "psnr"}]
    args.curves = [eval(x) for x in args.curves]  # WARNING: unsafe!
    args.curves += [curves_default] * (len(args.query) - len(args.curves))
    args.pareto += [False] * (len(args.query) - len(args.pareto))

    return args


def main(argv):
    args = build_args(argv)
    repo = aim.Repo(args.aim_repo)
    df = create_dataframe(repo, args)
    plot_dataframe(df, args)


if __name__ == "__main__":
    main(sys.argv[1:])
