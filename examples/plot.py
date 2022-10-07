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
import textwrap
from typing import Iterable

import aim
import pandas as pd

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
    [compressai_dataframe(name, dataset="image/kodak") for name in COMPRESSAI_CODECS]
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
        epoch="best",
    )
    if name:
        df["name"] = name
    df = format_dataframe(df, x, y, curves, skip_nan=True)
    df.sort_values(["name", x, y], inplace=True)
    df.reset_index(drop=True, inplace=True)
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
        from plotly.offline import plot

        plot(fig, auto_open=False, filename=args.out_html)

    if args.show:
        fig.show()


def build_args(argv):
    wrap = lambda s: "\n".join("\n".join(textwrap.wrap(x)) for x in s.splitlines())
    help = {
        "description": wrap(
            "Plot.\n"
            "\n"
            "Queries experiment tracker (Aim) repository for relevant metrics and plots. "
            "Users may specify what to plot using groups of --query, --curves, --name, and --pareto. "
            "If desired, one may plot multiple query groups within the same plot.\n"
        ),
        "show": "Show figure in browser.",
        "query": (
            "Query selector for relevant runs to aggregate plotting data from.\n"
            "\n"
            "Default: '' (i.e. uses all runs).\n"
            "\n"
            "Examples:\n"
            "  - 'run.model.name == \"bmshj2018-factorized\"'\n"
            "  - 'run.experiment.startswith(\"hindsight2020-best_model-\")'\n"
            "  - 'run.hash == \"e4e6d4d5e5c59c69f3bd7be2\"'\n"
            "  - 'run.created_at >= datetime(1970, 1, 1)'\n"
            "  - 'run.criterion.lmbda < 0.02 and run.hp.M == 3 * 2**6'\n"
            "  - 'random.random() > 1/6 and \"Russian roulette! Removes random runs.\"'\n"
        ),
        "curves": (
            wrap(
                "For the current query, specify a grouping and format for the curves. "
                "One may specify multiple such groupings for a given query within a list. "
                'The curve name is constructed from "name" + "suffix". '
                'If a key (e.g. "name", "suffix", "x", "y") is not specified, its default value is used.\n'
                "\n"
                'For "name" and "suffix", one may specify a hparam as by key via "{hparam}". '
                'There is also a "{name}" property that equals "{model.name}" by default, '
                "but this may be overridden via --name.\n"
            )
            + (
                "\n"
                "\n"
                'Default: [{"name": "{name}", "suffix": "", "x": args.x, "y": args.y}].\n'
                "\n"
                "Examples:\n"
                "  - Show both model name and experiment name:\n"
                '    [{"name": "{model.name} {experiment}"}]\n'
                "  - Group by hp.M:\n"
                '    [{"name": "{name} (M={hp.M})"}]\n'
                "  - Multiple metrics as separate curves:\n"
                "    [\n"
                '        {"suffix": " (full quality)", "y": "psnr_full"},\n'
                '        {"suffix": " (low quality)", "y": "psnr_low"},\n'
                "    ]\n"
                "  - Multi-rate models (e.g. G-VAE):\n"
                "    [{\n"
                '        "suffix": " {run.hash}",\n'
                '        "x": ["bpp_0", "bpp_1", "bpp_2", "bpp_3"],\n'
                '        "y": ["psnr_0", "psnr_1", "psnr_2", "psnr_3"],\n'
                "    }]\n"
            )
        ),
        "name": "Force override {name} property for respective query.",
        "pareto": (
            "Show only pareto-optimal points on curve for respective query.\n"
            "Default: False.\n"
        ),
    }

    parser = argparse.ArgumentParser(
        description=help["description"],
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--aim_repo", type=str, required=True)
    parser.add_argument("--out_html", type=str, default="plot_result.html")
    parser.add_argument("--out_csv", type=str, default="plot_result.csv")
    parser.add_argument("--show", action="store_true", help=help["show"])
    parser.add_argument("--x", "-x", type=str, default="bpp")
    parser.add_argument("--y", "-y", type=str, default="psnr")
    parser.add_argument(
        "--query", "-q", action="append", default=[], help=help["query"]
    )
    parser.add_argument(
        "--curves",
        "-c",
        action="append",
        default=[],
        help=help["curves"],
    )
    parser.add_argument("--name", "-n", action="append", default=[], help=help["name"])
    parser.add_argument("--pareto", action="append", default=[], help=help["pareto"])
    args = parser.parse_args(argv)

    if len(args.query) == 0:
        args.query = [""]
    num_queries = len(args.query)
    curves_default = {"name": "{name}", "suffix": "", "x": args.x, "y": args.y}
    args.curves = [eval(x) for x in args.curves]  # WARNING: unsafe!
    args.curves = [[{**curves_default, **x} for x in xs] for xs in args.curves]
    args.curves += [[curves_default]] * (num_queries - len(args.curves))
    args.name += [""] * (num_queries - len(args.name))
    args.pareto += [False] * (num_queries - len(args.pareto))

    return args


def main(argv):
    args = build_args(argv)
    repo = aim.Repo(args.aim_repo)
    df = create_dataframe(repo, args)
    plot_dataframe(df, args)


if __name__ == "__main__":
    main(sys.argv[1:])
