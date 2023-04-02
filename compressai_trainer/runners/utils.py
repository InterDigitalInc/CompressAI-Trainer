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

from collections import defaultdict
from typing import cast

import pandas as pd
import torch
from compressai.entropy_models import EntropyBottleneck
from PIL import Image

from compressai_trainer import plot
from compressai_trainer.plot import plot_entropy_bottleneck_distributions, plot_rd
from compressai_trainer.plot.featuremap import DEFAULT_COLORMAP
from compressai_trainer.utils.catalyst.loggers import (
    DistributionSuperlogger,
    FigureSuperlogger,
)
from compressai_trainer.utils.compressai.results import compressai_dataframe


class ChannelwiseBppMeter:
    """Log channel-wise rates (bpp)."""

    def __init__(self):
        self._chan_bpp = defaultdict(list)

    def update(self, out_net):
        _, _, h, w = out_net["x_hat"].shape
        chan_bpp = {
            k: l.detach().log2().sum(axis=(-2, -1)) / -(h * w)
            for k, l in out_net["likelihoods"].items()
        }
        for name, ch_bpp in chan_bpp.items():
            self._chan_bpp[name].extend(ch_bpp)

    def log(self, runner: DistributionSuperlogger):
        for name, ch_bpp_samples in self._chan_bpp.items():
            ch_bpp = torch.stack(ch_bpp_samples).mean(dim=0).to(torch.float16).cpu()
            c, *_ = ch_bpp.shape
            ch_bpp_sorted, _ = torch.sort(ch_bpp, descending=True)
            kwargs = dict(
                unused=None,
                scope="epoch",
                context={"name": name},
                bin_range=(0, c),
            )
            runner.log_distribution("chan_bpp_sorted", hist=ch_bpp_sorted, **kwargs)
            runner.log_distribution("chan_bpp_unsorted", hist=ch_bpp, **kwargs)


class DebugOutputsLogger:
    """Log ``debug_outputs`` from out_net/out_enc/out_dec,
    such as featuremaps of tensors.

    To use, add desired outputs to the ``debug_outputs`` key returned by
    ``forward``/``compress``/``decompress``. For example:

    .. code-block:: python

        class MyModel(compressai.models.google.FactorizedPrior):
            def compress(self, x):
                y = self.g_a(x)
                ...
                return {..., "debug_outputs": {"y": y}}
    """

    def log(self, out_infer, i, img_path_prefix):
        debug_outputs = {
            f"debug_{mode}_{k}": v
            for mode in ["net", "enc", "dec"]
            for k, v in out_infer[f"out_{mode}"].get("debug_outputs", {}).items()
        }

        for name, output in debug_outputs.items():
            if isinstance(output, torch.Tensor):
                Image.fromarray(
                    plot.featuremap_image(
                        output[i].cpu().numpy(), cmap=DEFAULT_COLORMAP
                    )
                ).save(f"{img_path_prefix}_{name}.png")


class EbDistributionsFigureLogger:
    """Log EntropyBottleneck (EB) distributions figure."""

    def log(self, runner: FigureSuperlogger, log_figure: bool = True, **kwargs):
        figs = {}

        for name, module in runner.model.named_modules():
            if not isinstance(module, EntropyBottleneck):
                continue

            fig = plot_entropy_bottleneck_distributions(module, **kwargs)
            figs[name] = fig

            if log_figure:
                context = {
                    "module": name,
                }
                runner.log_figure("pdf", fig, context=context)

        return figs


class RdFigureLogger:
    """Log RD figure."""

    def log(
        self,
        runner: FigureSuperlogger,
        df: pd.DataFrame,
        traces,
        codecs: list[str],
        dataset: str = "image/kodak",
        loader: str = "infer",
        metric: str = "psnr",
        opt_metric: str = "mse",
        log_figure: bool = True,
        **kwargs,
    ):
        hover_data = kwargs.get("scatter_kwargs", {}).get("hover_data", [])
        dfs = [
            compressai_dataframe(name, dataset=dataset, opt_metric=opt_metric)
            for name in codecs
        ]
        dfs.append(df)
        df = pd.concat(dfs)
        df = _reorder_dataframe_columns(df, hover_data)
        fig = plot_rd(df, metric=metric, **kwargs)
        for trace in traces:
            fig.add_trace(trace)
        if log_figure:
            context = {
                # "_" is used to order the figures in the experiment tracker.
                "_": int(metric != (opt_metric if opt_metric != "mse" else "psnr")),
                "dataset": dataset,
                "loader": loader,
                "metric": metric,
                "opt_metric": opt_metric,
            }
            runner.log_figure("rd-curves", fig, context=context)
        return fig


def _reorder_dataframe_columns(df: pd.DataFrame, head: list[str]) -> pd.DataFrame:
    head_set = set(head)
    columns = head + [x for x in df.columns if x not in head_set]
    return cast(pd.DataFrame, df[columns])
