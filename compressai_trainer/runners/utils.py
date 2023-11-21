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
from typing import Any, Optional, cast

import pandas as pd
import plotly.graph_objects as go
import torch
from catalyst import dl
from compressai.entropy_models import EntropyBottleneck
from PIL import Image

from compressai_trainer import plot
from compressai_trainer.plot import plot_entropy_bottleneck_distributions, plot_rd
from compressai_trainer.plot.featuremap import DEFAULT_COLORMAP
from compressai_trainer.utils.compressai.results import compressai_results_dataframe
from compressai_trainer.utils.utils import tensor_to_np_img


class GradientClipper:
    def __init__(self, runner: dl.Runner):
        self.runner = runner

    def __call__(self):
        grad_clip = self.runner.hparams["optimizer"].get("grad_clip", None)
        if grad_clip is None:
            return
        max_norm = grad_clip.get("max_norm", None)
        if max_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.runner.model.parameters(), max_norm)


class ChannelwiseBppMeter:
    """Log channel-wise rates (bpp)."""

    def __init__(self, runner):
        self.runner = runner
        self._chan_bpp = defaultdict(list)

    def update(self, out_net):
        _, _, h, w = out_net["x_hat"].shape
        chan_bpp = {
            k: l.detach().log2().sum(axis=(-2, -1)) / -(h * w)
            for k, l in out_net["likelihoods"].items()
        }
        for name, ch_bpp in chan_bpp.items():
            self._chan_bpp[name].extend(ch_bpp)

    def log(self, context=None):
        for name, ch_bpp_samples in self._chan_bpp.items():
            ch_bpp = torch.stack(ch_bpp_samples).mean(dim=0).to(torch.float16).cpu()
            c, *_ = ch_bpp.shape
            ch_bpp_sorted, _ = torch.sort(ch_bpp, descending=True)
            kw = dict(
                unused=None,
                scope="epoch",
                context={"name": name, **(context or {})},
                bin_range=(0, c),
            )
            self.runner.log_distribution("chan_bpp_sorted", hist=ch_bpp_sorted, **kw)
            self.runner.log_distribution("chan_bpp_unsorted", hist=ch_bpp, **kw)


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

    def __init__(self, runner):
        self.runner = runner

    def log(self, inputs, out_infer, context=None):
        # Collect items from out_infer[*]["debug_outputs"].
        debug_outputs = {
            (mode, k): v
            for mode, out in out_infer.items()
            if isinstance(out, dict) and "debug_outputs" in out
            for k, v in out.get("debug_outputs", {}).items()
        }

        # Ensure x_hat is logged, even if it is not specified in debug_outputs.
        if "x_hat" in out_infer.get("out_dec", {}):
            debug_outputs[("out_dec", "x_hat")] = out_infer["out_dec"]["x_hat"]

        for i in range(len(inputs)):
            sample_offset = (
                self.runner.loader_batch_step - 1
            ) * self.runner.loader_batch_size + 1
            sample_idx = sample_offset + i

            for (mode, key), outputs in debug_outputs.items():
                mode = mode.lstrip("out_")
                self._log_output(mode, key, inputs[i], outputs[i], sample_idx, context)

    def _log_output(self, mode, key, input, output, sample_idx, context):
        context = {"mode": mode, "key": key, **(context or {})}
        context_str = "_".join(f"{v}" for v in context.values())

        if key == "x_hat":
            arr = tensor_to_np_img(output.cpu())
        else:
            arr = plot.featuremap_image(output.cpu().numpy(), cmap=DEFAULT_COLORMAP)

        img_dir = self.runner.hparams["paths"]["images"]
        Image.fromarray(arr).save(f"{img_dir}/{sample_idx:06}_{context_str}.png")

        log_kwargs = dict(
            format="webp",
            lossless=True,
            quality=50,
            method=6,
            track_kwargs=dict(step=sample_idx),
        )
        self.runner.log_image("output", arr, context=context, **log_kwargs)


class EbDistributionsFigureLogger:
    """Log EntropyBottleneck (EB) distributions figure."""

    def __init__(self, runner):
        self.runner = runner

    def log(
        self,
        log_figure: bool = True,
        log_kwargs: Optional[dict[str, Any]] = None,
        context: Optional[dict[str, Any]] = None,
        **kwargs,
    ):
        log_kwargs = log_kwargs or {}
        figs = {}

        for name, module in self.runner.model.named_modules():
            if not isinstance(module, EntropyBottleneck):
                continue

            fig = plot_entropy_bottleneck_distributions(module, **kwargs)
            figs[name] = fig

            if log_figure:
                context_ = {"module": name, **(context or {})}
                self.runner.log_figure("pdf", fig, context=context_, **log_kwargs)

        return figs


class RdFigureLogger:
    """Log RD figure."""

    def __init__(self, runner):
        self.runner = runner

    def log(
        self,
        df: pd.DataFrame,
        traces,
        codecs: list[str],
        dataset: str = "image/kodak",
        loader: str = "infer",
        metric: str = "psnr",
        opt_metric: str = "mse",
        log_figure: bool = True,
        context: Optional[dict[str, Any]] = None,
        **kwargs,
    ):
        hover_data = kwargs.get("scatter_kwargs", {}).get("hover_data", [])
        dfs = [compressai_results_dataframe(filename) for filename in codecs]
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
                **(context or {}),
            }
            self.runner.log_figure("-rd-curves", fig, context=context)
        return fig

    def current_rd_traces(self, x: str, y: str, lmbda: float):
        num_points = len(self.runner._loader_metrics[x])
        samples_scatter = go.Scatter(
            x=self.runner._loader_metrics[x],
            y=self.runner._loader_metrics[y],
            mode="markers",
            name=f'{self.runner.hparams["model"]["name"]} {lmbda:.4f}',
            text=[f"lmbda={lmbda:.4f}\nsample_idx={i}" for i in range(num_points)],
            visible="legendonly",
        )
        return [samples_scatter]


class PdfSignaturesFigureLogger:
    """Log PDF signatures figure."""

    def __init__(self, runner):
        self.runner = runner

    def log(
        self,
        dataset: str = "image/kodak",
        loader: str = "infer",
        log_figure: bool = True,
        context: Optional[dict[str, Any]] = None,
        log_kwargs: Optional[dict[str, Any]] = None,
        batch_slice: slice = slice(None),
        **kwargs,
    ):
        def collect(key):
            return [
                value.detach().cpu().numpy()
                for value in self.runner._loader_metrics[key][batch_slice]
            ]

        x = collect("x")
        p = collect("p")
        p_hat = collect("p_hat")

        fig = plot.plot_pdf_signatures(x, p, p_hat, backend="plotly", **kwargs)
        fig.update_layout(autosize=False, width=700, height=150 * len(x))

        if log_figure:
            log_kwargs = log_kwargs or {}
            context = {
                "dataset": dataset,
                "loader": loader,
                **(context or {}),
            }
            self.runner.log_figure("pdf-signatures", fig, context=context, **log_kwargs)

        return fig


def _reorder_dataframe_columns(df: pd.DataFrame, head: list[str]) -> pd.DataFrame:
    head_set = set(head)
    columns = head + [x for x in df.columns if x not in head_set]
    return cast(pd.DataFrame, df[columns])
