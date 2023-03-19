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

import time
from collections import defaultdict
from typing import Any, cast

import pandas as pd
import plotly.graph_objects as go
import torch
import torch.nn.functional as F
from catalyst import metrics
from compressai.models.base import CompressionModel
from PIL import Image

from compressai_trainer import plot
from compressai_trainer.plot import plot_rd
from compressai_trainer.plot.featuremap import DEFAULT_COLORMAP
from compressai_trainer.registry import register_runner
from compressai_trainer.utils.catalyst.loggers import (
    DistributionSuperlogger,
    FigureSuperlogger,
)
from compressai_trainer.utils.compressai.results import compressai_dataframe
from compressai_trainer.utils.metrics import compute_metrics, db
from compressai_trainer.utils.utils import compute_padding, tensor_to_np_img

from .base import BaseRunner

METERS = [
    "loss",
    "aux_loss",
    "bpp_loss",
    "mse_loss",
]

INFER_METERS = [
    "bpp",
    "psnr",
    "ms-ssim",
]

RD_PLOT_METRICS = [
    "psnr",
    "ms-ssim",
    "ms-ssim-db",
]

RD_PLOT_DESCRIPTIONS = [
    "PSNR (RGB)",
    "MS-SSIM (RGB)",
    "MS-SSIM (RGB)",
]

RD_PLOT_TITLE = "Performance evaluation on {dataset} - {metric}"

RD_PLOT_SETTINGS_COMMON: dict[str, Any] = dict(
    codecs=[
        "bmshj2018-factorized",
        "bmshj2018-hyperprior",
        "mbt2018-mean",
        "mbt2018",
        "cheng2020-anchor",
        "vtm",
    ],
    scatter_kwargs=dict(
        hover_data=[
            "name",
            "bpp",
            "psnr",
            "ms-ssim",
            "loss",
            "epoch",
            "criterion.lmbda",
        ],
    ),
)


@register_runner("ImageCompressionRunner")
class ImageCompressionRunner(BaseRunner):
    """Runner for image compression experiments.

    Reimplementation of CompressAI's `examples/train.py
    <https://github.com/InterDigitalInc/CompressAI/blob/master/examples/train.py>`_,
    with additional functionality such as:

    - Plots RD curves.
    - Saves inference outputs including images (``_log_outputs``) and
      featuremaps (``_debug_outputs_logger``).
    - Metrics (e.g. ``loss``). See: ``METERS`` and ``INFER_METERS``.
    - Histograms for latent channel-wise rate distributions.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._debug_outputs_logger = DebugOutputsLogger()
        self._rd_figure_logger = RdFigureLogger()

    def on_loader_start(self, runner):
        super().on_loader_start(runner)
        self._setup_loader_metrics()
        self._setup_meters()

    def handle_batch(self, batch):
        if self.is_infer_loader:
            return self._handle_batch_infer(batch)

        x = batch
        out_net = self.model(x)
        out_criterion = self.criterion(out_net, x)
        loss = {}
        loss["net"] = out_criterion["loss"]

        if self.is_train_loader:
            loss["net"].backward()
            self._grad_clip()
            self.optimizer["net"].step()

        loss["aux"] = CompressionModel.aux_loss(self.model)  # type: ignore

        if self.is_train_loader:
            loss["aux"].backward()  # type: ignore
            self.optimizer["aux"].step()
            self.optimizer["net"].zero_grad()
            self.optimizer["aux"].zero_grad()

        batch_metrics = {
            "loss": loss["net"],
            "aux_loss": loss["aux"],
            **out_criterion,
        }
        self._update_batch_metrics(batch_metrics)

    def _handle_batch_infer(self, batch):
        x = batch.to(self.engine.device)
        out_infer = self.predict_batch(x)
        out_net = out_infer["out_net"]

        out_criterion = self.criterion(out_net, x)
        out_metrics = compute_metrics(x, out_net["x_hat"], ["psnr", "ms-ssim"])
        out_metrics["bpp"] = out_infer["bpp"]
        out_metrics["ms-ssim-db"] = db(1 - out_metrics["ms-ssim"])

        loss = {}
        loss["net"] = out_criterion["loss"]
        loss["aux"] = self.model_module.aux_loss()

        batch_metrics = {
            "loss": loss["net"],
            "aux_loss": loss["aux"],
            **out_criterion,
            **out_metrics,
            "bpp": out_infer["bpp"],
        }
        self._update_batch_metrics(batch_metrics)
        self._handle_custom_metrics(out_net, out_metrics)

        self._log_outputs(x, out_infer)

    def predict_batch(self, batch):
        x = batch.to(self.engine.device)
        out_infer = inference(self.model_module, x)
        return out_infer

    def on_loader_end(self, runner):
        super().on_loader_end(runner)
        if self.is_infer_loader:
            self._log_rd_curves()
            self._loader_metrics["chan_bpp"].log(self)

    @property
    def _current_dataframe(self):
        r = lambda x: float(f"{x:.4g}")
        d = {
            "name": self.hparams["model"]["name"] + "*",
            "epoch": self.epoch_step,
            "criterion.lmbda": self.hparams["criterion"]["lmbda"],
            "loss": r(self.loader_metrics["loss"]),
            "bpp": r(self.loader_metrics["bpp"]),
            "psnr": r(self.loader_metrics["psnr"]),
            "ms-ssim": r(self.loader_metrics["ms-ssim"]),
            # NOTE: The dB of the mean of MS-SSIM samples
            # is not the same as the mean of MS-SSIM dB samples.
            "ms-ssim-db": r(db(1 - self.loader_metrics["ms-ssim"])),
        }
        return pd.DataFrame.from_dict([d])

    def _current_rd_traces(self, metric):
        lmbda = self.hparams["criterion"]["lmbda"]
        num_points = len(self._loader_metrics["bpp"])
        samples_scatter = go.Scatter(
            x=self._loader_metrics["bpp"],
            y=self._loader_metrics[metric],
            mode="markers",
            name=f'{self.hparams["model"]["name"]} {lmbda:.4f}',
            text=[f"lmbda={lmbda:.4f}\nsample_idx={i}" for i in range(num_points)],
            visible="legendonly",
        )
        return [samples_scatter]

    def _grad_clip(self):
        grad_clip = self.hparams["optimizer"].get("grad_clip", None)
        if grad_clip is None:
            return
        max_norm = grad_clip.get("max_norm", None)
        if max_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)

    def _handle_custom_metrics(self, out_net, out_metrics):
        self._loader_metrics["chan_bpp"].update(out_net)
        for metric in ["bpp", *RD_PLOT_METRICS]:
            self._loader_metrics[metric].append(out_metrics[metric])

    def _log_outputs(self, x, out_infer):
        for i in range(len(x)):
            sample_idx = (self.loader_batch_step - 1) * self.loader_batch_size + i + 1
            img_path_prefix = f"{self.hparams['paths']['images']}/{sample_idx:06}"
            Image.fromarray(
                tensor_to_np_img(out_infer["out_dec"]["x_hat"][i].cpu())
            ).save(f"{img_path_prefix}_x_hat.png")
            self._debug_outputs_logger.log(out_infer, i, img_path_prefix)

    def _log_rd_curves(self):
        meta = self.hparams["dataset"]["infer"]["meta"]
        for metric, description in zip(RD_PLOT_METRICS, RD_PLOT_DESCRIPTIONS):
            self._rd_figure_logger.log(
                runner=self,
                df=self._current_dataframe,
                traces=self._current_rd_traces(metric),
                metric=metric,
                dataset=meta["identifier"],
                **RD_PLOT_SETTINGS_COMMON,
                layout_kwargs=dict(
                    title=RD_PLOT_TITLE.format(
                        dataset=meta["name"],
                        metric=description,
                    )
                ),
            )

    def _setup_loader_metrics(self):
        self._loader_metrics = {
            "chan_bpp": ChannelwiseBppMeter(),
            **{k: [] for k in ["bpp", *RD_PLOT_METRICS]},
        }

    def _setup_meters(self):
        keys = list(METERS)
        if self.is_infer_loader:
            keys += INFER_METERS
        self.batch_meters = {
            key: metrics.AdditiveMetric(compute_on_call=False) for key in keys
        }


@torch.no_grad()
def inference(
    model: CompressionModel,
    x: torch.Tensor,
    skip_decompress: bool = False,
    min_div: int = 64,
) -> dict[str, Any]:
    """Run compression model on image batch."""
    n, _, h, w = x.shape
    pad, unpad = compute_padding(h, w, min_div=min_div)
    x_padded = F.pad(x, pad, mode="constant", value=0)

    # Compress using forward.
    out_net = model(x_padded)
    out_net["x_hat"] = F.pad(out_net["x_hat"], unpad)

    # Compress using compress/decompress.
    start = time.time()
    out_enc = model.compress(x_padded)
    enc_time = time.time() - start

    if not skip_decompress:
        start = time.time()
        out_dec = model.decompress(out_enc["strings"], out_enc["shape"])
        dec_time = time.time() - start
        out_dec["x_hat"] = F.pad(out_dec["x_hat"], unpad)
    else:
        out_dec = dict(out_net)
        del out_dec["likelihoods"]
        dec_time = None

    # Compute bpp.
    num_bits = sum(sum(map(len, s)) for s in out_enc["strings"]) * 8.0
    num_pixels = n * h * w
    bpp = num_bits / num_pixels

    return {
        "out_net": out_net,
        "out_enc": out_enc,
        "out_dec": out_dec,
        "bpp": bpp,
        "encoding_time": enc_time,
        "decoding_time": dec_time,
    }


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
        context = {
            # "_" is used to order the figures in the experiment tracker.
            "_": int(metric != (opt_metric if opt_metric != "mse" else "psnr")),
            "dataset": dataset,
            "loader": loader,
            "metric": metric,
            "opt_metric": opt_metric,
        }
        runner.log_figure(f"rd-curves", fig, context=context)


def _reorder_dataframe_columns(df: pd.DataFrame, head: list[str]) -> pd.DataFrame:
    head_set = set(head)
    columns = head + [x for x in df.columns if x not in head_set]
    return cast(pd.DataFrame, df[columns])
