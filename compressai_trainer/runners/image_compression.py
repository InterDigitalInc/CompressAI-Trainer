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
from typing import Any, cast

import pandas as pd
import plotly.graph_objects as go
import torch
from catalyst import metrics
from compressai.models.base import CompressionModel
from PIL import Image

from compressai_trainer.plot import plot_rd
from compressai_trainer.registry import register_runner
from compressai_trainer.utils.metrics import compute_metrics
from compressai_trainer.utils.utils import (
    compressai_dataframe,
    inference,
    tensor_to_np_img,
)

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

RD_PLOT_SETTINGS: dict[str, Any] = dict(
    title="Performance evaluation on Kodak - PSNR (RGB)",
    dataset="image/kodak",
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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def on_loader_start(self, runner):
        super().on_loader_start(runner)
        self._setup_loader_metrics()
        self._setup_meters()

    def handle_batch(self, batch):
        if self.is_infer_loader:
            return self.predict_batch(batch)

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

    def predict_batch(self, batch):
        x = batch.to(self.engine.device)

        out_infer = inference(self.model_module, x, skip_decompress=True)
        out_net = out_infer["out_net"]
        out_criterion = self.criterion(out_net, x)
        out_metrics = compute_metrics(x, out_net["x_hat"], ["psnr", "ms-ssim"])
        out_metrics["bpp"] = out_infer["bpp"]

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

    def on_loader_end(self, runner):
        super().on_loader_end(runner)
        if self.is_infer_loader:
            self._log_rd_figure(**RD_PLOT_SETTINGS)
            self._log_chan_bpp()

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
        }
        return pd.DataFrame.from_dict([d])

    def _current_rd_traces(self):
        lmbda = self.hparams["criterion"]["lmbda"]
        num_points = len(self._loader_metrics["bpp"])
        samples_scatter = go.Scatter(
            x=self._loader_metrics["bpp"],
            y=self._loader_metrics["psnr"],
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
        likelihoods = out_net["likelihoods"]
        _, _, h, w = out_net["x_hat"].shape
        chan_bpp = {
            k: l.detach().log2().sum(axis=(-2, -1)) / -(h * w)
            for k, l in likelihoods.items()
        }
        for name, ch_bpp in chan_bpp.items():
            self._loader_metrics["chan_bpp"][name].extend(ch_bpp)
        self._loader_metrics["bpp"].append(out_metrics["bpp"])
        self._loader_metrics["psnr"].append(out_metrics["psnr"])

    def _log_chan_bpp(self):
        for name, ch_bpp in self._loader_metrics["chan_bpp"].items():
            ch_bpp = torch.stack(ch_bpp).mean(dim=0).to(torch.float16).cpu()
            c, *_ = ch_bpp.shape
            ch_bpp_sorted, _ = torch.sort(ch_bpp, descending=True)
            kwargs = dict(
                unused=None,
                scope="epoch",
                context={"name": name},
                bin_range=(0, c),
            )
            self.log_distribution("chan_bpp_sorted", hist=ch_bpp_sorted, **kwargs)
            self.log_distribution("chan_bpp_unsorted", hist=ch_bpp, **kwargs)

    def _log_outputs(self, x, out_infer):
        for i in range(len(x)):
            sample_idx = (self.loader_batch_step - 1) * self.loader_batch_size + i + 1
            img_path_prefix = f"{self.hparams['paths']['images']}/{sample_idx:06}"
            Image.fromarray(
                tensor_to_np_img(out_infer["out_dec"]["x_hat"][i].cpu())
            ).save(f"{img_path_prefix}_x_hat.png")

    def _log_rd_figure(self, codecs: list[str], dataset: str, **kwargs):
        hover_data = kwargs.get("scatter_kwargs", {}).get("hover_data", [])
        dfs = [compressai_dataframe(name, dataset=dataset) for name in codecs]
        dfs.append(self._current_dataframe)
        df = pd.concat(dfs)
        df = _reorder_dataframe_columns(df, hover_data)
        fig = plot_rd(df, **kwargs)
        for trace in self._current_rd_traces():
            fig.add_trace(trace)
        context = {
            "dataset": dataset,
            "loader": "infer",
            "metric": "psnr",
            "opt_metric": "mse",
        }
        self.log_figure(f"rd-curves", fig, context=context)

    def _setup_loader_metrics(self):
        self._loader_metrics = {
            "chan_bpp": defaultdict(list),
            "bpp": [],
            "psnr": [],
        }

    def _setup_meters(self):
        keys = list(METERS)
        if self.is_infer_loader:
            keys += INFER_METERS
        self.batch_meters = {
            key: metrics.AdditiveMetric(compute_on_call=False) for key in keys
        }


def _reorder_dataframe_columns(df: pd.DataFrame, head: list[str]) -> pd.DataFrame:
    head_set = set(head)
    columns = head + [x for x in df.columns if x not in head_set]
    return cast(pd.DataFrame, df[columns])
