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

from collections import defaultdict
from typing import Any

import pandas as pd
import torch
from catalyst import metrics
from compressai.models.google import CompressionModel

from compressai_train.registry import register_runner
from compressai_train.utils.metrics import compute_metrics
from compressai_train.utils.utils import inference

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
        "mbt2018",
        "cheng2020-anchor",
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
        self._loader_metrics = {}

    def on_loader_start(self, runner):
        super().on_loader_start(runner)
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
        self._handle_custom_metrics(out_net)

    def on_loader_end(self, runner):
        super().on_loader_end(runner)
        if self.is_infer_loader:
            self._log_rd_figure(**RD_PLOT_SETTINGS)
            self._log_chan_bpp()

    @property
    def _current_dataframe(self):
        r = lambda x: float(f"{x:.4g}")
        d = {
            "name": self.hparams["model"]["name"],
            "epoch": self.epoch_step,
            "criterion.lmbda": self.hparams["criterion"]["lmbda"],
            "loss": r(self.loader_metrics["loss"]),
            "bpp": r(self.loader_metrics["bpp"]),
            "psnr": r(self.loader_metrics["psnr"]),
            "ms-ssim": r(self.loader_metrics["ms-ssim"]),
        }
        return pd.DataFrame.from_dict([d])

    def _grad_clip(self):
        grad_clip = self.hparams["optimizer"].get("grad_clip", None)
        if grad_clip is None:
            return
        max_norm = grad_clip.get("max_norm", None)
        if max_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)

    def _handle_custom_metrics(self, out_net):
        likelihoods = out_net["likelihoods"]
        _, _, h, w = out_net["x_hat"].shape
        chan_bpp = {
            k: l.detach().log2().sum(axis=(-2, -1)) / -(h * w)
            for k, l in likelihoods.items()
        }
        if "chan_bpp" not in self._loader_metrics:
            self._loader_metrics["chan_bpp"] = defaultdict(list)
        for k, ch_bpp in chan_bpp.items():
            self._loader_metrics["chan_bpp"][k].extend(ch_bpp)

    def _log_chan_bpp(self):
        for k, ch_bpp in self._loader_metrics["chan_bpp"].items():
            ch_bpp = torch.stack(ch_bpp).mean(dim=0).to(torch.float16).cpu()
            c, *_ = ch_bpp.shape
            ch_bpp_sorted, _ = torch.sort(ch_bpp, descending=True)
            bin_edges = torch.arange(c + 1)
            kwargs = dict(
                unused=None,
                scope="epoch",
                context={"name": k},
                bin_edges=bin_edges.numpy(),
            )
            self.log_distribution("chan_bpp_sorted", hist=ch_bpp_sorted, **kwargs)
            self.log_distribution("chan_bpp_unsorted", hist=ch_bpp, **kwargs)

    def _setup_meters(self):
        keys = list(METERS)
        if self.is_infer_loader:
            keys += INFER_METERS
        self.batch_meters = {
            key: metrics.AdditiveMetric(compute_on_call=False) for key in keys
        }
