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
from typing import Any, Optional

import pandas as pd
import torch
import torch.nn.functional as F
from catalyst import metrics
from compressai.models.base import CompressionModel
from compressai.typing import TCriterion

from compressai_trainer.registry import register_runner
from compressai_trainer.utils.metrics import compute_metrics, db
from compressai_trainer.utils.utils import compute_padding

from .base import BaseRunner
from .utils import (
    ChannelwiseBppMeter,
    DebugOutputsLogger,
    EbDistributionsFigureLogger,
    GradientClipper,
    RdFigureLogger,
)

RD_PLOT_TITLE = "Performance evaluation on {dataset} - {metric}"

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

RD_PLOT_SETTINGS_COMMON: dict[str, Any] = dict(
    codecs=[
        "image/kodak/compressai-bmshj2018-factorized_mse_cuda.json",
        "image/kodak/compressai-bmshj2018-hyperprior_mse_cuda.json",
        "image/kodak/compressai-mbt2018-mean_mse_cuda.json",
        "image/kodak/compressai-mbt2018_mse_cuda.json",
        "image/kodak/compressai-cheng2020-anchor_mse_cuda.json",
        "image/kodak/vtm.json",
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

    - Plots RD curves, learned entropy bottleneck distributions,
      and histograms for latent channel-wise rate distributions.
    - Saves inference outputs including images and featuremaps.

    Set the input arguments by overriding the defaults in
    ``conf/runner/ImageCompressionRunner.yaml``.
    """

    def __init__(
        self,
        inference: dict[str, Any],
        meters: dict[str, list[str]],
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._inference_kwargs = inference
        self._meter_keys = meters
        self._grad_clip = GradientClipper(self)
        self._debug_outputs_logger = DebugOutputsLogger(self)
        self._eb_distributions_figure_logger = EbDistributionsFigureLogger(self)
        self._rd_figure_logger = RdFigureLogger(self)

    def on_loader_start(self, runner):
        super().on_loader_start(runner)
        self._setup_metrics()

    def handle_batch(self, batch):
        if self.loader_key == "infer":
            return self._handle_batch_infer(batch)

        x = batch
        out_net = self.model(x)
        out_criterion = self.criterion(out_net, x)
        loss = {
            "net": out_criterion["loss"],
            "aux": self.model_module.aux_loss(),
        }

        if self.loader_key == "train":
            loss["net"].backward()
            loss["aux"].backward()
            self._grad_clip()
            self.optimizer["net"].step()
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
        out_infer = self.predict_batch(x, **self._inference_kwargs)
        out_net = out_infer["out_net"]
        out_dec = out_infer["out_dec"]

        out_criterion = self.criterion(out_net, x)
        out_metrics = compute_metrics(x, out_dec["x_hat"], ["psnr", "ms-ssim"])
        out_metrics["bpp"] = out_infer["bpp"]
        out_metrics["ms-ssim-db"] = db(1 - out_metrics["ms-ssim"])

        loss = {
            "net": out_criterion["loss"],
            "aux": self.model_module.aux_loss(),
        }

        batch_metrics = {
            "loss": loss["net"],
            "aux_loss": loss["aux"],
            **out_criterion,
            **out_metrics,
            "bpp": out_infer["bpp"],
        }
        self._update_batch_metrics(batch_metrics)
        self._handle_custom_metrics(out_net, out_metrics)

        self._debug_outputs_logger.log(x, out_infer)

    def predict_batch(self, batch, **kwargs):
        x = batch.to(self.engine.device)
        return inference(self.model_module, x, criterion=self.criterion, **kwargs)

    def on_loader_end(self, runner):
        super().on_loader_end(runner)
        if self.loader_key == "infer":
            self._log_rd_curves()
            self._eb_distributions_figure_logger.log(
                log_kwargs=dict(track_kwargs=dict(step=0))
            )
            self._loader_metrics["chan_bpp"].log()

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

    def _log_rd_curves(self):
        meta = self.hparams["dataset"]["infer"]["meta"]
        for metric, description in zip(RD_PLOT_METRICS, RD_PLOT_DESCRIPTIONS):
            self._rd_figure_logger.log(
                df=self._current_dataframe,
                traces=self._rd_figure_logger.current_rd_traces(
                    x="bpp", y=metric, lmbda=self.hparams["criterion"]["lmbda"]
                ),
                metric=metric,
                dataset=meta["identifier"],
                **RD_PLOT_SETTINGS_COMMON,
                layout_kwargs=dict(
                    title=RD_PLOT_TITLE.format(
                        dataset=meta["name"],
                        metric=description,
                    ),
                ),
            )

    def _handle_custom_metrics(self, out_net, out_metrics):
        self._loader_metrics["chan_bpp"].update(out_net)
        for metric in ["bpp", *RD_PLOT_METRICS]:
            self._loader_metrics[metric].append(out_metrics[metric])

    def _setup_metrics(self):
        self.batch_meters = {
            key: metrics.AdditiveMetric(compute_on_call=False)
            for key in self._meter_keys[self.loader_key]
        }

        self._loader_metrics = {
            "chan_bpp": ChannelwiseBppMeter(self),
            **{k: [] for k in ["bpp", *RD_PLOT_METRICS]},
        }


@torch.no_grad()
def inference(
    model: CompressionModel,
    x: torch.Tensor,
    skip_compress: bool = False,
    skip_decompress: bool = False,
    criterion: Optional[TCriterion] = None,
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
    if not skip_compress:
        start = time.time()
        out_enc = model.compress(x_padded)
        enc_time = time.time() - start
    else:
        out_enc = {}
        enc_time = None

    if not skip_decompress:
        assert not skip_compress
        start = time.time()
        out_dec = model.decompress(out_enc["strings"], out_enc["shape"])
        dec_time = time.time() - start
        out_dec["x_hat"] = F.pad(out_dec["x_hat"], unpad)
    else:
        out_dec = dict(out_net)
        del out_dec["likelihoods"]
        dec_time = None

    # Compute bpp.
    if not skip_compress:
        num_bits = sum(sum(map(len, s)) for s in out_enc["strings"]) * 8.0
        num_pixels = n * h * w
        bpp = num_bits / num_pixels
    else:
        out_criterion = criterion(out_net, x)
        bpp = out_criterion["bpp_loss"].item()

    return {
        "out_net": out_net,
        "out_enc": out_enc,
        "out_dec": out_dec,
        "bpp": bpp,
        "encoding_time": enc_time,
        "decoding_time": dec_time,
    }
