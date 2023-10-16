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

import random
import time
from collections import defaultdict
from itertools import chain
from typing import Any, Dict, List, Optional, TypeVar, cast

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
from .image_compression import (
    RD_PLOT_DESCRIPTIONS,
    RD_PLOT_METRICS,
    RD_PLOT_SETTINGS_COMMON,
    RD_PLOT_TITLE,
)
from .utils import (
    ChannelwiseBppMeter,
    DebugOutputsLogger,
    EbDistributionsFigureLogger,
    GradientClipper,
    RdFigureLogger,
)

K = TypeVar("K")
V = TypeVar("V")


@register_runner("GVAEImageCompressionRunner")
class GVAEImageCompressionRunner(BaseRunner):
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

        # Choose random lambda for this batch.
        lmbda_idx = random.randint(0, len(self._lmbdas) - 1)

        x = batch
        out_net = self.model(x, lmbda_idx=lmbda_idx)
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
            "lmbda": self._lmbdas[lmbda_idx],
        }

        # Append suffixes, i.e. f"{...}_{lmbda_idx}".
        batch_metrics = self._flatten_batch_metricses([batch_metrics], [lmbda_idx])

        self._update_batch_metrics(batch_metrics)

    def _handle_batch_infer(self, batch):
        x = batch.to(self.engine.device)

        # Run inference for each lambda, then flatten results.
        batch_metrics = self._flatten_batch_metricses(
            [
                self._handle_batch_infer_lmbda(x, lmbda_idx=lmbda_idx)
                for lmbda_idx in self._lmbda_idxs
            ],
            self._lmbda_idxs,
        )

        self._update_batch_metrics(batch_metrics)

        # Save per-sample metrics, too.
        for metric in self._loader_metrics.keys():
            if metric in batch_metrics:
                self._loader_metrics[metric].append(batch_metrics[metric])

    def _handle_batch_infer_lmbda(self, x, lmbda_idx):
        out_infer = self.predict_batch(x, lmbda_idx=lmbda_idx, **self._inference_kwargs)
        out_net = out_infer["out_net"]

        out_criterion = self.criterion(out_net, x)
        out_metrics = compute_metrics(x, out_net["x_hat"], ["psnr", "ms-ssim"])
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
            "lmbda": self._lmbdas[lmbda_idx],
        }

        self._debug_outputs_logger.log(x, out_infer, context={"lmbda_idx": lmbda_idx})
        self._loader_metrics[f"chan_bpp_{lmbda_idx}"].update(out_net)

        return batch_metrics

    def predict_batch(self, batch, lmbda_idx=None, lmbda=None, **kwargs):
        x = batch.to(self.engine.device)

        if lmbda_idx is not None:
            assert lmbda is None or lmbda == self._lmbdas[lmbda_idx]
            lmbda = self._lmbdas[lmbda_idx]

        return inference(
            self.model_module, x, criterion=self.criterion, lmbda=lmbda, **kwargs
        )

    def on_loader_end(self, runner):
        super().on_loader_end(runner)
        if self.loader_key == "infer":
            self._log_rd_curves()
            self._eb_distributions_figure_logger.log(
                log_kwargs=dict(track_kwargs=dict(step=0))
            )
            for i in range(len(self._lmbdas)):
                self._loader_metrics[f"chan_bpp_{i}"].log(context={"lmbda_idx": i})

    @property
    def _current_dataframe(self):
        r = lambda x: float(f"{x:.4g}")
        d = {
            "name": [self.hparams["model"]["name"] + "*" for _ in self._lmbda_idxs],
            "epoch": [self.epoch_step for _ in self._lmbda_idxs],
            "criterion.lmbda": self._lmbdas,
            "loss": [r(self.loader_metrics[f"loss_{i}"]) for i in self._lmbda_idxs],
            "bpp": [r(self.loader_metrics[f"bpp_{i}"]) for i in self._lmbda_idxs],
            "psnr": [r(self.loader_metrics[f"psnr_{i}"]) for i in self._lmbda_idxs],
            "ms-ssim": [
                r(self.loader_metrics[f"ms-ssim_{i}"]) for i in self._lmbda_idxs
            ],
            # NOTE: The dB of the mean of MS-SSIM samples
            # is not the same as the mean of MS-SSIM dB samples.
            "ms-ssim-db": [
                r(db(1 - self.loader_metrics[f"ms-ssim_{i}"])) for i in self._lmbda_idxs
            ],
        }
        return pd.DataFrame.from_dict(d)

    def _log_rd_curves(self):
        meta = self.hparams["dataset"]["infer"]["meta"]
        for metric, description in zip(RD_PLOT_METRICS, RD_PLOT_DESCRIPTIONS):
            self._rd_figure_logger.log(
                df=self._current_dataframe,
                traces=[
                    trace
                    for lmbda_idx, lmbda in enumerate(self._lmbdas)
                    for trace in self._rd_figure_logger.current_rd_traces(
                        x=f"bpp_{lmbda_idx}",
                        y=f"{metric}_{lmbda_idx}",
                        lmbda=lmbda,
                    )
                ],
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

    def _flatten_batch_metricses(self, batch_metricses, lmbda_idxs):
        # Flatten metrics for each lambda.
        singles = {
            f"{metric_name}_{lmbda_idx}": value
            for lmbda_idx, batch_metrics in zip(lmbda_idxs, batch_metricses)
            for metric_name, value in batch_metrics.items()
        }

        # Average metrics over lambdas.
        averages = {
            metric_name: sum(values) / len(values)
            for metric_name, values in _transpose_list_dict(batch_metricses).items()
        }

        return {**singles, **averages}

    def _setup_metrics(self):
        # Expand any meters containing a * to work with multiple lmbdas.
        meter_keys = [
            [meter_name]
            if "*" not in meter_name
            else [meter_name.replace("*", str(i)) for i in self._lmbda_idxs]
            for meter_name in self._meter_keys[self.loader_key]
        ]
        meter_keys = list(chain(*meter_keys))  # Flatten list of lists.

        self.batch_meters = {
            key: metrics.AdditiveMetric(compute_on_call=False) for key in meter_keys
        }

        self._loader_metrics = {
            **{f"chan_bpp_{i}": ChannelwiseBppMeter(self) for i in self._lmbda_idxs},
            **{k: [] for k in ["bpp", *RD_PLOT_METRICS]},
            **{
                f"{meter_name}_{i}": []
                for meter_name in ["bpp", *RD_PLOT_METRICS]
                for i in self._lmbda_idxs
            },
        }

    @property
    def _lmbdas(self) -> List[float]:
        # Alternative:
        # return cast(List[float], list(self.hparams["hp"]["lambdas"]))
        return cast(List[float], list(self.model_module.lambdas))

    @property
    def _lmbda_idxs(self) -> List[int]:
        return list(range(len(self._lmbdas)))


@torch.no_grad()
def inference(
    model: CompressionModel,
    x: torch.Tensor,
    skip_compress: bool = False,
    skip_decompress: bool = False,
    criterion: Optional[TCriterion] = None,
    min_div: int = 64,
    *,
    lmbda: float = None,
) -> dict[str, Any]:
    """Run compression model on image batch."""
    n, _, h, w = x.shape
    pad, unpad = compute_padding(h, w, min_div=min_div)
    x_padded = F.pad(x, pad, mode="constant", value=0)

    # Compress using forward.
    out_net = model(x_padded, lmbda=lmbda)
    out_net["x_hat"] = F.pad(out_net["x_hat"], unpad)

    # Compress using compress/decompress.
    if not skip_compress:
        start = time.time()
        out_enc = model.compress(x_padded, lmbda=lmbda)
        enc_time = time.time() - start
    else:
        out_enc = {}
        enc_time = None

    if not skip_decompress:
        assert not skip_compress
        start = time.time()
        out_dec = model.decompress(out_enc["strings"], out_enc["shape"], lmbda=lmbda)
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
        out_criterion = criterion(out_net, x, lmbda=lmbda)
        bpp = out_criterion["bpp_loss"].item()

    return {
        "out_net": out_net,
        "out_enc": out_enc,
        "out_dec": out_dec,
        "bpp": bpp,
        "encoding_time": enc_time,
        "decoding_time": dec_time,
    }


def _transpose_list_dict(ds: List[Dict[K, V]]) -> Dict[K, List[V]]:
    result = defaultdict(list)
    for d in ds:
        for k, v in d.items():
            result[k].append(v)
    return result
