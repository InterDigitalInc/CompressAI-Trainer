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

import os
from types import ModuleType
from typing import cast

import compressai
import torch
from catalyst import dl, metrics
from catalyst.typing import TorchCriterion, TorchOptimizer
from compressai.models.google import CompressionModel
from torch.nn.parallel import DataParallel, DistributedDataParallel

import compressai_train
from compressai_train.utils.metrics import compute_metrics
from compressai_train.utils.utils import inference


class ImageCompressionRunner(dl.Runner):
    criterion: TorchCriterion
    model: CompressionModel | DataParallel | DistributedDataParallel
    optimizer: dict[str, TorchOptimizer]
    metrics: dict[str, metrics.IMetric]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def on_experiment_start(self, runner):
        super().on_experiment_start(runner)
        self._log_git_diff(compressai)
        self._log_git_diff(compressai_train)

    def on_loader_start(self, runner):
        super().on_loader_start(runner)
        keys = ["loss", "aux_loss", "bpp_loss", "mse_loss", "lmbda"]
        if self.is_infer_loader:
            keys += ["psnr", "ms-ssim"]
            keys += ["bpp"]
            self.model_module.update()
        self.meters = {
            key: metrics.AdditiveMetric(compute_on_call=False) for key in keys
        }

    def handle_batch(self, batch):
        if self.is_infer_loader:
            return self.predict_batch(batch)

        x = batch
        out_net = self.model(x)
        out_criterion = self.criterion(out_net, x)
        loss = out_criterion["loss"]

        if self.is_train_loader:
            loss.backward()
            self._grad_clip()
            self.optimizer["net"].step()

        aux_loss = CompressionModel.aux_loss(self.model)  # type: ignore

        if self.is_train_loader:
            aux_loss.backward()
            self.optimizer["aux"].step()
            self.optimizer["net"].zero_grad()
            self.optimizer["aux"].zero_grad()

        d = {
            "loss": loss,
            "aux_loss": aux_loss,
            **out_criterion,
            "lmbda": self.criterion.lmbda,
        }

        self.batch_metrics.update(d)

        for key in self.meters.keys():
            self.meters[key].update(
                _coerce_item(self.batch_metrics[key]),
                self.batch_size,
            )

    def predict_batch(self, batch):
        x = batch.to(self.engine.device)

        out_infer = inference(self.model_module, x, skip_decompress=True)
        out_net = out_infer["out_net"]
        out_criterion = self.criterion(out_net, x)
        out_metrics = compute_metrics(x, out_net["x_hat"], ["psnr", "ms-ssim"])

        loss = out_criterion["loss"]
        aux_loss = self.model_module.aux_loss()

        d = {
            "loss": loss,
            "aux_loss": aux_loss,
            **out_criterion,
            "lmbda": self.criterion.lmbda,
            **out_metrics,
            "bpp": out_infer["bpp"],
        }

        self.batch_metrics.update(d)

        for key in self.meters.keys():
            self.meters[key].update(
                _coerce_item(self.batch_metrics[key]),
                self.batch_size,
            )

    def on_loader_end(self, runner):
        for key in self.meters.keys():
            self.loader_metrics[key] = self.meters[key].compute()[0]
        super().on_loader_end(runner)

    def on_epoch_end(self, runner):
        self.epoch_metrics["_epoch_"]["epoch"] = self.epoch_step
        super().on_epoch_end(runner)

    @property
    def model_module(self) -> CompressionModel:
        if isinstance(self.model, (DataParallel, DistributedDataParallel)):
            return cast(CompressionModel, self.model.module)
        return self.model

    def _grad_clip(self):
        grad_clip = self.hparams["optimizer"].get("grad_clip", None)
        if grad_clip is None:
            return
        max_norm = grad_clip.get("max_norm", None)
        if max_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)

    def _log_git_diff(self, package: ModuleType):
        src_root = self.hparams["paths"]["src"]
        diff_path = os.path.join(src_root, f"{package.__name__}.patch")
        self.log_artifact(f"{package.__name__}_git_diff", path_to_artifact=diff_path)


def _coerce_item(x):
    return x.item() if hasattr(x, "item") else x
