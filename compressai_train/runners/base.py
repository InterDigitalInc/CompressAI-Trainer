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
from catalyst import dl
from catalyst.typing import TorchCriterion, TorchOptimizer
from compressai.models.google import CompressionModel
from torch.nn.parallel import DataParallel, DistributedDataParallel

import compressai_train


class BaseRunner(dl.Runner):
    criterion: TorchCriterion
    model: CompressionModel | DataParallel | DistributedDataParallel
    optimizer: dict[str, TorchOptimizer]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def on_experiment_start(self, runner):
        super().on_experiment_start(runner)
        self._log_git_diff(compressai)
        self._log_git_diff(compressai_train)

    def on_epoch_start(self, runner):
        super().on_epoch_start(runner)

    def on_loader_start(self, runner):
        super().on_loader_start(runner)
        if self.is_infer_loader:
            self.model_module.update()
        self.batch_meters = {}

    def on_loader_end(self, runner):
        for key in self.batch_meters.keys():
            self.loader_metrics[key] = self.batch_meters[key].compute()[0]
        super().on_loader_end(runner)

    def on_epoch_end(self, runner):
        self.epoch_metrics["_epoch_"]["epoch"] = self.epoch_step
        super().on_epoch_end(runner)

    def on_experiment_end(self, runner):
        super().on_experiment_end(runner)

    @property
    def model_module(self) -> CompressionModel:
        """Returns model instance."""
        if isinstance(self.model, (DataParallel, DistributedDataParallel)):
            return cast(CompressionModel, self.model.module)
        return self.model

    def _update_batch_metrics(self, batch_metrics):
        self.batch_metrics.update(batch_metrics)
        for key in self.batch_meters.keys():
            self.batch_meters[key].update(
                _coerce_item(self.batch_metrics[key]),
                self.batch_size,
            )

    def _log_git_diff(self, package: ModuleType):
        src_root = self.hparams["paths"]["src"]
        diff_path = os.path.join(src_root, f"{package.__name__}.patch")
        self.log_artifact(f"{package.__name__}_git_diff", path_to_artifact=diff_path)


def _coerce_item(x):
    return x.item() if hasattr(x, "item") else x
