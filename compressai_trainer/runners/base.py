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

import os
from types import ModuleType
from typing import cast

import compressai
import yaml
from catalyst import dl, metrics
from catalyst.typing import TorchCriterion, TorchOptimizer
from compressai.models.base import CompressionModel
from torch.nn.parallel import DataParallel, DistributedDataParallel

import compressai_trainer
from compressai_trainer.utils.catalyst.loggers import AllSuperlogger
from compressai_trainer.utils.utils import num_parameters


class BaseRunner(dl.Runner, AllSuperlogger):
    """Generic runner for all CompressAI Trainer experiments.

    See the ``catalyst.dl.Runner`` documentation for info on runners.

    ``BaseRunner`` provides functionality for common tasks such as:

    - Logging environment: git hashes/diff, pip list, YAML config.
    - Logging model basic info: num params, weight shapes, etc.
    - Batch meters that aggregate (e.g. average) per-loader metrics
      (e.g. loss) which are collected per-batch.
    - Calls ``model.update()`` before inference (i.e. test).
    """

    criterion: TorchCriterion
    model: CompressionModel | DataParallel | DistributedDataParallel
    optimizer: dict[str, TorchOptimizer]
    batch_meters: dict[str, metrics.IMetric]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._has_started = False

    def on_experiment_start(self, runner):
        super().on_experiment_start(runner)
        self._log_config()
        self._log_git_diff(compressai)
        self._log_git_diff(compressai_trainer)
        self._log_pip()
        self._log_model_info()

    def on_epoch_start(self, runner):
        if not self._has_started:
            self._has_started = True
            self._log_state()
        super().on_epoch_start(runner)

    def on_loader_start(self, runner):
        super().on_loader_start(runner)
        if self.is_infer_loader:
            self.model_module.update(force=True)
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

    def log_image(self, *args, **kwargs):
        AllSuperlogger.log_image(self, *args, **kwargs)

    def _update_batch_metrics(self, batch_metrics):
        self.batch_metrics.update(batch_metrics)
        for key in batch_metrics.keys():
            if key not in self.batch_meters:
                continue
            self.batch_meters[key].update(
                _coerce_item(self.batch_metrics[key]),
                self.batch_size,
            )

    def _log_artifact(self, tag: str, filename: str, dir_key: str):
        root = self.hparams["paths"][dir_key]
        dest_path = os.path.join(root, filename)
        self.log_artifact(tag, path_to_artifact=dest_path)

    def _log_config(self):
        self._log_artifact("config.yaml", "config.yaml", "configs")

    def _log_pip(self):
        self._log_artifact("pip_list.txt", "pip_list.txt", "src")
        self._log_artifact("requirements.txt", "requirements.txt", "src")

    def _log_git_diff(self, package: ModuleType):
        self._log_artifact(
            f"{package.__name__}_git_diff", f"{package.__name__}.patch", "src"
        )

    def _log_state(self):
        state = {
            "epoch_step": self.epoch_step,
        }
        self.loggers["aim"].log_artifact(
            "state",
            artifact=yaml.safe_dump(state),
            kind="text",  # type: ignore
            runner=self,
        )

    def _log_model_info(self):
        stats = {
            "num_params": num_parameters(self.model),
        }
        self.log_hparams({"stats": stats})
        print("\nModel:")
        print(self.model_module)
        print("\nModel state dict:")
        for k, v in self.model_module.state_dict().items():
            print(f"{str(list(v.shape)): <24} {k}")
        print("")


def _coerce_item(x):
    return x.item() if hasattr(x, "item") else x
