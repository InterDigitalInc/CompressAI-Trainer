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
import warnings
from shlex import quote
from types import ModuleType
from typing import TYPE_CHECKING, Any, Mapping, OrderedDict, cast

import compressai
import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor

import compressai_trainer
from compressai_trainer.registry import MODELS
from compressai_trainer.typing import TModel
from compressai_trainer.utils import git

from . import outputs
from .config import create_model

if TYPE_CHECKING:
    import torch.nn as nn


def load_config(run_root: str) -> DictConfig:
    """Returns config file given run root path.

    Example of run root path: ``/path/to/runs/e4e6d4d5e5c59c69f3bd7be2``.
    """
    config_path = os.path.join(run_root, outputs.CONFIG_DIR, outputs.CONFIG_NAME)
    conf = OmegaConf.load(config_path)
    return cast(DictConfig, conf)


def load_checkpoint(
    conf: DictConfig, *, epoch: int | str = "best", warn_only: bool = True
) -> nn.Module:
    """Loads particular checkpoint for given conf.

    A particular model is a function of:

        - Hyperparameters/configuration
        - Source code
        - Checkpoint file

    This tries to reassemble/verify the same environment.
    """
    # If git hashes are different, raise error/warning.
    _check_git_commit_version(conf, compressai, warn_only=warn_only)
    _check_git_commit_version(conf, compressai_trainer, warn_only=warn_only)

    device = torch.device(conf.misc.device)
    model = create_model(conf)
    ckpt_path = get_checkpoint_path(conf, epoch)
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = state_dict_from_checkpoint(ckpt)
    model.load_state_dict(state_dict)

    return model


def load_model(conf: DictConfig) -> TModel:
    """Load a model from one of various sources.

    The source is determined by setting the config setting
    ``model.source`` to one of the following:

    - "config":
        Uses CompressAI Trainer standard config.
        (e.g. ``hp``, ``paths.model_checkpoint``, etc.)

    - "from_state_dict":
        Uses model's ``from_state_dict()`` factory method.
        Requires ``model.name`` and ``paths.model_checkpoint`` to be set.
        For example:

        .. code-block:: yaml

            model:
              name: "bmshj2018-factorized"
            paths:
              model_checkpoint: "/home/user/.cache/torch/hub/checkpoints/bmshj2018-factorized-prior-3-5c6f152b.pth.tar"

    - "zoo":
        Uses CompressAI's zoo of models.
        Requires ``model.name``, ``model.metric``, ``model.quality``,
        and ``model.pretrained`` to be set.
        For example:

        .. code-block:: yaml

            model:
              name: "bmshj2018-factorized"
              metric: "mse"
              quality: 3
              pretrained: True
    """
    source = conf.model.get("source", None)

    if source is None:
        raise ValueError(
            "Please override model.source with one of "
            '"config", "from_state_dict", or "zoo".\n'
            "\nExample: "
            '++model.source="config"'
        )

    if source is None or source == "config":
        if not conf.paths.model_checkpoint:
            raise ValueError(
                "Please override paths.model_checkpoint.\nExample: "
                "++paths.model_checkpoint='${paths.checkpoints}/runner.last.pth'"
            )
        return create_model(conf)

    if source == "from_state_dict":
        return _load_checkpoint_from_state_dict(
            conf.model.name,
            conf.paths.model_checkpoint,
        ).to(conf.misc.device)

    if source == "zoo":
        return compressai.zoo.image._load_model(
            conf.model.name,
            conf.model.metric,
            conf.model.quality,
            conf.model.pretrained,
        ).to(conf.misc.device)

    raise ValueError(f"Unknown model.source: {source}")


def _load_checkpoint_from_state_dict(arch: str, checkpoint_path: str):
    ckpt = torch.load(checkpoint_path)
    state_dict = state_dict_from_checkpoint(ckpt)
    state_dict = compressai.zoo.load_state_dict(state_dict)  # for pre-trained models
    model = MODELS[arch].from_state_dict(state_dict)
    return model


def get_checkpoint_path(conf: Mapping[str, Any], epoch: int | str = "best") -> str:
    """Returns checkpoint path for given conf.

    Args:
        conf: Configuration.
        epoch: Exact epoch (int) or "best" or "last" (str).
    """
    ckpt_dir = conf["paths"]["checkpoints"]
    epoch_str = f"{epoch:04}" if isinstance(epoch, int) else epoch
    return os.path.join(ckpt_dir, f"runner.{epoch_str}.pth")


def state_dict_from_checkpoint(ckpt) -> OrderedDict[str, Tensor]:
    "Gets model state dict, with fallback for non-Catalyst trained models."
    return (
        ckpt["model_state_dict"]
        if "model_state_dict" in ckpt
        else ckpt["state_dict"]
        if "state_dict" in ckpt
        else ckpt
    )


def _check_git_commit_version(conf: DictConfig, package: ModuleType, warn_only: bool):
    name = package.__name__
    path = package.__path__[0]
    expected = conf.env.git[name].version
    expected_hash = _git_commit_version_to_hash(expected)
    actual = git.commit_commit_version(root=path)

    if expected == actual:
        return

    message = (
        f"Git commit version for {name} does not match config.\n"
        f"Expected: {expected}.\n"
        f"Actual: {actual}.\n"
        f"Please run: (cd {quote(path)} && git checkout {expected_hash})"
    )

    if warn_only:
        warnings.warn(message)
    else:
        raise ValueError(message)


def _git_commit_version_to_hash(version: str) -> str:
    return version.rstrip("-dirty").rpartition("-")[-1].lstrip("g")
