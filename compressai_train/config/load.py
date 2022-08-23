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
import warnings
from shlex import quote
from types import ModuleType
from typing import TYPE_CHECKING, Any, Mapping, OrderedDict, cast

import compressai
import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor

import compressai_train
from compressai_train.config.config import create_model
from compressai_train.utils import git

if TYPE_CHECKING:
    import torch.nn as nn


def load_config(run_root: str) -> DictConfig:
    """Returns config file given run root path.

    Example of run root path: `/path/to/runs/e4e6d4d5e5c59c69f3bd7be2`.
    """
    config_path = os.path.join(run_root, "configs", "config.yaml")
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
    _check_git_hash(conf, compressai, warn_only=warn_only)
    _check_git_hash(conf, compressai_train, warn_only=warn_only)

    device = torch.device(conf.misc.device)
    model = create_model(conf)
    ckpt_path = get_checkpoint_path(conf, epoch)
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = state_dict_from_checkpoint(ckpt)
    model.load_state_dict(state_dict)

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


def _check_git_hash(conf: DictConfig, package: ModuleType, warn_only: bool):
    name = package.__name__
    path = package.__path__[0]
    expected = conf.env.git[name].hash
    actual = git.commit_hash(root=path)
    hash_len = min(len(expected), len(actual))
    assert hash_len >= 7

    if expected[:hash_len] == actual[:hash_len]:
        return

    message = (
        f"Git hash for {name} does not match config.\n"
        f"Expected: {expected}.\n"
        f"Actual: {actual}.\n"
        f"Please run: (cd {quote(path)} && git checkout {expected})"
    )

    if warn_only:
        warnings.warn(message)
    else:
        raise ValueError(message)
