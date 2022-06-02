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
from typing import Any, Dict, cast

from omegaconf import DictConfig, OmegaConf

from compressai_train.registry.torch import CRITERIONS, MODELS, OPTIMIZERS, SCHEDULERS
from compressai_train.typing.torch import (
    TCriterion,
    TDataLoader,
    TModel,
    TOptimizer,
    TScheduler,
)

from .dataset import create_dataset_tuple


def create_criterion(conf: DictConfig) -> TCriterion:
    kwargs = OmegaConf.to_container(conf, resolve=True)
    kwargs = cast(Dict[str, Any], kwargs)
    del kwargs["type"]
    criterion = CRITERIONS[conf.type](**kwargs)
    return criterion


def create_dataloaders(conf: DictConfig) -> dict[str, TDataLoader]:
    return {
        key: create_dataset_tuple(conf.dataset[key], conf.misc.device).loader
        for key in ["train", "valid", "infer"]
    }


def create_model(conf: DictConfig) -> TModel:
    model = MODELS[conf.model.name](**conf.hp)
    model = model.to(conf.misc.device)
    return model


def create_optimizer(conf: DictConfig, net: TModel) -> TOptimizer:
    return OPTIMIZERS[conf.type](conf, net)


def create_scheduler(conf: DictConfig, optimizer: TOptimizer) -> dict[str, TScheduler]:
    scheduler = {}
    for optim_key, optim_conf in conf.items():
        optim_key = cast(str, optim_key)
        kwargs = OmegaConf.to_container(optim_conf, resolve=True)
        kwargs = cast(Dict[str, Any], kwargs)
        del kwargs["type"]
        kwargs["optimizer"] = optimizer[optim_key]
        scheduler[optim_key] = SCHEDULERS[optim_conf.type](**kwargs)
    return scheduler


def write_config(conf: DictConfig):
    logdir = conf.misc.config_logdir
    filename = "config.yaml"
    s = OmegaConf.to_yaml(conf, resolve=False)
    os.makedirs(logdir, exist_ok=True)
    with open(os.path.join(logdir, filename), "w") as f:
        f.write(s)
