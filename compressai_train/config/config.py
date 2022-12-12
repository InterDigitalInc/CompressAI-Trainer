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

import math
from typing import Any, Dict, cast

from catalyst.utils.torch import load_checkpoint
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
from .env import get_env


def configure_conf(conf: DictConfig):
    conf.env = get_env(conf)
    for loader in conf.dataset:
        conf_loader = conf.dataset[loader]
        drop_last = conf_loader.loader.get("drop_last", False)
        to_int = math.floor if drop_last else math.ceil
        conf_loader.meta.steps_per_epoch = to_int(
            conf_loader.meta.num_samples / conf_loader.loader.batch_size
        )


def create_criterion(conf: DictConfig) -> TCriterion:
    kwargs = OmegaConf.to_container(conf, resolve=True)
    kwargs = cast(Dict[str, Any], kwargs)
    del kwargs["type"]
    criterion = CRITERIONS[conf.type](**kwargs)
    return criterion


def create_dataloaders(conf: DictConfig) -> dict[str, TDataLoader]:
    return {
        loader: create_dataset_tuple(conf.dataset[loader], conf.misc.device).loader
        for loader in conf.dataset
    }


def create_model(conf: DictConfig) -> TModel:
    import compressai.models as _

    model = MODELS[conf.model.name](**conf.hp)
    model = model.to(conf.misc.device)

    if conf.paths.model_checkpoint:
        from compressai_train.config.load import state_dict_from_checkpoint

        checkpoint = load_checkpoint(conf.paths.model_checkpoint)
        state_dict = state_dict_from_checkpoint(checkpoint)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        print(f"Missing keys: {missing_keys}\nUnexpected keys: {unexpected_keys}")

    return model


def create_optimizer(conf: DictConfig, net: TModel) -> TOptimizer:
    return OPTIMIZERS[conf.type](net, conf)


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
