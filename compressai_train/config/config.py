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

from typing import Any, Dict, cast

import aim
from catalyst import dl
from omegaconf import DictConfig, OmegaConf

from compressai_train.registry.catalyst import CALLBACKS
from compressai_train.registry.torch import CRITERIONS, MODELS, OPTIMIZERS, SCHEDULERS
from compressai_train.typing.catalyst import TCallback
from compressai_train.typing.torch import (
    TCriterion,
    TDataLoader,
    TModel,
    TOptimizer,
    TScheduler,
)
from compressai_train.utils.catalyst import AimLogger

from .dataset import create_dataset_tuple


def create_callback(conf: DictConfig) -> TCallback:
    kwargs = OmegaConf.to_container(conf, resolve=True)
    kwargs = cast(Dict[str, Any], kwargs)
    del kwargs["type"]
    callback = CALLBACKS[conf.type](**kwargs)
    return callback


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


def configure_engine(conf: DictConfig) -> dict[str, Any]:
    engine_kwargs = OmegaConf.to_container(conf.engine, resolve=True)
    engine_kwargs = cast(Dict[str, Any], engine_kwargs)
    engine_kwargs["callbacks"] = [
        create_callback(cb_conf) for cb_conf in conf.engine.callbacks
    ]
    engine_kwargs["hparams"] = OmegaConf.to_container(conf, resolve=True)
    engine_kwargs["loggers"] = {
        "aim": AimLogger(
            experiment=conf.exp.name,
            run_hash=conf.env.aim.run_hash,
            repo=aim.Repo(
                conf.env.aim.repo,
                init=not aim.Repo.exists(conf.env.aim.repo),
            ),
            **conf.engine.loggers.aim,
        ),
        "tensorboard": dl.TensorboardLogger(
            **conf.engine.loggers.tensorboard,
        ),
    }
    return engine_kwargs
