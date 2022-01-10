from __future__ import annotations

from typing import Any, Dict, cast

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

from .dataset import create_dataset_tuple


def create_callback(conf: DictConfig) -> TCallback:
    kwargs = OmegaConf.to_container(conf)
    kwargs = cast(Dict[str, Any], kwargs)
    del kwargs["type"]
    callback = CALLBACKS[conf.type](**kwargs)
    return callback


def create_criterion(conf: DictConfig) -> TCriterion:
    kwargs = OmegaConf.to_container(conf)
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
        kwargs = OmegaConf.to_container(optim_conf)
        kwargs = cast(Dict[str, Any], kwargs)
        del kwargs["type"]
        kwargs["optimizer"] = optimizer[optim_key]
        scheduler[optim_key] = SCHEDULERS[optim_conf.type](**kwargs)
    return scheduler


def configure_engine(conf: DictConfig) -> dict[str, Any]:
    engine_kwargs = OmegaConf.to_container(conf.engine)
    engine_kwargs = cast(Dict[str, Any], engine_kwargs)
    engine_kwargs["callbacks"] = [
        create_callback(cb_conf) for cb_conf in conf.engine.callbacks
    ]
    engine_kwargs["hparams"] = OmegaConf.to_container(conf)
    return engine_kwargs
