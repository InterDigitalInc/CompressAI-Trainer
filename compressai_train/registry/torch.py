from __future__ import annotations

from typing import Callable

import compressai.zoo.image as cai_zoo_img
from torch.optim import lr_scheduler

from compressai_train.typing import (
    TCriterion,
    TCriterion_b,
    TDataset,
    TDataset_b,
    TModel,
    TModel_b,
    TOptimizer,
    TOptimizer_b,
    TScheduler,
    TScheduler_b,
)

CRITERIONS: dict[str, Callable[..., TCriterion]] = {}
DATASETS: dict[str, Callable[..., TDataset]] = {}
MODELS: dict[str, Callable[..., TModel]] = cai_zoo_img.model_architectures
OPTIMIZERS: dict[str, Callable[..., TOptimizer]] = {}
SCHEDULERS: dict[str, Callable[..., TScheduler]] = {
    k: v for k, v in lr_scheduler.__dict__.items() if k[0].isupper()
}


def register_criterion(name: str):
    """Decorator for registering a criterion."""

    def decorator(cls: type[TCriterion_b]) -> type[TCriterion_b]:
        CRITERIONS[name] = cls
        return cls

    return decorator


def register_dataset(name: str):
    """Decorator for registering a dataset."""

    def decorator(cls: type[TDataset_b]) -> type[TDataset_b]:
        DATASETS[name] = cls
        return cls

    return decorator


def register_model(name: str):
    """Decorator for registering a model."""

    def decorator(cls: type[TModel_b]) -> type[TModel_b]:
        MODELS[name] = cls
        return cls

    return decorator


def register_optimizer(name: str):
    """Decorator for registering a optimizer."""

    def decorator(cls: Callable[..., TOptimizer_b]) -> Callable[..., TOptimizer_b]:
        OPTIMIZERS[name] = cls
        return cls

    return decorator


def register_scheduler(name: str):
    """Decorator for registering a scheduler."""

    def decorator(cls: type[TScheduler_b]) -> type[TScheduler_b]:
        SCHEDULERS[name] = cls
        return cls

    return decorator
