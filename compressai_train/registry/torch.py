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

from typing import Callable, TypeVar

import compressai.zoo.image as cai_zoo_img
from torch.optim import lr_scheduler

from compressai_train.typing import (
    TCriterion,
    TDataset,
    TModel,
    TOptimizer,
    TScheduler,
)

CRITERIONS: dict[str, Callable[..., TCriterion]] = {}
DATASETS: dict[str, Callable[..., TDataset]] = {}
MODELS: dict[str, Callable[..., TModel]] = cai_zoo_img.model_architectures
OPTIMIZERS: dict[str, Callable[..., TOptimizer]] = {}
SCHEDULERS: dict[str, Callable[..., TScheduler]] = {
    k: v for k, v in lr_scheduler.__dict__.items() if k[0].isupper()
}

TCriterion_b = TypeVar("TCriterion_b", bound=TCriterion)
TDataset_b = TypeVar("TDataset_b", bound=TDataset)
TModel_b = TypeVar("TModel_b", bound=TModel)
TOptimizer_b = TypeVar("TOptimizer_b", bound=TOptimizer)
TScheduler_b = TypeVar("TScheduler_b", bound=TScheduler)


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
