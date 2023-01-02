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

from dataclasses import dataclass
from typing import Callable, cast

from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torchvision import transforms

from compressai_trainer.registry.torch import DATASETS
from compressai_trainer.registry.torchvision import TRANSFORMS
from compressai_trainer.typing.torch import TDataLoader, TDataset


@dataclass
class DatasetTuple:
    transform: transforms.Compose
    dataset: TDataset
    loader: TDataLoader


def create_data_transform(transform_conf: DictConfig) -> Callable:
    name, kwargs = next(iter(transform_conf.items()))
    name = cast(str, name)
    return TRANSFORMS[name](**kwargs)


def create_data_transform_composition(conf: DictConfig) -> transforms.Compose:
    return transforms.Compose(
        [create_data_transform(transform_conf) for transform_conf in conf.transforms]
    )


def create_dataset(conf: DictConfig, transform: Callable) -> TDataset:
    return DATASETS[conf.type](**conf.config, transform=transform)


def create_dataloader(conf: DictConfig, dataset: TDataset, device: str) -> TDataLoader:
    return DataLoader(dataset, **conf.loader, pin_memory=(device == "cuda"))


def create_dataset_tuple(conf: DictConfig, device: str) -> DatasetTuple:
    transform = create_data_transform_composition(conf)
    dataset = create_dataset(conf, transform)
    loader = create_dataloader(conf, dataset, device)
    return DatasetTuple(transform=transform, dataset=dataset, loader=loader)
