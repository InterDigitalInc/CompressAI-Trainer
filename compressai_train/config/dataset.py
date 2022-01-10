from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, cast

from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torchvision import transforms

from compressai_train.registry.torch import DATASETS
from compressai_train.registry.torchvision import TRANSFORMS
from compressai_train.typing.torch import TDataLoader, TDataset


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
