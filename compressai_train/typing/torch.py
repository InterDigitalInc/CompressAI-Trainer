from __future__ import annotations

from typing import Dict, TypeVar, Union

import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau, _LRScheduler
from torch.utils.data import DataLoader, Dataset

TCriterion = nn.Module
TDataLoader = DataLoader
TDataset = Dataset
TModel = nn.Module
TOptimizer = Dict[str, Optimizer]
TScheduler = Union[ReduceLROnPlateau, _LRScheduler]
TSchedulerDict = Dict[str, TScheduler]

TCriterion_b = TypeVar("TCriterion_b", bound=TCriterion)
TDataLoader_b = TypeVar("TDataLoader_b", bound=TDataLoader)
TDataset_b = TypeVar("TDataset_b", bound=TDataset)
TModel_b = TypeVar("TModel_b", bound=TModel)
TOptimizer_b = TypeVar("TOptimizer_b", bound=TOptimizer)
TScheduler_b = TypeVar("TScheduler_b", bound=TScheduler)
