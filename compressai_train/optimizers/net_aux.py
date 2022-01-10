from __future__ import annotations

from typing import Dict, cast

import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig

from compressai_train.registry import register_optimizer


@register_optimizer("net_aux")
def net_aux_optimizer(conf: DictConfig, net: nn.Module) -> dict[str, optim.Optimizer]:
    """Returns separate optimizers for net and auxiliary losses.

    Each optimizer operates on a mutually exclusive set of parameters.
    """

    parameters = {
        "net": {
            name
            for name, param in net.named_parameters()
            if param.requires_grad and not name.endswith(".quantiles")
        },
        "aux": {
            name
            for name, param in net.named_parameters()
            if param.requires_grad and name.endswith(".quantiles")
        },
    }

    # Make sure we don't have an intersection of parameters
    params_dict = dict(net.named_parameters())
    inter_params = parameters["net"] & parameters["aux"]
    union_params = parameters["net"] | parameters["aux"]
    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    optimizer = {
        key: optim.Adam(
            (params_dict[name] for name in sorted(parameters[key])),
            lr=conf[key].lr,
        )
        for key in ["net", "aux"]
    }

    return cast(Dict[str, optim.Optimizer], optimizer)
