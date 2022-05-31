from __future__ import annotations

from typing import Any

import catalyst
import catalyst.callbacks
import catalyst.utils
import hydra
from omegaconf import DictConfig

from compressai_train.config import (
    configure_engine,
    create_criterion,
    create_dataloaders,
    create_model,
    create_optimizer,
    create_scheduler,
    get_env,
)
from compressai_train.runners import ImageCompressionRunner


def setup(conf: DictConfig) -> dict[str, Any]:
    catalyst.utils.set_global_seed(conf.misc.seed)
    catalyst.utils.prepare_cudnn(benchmark=True)

    conf.env = get_env(conf)

    model = create_model(conf)
    criterion = create_criterion(conf.criterion)
    optimizer = create_optimizer(conf.optimizer, model)
    scheduler = create_scheduler(conf.scheduler, optimizer)
    loaders = create_dataloaders(conf)

    d = dict(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        loaders=loaders,
    )

    return {**d, **configure_engine(conf)}


@hydra.main(version_base=None)
def main(conf: DictConfig):
    engine_kwargs = setup(conf)
    runner = ImageCompressionRunner()
    runner.train(**engine_kwargs)


if __name__ == "__main__":
    main()
