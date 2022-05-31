from __future__ import annotations

import argparse
import sys
from typing import Any, cast

import aim
import catalyst
import catalyst.callbacks
import catalyst.utils
import hydra
from catalyst import dl
from omegaconf import DictConfig, OmegaConf

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
from compressai_train.utils.catalyst import AimLogger


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


def build_args(parser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config file",
    )
    return parser


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Example training script.")
    build_args(parser)
    args = parser.parse_args(argv)
    return args


@hydra.main(version_base=None)
def main(conf: DictConfig):
    engine_kwargs = setup(conf)

    runner = ImageCompressionRunner()

    runner.train(
        **engine_kwargs,
        loggers={
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
        },
    )


if __name__ == "__main__":
    main()
