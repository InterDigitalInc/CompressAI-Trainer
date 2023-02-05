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

from typing import Any, Dict, cast

import aim
from catalyst import dl
from omegaconf import DictConfig, OmegaConf

from compressai_trainer.registry.catalyst import CALLBACKS, RUNNERS
from compressai_trainer.typing.catalyst import TCallback, TRunner
from compressai_trainer.utils.catalyst.loggers import AimLogger

PRIMARY_LOGGER = "aim"


def create_callback(conf: DictConfig) -> TCallback:
    kwargs = OmegaConf.to_container(conf, resolve=True)
    kwargs = cast(Dict[str, Any], kwargs)
    del kwargs["type"]
    callback = CALLBACKS[conf.type](**kwargs)
    return callback


def create_logger(conf: DictConfig, logger_type: str) -> dl.ILogger:
    if logger_type == "aim":
        logger = AimLogger(
            experiment=conf.exp.name,
            run_hash=conf.env.aim.run_hash,
            repo=aim.Repo(
                conf.env.aim.repo,
                init=not aim.Repo.exists(conf.env.aim.repo),
            ),
            **conf.engine.loggers.aim,
        )
        conf.env.aim.run_hash = logger.run.hash
        return logger
    if logger_type == "tensorboard":
        assert conf.env[PRIMARY_LOGGER].run_hash is not None
        return dl.TensorboardLogger(
            **conf.engine.loggers.tensorboard,
        )
    raise ValueError(f"Unknown logger type: {logger_type}")


def create_runner(conf: DictConfig) -> TRunner:
    kwargs = OmegaConf.to_container(conf, resolve=True)
    kwargs = cast(Dict[str, Any], kwargs)
    del kwargs["type"]
    runner = RUNNERS[conf.type](**kwargs)
    return runner


def configure_engine(conf: DictConfig) -> dict[str, Any]:
    engine_kwargs = OmegaConf.to_container(conf.engine, resolve=True)
    engine_kwargs = cast(Dict[str, Any], engine_kwargs)
    logger_types = [
        PRIMARY_LOGGER,
        *[k for k in conf.engine.loggers.keys() if k != PRIMARY_LOGGER],
    ]
    engine_kwargs["loggers"] = {
        logger_type: create_logger(conf, logger_type) for logger_type in logger_types
    }
    engine_kwargs["callbacks"] = [
        create_callback(cb_conf) for cb_conf in conf.engine.callbacks
    ]
    engine_kwargs["hparams"] = OmegaConf.to_container(conf, resolve=True)
    return engine_kwargs
