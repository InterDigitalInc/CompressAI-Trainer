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

import os
from types import ModuleType
from typing import Any, Mapping

import compressai
from omegaconf import DictConfig, OmegaConf

import compressai_train
from compressai_train.utils import git


def write_outputs(conf: DictConfig):
    write_config(conf)
    write_git_diff(conf, compressai_train)
    write_git_diff(conf, compressai)


def write_config(conf: DictConfig):
    logdir = conf.misc.config_logdir
    filename = "config.yaml"
    s = OmegaConf.to_yaml(conf, resolve=False)
    os.makedirs(logdir, exist_ok=True)
    with open(os.path.join(logdir, filename), "w") as f:
        f.write(s)


def write_git_diff(conf: Mapping[str, Any], package: ModuleType) -> str:
    src_root = conf["paths"]["src"]
    diff_path = os.path.join(src_root, f"{package.__name__}.patch")
    os.makedirs(src_root, exist_ok=True)
    with open(diff_path, "w") as f:
        f.write(git.diff(root=package.__path__[0]))
    return diff_path
