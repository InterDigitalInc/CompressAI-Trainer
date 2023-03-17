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

import re

import compressai.utils.eval_model.__main__ as eval_model_main
import torch
import torch.nn as nn
from compressai.registry import MODELS
from compressai.zoo import load_state_dict

from compressai_trainer.config import load_checkpoint as load_checkpoint_from_config
from compressai_trainer.config import load_config, state_dict_from_checkpoint


def load_checkpoint(arch: str, no_update: bool, checkpoint_path: str) -> nn.Module:
    # NOTE a bit hacky
    pattern = r"^(?P<run_root>.*/runs/(?P<run_hash>[^/]*))/checkpoints/.*$"
    m = re.match(pattern, checkpoint_path)

    if m is None:
        model = load_checkpoint_from_state_dict(arch, checkpoint_path)
    else:
        run_root = m.group("run_root")
        conf = load_config(run_root)
        model = load_checkpoint_from_config(conf)

    if not no_update:
        model.update(force=True)

    model = model.eval()

    return model


def load_checkpoint_from_state_dict(arch: str, checkpoint_path: str):
    ckpt = torch.load(checkpoint_path)
    state_dict = state_dict_from_checkpoint(ckpt)
    state_dict = load_state_dict(state_dict)  # for zoo models
    model = MODELS[arch].from_state_dict(state_dict)
    return model


def main(argv):
    eval_model_main.__dict__["load_checkpoint"] = load_checkpoint
    eval_model_main.main(argv)
