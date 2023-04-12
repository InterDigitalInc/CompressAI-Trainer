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

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from compressai.entropy_models import EntropyBottleneck
from compressai.latent_codecs import EntropyBottleneckLatentCodec
from compressai.models.base import SimpleVAECompressionModel
from compressai.registry import register_dataset, register_model
from torch import Tensor
from torch.utils.data import Dataset

thisdir = Path(__file__).parent
with open(thisdir.joinpath("SyntheticDataset.yaml")) as f:
    SYNTHETIC_DATASET_CONFIG = yaml.safe_load(f)


@register_model("fast-test-model")
class FastTestModel(SimpleVAECompressionModel):
    def __init__(self, N, M, **kwargs):
        super().__init__(**kwargs)
        K = 1
        S = 6
        self.g_a = nn.Sequential(
            nn.Conv2d(3, M, K, 2**S, K // 2),
        )
        self.g_s = nn.Sequential(
            nn.ConvTranspose2d(M, 3, K, 2**S, K // 2, 2**S - 1),
        )
        self.latent_codec = EntropyBottleneckLatentCodec(channels=M)


@register_dataset("SyntheticDataset")
class SyntheticDataset(Dataset):
    def __init__(self, image_size=(256, 256), num_samples=1024, transform=None):
        self.image_size = image_size
        self.num_samples = num_samples
        self.transform = transform

    def __getitem__(self, index):
        img = self._generate_sample(shape=(3, *self.image_size))

        if self.transform:
            return self.transform(img)

        return img

    def __len__(self):
        return self.num_samples

    def _generate_sample(self, shape) -> Tensor:
        noise_gain = 0.25 * np.random.rand()
        signal_x_gain = np.random.rand()
        signal_y_gain = np.random.rand()
        freq_x = np.abs(np.random.normal(0, 10 * np.pi))
        freq_y = np.abs(np.random.normal(0, 10 * np.pi))

        _, h, w = shape

        xx, yy = np.meshgrid(np.arange(w), np.arange(h))
        xxt = freq_x * xx[np.newaxis, :] / w
        yyt = freq_y * yy[np.newaxis, :] / h

        signal_x = 0.5 + 0.5 * np.sin(xxt)
        signal_y = 0.5 + 0.5 * np.sin(yyt)
        signal = signal_x_gain * signal_x + signal_y_gain * signal_y

        noise = noise_gain * np.random.normal(0, 1, size=(1, h, w))

        x = (signal + noise).clip(0, 1)
        x = x.astype(np.float32)
        x = np.broadcast_to(x, shape)
        x = torch.from_numpy(x)

        return x
