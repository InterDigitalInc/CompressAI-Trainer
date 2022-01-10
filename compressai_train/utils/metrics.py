from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from pytorch_msssim import ms_ssim


def compute_metrics(x: torch.Tensor, x_hat: torch.Tensor, metrics: list[str]):
    return {metric: _METRICS[metric](x, x_hat) for metric in metrics}


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = F.mse_loss(a, b).item()
    return -10 * math.log10(mse)


def msssim(a: torch.Tensor, b: torch.Tensor) -> float:
    return ms_ssim(a, b, data_range=1.0).item()


_METRICS = {
    "psnr": psnr,
    "msssim": msssim,
    "ms-ssim": msssim,
    "ms_ssim": msssim,
}
