from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def preprocess_img(img: np.ndarray) -> torch.Tensor:
    x = (img.transpose(2, 0, 1) / 255).astype(np.float32)
    x = torch.from_numpy(x).unsqueeze(0)
    return x


def postprocess_img(x_hat: torch.Tensor) -> np.ndarray:
    x = x_hat.squeeze(0).numpy()
    x = (x.transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
    return x


@torch.no_grad()
def inference_single_image_uint8(
    model, img: np.ndarray, device=None
) -> tuple[np.ndarray, list[bytes]]:
    """Run inference on a single HWC uint8 RGB image."""
    x = preprocess_img(img)
    x = x.to(device=device)
    result = inference(model, x, skip_decompress=False)
    x_hat = result["out_dec"]["x_hat"].cpu()
    img_rec = postprocess_img(x_hat)
    encoded = [s[0] for s in result["out_enc"]["strings"]]
    return img_rec, encoded


@torch.no_grad()
def inference(model, x: torch.Tensor, skip_decompress: bool = False) -> dict[str, Any]:
    """Run compression model on image batch."""
    n, _, h, w = x.shape
    pad, unpad = _get_pad(h, w)

    x_padded = F.pad(x, pad, mode="constant", value=0)
    out_net = model(x_padded)
    out_net["x_hat"] = F.pad(out_net["x_hat"], unpad)
    out_enc = model.compress(x_padded)
    if skip_decompress:
        out_dec = dict(out_net)
        del out_dec["likelihoods"]
    else:
        out_dec = model.decompress(out_enc["strings"], out_enc["shape"])
    out_dec["x_hat"] = F.pad(out_dec["x_hat"], unpad)

    num_pixels = n * h * w
    num_bits = sum(sum(map(len, s)) for s in out_enc["strings"]) * 8.0
    bpp = num_bits / num_pixels

    return {
        "out_net": out_net,
        "out_enc": out_enc,
        "out_dec": out_dec,
        "bpp": bpp,
    }


def _get_pad(h, w):
    p = 64  # maximum 6 strides of 2

    new_h = (h + p - 1) // p * p
    new_w = (w + p - 1) // p * p

    left = (new_w - w) // 2
    right = new_w - w - left
    top = (new_h - h) // 2
    bottom = new_h - h - top

    pad = (left, right, top, bottom)
    unpad = (-left, -right, -top, -bottom)

    return pad, unpad
