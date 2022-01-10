from __future__ import annotations

from typing import Callable

from torchvision import transforms

TRANSFORMS: dict[str, Callable[..., Callable]] = {
    k: v for k, v in transforms.__dict__.items() if k[0].isupper()
}
