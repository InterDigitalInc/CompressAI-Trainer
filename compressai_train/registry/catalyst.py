from __future__ import annotations

import catalyst.callbacks

CALLBACKS: dict[str, type[catalyst.callbacks.Callback]] = {
    k: v for k, v in catalyst.callbacks.__dict__.items() if k[0].isupper()
}
