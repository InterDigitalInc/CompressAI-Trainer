import compressai.models.waseda as waseda

from compressai_train.models.compressai.google import (
    JointAutoregressiveHierarchicalPriors,
)
from compressai_train.registry import register_model

__all__ = [
    "Cheng2020Anchor",
    "Cheng2020Attention",
]


@register_model("cheng2020-anchor")
class Cheng2020Anchor(
    JointAutoregressiveHierarchicalPriors,
    waseda.Cheng2020Anchor,
):
    """Anchor model variant from `"Learned Image Compression with
    Discretized Gaussian Mixture Likelihoods and Attention Modules"
    <https://arxiv.org/abs/2001.01568>`_, by Zhengxue Cheng, Heming Sun, Masaru
    Takeuchi, Jiro Katto.

    Uses residual blocks with small convolutions (3x3 and 1x1), and sub-pixel
    convolutions for up-sampling.

    Args:
        N (int): Number of channels
    """

    def __init__(self, N=192, **kwargs):
        waseda.Cheng2020Anchor.__init__(self, N=N, **kwargs)

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a.0.conv1.weight"].size(0)
        net = cls(N)
        net.load_state_dict(state_dict)
        return net


@register_model("cheng2020-attn")
class Cheng2020Attention(
    Cheng2020Anchor,
    waseda.Cheng2020Attention,
):
    """Self-attention model variant from `"Learned Image Compression with
    Discretized Gaussian Mixture Likelihoods and Attention Modules"
    <https://arxiv.org/abs/2001.01568>`_, by Zhengxue Cheng, Heming Sun, Masaru
    Takeuchi, Jiro Katto.

    Uses self-attention, residual blocks with small convolutions (3x3 and 1x1),
    and sub-pixel convolutions for up-sampling.

    Args:
        N (int): Number of channels
    """

    def __init__(self, N=192, **kwargs):
        waseda.Cheng2020Attention.__init__(self, N=N, **kwargs)
