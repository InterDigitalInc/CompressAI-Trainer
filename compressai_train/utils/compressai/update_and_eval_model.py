import compressai.utils.eval_model.__main__ as eval_model_main
import torch
import torch.nn as nn
from compressai.zoo import load_state_dict
from compressai.zoo.image import model_architectures as architectures


def load_checkpoint(arch: str, checkpoint_path: str) -> nn.Module:
    ckpt = torch.load(checkpoint_path)
    state_dict = (
        ckpt["model_state_dict"]
        if "model_state_dict" in ckpt
        else ckpt["state_dict"]
        if "state_dict" in ckpt
        else ckpt
    )
    state_dict = load_state_dict(state_dict)
    model = architectures[arch].from_state_dict(state_dict).eval()
    model.update(force=True)
    return model


def main(argv):
    eval_model_main.__dict__["load_checkpoint"] = load_checkpoint
    eval_model_main.main(argv)
