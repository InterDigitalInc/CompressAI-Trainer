from pathlib import Path

from compressai.datasets.image import ImageFolder
from PIL import Image
from torch.utils.data import Dataset

from compressai_train.registry import register_dataset

ImageFolder = register_dataset("ImageFolder")(ImageFolder)


@register_dataset("Vimeo90kDataset")
class Vimeo90kDataset(Dataset):
    """Load a Vimeo-90K structured dataset.

    Vimeo-90K dataset from
    Tianfan Xue, Baian Chen, Jiajun Wu, Donglai Wei, William T. Freeman:
    `"Video Enhancement with Task-Oriented Flow"
    <https://arxiv.org/abs/1711.09078>`_,
    International Journal of Computer Vision (IJCV), 2019.

    Training and testing image samples are respectively stored in
    separate directories:

    .. code-block::

        - rootdir/
            - sequence/
                - 00001/001/im1.png
                - 00001/001/im2.png
                - 00001/001/im3.png

    Args:
        root (string): root directory of the dataset
        transform (callable, optional): a function or transform that takes in a
            PIL image and returns a transformed version
        split (string): split mode ('train' or 'valid')
        tuplet (int): order of dataset tuplet (e.g. 3 for "triplet" dataset)
    """

    def __init__(self, root, transform=None, split="train", tuplet=3):
        list_path = Path(root) / self._list_filename(split, tuplet)

        with open(list_path) as f:
            self.samples = [
                f"{root}/sequences/{line.rstrip()}/im{idx}.png"
                for line in f
                for idx in range(1, tuplet + 1)
            ]

        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
        """
        img = Image.open(self.samples[index]).convert("RGB")
        if self.transform:
            return self.transform(img)
        return img

    def __len__(self):
        return len(self.samples)

    def _list_filename(self, split: str, tuplet: int) -> str:
        tuplet_prefix = {3: "tri", 7: "sep"}[tuplet]
        list_suffix = {"train": "trainlist", "valid": "testlist"}[split]
        return f"{tuplet_prefix}_{list_suffix}.txt"
