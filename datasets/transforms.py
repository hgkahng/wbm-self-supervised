# -*- coding: utf-8 -*-

import torch

from torchvision.transforms import Compose
from torchvision.transforms import Resize, ToTensor, Lambda
from torchvision.transforms import RandomRotation, RandomHorizontalFlip, RandomVerticalFlip
from PIL import Image


class BasicTransform(object):
    """Add class docstring."""
    @classmethod
    def get(cls, size: tuple):
        transforms = [
            Resize(size=size, interpolation=Image.NEAREST),
            ToTensor(),
            Lambda(lambda x: torch.ceil(x * 2)),
        ]

        return Compose(transforms)


class RotationTransform(object):
    """Add class docstring."""
    @classmethod
    def get(cls, size: tuple):
        transforms = [
            Resize(size=size, interpolation=Image.NEAREST),
            RandomRotation(360, fill=(0, )),
            RandomHorizontalFlip(.5),
            RandomVerticalFlip(.5),
            ToTensor(),
            Lambda(lambda x: torch.ceil(x * 2)),
        ]

        return Compose(transforms)
