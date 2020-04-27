# -*- coding: utf-8 -*-

import torch
from PIL import Image
from torchvision.utils import make_grid


def save_image_dpi(tensor, fp, nrow=8, padding=2,
                   normalize=False, range=None, scale_each=False, pad_value=0, format=None, dpi=(500, 500)):
    """A modified version of `torchvision.utils.save_image`."""
    grid = make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
                        normalize=normalize, range=range, scale_each=scale_each)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    if isinstance(dpi, int):
        dpi = (dpi, dpi)
    im.save(fp, format=format, dpi=dpi)
