# -*- coding: utf-8 -*-

"""
    Base classes.
"""

import torch
import torch.nn as nn


class BackboneBase(nn.Module):
    def __init__(self, layer_config: list, in_channels: int):
        super(BackboneBase, self).__init__()
        assert isinstance(layer_config, (str, list, dict))
        assert in_channels in [1, 2, 3]

    def forward(self, x):
        raise NotImplementedError

    def freeze_weights(self, to_freeze: list = []):
        """
        Freeze layers by their names. Searches through
        the hierarchy of layers to find an exact match.
        """
        for p in self.parameters():
            p.requires_grad = False

    def save_weights(self, path: str):
        """Save weights to a file with weights only."""
        torch.save(self.state_dict(), path)

    def load_weights(self, path: str):
        """Load weights from a file with weights only."""
        self.load_state_dict(torch.load(path))

    def load_weights_from_checkpoint(self, path: str, key: str):
        """
        Load weights from a checkpoint.
        Arguments:
            path: str, path to pretrained `.pt` file.
            key: str, key to retrieve the model from a state dictionary of pretrained modules.
        """
        ckpt = torch.load(path, map_location='cpu')
        self.load_state_dict(ckpt[key])
        # self.to(self.device)


class HeadBase(nn.Module):
    def __init__(self, output_size: int):
        super(HeadBase, self).__init__()
        assert isinstance(output_size, int)

    def save_weights(self, path: str):
        """Save weights to a file with weights only."""
        torch.save(self.state_dict(), path)

    def load_weights(self, path: str):
        """Load weights from a file with weights only."""
        self.load_state_dict(torch.load(path))

    def load_weights_from_checkpoint(self, path: str, key: str):
        """
        Load weights from a checkpoint.
        Arguments:
            path: str, path to pretrained `.pt` file.
            key: str, key to retrieve the model from a state dictionary of pretrained modules.
        """
        ckpt = torch.load(path, map_location='cpu')
        self.load_state_dict(ckpt[key])
        # self.to(self.device)


class ConvHeadBase(HeadBase):
    def __init__(self, input_shape: tuple, output_size: int):
        super(ConvHeadBase, self).__init__(output_size)
        assert len(input_shape) == 3, "(C, H, W)"


class FlattenHeadBase(HeadBase):
    def __init__(self, input_shape: tuple, output_size: int):
        super(FlattenHeadBase, self).__init__(output_size)
        assert len(input_shape) == 3, "(C, H, W)"


class GAPHeadBase(HeadBase):
    def __init__(self, in_channels: int, output_size: int):
        super(GAPHeadBase, self).__init__(output_size)
        assert isinstance(in_channels, int), "Number of output feature maps of backbone."


class DecoderBase(nn.Module):
    """Base class for decoders."""
    def __init__(self, layer_config: dict, input_shape: tuple, output_shape: tuple):
        super(DecoderBase, self).__init__()
        assert isinstance(layer_config, dict)
        assert len(input_shape) == 3, "(C, H, W)"
        assert len(output_shape) == 3, "(C, H, W)"

    def forward(self, x):
        raise NotImplementedError


class GeneratorBase(nn.Module):
    """Base class for generators."""
    def __init__(self, model_type: str, layer_config: list, latent_size: int, output_shape: tuple):
        super(GeneratorBase, self).__init__()
        assert model_type in ['dcgan', 'vgg']
        assert isinstance(layer_config, list)
        assert isinstance(latent_size, int)
        assert len(output_shape) == 3, "(C, H, W)"

    def forward(self, z):
        raise NotImplementedError


class DiscriminatorBase(nn.Module):
    """Base class for discriminators."""
    def __init__(self, model_type: str, layer_config: list, in_channels: int, latent_size: int):
        super(DiscriminatorBase, self).__init__()
        assert model_type in ['dcgan', 'vgg']
        assert isinstance(layer_config, list)
        assert in_channels in [1, 2]
        assert isinstance(latent_size, int)

    def forward(self, x, z):
        raise NotImplementedError
