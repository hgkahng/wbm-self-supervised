# -*- coding: utf-8 -*-

import collections

import torch
import torch.nn as nn

from models.base import GeneratorBase
from models.base import DiscriminatorBase
from utils.initialization import initialize_weights


class GAN_TYPES(object):
    dcgan = {
        'kernel_size': 4,
        'padding': 1,
        'activation': nn.LeakyReLU(.2, inplace=True),
        'batch_norm': True,
    }
    vgg = {
        'kernel_size': 3,
        'padding': 1,
        'activation': nn.ReLU(inplace=True),
        'batch_norm': False,
    }

    @classmethod
    def get(cls, model_type: str):
        if model_type == 'dcgan':
            return cls.dcgan
        elif model_type == 'vgg':
            return cls.vgg
        else:
            raise NotImplementedError


class Generator(GeneratorBase):
    """Add class docstring."""
    def __init__(self, model_type, layer_config, latent_size, output_shape):
        super(Generator, self).__init__(model_type, layer_config, latent_size, output_shape)

        for k, v in GAN_TYPES.get(model_type).items():
            setattr(self, k, v)

        self.model_type = model_type
        self.layer_config = layer_config
        self.latent_size = latent_size
        self.output_shape = output_shape
        self.layers = self.make_layers(
            layer_cfg=self.layer_config,
            latent_size=self.latent_size,
            out_channels=self.output_shape[0],
            batch_norm=self.batch_norm,
            kernel_size=self.kernel_size,
            padding=self.padding,
            activation=self.activation
            )

        initialize_weights(self.layers, activation='relu')

    def forward(self, z: torch.Tensor):
        return (torch.tanh(self.layers(z)) + 1) / 2

    @staticmethod
    def make_layers(layer_cfg: list,
                    latent_size: int,
                    out_channels: int,
                    batch_norm: bool,
                    kernel_size: int,
                    padding: int,
                    activation
                    ):

        def find_channel_dimension(cfg: list):
            """Finds first integer in a nested configuration list."""
            flat_cfg = [v for block_cfg in cfg for v in block_cfg]
            for v in flat_cfg:
                if isinstance(v, int):
                    return v

        v = find_channel_dimension(layer_cfg)

        layers = nn.Sequential()
        block = nn.Sequential()
        block.add_module('conv1', nn.ConvTranspose2d(latent_size, v, kernel_size=5, stride=1))
        if batch_norm:
            block.add_module('bnorm1', nn.BatchNorm2d(v))
        block.add_module('relu1', activation)
        layers.add_module('block0', block)

        in_channels = v

        for i, block_cfg in enumerate(layer_cfg, 1):

            block = nn.Sequential()
            for j, v in enumerate(block_cfg, 1):
                stride = 2 if j == len(block_cfg) else 1
                conv = nn.ConvTranspose2d(in_channels, v, kernel_size=kernel_size, stride=stride, padding=padding)
                block.add_module(f'conv{j}', conv)
                block.add_module(f'relu{j}', activation)
                if batch_norm:
                    block.add_module(f'bnorm{j}', nn.BatchNorm2d(v))
                
                in_channels = v

            layers.add_module(f'block{i}', block)

        layers.add_module('logits', nn.Conv2d(v, out_channels, kernel_size=1, stride=1))

        return layers

    @property
    def in_channels(self):
        return self.input_shape[0]

    @property
    def input_size(self):
        return self.input_shape[1:]

    @property
    def input_shape(self):
        return (self.latent_size, 1 , 1)

    @property
    def out_channels(self):
        return self.output_shape[0]

    @property
    def output_size(self):
        return self.output_shape[1:]

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.layers.parameters() if p.requires_grad)


class Discriminator(DiscriminatorBase):
    """Add class docstring."""
    def __init__(self, model_type, layer_config, in_channels, latent_size):
        super(Discriminator, self).__init__(model_type, layer_config, in_channels, latent_size)

        for k, v in GAN_TYPES.get(model_type).items():
            setattr(self, k, v)

        self.model_type = model_type
        self.layer_config = layer_config
        self.in_channels = in_channels
        self.latent_size = latent_size

        self.conv, self.out_channels = self.make_conv_layers(
            layer_cfg=self.layer_config,
            in_channels=self.in_channels,
            batch_norm=self.batch_norm,
            kernel_size=self.kernel_size,
            padding=self.padding,
            activation=self.activation,
        )

        self.fc = nn.Sequential(
            collections.OrderedDict(
                [
                    ('conv1', nn.Conv2d(self.latent_size, self.out_channels, 1, 1)),
                    ('relu1', self.activation),
                ]
            )
        )

        self.head = nn.Sequential(
            collections.OrderedDict(
                [
                    ('linear', nn.Linear(self.out_channels * 2, self.out_channels)),
                    ('relu', self.activation),
                    ('logit', nn.Linear(self.out_channels, 1))
                ]
            )
        )

        initialize_weights(self)

    def forward(self, x: torch.Tensor, z: torch.Tensor):
        assert x.ndim == 4, "(B, C, H, W)"
        if z.ndim == 2:
            z = z.view(*z.size(), 1, 1)

        b, _, _, _ = x.size()
        o_x = self.conv(x)  # (B, C, 1, 1)
        o_z = self.fc(z)    # (B, C, 1, 1)

        return self.head(torch.cat([o_x.view(b, -1), o_z.view(b, -1)], dim=1))  # logit

    @staticmethod
    def make_conv_layers(layer_cfg, in_channels, batch_norm, kernel_size, padding, activation):
        """Convolutional layers for x."""

        layers = nn.Sequential()
        for i, block_cfg in enumerate(layer_cfg, 1):

            block = nn.Sequential()
            for j, v in enumerate(block_cfg):
                stride = 2 if j == len(block_cfg) else 1
                conv = nn.Conv2d(in_channels, v, kernel_size, stride, padding=padding)
                block.add_module(f'conv{j}', conv)
                block.add_module(f'relu{j}', activation)
                if batch_norm:
                    block.add_module(f'bnorm{j}', nn.BatchNorm2d(v))
                
                in_channels = v

            layers.add_module(f'block{i}', block)

        layers.add_module('gap', nn.AdaptiveAvgPool2d(1))

        return layers, in_channels

    @property
    def num_parameters(self):
        return sum(p.numel() for l in [self.conv, self.head, self.fc] \
            for p in l.parameters() if p.requires_grad)
