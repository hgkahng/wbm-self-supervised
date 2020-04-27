# -*- coding: utf-8 -*-

import torch.nn as nn

from models.base import DecoderBase
from utils.initialization import initialize_weights


class VGGDecoder(DecoderBase):

    def __init__(self, layer_config, input_shape, output_shape, batch_norm: bool = True):
        super(VGGDecoder, self).__init__(layer_config, input_shape, output_shape)

        self.layer_config = layer_config
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.batch_norm = batch_norm
        self.layers = self.make_layers(
            cfg=self.layer_config,
            in_channels=self.input_shape[0],
            out_channels=self.output_shape[0],
            batch_norm=self.batch_norm
        )

        initialize_weights(self.layers, activation='relu')

    def forward(self, x):
        return self.layers(x)

    @staticmethod
    def make_layers(cfg, in_channels, out_channels, batch_norm: bool = True):
        layers = nn.Sequential()
        for i, block_cfg in enumerate(cfg, 1):
            block = nn.Sequential()
            p_idx = c_idx = 1
            for _, v in enumerate(block_cfg, 1):
                if v == 'U':
                    block.add_module(f'upsample{p_idx}', nn.Upsample(scale_factor=(2, 2), mode='nearest'))
                    p_idx += 1
                else:
                    conv2d = nn.ConvTranspose2d(in_channels, v, kernel_size=3, padding=1, bias=not batch_norm)
                    block.add_module(f'conv{c_idx}', conv2d)
                    if batch_norm:
                        block.add_module(f'bnorm{c_idx}', nn.BatchNorm2d(v))
                    block.add_module(f'relu{c_idx}', nn.ReLU(inplace=False))
                    in_channels = v
                    c_idx += 1
            layers.add_module(f'block{i}', block)

        logits = nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0)
        layers.add_module('logits', logits)

        return layers

    @property
    def in_channels(self):
        return self.input_shape[0]

    @property
    def input_size(self):
        return self.input_shape[1:]

    @property
    def out_channels(self):
        return self.output_shape[0]

    @property
    def output_size(self):
        return self.output_shape[1:]

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.layers.parameters() if p.requires_grad)
