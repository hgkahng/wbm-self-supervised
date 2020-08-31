# -*- coding: utf-8 -*-

import collections
import torch.nn as nn

from models.base import DecoderBase
from utils.initialization import initialize_weights

def convt3x3(in_channels: int, out_channels: int, stride: int = 1):
    convt_kws = dict(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )
    if stride == 1:
        return nn.ConvTranspose2d(**convt_kws)
    elif stride == 2:
        scale_factor = stride
        return nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=scale_factor),
            nn.ConvTranspose2d(**convt_kws)
        )
    else:
        raise NotImplementedError


def convt1x1(in_channels: int, out_channels: int, stride: int = 1):
    if stride != 1:
        raise ValueError("Only `stride`=1 is supported.")
    convt_kws = dict(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        stride=stride,
        padding=0,
        bias=False
    )
    return nn.ConvTranspose2d(**convt_kws)


def upsample1x1(in_channels: int, out_channels: int , scale_factor: int = 2):
    """Nearest neighbor-based upsampling & 1x1 kernels for transpose convolution."""
    return nn.Sequential(
        nn.UpsamplingNearest2d(scale_factor=scale_factor),
        convt1x1(in_channels, out_channels, stride=1),
        nn.BatchNorm2d(out_channels)
    )


class BasicBlockDec(nn.Module):
    expansion = 1
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, **kwargs):
        super(BasicBlockDec, self).__init__()

        self.widening_factor = wf = kwargs.get('widening_factor', 1)

        self.conv1 = convt3x3(in_channels, out_channels * wf, stride=stride)  # possible upsampling
        self.bnorm1 = nn.BatchNorm2d(out_channels * wf)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = convt3x3(out_channels * wf, out_channels, stride=1)      # equal output shape
        self.bnorm2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

        self.stride = stride
        if self.stride != 1:
            self.upsample = upsample1x1(in_channels, out_channels, scale_factor=stride)
        else:
            self.upsample = None  # FIXME

    def forward(self, x):

        if self.upsample is not None:  # stride > 1
            identity = self.upsample(x)
        else:                          # stride = 1 
            identity = x

        out = self.bnorm1(self.conv1(x))
        out = self.relu1(out)
        out = self.bnorm2(self.conv2(out))

        return self.relu2(out + identity)


class BottleNeckDec(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1, **kwargs):
        super(BottleNeckDec, self).__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError


class ResNetDecoder(DecoderBase):
    blocks = {
        'basic': BasicBlockDec,
        'bottleneck': BottleNeckDec,
    }
    def __init__(self, layer_config: dict, input_shape: tuple, output_shape: tuple):
        super(ResNetDecoder, self).__init__(layer_config, input_shape, output_shape)
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.layer_config = layer_config
        self.layers = self.make_layers(
            layer_cfg=self.layer_config,
            input_shape=self.input_shape,
            output_shape=self.output_shape,
            widening_factor=self.layer_config.get('widening_factor', 1)
        )

        initialize_weights(self.layers, activation='relu')


    def forward(self, x):
        return self.layers(x)

    @classmethod
    def make_layers(cls, layer_cfg: dict, input_shape: tuple, output_shape: tuple, **kwargs):

        in_channels = input_shape[0]
        out_channels = output_shape[0]

        layers = nn.Sequential()
        Block = cls.blocks[layer_cfg['block_type']]

        for i, v in enumerate(layer_cfg['channels']):
            stride = layer_cfg['strides'][i]
            layers.add_module(f'block{i}', Block(in_channels, v, stride=stride, **kwargs))
            in_channels = v * Block.expansion

        block_out = nn.Sequential(
            collections.OrderedDict(
                [
                    ('upsample', nn.UpsamplingNearest2d(scale_factor=2)),
                    ('out', convt1x1(in_channels, out_channels, stride=1)),
                ]
            )
        )
        layers.add_module('block_out', block_out)

        return layers

    @property
    def expansion(self):
        return self.blocks[self.layer_config['block_type']].expansion

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
        return self.input_shape[1:]

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.layers.parameters() if p.requires_grad)
