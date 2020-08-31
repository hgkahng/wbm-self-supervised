# -*- coding: utf-8 -*-

import collections
import torch
import torch.nn as nn
from layers.core import Flatten


class CNNWDI(nn.Module):
    def __init__(self, in_channels: int = 2, out_channels: int = 128):
        super(CNNWDI, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.layers = nn.Sequential(
            collections.OrderedDict(
                [
                    ('C1', self.conv_bn_relu(2, 16, 3, 1, 0)),
                    ('P1', nn.MaxPool2d(2, 2)),
                    ('C2', self.conv_bn_relu(16, 16, 3, 1, 1)),
                    ('C3', self.conv_bn_relu(16, 32, 3, 1, 1)),
                    ('P2', nn.MaxPool2d(2, 2)),
                    ('C4', self.conv_bn_relu(32, 32, 3, 1, 1)),
                    ('C5', self.conv_bn_relu(32, 64, 3, 1, 1)),
                    ('P3', nn.MaxPool2d(2, 2)),
                    ('C6', self.conv_bn_relu(64, 64, 3, 1, 1)),
                    ('C7', self.conv_bn_relu(64, 128, 3, 1, 1)),
                    ('P4', nn.MaxPool2d(2, 2)),
                    ('C8', self.conv_bn_relu(128, self.out_channels, 3, 1, 1)),
                    ('DO', nn.Dropout2d(.2)),
                    ('P5', nn.MaxPool2d(2, 2)),
                ]
            )
        )

    def forward(self, x):
        return self.layers(x)

    @staticmethod
    def conv_bn_relu(in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
