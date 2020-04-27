# -*- coding: utf-8 -*-

import functools
import collections

import torch
import torch.nn as nn

from models.base import HeadBase, FlattenHeadBase, PoolingHeadBase
from layers.core import Flatten
from utils.initialization import initialize_weights


class LinearClassifier(FlattenHeadBase):
    def __init__(self, input_shape: tuple, num_classes: int, dropout: float = 0.0):
        super(LinearClassifier, self).__init__(input_shape, num_classes)

        self.input_shape = input_shape
        self.num_classes = num_classes
        self.dropout = dropout
        self.layers = self.make_layers(
            num_features=functools.reduce(lambda a, b: a * b, self.input_shape),
            num_classes=self.num_classes,
            dropout=self.dropout,
        )

        initialize_weights(self.layers, activation='relu')

    @staticmethod
    def make_layers(num_features: int, num_classes: int, dropout: float = 0.0):
        layers = nn.Sequential(
            collections.OrderedDict(
                [
                    ('flatten', Flatten()),
                    ('dropout', nn.Dropout(p=dropout)),
                    ('linear', nn.Linear(num_features, num_classes))
                ]
            )
        )

        return layers

    def forward(self, x: torch.Tensor):
        return self.layers(x)

    @property
    def in_channels(self):
        return self.input_shape[0]

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.layers.parameters() if p.requires_grad)


class GAPClassifier(PoolingHeadBase):
    def __init__(self, in_channels: int, num_classes: int, dropout: float = 0.0):
        """
        Arguments:
            input_shape: list or tuple of length 3, (C, H, W).
            num_classes: int, number of target classes.
        """
        super(GAPClassifier, self).__init__(in_channels, num_classes)

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.dropout = dropout
        self.layers = self.make_layers(
            in_channels=self.in_channels,
            num_classes=self.num_classes,
            dropout=self.dropout,
        )

        initialize_weights(self.layers)

    @staticmethod
    def make_layers(in_channels: int, num_classes: int, dropout: float = 0.0):
        layers = nn.Sequential(
            collections.OrderedDict(
                [
                    ('gap', nn.AdaptiveAvgPool2d(1)),
                    ('flatten', Flatten()),
                    ('dropout', nn.Dropout(p=dropout)),
                    ('linear', nn.Linear(in_channels, num_classes))
                ]
            )
        )

        return layers

    def forward(self, x: torch.Tensor):
        return self.layers(x)

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.layers.parameters() if p.requires_grad)


class LinearProjector(LinearClassifier):
    def __init__(self, input_shape: tuple, num_features: int):
        super(LinearProjector, self).__init__(input_shape, num_features, dropout=0.)


class GAPProjector(GAPClassifier):
    def __init__(self, in_channels: int, num_features: int):
        """
        Arguments:
            input_shape: list or tuple of shape (C, H, W).
            num_features: int, number of output units.
        """
        super(GAPProjector, self).__init__(in_channels, num_features, dropout=0.)


class NonlinearProjector(PoolingHeadBase):
    def __init__(self, in_channels: int, num_features: int):
        """
        Arguments:
            input_shape: list or tuple of shape (C, H, W).
            num_features: int, number of output units.
        """
        super(NonlinearProjector, self).__init__(in_channels, num_features)

        self.in_channels = in_channels
        self.num_features = num_features
        self.layers = self.make_layers(
            in_channels=self.in_channels,
            num_features=self.num_features,
        )

    @staticmethod
    def make_layers(in_channels: int, num_features: int):
        layers = nn.Sequential(
            collections.OrderedDict(
                [
                    ('gap', nn.AdaptiveAvgPool2d(1)),
                    ('flatten', Flatten()),
                    ('linear1', nn.Linear(in_channels, in_channels)),
                    ('relu1', nn.ReLU(inplace=True)),
                    ('linear2', nn.Linear(in_channels, num_features))
                ]
            )
        )

        return layers

    def forward(self, x: torch.Tensor):
        return self.layers(x)

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.layers.parameters() if p.requires_grad)


class PatchGAPClassifier(PoolingHeadBase):
    def __init__(self, num_patches: int, in_channels: int, num_classes: int, dropout: float = 0.0):

        super(PatchGAPClassifier, self).__init__(in_channels, num_classes)
        self.num_patches = num_patches
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.dropout = dropout

        self.layers = self.make_layers(
            num_patches=self.num_patches,
            in_channels=self.in_channels,
            num_classes=self.num_classes,
            dropout=self.dropout
        )

        initialize_weights(self.layers)

    def forward(self, x: torch.Tensor):
        assert x.ndim == 5, "(B, P, C', H', W')"
        b, p, c, h, w = x.size()
        x = x.view(b, p * c, h, w)
        return self.layers(x)

    @staticmethod
    def make_layers(num_patches: int, in_channels: int, num_classes: int, dropout: float):
        layers = nn.Sequential(
            collections.OrderedDict(
                [
                    ('gap', nn.AdaptiveAvgPool2d(1)),
                    ('flatten', Flatten()),
                    ('dropout1', nn.DRopout(p=dropout)),
                    ('linear1', nn.Linear(num_patches * in_channels, in_channels)),
                    ('relu1', nn.ReLU(inplace=True)),
                    ('dropout2', nn.Dropout(p=dropout)),
                    ('linear2', nn.Linear(in_channels, num_classes))
                ]
            )
        )

        return layers

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.layers.parameters() if p.requires_grad)
