# -*- coding: utf-8 -*-

import math
import functools
import collections

import torch
import torch.nn as nn

# from entmax import sparsemax

from models.base import FlattenHeadBase, GAPHeadBase
from layers.core import Flatten
from utils.initialization import initialize_weights


class LinearClassifier(FlattenHeadBase):
    def __init__(self, input_shape: tuple, num_classes: int, dropout: float = 0.0):
        super(LinearClassifier, self).__init__(input_shape, num_classes)

        self.input_shape = input_shape  # (C, H, W)
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


class GAPClassifier(GAPHeadBase):
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

    @property
    def num_features(self):
        return self.num_classes


class NonlinearProjector(GAPHeadBase):
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
                    ('linear1', nn.Linear(in_channels, in_channels, bias=False)),
                    ('bnorm1', nn.BatchNorm1d(in_channels)),
                    ('relu1', nn.ReLU(inplace=True)),
                    ('linear2', nn.Linear(in_channels, num_features, bias=True))
                ]
            )
        )

        return layers

    def forward(self, x: torch.Tensor):
        return self.layers(x)

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.layers.parameters() if p.requires_grad)


class AttentionProjector(GAPHeadBase):
    def __init__(self, in_channels: int, num_features: int, dropout: float = 0.1, temperature: float = None):
        super(AttentionProjector, self).__init__(in_channels, num_features)

        self.in_channels = in_channels
        self.num_features = num_features
        self.temperature = temperature if isinstance(temperature, float) else math.sqrt(num_features)

        self.nonlinear = nn.Sequential(
            collections.OrderedDict(
                [
                    ('gap', nn.AdaptiveAvgPool2d(1)),
                    ('flatten', Flatten()),
                    ('linear', nn.Linear(in_channels, num_features)),
                    ('relu', nn.ReLU(inplace=True))
                ]
            )
        )

        self.q_linear = nn.Linear(num_features, num_features, bias=False)
        self.k_linear = nn.Linear(num_features, num_features, bias=False)
        self.v_linear = nn.Linear(num_features, num_features, bias=True)

        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(dropout) if isinstance(dropout, float) else None

        self.initialize_weights()

    def forward(self, x: torch.Tensor):

        if x.ndim != 4:
            raise ValueError(f"Expecting 4d input of shape (num_views x B, C, H, W).")

        h = self.nonlinear(x)          # (NxB,   F)

        Q = self.q_linear(h)           # (NxB,   F)
        K = self.k_linear(h)           # (NxB,   F)
        V = self.v_linear(h)           # (NxB,   F)

        energy = torch.matmul(Q, K.T)  # (NxB, NxB)
        energy.div_(self.temperature)
        attention = self.softmax(energy)

        if self.dropout is not None:
            attention = self.dropout(attention)

        out = torch.matmul(attention, V)
        out = nn.functional.relu(out)

        return out, attention         # (NxB, F), (NxB, NxB)

    @property
    def num_parameters(self):
        n = 0
        for layer in [self.q_linear, self.k_linear, self.v_linear]:
            for p in layer.parameters():
                if p.requires_grad:
                    n += p.numel()
        return n

    def initialize_weights(self):
        for _, m in self.named_modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 1)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 1)
            else:
                pass


class PatchGAPClassifier(GAPHeadBase):
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
                    ('dropout1', nn.Dropout(p=dropout)),
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
