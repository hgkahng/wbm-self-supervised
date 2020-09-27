# -*- coding: utf-8 -*-

import math
import collections

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base import GAPHeadBase
from layers.core import Flatten
from layers.attention import CBAM
from utils.initialization import initialize_weights


class LinearHead(GAPHeadBase):
    def __init__(self, in_channels: int, num_features: int, dropout: float = 0.0):
        """
        Arguments:
            in_channels: int, number of input feature maps.
            num_features: int, number of output features.
        """
        super(LinearHead, self).__init__(in_channels, num_features)

        self.in_channels = in_channels
        self.num_features = num_features
        self.dropout = dropout
        self.layers = self.make_layers(
            in_channels=self.in_channels,
            num_features=self.num_features,
            dropout=self.dropout,
        )
        initialize_weights(self.layers)

    @staticmethod
    def make_layers(in_channels: int, num_features: int, dropout: float = 0.0):
        layers = nn.Sequential(
            collections.OrderedDict(
                [
                    ('gap', nn.AdaptiveAvgPool2d(1)),
                    ('flatten', Flatten()),
                    ('dropout', nn.Dropout(p=dropout)),
                    ('linear', nn.Linear(in_channels, num_features))
                ]
            )
        )

        return layers

    def forward(self, x: torch.Tensor):
        return self.layers(x)

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class LinearClassifier(LinearHead):
    def __init__(self, in_channels: int, num_classes: int, dropout: float = 0.):
        """
        Arguments:
            in_channels: int, number of input feature maps.
            num_classes: int, number of classes.
        """
        super(LinearClassifier, self).__init__(in_channels, num_classes, dropout)

    @property
    def num_classes(self):
        return self.num_features


class MLPHead(GAPHeadBase):
    def __init__(self, in_channels: int, num_features: int):
        """
        Arguments:
            in_channels: int, number of input feature maps.
            num_features: int, number of output units.
        """
        super(MLPHead, self).__init__(in_channels, num_features)

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


class CBAMHead(GAPHeadBase):
    def __init__(self, in_channels: int, num_features: int, input_size: tuple, **kwargs):
        super(CBAMHead, self).__init__(in_channels, num_features)
        self.in_channels = in_channels
        self.num_features = num_features
        self.input_size = input_size
        self.cbam = CBAM(self.in_channels, spatial_conv_size=3)
        self.z_linear = nn.Sequential(
            collections.OrderedDict(
                [
                    ('gap', nn.AdaptiveAvgPool2d(1)),
                    ('flatten', Flatten()),
                    ('linear', nn.Linear(in_channels, num_features)),  # 512 -> 128
                ]
            )
        )

        self.depth_linear = nn.Sequential(
            collections.OrderedDict(
                [
                    ('flatten', Flatten()),
                    ('linear', nn.Linear(in_channels, num_features)),  # for resnet18, 512 -> 128
                ]
            )
        )

        self.spatial_linear = nn.Sequential(
            collections.OrderedDict(
                [
                    ('flatten', nn.Conv2d(1, self.num_features, kernel_size=self.input_size, stride=1, padding=0)),
                    ('flatten', Flatten()),
                ]
            )
        )

    def forward(self, x: torch.FloatTensor):
        z, depth_attn, spatial_attn = self.cbam(x)
        a = self.z_linear(F.relu(z + x))
        b = self.depth_linear(depth_attn)
        c = self.spatial_linear(spatial_attn)
        return a, b, c

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SelfAttentionHead(GAPHeadBase):
    def __init__(self, in_channels: int, num_features: int):
        super(SelfAttentionHead, self).__init__(in_channels, num_features)
        self.in_channels = in_channels
        self.num_features = num_features
        conv_kwargs = {
            'kernel_size': 1,
            'stride': 1,
            'padding': 0
        }
        self.q_conv = nn.Conv2d(in_channels, in_channels//8, **conv_kwargs)
        self.k_conv = nn.Conv2d(in_channels, in_channels//8, **conv_kwargs)
        self.v_conv = nn.Conv2d(in_channels, in_channels//8, **conv_kwargs)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        self.linear()

    def forward(self, x1: torch.FloatTensor, x2: torch.FloatTensor):

        b, _, h, w = x1.size()
        a1 = self.compute_attention_scores(x1)   # (B, P, P); P=h*w
        a2 = self.compute_attention_scores(x2)   # (B, P, P)
        v1 = self.v_conv(x1).view(b, -1, h * w)  # (B, C, P)
        v2 = self.v_conv(x2).view(b, -1, h * w)  # (B, C, P)

        z1 = torch.bmm(v1, a2.permute(0, 2, 1))  # (B, C, P)
        z2 = torch.bmm(v2, a1.permute(0, 2, 1))  # (B, C, P)

        z1 = z1.view(b, z1.size(1), h, w)        # (B, C, H, W)
        z2 = z2.view(b, z2.size(1), h, w)        # (B, C, H, W)

        z1 = self.gamma * z1 + x1
        z2 = self.gamma * z2 + x2

        return (z1, z2), (a1, a2)                # (B, C, H, W), (B, C, )

    def compute_attention_scores(self, x: torch.FloatTensor):
        b, _, h, w = x.size()
        q = self.q_conv(x).view(b, -1, h * w).permute(0, 2, 1)  # (B, P, C)
        k = self.k_conv(x).view(b, -1, h * w)                   # (B, C, P)
        energy = torch.bmm(q, k)                                # (B, P, P)
        attention = self.softmax(energy)                        # (B, P, P); normalized along the last dim.
        return attention


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
