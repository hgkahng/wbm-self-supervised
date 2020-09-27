# -*- coding: utf-8 -*-

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.core import Flatten
from utils.initialization import initialize_weights


class BasicConv(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 groups: int = 1,
                 relu: bool = True,
                 bn: bool = True,
                 bias: bool = False):
        super(BasicConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.conv = nn.Conv2d(in_channels=self.in_channels,
                              out_channels=self.out_channels,
                              kernel_size=self.kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              groups=groups,
                              bias=bias)

        self.bn = nn.BatchNorm2d(self.out_channels,
                                 eps=1e-5,
                                 momentum=0.01,
                                 affine=True) if bn else None

        self.relu = nn.ReLU if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)

        return x


class ChannelGate(nn.Module):
    def __init__(self,
                 gate_channels: int,
                 reduction_ratio: int = 16,
                 pool_types: list = ['avg', 'max']):
        super(ChannelGate, self).__init__()

        self.gate_channels = gate_channels
        self.hidden_channels = gate_channels // reduction_ratio
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(self.gate_channels, self.hidden_channels),
            nn.ReLU(),
            nn.Linear(self.hidden_channels, self.gate_channels)
        )
        self.pool_types = pool_types

    def forward(self, x: torch.FloatTensor, return_attention: bool = False):
        b, c, h, w = x.size()

        channel_attn_sum = None

        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d(x, (h, w), stride=(h, w))
                channel_attn_raw = self.mlp(avg_pool)

            elif pool_type=='max':
                max_pool = F.max_pool2d(x, (h, w), stride=(h, w))
                channel_attn_raw = self.mlp(max_pool)

            if channel_attn_sum is None:
                channel_attn_sum = channel_attn_raw
            else:
                channel_attn_sum = channel_attn_sum + channel_attn_raw

        attention = F.sigmoid(channel_attn_sum)  # (B, C)
        x_out = x * attention.view(b, c, 1, 1).expand_as(x)

        if return_attention:
            return x_out, attention
        else:
            return x_out


class ChannelPool(nn.Module):
    def __init__(self):
        super(ChannelPool, self).__init__()

    def forward(self, x: torch.FloatTensor):
        max_along_depth = torch.max(x, 1)[0].unsqueeze(1)
        avg_along_depth = torch.mean(x, 1).unsqueeze(1)
        return torch.cat([max_along_depth, avg_along_depth], dim=1)


class SpatialGate(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super(SpatialGate, self).__init__()

        self.kernel_size = kernel_size
        self.padding = (self.kernel_size - 1) // 2   # 7 -> 3 or 3 -> 1
        self.compress = ChannelPool()
        self.spatial = BasicConv(in_channels=2,
                                 out_channels=1,
                                 kernel_size=self.kernel_size,
                                 stride=1,
                                 padding=self.padding,
                                 relu=False)

    def forward(self, x: torch.FloatTensor, return_attention: bool = False):
        attention = self.spatial(self.compress(x))
        attention = F.sigmoid(attention)
        x_out = x * attention
        if return_attention:
            return x_out, attention
        else:
            return x_out


class CBAM(nn.Module):
    def __init__(self,
                 gate_channels: int,
                 reduction_ratio: int = 16,
                 pool_types: list = ['avg', 'max'],
                 spatial_conv_size: int = 7):
        super(CBAM, self).__init__()

        self.gate_channels = gate_channels
        self.reduction_ratio = reduction_ratio
        self.spatial_conv_size = spatial_conv_size
        self.depth_gate = ChannelGate(
            gate_channels=self.gate_channels,
            reduction_ratio=reduction_ratio,
            pool_types=pool_types
        )
        self.spatial_gate = SpatialGate(kernel_size=self.spatial_conv_size)

    def forward(self, x):
        z, depth_attn = self.depth_gate(x, return_attention=True)
        z, spatial_attn = self.spatial_gate(z, return_attention=True)
        return z, depth_attn, spatial_attn


class BatchQKVAttention(nn.Module):
    def __init__(self, in_features: int, out_features: int, temperature : float = None):
        super(BatchQKVAttention, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.temperature = math.sqrt(out_features) if temperature is None else temperature

        self.query_proj = nn.Linear(in_features, out_features, bias=True)
        self.key_proj   = nn.Linear(in_features, out_features, bias=True)
        self.value_proj = nn.Linear(in_features, out_features, bias=True)

        self.softmax = nn.Softmax(dim=-1)

        # self.layer_norm = nn.LayerNorm(out_features)
        # initialize_weights(self)

    def forward(self, x: torch.Tensor):
        """
        Arguments:
            x: 3d tensor of shape (B, N, F), where N is the number of different views.
        Returns:
            V: 3d tensor of shape (B, N, F).
            attention: 2d tensor of shape (N x B, N x B).
        """
        if x.ndim != 3:
            raise ValueError(f"Expecting 3d input for `x`. Received {x.ndim}d.")
        b, n, _ = x.size()
        x = x.view(b * n, -1)          # (NB, F)

        Q = self.query_proj(x)         # (NB, F)
        K = self.key_proj(x)           # (NB, F)
        V = self.value_proj(x)         # (NB, F)
        energy = torch.matmul(Q, K.T)  # (NB, NB)
        energy.div_(self.temperature)
        attention = self.softmax(energy)
        # V = x + torch.matmul(attention, V)
        # V = self.layer_norm(V)

        return V.view(b, n, -1), attention  # (B, N, F), (NxB, NxB)

    @property
    def num_parameters(self):
        n = 0
        for layer in [self.query_proj, self.key_proj, self.value_proj]:
            for p in layer.parameters():
                if p.requires_grad:
                    n += p.numel()
        return n
