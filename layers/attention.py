# -*- coding: utf-8 -*-

import math

import torch
import torch.nn as nn

from utils.initialization import initialize_weights


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
