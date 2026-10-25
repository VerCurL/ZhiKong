"""MLP blocks adapted from LAGCFC's MAPPO implementation."""

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, depth=2, normalize=True):
        super().__init__()
        layers = []
        width = input_dim
        for _ in range(depth):
            layers.extend([nn.Linear(width, hidden_dim), nn.ReLU()])
            if normalize:
                layers.append(nn.LayerNorm(hidden_dim))
            width = hidden_dim
        self.net = nn.Sequential(*layers)
        self.output_dim = width

    def forward(self, features):
        return self.net(features)
