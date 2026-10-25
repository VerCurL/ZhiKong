"""Recurrent layer with explicit time-major training sequences."""

import torch
from torch import nn


class GRULayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(self, features, hidden, reset_mask):
        single = features.ndim == 2
        if single:
            hidden = hidden * reset_mask
            output, next_hidden = self.gru(features.unsqueeze(0), hidden.unsqueeze(0))
            return self.norm(output.squeeze(0)), next_hidden.squeeze(0)
        if hidden.ndim == 2:
            hidden = hidden.unsqueeze(0)
        outputs = []
        for features_t, mask_t in zip(features, reset_mask):
            hidden = hidden * mask_t.reshape(1, -1, 1)
            output, hidden = self.gru(features_t.unsqueeze(0), hidden)
            outputs.append(self.norm(output.squeeze(0)))
        result = torch.stack(outputs)
        return (result.squeeze(0) if single else result), hidden.squeeze(0)
