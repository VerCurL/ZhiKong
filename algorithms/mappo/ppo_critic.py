"""Centralized recurrent value network adapted from LAGCFC MAPPO."""

from torch import nn

from algorithms.utils.gru import GRULayer
from algorithms.utils.mlp import MLP


class PPOCritic(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.base = MLP(state_dim, hidden_dim)
        self.rnn = GRULayer(hidden_dim, hidden_dim)
        self.value_base = MLP(hidden_dim, hidden_dim)
        self.value_out = nn.Linear(hidden_dim, 1)

    def forward(self, state, hidden, reset_mask):
        features = self.base(state)
        recurrent, next_hidden = self.rnn(features, hidden, reset_mask)
        return self.value_out(self.value_base(recurrent)), next_hidden
