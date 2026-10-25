"""Shared recurrent Actor adapted from LAGCFC MAPPO."""

import torch
from torch import nn

from algorithms.utils.act import ACTLayer
from algorithms.utils.gru import GRULayer
from algorithms.utils.mlp import MLP


class PPOActor(nn.Module):
    def __init__(self, obs_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.base = MLP(obs_dim, hidden_dim)
        self.rnn = GRULayer(hidden_dim, hidden_dim)
        self.action = ACTLayer(hidden_dim, hidden_dim)

    def forward(self, obs, hidden, reset_mask, action_mask, deterministic=False):
        features = self.base(obs)
        recurrent, next_hidden = self.rnn(features, hidden, reset_mask)
        actions, log_prob, entropy = self.action(recurrent, action_mask, deterministic)
        return actions, log_prob, entropy, next_hidden

    def evaluate_actions(self, obs, hidden, reset_mask, action_mask, actions):
        features = self.base(obs)
        recurrent, _ = self.rnn(features, hidden, reset_mask)
        return self.action.evaluate_actions(recurrent, actions, action_mask)
