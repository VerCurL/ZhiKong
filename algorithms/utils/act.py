"""Four independent discrete heads with joint log probability."""

import torch
from torch import nn

from .distributions import categorical
from .mlp import MLP
from .action_spec import ACTION_DIMS


class ACTLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.base = MLP(input_dim, hidden_dim, depth=2)
        self.heads = nn.ModuleList(nn.Linear(hidden_dim, choices) for choices in ACTION_DIMS)
        for head in self.heads:
            nn.init.orthogonal_(head.weight, gain=0.01)
            nn.init.zeros_(head.bias)

    def _distributions(self, features, mask):
        features = self.base(features)
        pieces = torch.split(mask, ACTION_DIMS, dim=-1)
        return [categorical(head(features), valid) for head, valid in zip(self.heads, pieces)]

    def forward(self, features, mask, deterministic=False):
        dists = self._distributions(features, mask)
        actions = [dist.probs.argmax(-1) if deterministic else dist.sample() for dist in dists]
        action = torch.stack(actions, dim=-1)
        log_prob = sum(dist.log_prob(choice) for dist, choice in zip(dists, actions))
        entropy = sum(dist.entropy() for dist in dists)
        return action, log_prob.unsqueeze(-1), entropy.unsqueeze(-1)

    def evaluate_actions(self, features, actions, mask):
        dists = self._distributions(features, mask)
        log_prob = sum(dist.log_prob(actions[..., head]) for head, dist in enumerate(dists))
        entropy = sum(dist.entropy() for dist in dists)
        return log_prob.unsqueeze(-1), entropy.unsqueeze(-1)
