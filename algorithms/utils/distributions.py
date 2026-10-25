"""Masked categorical action heads."""

import torch


def categorical(logits, mask):
    if logits.shape != mask.shape:
        raise ValueError("Action mask shape does not match logits")
    if not torch.all(mask.any(dim=-1)):
        raise ValueError("Every action head needs at least one legal action")
    return torch.distributions.Categorical(logits=logits.masked_fill(~mask.bool(), -1e9))
