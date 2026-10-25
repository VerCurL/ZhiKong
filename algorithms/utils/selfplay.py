"""Frozen historical actors, following LAGCFC's FSP opponent pool."""

from copy import deepcopy
import random


class HistoricalActorPool:
    def __init__(self, history_limit, latest_probability, seed):
        self.history_limit = int(history_limit)
        self.latest_probability = float(latest_probability)
        self.rng = random.Random(seed)
        self.history = []

    def add(self, actor, normalizer):
        self.history.append({
            "actor": {key: tensor.detach().cpu().clone() for key, tensor in actor.state_dict().items()},
            "normalizer": normalizer.state_dict(),
        })
        self.history = self.history[-self.history_limit:]

    def sample_into(self, target_actor, target_normalizer, latest_actor, latest_normalizer):
        if not self.history or self.rng.random() < self.latest_probability:
            target_actor.load_state_dict(latest_actor.state_dict())
            target_normalizer.load_state_dict(latest_normalizer.state_dict())
            return "latest"
        index = self.rng.randrange(len(self.history))
        target_actor.load_state_dict(self.history[index]["actor"])
        target_normalizer.load_state_dict(self.history[index]["normalizer"])
        return f"history_{index}"

    def state_dict(self):
        return {"history": deepcopy(self.history), "rng": self.rng.getstate()}

    def load_state_dict(self, state):
        self.history = state["history"]
        self.rng.setstate(state["rng"])
