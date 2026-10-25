"""Compatibility statistics; LAGCFC actor inputs use schema scaling directly."""

import numpy as np


class RunningMoments:
    def __init__(self, size):
        self.mean = np.zeros(size, dtype=np.float64)
        self.var = np.ones(size, dtype=np.float64)
        self.count = 1e-4

    def update(self, samples):
        samples = np.asarray(samples, dtype=np.float64).reshape(-1, self.mean.size)
        if samples.size == 0:
            return
        n = samples.shape[0]
        sample_mean = samples.mean(0)
        sample_var = samples.var(0)
        delta = sample_mean-self.mean
        total = self.count+n
        self.mean += delta*n/total
        self.var = (self.var*self.count + sample_var*n + delta**2*self.count*n/total)/total
        self.count = total

    def normalize(self, samples):
        # The LAGCFC contract already scales every feature in the observation
        # encoder and does not apply an additional running feature transform.
        return np.asarray(samples, dtype=np.float32)

    def merge(self, other):
        """Commit observations collected while the behavior normalizer was frozen."""
        delta = other.mean-self.mean
        total = self.count+other.count
        self.var = (self.var*self.count + other.var*other.count +
                    delta**2*self.count*other.count/total)/total
        self.mean = self.mean+delta*other.count/total
        self.count = total

    def state_dict(self):
        return {"mean": self.mean.copy(), "var": self.var.copy(), "count": self.count}

    def load_state_dict(self, state):
        self.mean = np.asarray(state["mean"], dtype=np.float64)
        self.var = np.asarray(state["var"], dtype=np.float64)
        self.count = float(state["count"])
