"""Bounded progress in observable proxy energy; independent of ground coordinates."""

import math
from ...utils.utils import clip

WEIGHT = 0.002
GRAVITY = 9.80665
# Training preferences, not a claim about the aircraft's optimal measured envelope.
REFERENCE_HEIGHT_M = 6000.0
REFERENCE_SPEED_MPS = 250.0
ENERGY_WIDTH_M = 6000.0
SCORE_RATE_SCALE = 0.1


class EnergyReward:
    weight = WEIGHT

    def __init__(self):
        self.reference = REFERENCE_HEIGHT_M+REFERENCE_SPEED_MPS**2/(2*GRAVITY)
        self.reset()

    def reset(self):
        self.previous = None

    def compute(self, frame, dt):
        if not frame["alive"]:
            self.reset()
            return 0.0
        energy = frame["height"]+frame["body_speed"]**2/(2*GRAVITY)
        score = math.exp(-((energy-self.reference)/ENERGY_WIDTH_M)**2)
        signal = 0.0 if self.previous is None else clip((score-self.previous)/(SCORE_RATE_SCALE*dt))
        self.previous = score
        return signal
