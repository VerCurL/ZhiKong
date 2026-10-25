"""FKR angular/range score, two softmax pools, timed EMA and bounded trend."""

import math
from ...utils.utils import clip, ema, sigmoid, softmax_score
from .distance_reward import DISTANCE_BANDS

WEIGHT = 0.005
TYPE_SETTINGS = {"short": {"soft_width_m": 1000.0, "weights": (.45, .30, .25)},
                 "medium": {"soft_width_m": 3000.0, "weights": (.55, .10, .35)}}
DISTANCE_SIGMA_FRACTION = .35
AO_SIGMA_RAD = math.radians(35)
TA_SIGMA_RAD = math.radians(120)
TYPE_BETA = TARGET_BETA = 4.0
EMA_TAU_SECONDS = .9
ABSOLUTE_WEIGHT, TREND_WEIGHT, TREND_RATE_SCALE = .2, .8, .25


class AttackWindowReward:
    weight = WEIGHT

    def __init__(self):
        self.windows = {kind: ((lo+hi)/2, DISTANCE_SIGMA_FRACTION*(hi-lo), hi)
                        for kind, (lo, hi) in DISTANCE_BANDS.items()}
        self.reset()

    def reset(self):
        self.signature = None
        self.score = None

    def compute(self, frame, dt):
        kinds = [kind for kind in ("short", "medium") if frame[kind+"_ammo"] > 0]
        if not frame["alive"] or not kinds or not frame["enemies"]:
            self.reset()
            return 0.0
        scores = []
        for target in frame["enemies"].values():
            types = []
            angular = (math.exp(-(target["ao"]/AO_SIGMA_RAD)**2), math.exp(-(target["ta"]/TA_SIGMA_RAD)**2))
            for kind in kinds:
                band = TYPE_SETTINGS[kind]
                center, sigma, maximum = self.windows[kind]
                terms = (*angular, math.exp(-((target["distance"]-center)/sigma)**2))
                gate = sigmoid((maximum-target["distance"])/band["soft_width_m"])
                types.append(gate*sum(w*x for w, x in zip(band["weights"], terms)))
            scores.append(softmax_score(types, TYPE_BETA))
        current = softmax_score(scores, TARGET_BETA)
        signature = (frame["enemy_signature"], tuple(kinds))
        trend = 0.0
        if self.score is not None and signature == self.signature:
            smoothed = ema(self.score, current, dt, EMA_TAU_SECONDS)
            trend = math.tanh((smoothed-self.score)/(TREND_RATE_SCALE*dt))
        else:
            smoothed = current
        self.score, self.signature = smoothed, signature
        return clip(ABSOLUTE_WEIGHT*smoothed+TREND_WEIGHT*trend)
