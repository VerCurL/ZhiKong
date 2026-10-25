"""FKR angular/range score, two softmax pools, timed EMA and bounded trend."""

import math
from ...utils.utils import clip, ema, sigmoid, softmax_score
from .distance_reward import DISTANCE_BANDS

WEIGHT = 0.005
TYPE_SETTINGS = {"short": {"soft_width_m": 1000.0, "weights": (.45, .30, .25), "ao_sigma_rad": math.radians(35)},
                 "medium": {"soft_width_m": 2500.0, "weights": (.62, .08, .30), "ao_sigma_rad": math.radians(17.5)}}
DISTANCE_SIGMA_FRACTION = .35
TA_SIGMA_RAD = math.radians(120)
CRM_LOCK_DISTANCE_M = 37040.0
CRM_LOCK_HALF_ANGLE_RAD = math.radians(35.0)
CRM_LOCK_SOFT_EDGE_RAD = math.radians(3.0)
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
            for kind in kinds:
                band = TYPE_SETTINGS[kind]
                angular = (math.exp(-(target["ao"]/band["ao_sigma_rad"])**2),
                           math.exp(-(target["ta"]/TA_SIGMA_RAD)**2))
                center, sigma, maximum = self.windows[kind]
                terms = (*angular, math.exp(-((target["distance"]-center)/sigma)**2))
                gate = sigmoid((maximum-target["distance"])/band["soft_width_m"])
                score = gate*sum(w*x for w, x in zip(band["weights"], terms))
                if kind == "medium":
                    # The platform's CRM lock is a 37.04 km, 35 degree
                    # 3-D cone with a finite establishment time. The raw
                    # lock flag is a stronger signal than a guessed geometry.
                    cone_gate = sigmoid((CRM_LOCK_HALF_ANGLE_RAD-target["ao"])/CRM_LOCK_SOFT_EDGE_RAD)
                    if target.get("stable_radar_lock", False):
                        lock_factor = 1.0
                    elif target.get("radar_lock", False):
                        lock_factor = .35
                    elif target.get("in_attack_range", False):
                        lock_factor = .15
                    else:
                        lock_factor = .05
                    score *= cone_gate*lock_factor
                types.append(score)
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
