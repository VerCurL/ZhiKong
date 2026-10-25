"""Single observed missile: medium-range retreat and LAGCFC short-range shaping."""

import math
from ...utils.utils import clip

# Self-destruct timers: official competition scenario, page 1 (AIM-9M/120B).
# Other values are provisional training preferences, not guaranteed safe envelopes.
THREAT_POLICY = {"short_lifetime_s": 60.0, "medium_lifetime_s": 80.0,
                 "release_margin_s": 3.0, "imminent_margin_s": 2.0}
WEIGHT = 0.010
MEDIUM_SAFE_DISTANCE_M = 12000.0
MEDIUM_DISTANCE_WIDTH_M = 3000.0
CLOSING_RATE_SCALE = 400.0
DISTANCE_RATE_SCALE = 400.0
SHORT_WEIGHTS = (0.8, 1.2, 0.8)
MEDIUM_WEIGHTS = (0.4, 0.2, 0.2, 0.2)


def threat_active(kind, age, distance, closing):
    """Retire old missiles before slot selection; protect imminent approaches."""
    if kind not in (0, 1) or not all(math.isfinite(v) for v in (age, distance, closing)) or age < 0:
        return False
    lifetime = THREAT_POLICY["medium_lifetime_s" if kind else "short_lifetime_s"]
    if age >= lifetime:
        return False
    if not kind or age < lifetime-THREAT_POLICY["release_margin_s"]:
        return True
    remaining = lifetime-age
    return closing > 0 and distance/closing <= remaining+THREAT_POLICY["imminent_margin_s"]


class MissileAvoidReward:
    weight = WEIGHT

    def __init__(self):
        self.reset()

    def reset(self):
        self.previous = None

    def compute(self, frame, dt):
        missile = frame["missile"]
        if not frame["alive"] or missile is None:
            self.reset()
            return 0.0
        identity = frame["missile_id"]
        before = self.previous
        dc = dr = 0.0
        if before and identity is not None and identity == before[0] and missile["kind"] == before[1]["kind"]:
            dc = math.tanh((before[1]["closing"]-missile["closing"])/(CLOSING_RATE_SCALE*dt))
            dr = math.tanh((missile["distance"]-before[1]["distance"])/(DISTANCE_RATE_SCALE*dt))
        self.previous = (identity, missile.copy())
        if missile["kind"] == 0:
            lateral = math.sin(missile["ao"])**2
            return clip(sum(w*v for w, v in zip(SHORT_WEIGHTS, (lateral, dc, dr))), -2.5, 2.5)/2.5
        back = -math.cos(missile["ao"])
        separation = math.tanh((missile["distance"]-MEDIUM_SAFE_DISTANCE_M)/MEDIUM_DISTANCE_WIDTH_M)
        return clip(sum(w*v for w, v in zip(MEDIUM_WEIGHTS, (back, separation, dc, dr))))
