"""Single observed missile: medium-range retreat and LAGCFC short-range shaping."""

import math
from ...utils.utils import clip

# Self-destruct timers: official competition scenario, page 1 (AIM-9M/120B).
# Other values are provisional training preferences, not guaranteed safe envelopes.
THREAT_POLICY = {"short_lifetime_s": 60.0, "medium_lifetime_s": 80.0,
                 "release_margin_s": 3.0, "imminent_margin_s": 2.0}
WEIGHT = 0.014
# Measured at a high-speed probe condition; use these as directional priors,
# not exact platform invariants for every 4v4 initial speed.
MEDIUM_HEADON_NEZ_M = 37500.0
MEDIUM_TAIL_NEZ_M = 26500.0
MEDIUM_DISTANCE_WIDTH_M = 4500.0
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

    @staticmethod
    def _medium_nez(missile):
        # AO≈0 means the threat is in front of the aircraft; AO≈pi is a
        # rear-quarter/tail threat. Interpolate the measured head-on/tail
        # boundaries so the reward does not hard-switch at 90 degrees.
        front = max(0.0, min(1.0, (1.0+math.cos(missile["ao"]))/2.0))
        return MEDIUM_TAIL_NEZ_M+(MEDIUM_HEADON_NEZ_M-MEDIUM_TAIL_NEZ_M)*front

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
        nez = self._medium_nez(missile)
        separation = math.tanh((missile["distance"]-nez)/MEDIUM_DISTANCE_WIDTH_M)
        # The measured medium-missile tests favor full-throttle energy
        # preservation. Do not teach strong lateral-turn shaping inside NEZ.
        energy = math.tanh((frame["body_speed"]-170.0)/35.0)
        if missile["distance"] <= nez:
            return clip(.65*energy+.20*dc+.15*dr)
        return clip(.30*separation+.25*energy+.25*dc+.20*dr)
