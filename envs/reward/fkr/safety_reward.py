"""Always-on altitude and CAS soft constraints; no map-position shaping."""

from ...utils.utils import clip

WEIGHT = 0.020
SOFT_CEILING_MARGIN_M = 500.0
HEIGHT_ERROR_M = 1000.0
# Conservative initial preferences, not a calibrated F16 stall/structural envelope.
MIN_CAS_MPS = 80.0
MAX_CAS_MPS = 350.0
SPEED_ERROR_MPS = 50.0


class SafetyReward:
    weight = WEIGHT

    def __init__(self, profile):
        self.floor = profile["limits"]["minimum_safe_altitude_m"]
        self.ceiling = profile["limits"]["maximum_altitude_m"]-SOFT_CEILING_MARGIN_M

    def compute(self, frame):
        if not frame["alive"]:
            return 0.0
        height, speed = frame["height"], frame["cas"]
        penalty = (max(0.0, self.floor-height)/HEIGHT_ERROR_M)**2
        penalty += (max(0.0, height-self.ceiling)/HEIGHT_ERROR_M)**2
        penalty += (max(0.0, MIN_CAS_MPS-speed)/SPEED_ERROR_MPS)**2
        penalty += (max(0.0, speed-MAX_CAS_MPS)/SPEED_ERROR_MPS)**2
        return clip(penalty, 0.0, 1.0)
