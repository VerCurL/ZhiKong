"""Signed progress toward the public ammunition-dependent engagement band."""

from ...utils.utils import clip, interval_distance

WEIGHT = 0.003
PROGRESS_SPEED_MPS = 300.0
# CRM/AMRAAM should be used before the measured medium-missile NEZ, rather
# than rewarding the aircraft for closing into the old 24 km lower bound.
DISTANCE_BANDS = {"medium": (30000.0, 37040.0), "short": (4000.0, 9000.0)}


class DistanceReward:
    weight = WEIGHT

    def __init__(self):
        self.reset()

    def reset(self):
        self.signature = None
        self.distance = None

    def compute(self, frame, dt):
        kind = "medium" if frame["medium_ammo"] > 0 else "short" if frame["short_ammo"] > 0 else None
        if not frame["alive"] or not frame["enemies"] or kind is None:
            self.reset()
            return 0.0
        target = min(frame["enemies"].values(), key=lambda item: (item["distance"], item["id"]))
        band = DISTANCE_BANDS[kind]
        signature = (target["id"], kind, frame["enemy_signature"])
        now = interval_distance(target["distance"], *band)
        value = clip((self.distance-now)/(PROGRESS_SPEED_MPS*dt)) if signature == self.signature else 0.0
        self.signature, self.distance = signature, now
        return value
