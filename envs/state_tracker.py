"""Short lived per-aircraft state used by observation and control adapters."""

from collections import defaultdict


class StateTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.last_seen = {}
        self.last_alive = {}
        self.last_launch = {}
        self.targets = {}
        self.previous = None
        self.step = 0

    def update(self, state, sim_time):
        if self.previous is not None:
            for side in ("red", "blue"):
                for name, plane in state.get(side, {}).items():
                    old = self.previous.get(side, {}).get(name, {})
                    old_am = int(old.get("AMRAAMCurrentNum", 0))
                    old_sr = int(old.get("SRAAMCurrentNum", 0))
                    if int(plane.get("AMRAAMCurrentNum", 0)) < old_am or int(plane.get("SRAAMCurrentNum", 0)) < old_sr:
                        self.last_launch[name] = sim_time
        self.previous = state
        self.step += 1

    def mark_seen(self, observer, target, sim_time):
        self.last_seen[(observer, target)] = sim_time

    def mark_alive(self, observer, target, living):
        self.last_alive[(observer, target)] = bool(living)

    def target_alive(self, observer, target):
        return self.last_alive.get((observer, target), False)

    def track_age(self, observer, target, sim_time):
        last = self.last_seen.get((observer, target))
        return 60.0 if last is None else max(0.0, min(60.0, sim_time - last))

    def launch_age(self, name, sim_time):
        last = self.last_launch.get(name)
        return 80.0 if last is None else max(0.0, min(80.0, sim_time - last))
