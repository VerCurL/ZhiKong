"""Six reward classes, shared team outcomes and observable continuous shaping."""

import math
import numpy as np
from ...utils.utils import clip
from ...termination import outcome
from .event_driven_reward import EventDrivenReward, EVENT_NAMES, EVENT_VALUES
from .safety_reward import SafetyReward
from .distance_reward import DistanceReward
from .attack_window_reward import AttackWindowReward
from .missile_avoid_reward import MissileAvoidReward
from .energy_reward import EnergyReward

REWARD_VERSION = 4
DENSE_CLIP_PER_SECOND = .02
DENSE_NAMES = ("safety", "distance", "attack", "defense", "energy")
REWARD_METRIC_COLUMNS = (
    tuple("reward_"+n for n in ("terminal", "event", "dense", "total")) +
    tuple("reward_"+n for n in DENSE_NAMES) + tuple("signal_"+n for n in DENSE_NAMES) +
    tuple("event_"+n for n in EVENT_NAMES) +
    ("aircraft_seconds", "dense_clip_seconds", "threat_seconds", "short_threat_seconds", "medium_threat_seconds",
     "control_policy_steps", "control_center_steps", "control_boundary_steps"))
REWARD_RATIO_COLUMNS = ("policy_control_fraction", "dense_clip_fraction")


def summarize_reward_metrics(values):
    result = {key: float(values.get(key, 0)) for key in REWARD_METRIC_COLUMNS}
    steps = sum(result["control_"+name+"_steps"] for name in ("policy", "center", "boundary"))
    result["policy_control_fraction"] = result["control_policy_steps"]/steps if steps else 0.0
    result["dense_clip_fraction"] = result["dense_clip_seconds"]/result["aircraft_seconds"] if result["aircraft_seconds"] else 0.0
    return result


class MultipleCombatReward:
    required_observation_version = 6

    def __init__(self, profile, encoder):
        if encoder.schema_version != self.required_observation_version:
            raise ValueError("Reward requires the current compact observation contract")
        self.encoder, self.n = encoder, encoder.num_agents
        self.safety = SafetyReward(profile)
        self.events = EventDrivenReward(encoder.life_max)
        self.components = {side: [{"distance": DistanceReward(), "attack": AttackWindowReward(),
                                   "defense": MissileAvoidReward(), "energy": EnergyReward()}
                                  for _ in range(self.n)] for side in ("red", "blue")}
        self.last_time = None

    def reset(self, sim_time, state):
        if not math.isfinite(sim_time):
            raise ValueError("Reward reset time must be finite")
        self.last_time = float(sim_time)
        self.events.reset(state)
        for components in self.components.values():
            for parts in components:
                for component in parts.values():
                    component.reset()

    def calculate(self, frames, sim_time, state, terminal=False, control_masks=None, control_modes=None):
        if self.last_time is None:
            raise RuntimeError("Reset rewards before calculating a transition")
        dt = float(sim_time)-self.last_time
        if not math.isfinite(dt) or dt < 0 or (dt == 0 and not terminal):
            raise ValueError("Reward requires a positive simulation time step, or zero for a terminal event")
        events = self.events.compute(state, terminal)
        detail, rewards = {"dt": dt, "team": {}, "aircraft": {}, "winner": outcome(state) if terminal else None}, {}
        for side, components in self.components.items():
            metrics = dict.fromkeys(REWARD_METRIC_COLUMNS, 0.0)
            dense, aircraft = [], []
            for index, (parts, frame) in enumerate(zip(components, frames[side])):
                policy = True if control_masks is None else bool(control_masks[side][index, 0])
                mode = ("policy" if policy else "center") if control_modes is None else control_modes[side][index]
                # Control belongs to s_t, including an aircraft killed in s_(t+1).
                if mode in ("policy", "center", "boundary"):
                    metrics["control_"+mode+"_steps"] += 1
                signals = dict.fromkeys(DENSE_NAMES, 0.0)
                if not frame["alive"] or dt == 0:
                    for part in parts.values():
                        part.reset()
                    dense.append(0.0)
                    aircraft.append({"signals": signals, "dense": 0.0})
                    continue
                metrics["aircraft_seconds"] += dt
                threatened = frame["missile"] is not None
                if threatened:
                    metrics["threat_seconds"] += dt
                    metrics["medium_threat_seconds" if frame["missile"]["kind"] else "short_threat_seconds"] += dt
                signals["safety"] = self.safety.compute(frame)
                if policy:
                    signals["defense"] = parts["defense"].compute(frame, dt)
                    if threatened:
                        for name in ("distance", "attack", "energy"):
                            if name != "energy":
                                parts[name].reset()
                        signals["energy"] = parts["energy"].compute(frame, dt)
                    else:
                        for name in ("distance", "attack", "energy"):
                            signals[name] = parts[name].compute(frame, dt)
                else:
                    for part in parts.values():
                        part.reset()
                rates = {name: (self.safety.weight if name == "safety" else parts[name].weight)*signals[name]
                         for name in DENSE_NAMES}
                rates["safety"] *= -1
                rate = sum(rates.values())
                value = dt*clip(rate, -DENSE_CLIP_PER_SECOND, DENSE_CLIP_PER_SECOND)
                dense.append(value)
                aircraft.append({"signals": signals, "rates": rates, "dense": value})
                metrics["dense_clip_seconds"] += dt*float(abs(rate) > DENSE_CLIP_PER_SECOND)
                for name in DENSE_NAMES:
                    metrics["reward_"+name] += dt*rates[name]/self.n
                    metrics["signal_"+name] += dt*signals[name]/self.n
            terminal_value = events[side]["terminal"]
            event_value = sum(events[side].values())-terminal_value
            team = {"terminal": terminal_value, "event": event_value, "dense": sum(dense)/self.n,
                    "events": events[side], "metrics": metrics}
            team["total"] = team["terminal"]+team["event"]+team["dense"]
            for name in ("terminal", "event", "dense", "total"):
                metrics["reward_"+name] = team[name]
            for name in EVENT_NAMES:
                metrics["event_"+name] = events[side][name]
            detail["team"][side], detail["aircraft"][side] = team, aircraft
            rewards[side] = np.full(self.n, team["total"], dtype=np.float64)
        self.last_time = float(sim_time)
        return rewards, detail
