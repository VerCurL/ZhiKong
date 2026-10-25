"""Move toward formation slots after the official platform reset, never teleport."""

from itertools import permutations
import math
import numpy as np
from algorithms.utils.action_spec import ACTION_DIMS, ACTION_NAMES, NEUTRAL_ACTION, FLIGHT_CONTROLLER, WEAPON_HEAD, WEAPON_CHOICES
from .catalogue import FORMATION_SCHEMA


class FormationController:
    def __init__(self, profile, schema=None):
        self.profile = profile
        self.schema = schema or FORMATION_SCHEMA
        self.contexts = {}

    def clear(self):
        self.contexts.clear()

    def _position(self, plane):
        field, units = self.profile["battlefield"], self.profile["units"]
        return np.array([(float(plane["position/lat-geod-deg"])-field["center_latitude_deg"])*units["earth_meters_per_degree"],
                         (float(plane["position/long-gc-deg"])-field["center_longitude_deg"])*units["earth_meters_per_degree"]*math.cos(math.radians(field["center_latitude_deg"])),
                         float(plane["position/h-sl-ft"])*units["feet_to_meters"]])

    def reset(self, state, side, name, sim_time):
        if name not in self.schema["templates"]:
            raise ValueError(f"Unknown formation {name!r}")
        if self.schema["templates"][name] is None:
            # Use the original center rule, including its speed/altitude actions.
            self.contexts.pop(side, None)
            return
        team = state.get(side, {})
        positions = [self._position(p) for p in team.values() if all(k in p for k in
                     ("position/lat-geod-deg", "position/long-gc-deg", "position/h-sl-ft"))]
        if not positions:
            self.contexts.pop(side, None)
            return
        center = np.mean(positions, axis=0)
        distance = float(np.linalg.norm(center[:2]))
        forward = -center[:2]/distance if distance > 1e-6 else np.array([1.0, 0.0])
        right = np.array([-forward[1], forward[0]])
        starts = np.stack([self._position(team[f"{side}_{i}"]) if all(k in team.get(f"{side}_{i}", {}) for k in
                           ("position/lat-geod-deg", "position/long-gc-deg", "position/h-sl-ft")) else center.copy()
                           for i in range(4)])
        local = np.asarray(self.schema["templates"][name], dtype=np.float64)
        p = self.schema["parameters"]
        offsets = np.zeros((4, 3))
        offsets[:, :2] = (local[:, 0, None]*p["depth_m"]*forward+
                          local[:, 1, None]*p["lateral_span_m"]*right)
        offsets[:, 2] = local[:, 2]*p["height_scale_m"]
        # Assign once, minimizing crossover from the official starting line.
        order = min(permutations(range(4)), key=lambda indices: float(np.sum((starts-center-offsets[list(indices)])**2)))
        speed = np.mean([float(v.get("velocities/u-fps", 0))*self.profile["units"]["feet_to_meters"] for v in team.values()])
        self.contexts[side] = {"name": name, "time": float(sim_time), "center": center, "distance": distance,
                               "forward": forward, "offsets": offsets[list(order)], "starts": starts-center,
                               "speed": max(float(speed), 1.0), "assignment": tuple(order)}

    def targets(self, side, sim_time):
        context = self.contexts.get(side)
        if context is None:
            return None
        p = self.schema["parameters"]
        elapsed = max(0.0, float(sim_time)-context["time"])
        progress = context["speed"]*elapsed
        if progress >= max(0.0, context["distance"]-p["center_release_radius_m"]):
            # Fixed-wing aircraft cannot stop at stationary center slots. End
            # this opening maneuver and let the existing center rule take over.
            return None
        fraction = min(elapsed/p["blend_seconds"], 1.0)
        blend = fraction*fraction*(3-2*fraction)
        center = context["center"].copy()
        center[:2] += progress*context["forward"]
        return center+(1-blend)*context["starts"]+blend*context["offsets"]

    def action(self, state, side, index, sim_time):
        targets = self.targets(side, sim_time)
        if targets is None:
            return None
        context, p = self.contexts[side], self.schema["parameters"]
        plane = state[side][f"{side}_{index}"]
        error = targets[index]-self._position(plane)
        correction = p["position_gain_per_s"]*error[:2]
        norm = float(np.linalg.norm(correction))
        if norm > p["max_correction_mps"]:
            correction *= p["max_correction_mps"]/norm
        command_velocity = context["speed"]*context["forward"]+correction
        desired = math.degrees(math.atan2(command_velocity[1], command_velocity[0]))
        heading_error = (desired-float(plane.get("attitude/psi-deg", 0))+180) % 360-180
        action = np.asarray(NEUTRAL_ACTION, dtype=np.int64).copy()
        action[WEAPON_HEAD] = WEAPON_CHOICES.index("none")
        if abs(heading_error) > p["heading_deadband_deg"]:
            turns = np.asarray(FLIGHT_CONTROLLER["delta_heading_deg"])
            candidates = np.flatnonzero(turns*heading_error > 0)
            action[ACTION_NAMES.index("turn")] = int(min(candidates, key=lambda j: abs(turns[j]-heading_error)))
        speed_error = float(np.linalg.norm(command_velocity))-float(plane["velocities/u-fps"])*self.profile["units"]["feet_to_meters"]
        if abs(speed_error) > p["speed_deadband_mps"]:
            deltas = FLIGHT_CONTROLLER["delta_speed_mps"]
            action[ACTION_NAMES.index("speed")] = int(np.argmax(deltas) if speed_error > 0 else np.argmin(deltas))
        if abs(error[2]) > p["height_deadband_m"]:
            deltas = FLIGHT_CONTROLLER["delta_altitude_m"]
            action[ACTION_NAMES.index("altitude")] = int(np.argmax(deltas) if error[2] > 0 else np.argmin(deltas))
        return action
