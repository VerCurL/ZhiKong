"""Shared training/deployment routing: boundary, contact, missile, then navigation."""

import math
import numpy as np
from algorithms.utils.action_spec import ACTION_DIMS, ACTION_NAMES, NEUTRAL_ACTION, FLIGHT_CONTROLLER, WEAPON_HEAD, WEAPON_CHOICES
from .reward.fkr.missile_avoid_reward import THREAT_POLICY
from .formations import FormationController

CONTROL_SCHEMA = {"version": 1, "priority": ["boundary", "enemy", "missile", "center"],
                  "boundary_margin_m": 0.0, "navigation_weapon": "none"}
EXECUTION_SCHEMA = {**CONTROL_SCHEMA, "threat_policy": dict(THREAT_POLICY)}


class ControlRules:
    def __init__(self, profile, formation_schema=None):
        self.profile = profile
        self.formation = FormationController(profile, formation_schema)

    def reset(self, state, formations, sim_time):
        self.formation.clear()
        for side, name in formations.items():
            self.formation.reset(state, side, name, sim_time)

    def _outside(self, plane):
        field = self.profile["battlefield"]
        scale = self.profile["units"]["earth_meters_per_degree"]
        north = (float(plane["position/lat-geod-deg"])-field["center_latitude_deg"])*scale
        east = (float(plane["position/long-gc-deg"])-field["center_longitude_deg"])*scale*math.cos(math.radians(field["center_latitude_deg"]))
        return (bool(plane.get("IsOutOfValidBattleArea", 0)) or
                abs(north) >= field["height_m"]/2-CONTROL_SCHEMA["boundary_margin_m"] or
                abs(east) >= field["width_m"]/2-CONTROL_SCHEMA["boundary_margin_m"])

    def _center_action(self, plane):
        field = self.profile["battlefield"]
        north = field["center_latitude_deg"]-float(plane["position/lat-geod-deg"])
        east = (field["center_longitude_deg"]-float(plane["position/long-gc-deg"]))*math.cos(math.radians(field["center_latitude_deg"]))
        action = np.asarray(NEUTRAL_ACTION, dtype=np.int64).copy()
        action[WEAPON_HEAD] = WEAPON_CHOICES.index(CONTROL_SCHEMA["navigation_weapon"])
        if north != 0 or east != 0:
            desired = math.degrees(math.atan2(east, north)) % 360
            error = (desired-float(plane.get("attitude/psi-deg", 0))+180) % 360-180
            action[ACTION_NAMES.index("turn")] = int(np.argmin(np.abs(np.asarray(FLIGHT_CONTROLLER["delta_heading_deg"])-error)))
        return action

    def route(self, state, side, frames, actions):
        actions = np.asarray(actions, dtype=np.int64)
        if actions.shape != (len(frames), len(ACTION_DIMS)) or np.any(actions < 0) or np.any(actions >= np.asarray(ACTION_DIMS)):
            raise ValueError("Invalid high-level actions for rule routing")
        executed, masks, modes = actions.copy(), np.zeros((len(frames), 1), np.float32), []
        sim_time = max((float(p.get("simulation/sim-time-sec", 0)) for p in state.get(side, {}).values()), default=0.0)
        for index, frame in enumerate(frames):
            plane = state.get(side, {}).get(f"{side}_{index}", {})
            if not frame["alive"]:
                executed[index] = NEUTRAL_ACTION
                modes.append("dead")
            elif self._outside(plane):
                executed[index] = self._center_action(plane)
                modes.append("boundary")
            elif frame["enemies"] or frame["missile"] is not None:
                masks[index] = 1
                modes.append("policy")
            else:
                formation_action = self.formation.action(state, side, index, sim_time)
                executed[index] = self._center_action(plane) if formation_action is None else formation_action
                modes.append("center")
        return executed, masks, modes
