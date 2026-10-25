"""Map high-level MAPPO actions to platform flight-control commands."""

import numpy as np

from .observation_encoder import alive
from .weapon_control import WeaponController
from .model.flight_controller import FrozenFlightController
from .model.pid.controller import PIDFlightController
from .model import validate_flight_controller
from algorithms.utils.action_spec import ACTION_DIMS, NEUTRAL_ACTION, WEAPON_HEAD


class ActionAdapter:
    def __init__(self, encoder, profile, model_path=None, flight_actor=None, flight_controller="baseline"):
        self.encoder = encoder
        self.profile = profile
        self.ft = profile["units"]["feet_to_meters"]
        self.flight_controller_name = validate_flight_controller(flight_controller)
        controller = FrozenFlightController if self.flight_controller_name == "baseline" else PIDFlightController
        self.flight_controller = controller(self.ft, model_path=model_path, actor=flight_actor)
        self.weapons = WeaponController(encoder, profile)

    def reset(self):
        self.flight_controller.reset()
        self.weapons.reset()

    def flight_inputs(self, state, side, actions):
        actions = np.asarray(actions, dtype=np.int64)
        if actions.shape != (self.encoder.num_agents, len(ACTION_DIMS)):
            raise ValueError("Expected one high-level action per aircraft with the configured action heads")
        if np.any(actions < 0) or np.any(actions >= np.asarray(ACTION_DIMS)):
            raise ValueError(f"High-level action is outside {ACTION_DIMS}")
        names = [f"{side}_{i}" for i in range(self.encoder.num_agents)]
        live_indices = [i for i, name in enumerate(names) if alive(state.get(side, {}).get(name, {}))]
        return ([names[i] for i in live_indices], [state[side][names[i]] for i in live_indices], actions[live_indices])

    def side_commands(self, state, side, actions, tracker, flight_controls=None, weapon_enabled=None):
        live_names, planes, live_actions = self.flight_inputs(state, side, actions)
        actions = np.asarray(actions, dtype=np.int64)
        live_indices = [int(name.rsplit("_", 1)[1]) for name in live_names]
        if flight_controls is None:
            flight_controls = self.flight_controller.controls(live_names, planes, live_actions)
        else:
            flight_controls = np.asarray(flight_controls, dtype=np.float64)
            if flight_controls.shape != (len(live_indices), 4) or not np.all(np.isfinite(flight_controls)):
                raise ValueError("Precomputed flight controls must match the live aircraft")
        low_level = np.zeros((self.encoder.num_agents, 4), dtype=np.float64)
        low_level[live_indices] = flight_controls
        command = {}
        for i in range(self.encoder.num_agents):
            name = f"{side}_{i}"
            plane = state.get(side, {}).get(name, {})
            action = actions[i] if alive(plane) else NEUTRAL_ACTION
            if alive(plane):
                height_delta, heading_delta, speed_delta = self.flight_controller.deltas(action)
                tracker.targets[name] = ((float(plane.get("attitude/psi-deg", 0.0))+heading_delta) % 360.0,
                                         float(plane["position/h-sl-ft"])+height_delta/self.ft,
                                         float(plane["velocities/u-fps"])+speed_delta/self.ft)
            else:
                self.flight_controller.forget(name)
                tracker.targets.pop(name, None)
            aileron, elevator, rudder, throttle = map(float, low_level[i])
            weapon = self.weapons.command(state, side, name, int(action[WEAPON_HEAD]),
                                          enabled=weapon_enabled is None or bool(weapon_enabled[i]))
            command[name] = {
                "mode": 0,
                "fcs/aileron-cmd-norm": aileron if alive(plane) else 0.0,
                "fcs/elevator-cmd-norm": elevator if alive(plane) else 0.0,
                "fcs/rudder-cmd-norm": rudder if alive(plane) else 0.0,
                "fcs/throttle-cmd-norm": throttle if alive(plane) else 0.0,
                "fcs/weapon-launch": weapon["fcs/weapon-launch"],
                "switch-missile": weapon["switch-missile"],
                "switch-acmType": 0,
                "change-target": weapon["change-target"],
            }
        return command
