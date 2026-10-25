"""Fixed F16 controller; match LAGCFC's 12 inputs and discrete control decoding."""

import math
from pathlib import Path

import numpy as np
import torch

from algorithms.utils.action_spec import ACTION_NAMES, FLIGHT_CONTROLLER
from .baseline.baseline_actor import BaselineActor


def calibrated_airspeed_mps(plane, feet_to_meters):
    """Prefer measured CAS. Otherwise estimate static pressure with ISA from height.

    The official socket table supplies Mach and altitude, but does not list CAS
    or static pressure. This fallback assumes standard atmosphere; it does not
    substitute ground speed or true airspeed for calibrated airspeed.
    """
    if "velocities/vc-fps" in plane:
        return float(plane["velocities/vc-fps"])*feet_to_meters
    if "velocities/vc-kts" in plane:
        return float(plane["velocities/vc-kts"])*1852.0/3600.0
    if "velocities/mach" not in plane:
        raise ValueError("Baseline flight controller needs velocities/vc-fps, vc-kts or mach")
    mach = max(0.0, float(plane["velocities/mach"]))
    height = float(plane["position/h-sl-ft"])*feet_to_meters
    # ISA geopotential height, troposphere and lower isothermal stratosphere.
    height = 6356766.0*height/(6356766.0+height)
    troposphere = min(height, 11000.0)
    temperature = 288.15-0.0065*troposphere
    pressure_ratio = (temperature/288.15)**(9.80665/(287.05287*0.0065))
    if height > 11000.0:
        pressure_ratio *= math.exp(-9.80665*(height-11000.0)/(287.05287*temperature))

    def impact_ratio(value):
        if value <= 1.0:
            return (1.0+0.2*value*value)**3.5-1.0
        # Rayleigh pitot relation includes the normal shock at supersonic speed.
        return (1.2*value*value)**3.5*(2.4/(2.8*value*value-0.4))**2.5-1.0

    ratio = pressure_ratio*impact_ratio(mach)
    if ratio <= impact_ratio(1.0):
        calibrated_mach = math.sqrt(max(0.0, 5.0*((ratio+1.0)**(2.0/7.0)-1.0)))
    else:
        lower, upper = 1.0, max(2.0, mach)
        while impact_ratio(upper) < ratio:
            upper *= 2.0
        for _ in range(48):
            middle = (lower+upper)/2.0
            if impact_ratio(middle) < ratio:
                lower = middle
            else:
                upper = middle
        calibrated_mach = (lower+upper)/2.0
    return calibrated_mach*math.sqrt(1.4*287.05287*288.15)


class FrozenFlightController:
    def __init__(self, feet_to_meters, model_path=None, actor=None):
        self.ft = feet_to_meters
        self.settings = FLIGHT_CONTROLLER
        path = Path(model_path) if model_path is not None else Path(__file__).resolve().parents[2]/self.settings["model_path"]
        self.model_path = path
        if actor is None:
            self.actor = BaselineActor()
            self.actor.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
            self.actor.eval()
            self.actor.requires_grad_(False)
        else:
            # Share immutable weights only; recurrent state remains in this instance.
            self.actor = actor
        self.hidden_dim = self.actor.rnn.gru.hidden_size
        self.layers = self.actor.rnn.gru.num_layers
        self.action_dims = np.asarray([head.logits_net.out_features for head in self.actor.act.action_outs])
        self.reset()

    def reset(self):
        self.hidden = {}

    def forget(self, name):
        self.hidden.pop(name, None)

    def deltas(self, action):
        return (self.settings["delta_altitude_m"][int(action[ACTION_NAMES.index("altitude")])],
                self.settings["delta_heading_deg"][int(action[ACTION_NAMES.index("turn")])],
                self.settings["delta_speed_mps"][int(action[ACTION_NAMES.index("speed")])])

    def observation(self, plane, action):
        required = ("position/h-sl-ft", "attitude/roll-rad", "attitude/pitch-rad",
                    "velocities/u-fps", "velocities/v-fps", "velocities/w-fps")
        missing = [key for key in required if key not in plane]
        if missing:
            raise ValueError(f"Live aircraft is missing baseline flight-controller inputs: {missing}")
        height, heading, speed = self.deltas(action)
        scales = self.settings["observation_scales"]
        roll, pitch = float(plane["attitude/roll-rad"]), float(plane["attitude/pitch-rad"])
        own = np.asarray([float(plane["position/h-sl-ft"])*self.ft/scales["altitude_m"],
                          math.sin(roll), math.cos(roll), math.sin(pitch), math.cos(pitch),
                          *(float(plane[f"velocities/{axis}-fps"])*self.ft/scales["speed_mps"] for axis in ("u", "v", "w")),
                          calibrated_airspeed_mps(plane, self.ft)/scales["speed_mps"]])
        # LAGCFC clips the nine own-aircraft features before inserting deltas.
        return np.concatenate(([height/scales["altitude_error_m"], math.radians(heading), speed/scales["speed_mps"]],
                               np.clip(own, -scales["clip"], scales["clip"]))).astype(np.float32)

    @torch.no_grad()
    def controls(self, names, planes, actions):
        return self.controls_many([(self, names, planes, actions)])[0]

    @staticmethod
    @torch.no_grad()
    def controls_many(requests):
        """One forward for multiple controllers; scatter by controller and name."""
        observations, hidden, rows, outputs = [], [], [], []
        actor = requests[0][0].actor if requests else None
        seen = set()
        for controller, names, planes, actions in requests:
            if controller.actor is not actor:
                raise ValueError("Batched flight controllers must share the same frozen actor")
            if len(names) != len(planes) or len(names) != len(actions):
                raise ValueError("Each flight request needs matching names, planes and actions")
            outputs.append(np.empty((len(names), len(controller.action_dims)), dtype=np.float64))
            for name, plane, action in zip(names, planes, actions):
                identity = (id(controller), name)
                if identity in seen:
                    raise ValueError("Aircraft appears twice in the same flight-control batch")
                seen.add(identity)
                observations.append(controller.observation(plane, action))
                value = controller.hidden.get(name)
                hidden.append(np.zeros((controller.layers, controller.hidden_dim), dtype=np.float32) if value is None else value)
                rows.append((controller, name))
        if not rows:
            return outputs
        output, next_hidden = actor(np.stack(observations), np.stack(hidden))
        controls = output.cpu().numpy().astype(np.float64)
        next_hidden = next_hidden.cpu().numpy()
        for index, (controller, name) in enumerate(rows):
            controller.hidden[name] = next_hidden[index].copy()
        start = 0
        for index, (controller, names, _, _) in enumerate(requests):
            count = len(names)
            values = controls[start:start+count]
            values[:, :3] = values[:, :3]*2.0/(controller.action_dims[:3]-1)-1.0
            values[:, 3] = (values[:, 3]/(controller.action_dims[3]-1)*
                           (controller.settings["throttle_max"]-controller.settings["throttle_min"])+controller.settings["throttle_min"])
            outputs[index] = values
            start += count
        return outputs
