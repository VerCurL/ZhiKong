"""Adapt project platform telemetry and high-level deltas to the PID package.

The copied PID implementation keeps its own EAS state and 0.2 s history for
each aircraft.  The adapter supplies the original named native measurements;
it does not replace them with the MAPPO observation vector.
"""

import json
import math
from pathlib import Path

import numpy as np

from .contracts import EASReference, EASState
from .pid_speed_proportional import make_controller

_DELTA_ALTITUDE_M = (100.0, 0.0, -100.0)
_DELTA_HEADING_DEG = (-30.0, -15.0, 0.0, 15.0, 30.0)
_DELTA_SPEED_MPS = (17.0, 0.0, -17.0)
_ACTION_ALTITUDE = 0
_ACTION_TURN = 1
_ACTION_SPEED = 2


class PIDFlightController:
    """One independently stateful C50-M1 PID controller per aircraft."""

    def __init__(self, feet_to_meters, model_path=None, actor=None):
        del actor  # PID has no neural actor; kept for the common constructor API.
        self.ft = float(feet_to_meters)
        self.model_path = Path(model_path) if model_path is not None else Path(__file__).with_name("model.json")
        with self.model_path.open("r", encoding="utf-8") as stream:
            self.model = json.load(stream)
        self.controllers = {}
        self.control_steps = {}
        self.reset()

    def reset(self):
        self.controllers.clear()
        self.control_steps.clear()

    def forget(self, name):
        self.controllers.pop(name, None)
        self.control_steps.pop(name, None)

    def deltas(self, action):
        return (_DELTA_ALTITUDE_M[int(action[_ACTION_ALTITUDE])],
                _DELTA_HEADING_DEG[int(action[_ACTION_TURN])],
                _DELTA_SPEED_MPS[int(action[_ACTION_SPEED])])

    def _controller(self, name):
        if name not in self.controllers:
            self.controllers[name] = make_controller("pid", self.model)
        return self.controllers[name]

    @staticmethod
    def _finite(plane, key):
        if key not in plane:
            raise ValueError(f"PID flight-controller input is missing: {key}")
        value = float(plane[key])
        if not math.isfinite(value):
            raise ValueError(f"PID flight-controller input is non-finite: {key}")
        return value

    def _state(self, name, plane):
        # These are the PID package's native inputs. EAS is deliberately read
        # from velocities/ve-fps; it is not substituted with body-u/TAS.
        # The platform timestamp is a binary32 accumulation.  The PID contract
        # requires the logical controller clock to be index*0.2 exactly, so the
        # adapter validates that the raw field is present but supplies its own
        # per-aircraft logical grid to avoid accumulated timestamp drift.
        raw_time = self._finite(plane, "simulation/sim-time-sec")
        del raw_time
        logical_time = self.control_steps.get(name, 0)*.2
        return EASState(
            time_sec=logical_time,
            altitude_m=self._finite(plane, "position/h-sl-ft")*self.ft,
            eas_mps=self._finite(plane, "velocities/ve-fps")*self.ft,
            roll_rad=self._finite(plane, "attitude/roll-rad"),
            pitch_rad=self._finite(plane, "attitude/pitch-rad"),
            heading_rad=math.radians(self._finite(plane, "attitude/psi-deg")),
            alpha_rad=math.radians(self._finite(plane, "aero/alpha-deg")),
            beta_rad=math.radians(self._finite(plane, "aero/beta-deg")),
            p_radps=self._finite(plane, "velocities/p-rad_sec"),
            q_radps=self._finite(plane, "velocities/q-rad_sec"),
            r_radps=self._finite(plane, "velocities/r-rad_sec"),
            v_north_mps=self._finite(plane, "velocities/v-north-fps")*self.ft,
            v_east_mps=self._finite(plane, "velocities/v-east-fps")*self.ft,
            v_down_mps=self._finite(plane, "velocities/v-down-fps")*self.ft,
        )

    def _reference(self, plane, action, state):
        height_delta, heading_delta, speed_delta = self.deltas(action)
        heading = (self._finite(plane, "attitude/psi-deg") + heading_delta) % 360.0
        return EASReference(math.radians(heading), state.altitude_m+height_delta,
                            max(1e-3, state.eas_mps+speed_delta))

    def controls(self, names, planes, actions):
        if len(names) != len(planes) or len(names) != len(actions):
            raise ValueError("Each PID flight request needs matching names, planes and actions")
        outputs = np.empty((len(names), 4), dtype=np.float64)
        for index, (name, plane, action) in enumerate(zip(names, planes, actions)):
            state = self._state(name, plane)
            output, _ = self._controller(name).step(state, self._reference(plane, action, state), .2)
            outputs[index] = output.commands
            self.control_steps[name] = self.control_steps.get(name, 0)+1
        return outputs

    @staticmethod
    def controls_many(requests):
        outputs = []
        for controller, names, planes, actions in requests:
            outputs.append(controller.controls(names, planes, actions))
        return outputs
