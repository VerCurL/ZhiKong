"""Dual-socket environments and a pool of independent platform instances."""

from copy import deepcopy
import math
import random
import os
import socket
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_EXCEPTION

import numpy as np

from comm_interface import env as PlatformClient
from .action_adapter import ActionAdapter
from .control_rules import ControlRules
from .formations import resolve_formations, FORMATION_SCHEMA
from .utils.observations import decode_actor
from .model.flight_controller import FrozenFlightController
from .action_mask import masks_for_side
from .observation_encoder import ObservationEncoder, alive
from .reward.fkr.multiplecombat_reward import MultipleCombatReward
from .state_tracker import StateTracker
from .termination import outcome, round_done, frame_time, team_score, in_flight_missiles


class CompetitionEnv:
    def __init__(self, profile, scenario, side="red", host=None,
                 port=None, red_port=None, blue_port=None, seed=1, client=None,
                 connect_timeout=None, socket_timeout=None, flight_controller_path=None, flight_actor=None,
                 flight_controller="baseline", training_start_mode="official",
                 combat_start_seconds=200.0):
        if side not in ("red", "blue"):
            raise ValueError("side must be red or blue")
        self.num_agents = int(scenario["aircraft_per_side"])
        if self.num_agents != 4:
            raise ValueError("Only 4v4 is supported in this phase")
        self.profile = profile
        self.scenario = scenario
        self.side = side
        if training_start_mode not in ("official", "combat_start"):
            raise ValueError("training_start_mode must be official or combat_start")
        self.training_start_mode = training_start_mode
        self.combat_start_seconds = float(combat_start_seconds)
        if self.combat_start_seconds < 0:
            raise ValueError("combat_start_seconds must be nonnegative")
        self.rng = random.Random(seed)
        self.seed_formations(seed)
        self.formations = {}
        self.client = client
        self.dual_socket = client is None
        if self.dual_socket:
            if red_port is None and port is None:
                raise ValueError("Provide red_port/blue_port from train_mappo.yaml")
            actual_red_port = red_port if red_port is not None else port
            self.red_client = PlatformClient(host, actual_red_port, connect_timeout, socket_timeout)
            try:
                self.blue_client = PlatformClient(host, blue_port if blue_port is not None else actual_red_port + 1,
                                                  connect_timeout, socket_timeout)
            except BaseException:
                self.red_client.close()
                raise
        else:
            self.red_client = None
            self.blue_client = None
        self.encoder = ObservationEncoder(
            short_mode=profile["weapon"]["aim_mode_short"],
            detection_m=profile["limits"]["enemy_detection_m"],
            missile_detection_m=profile["limits"]["missile_detection_m"],
            num_agents=self.num_agents,
            feet_to_meters=profile["units"]["feet_to_meters"],
            earth_meters_per_degree=profile["units"]["earth_meters_per_degree"],
            life_max=profile["aircraft"]["initial_life"],
            sraam_capacity=profile["aircraft"]["sraam_capacity"],
            amraam_capacity=profile["aircraft"]["amraam_capacity"],
            filter_visibility=not self.dual_socket,
            battle_seconds=profile["limits"]["battle_seconds"],
        )
        self.adapter = ActionAdapter(self.encoder, profile, model_path=flight_controller_path,
                                     flight_actor=flight_actor, flight_controller=flight_controller)
        self.reward = MultipleCombatReward(profile, self.encoder)
        self.rules = ControlRules(profile)
        self._frames = {}
        self._pending_routing = None
        self.tracker = StateTracker()
        self.state = None
        self._side_states = {"red": None, "blue": None}
        self.episode_done = False
        self.last_time = 0.0
        self._frame_time = None
        self._clock_snapshot = {}
        self.steps = 0
        self._missile_cache = {}

    def seed_formations(self, seed):
        # A separate stream keeps template selection out of official init draws.
        self.formation_rng = random.Random(f"formations:{seed}")

    @staticmethod
    def _merge_payloads(red_payload, blue_payload, previous=None):
        """Merge side-only platform responses while preserving sparse plane fields."""
        merged = {"red": {}, "blue": {}, "missile": {}}
        previous = previous or {}
        # A side-only channel can still include sparse enemy groups. Merge every
        # group from both payloads, retaining fields omitted after a plane dies.
        for payload in (red_payload or {}, blue_payload or {}):
            if not isinstance(payload, dict):
                continue
            for side in ("red", "blue"):
                for name, fields in payload.get(side, {}).items():
                    merged[side][name] = dict(merged[side].get(name, previous.get(side, {}).get(name, {})))
                    if isinstance(fields, dict):
                        merged[side][name].update(fields)
        for side in ("red", "blue"):
            for name, fields in previous.get(side, {}).items():
                merged[side].setdefault(name, dict(fields))
        # The owner's Socket is authoritative. Sparse enemy copies from the
        # other Socket must not overwrite fresh life/status/clock fields.
        for side, payload in (("red", red_payload), ("blue", blue_payload)):
            for name, fields in (payload or {}).get(side, {}).items():
                if isinstance(fields, dict):
                    merged[side][name].update(fields)
        for payload in (red_payload or {}, blue_payload or {}):
            if isinstance(payload, dict):
                for name, fields in payload.get("missile", {}).items():
                    merged["missile"][name] = dict(fields) if isinstance(fields, dict) else fields
        return merged

    @staticmethod
    def _side_view(payload, side, previous=None):
        """Normalize one side-only response without applying visibility filtering."""
        previous = previous or {"red": {}, "blue": {}, "missile": {}}
        payload = payload if isinstance(payload, dict) else {}
        view = {"red": {}, "blue": {}, "missile": {}}
        if any(str(key).startswith(("red_", "blue_")) for key in payload):
            view[side] = {name: dict(fields) for name, fields in payload.items()
                          if str(name).startswith(f"{side}_") and isinstance(fields, dict)}
        for group in ("red", "blue"):
            for name, fields in payload.get(group, {}).items():
                view[group][name] = dict(fields) if isinstance(fields, dict) else fields
        # Preserve sparse fields only for our own aircraft. Enemy slots and
        # missiles are direct current-frame platform data; omitted means unseen.
        for name, fields in previous.get(side, {}).items():
            view[side][name] = {**fields, **view[side].get(name, {})}
        for name, fields in payload.get("missile", {}).items():
            view["missile"][name] = dict(fields) if isinstance(fields, dict) else fields
        return view

    def _clock_error(self, message, raw=None, current=None):
        state = self.state if raw is None else raw
        state = state or {}
        ports = {side: getattr(getattr(self, side+"_client", None), "PORT", None) for side in ("red", "blue")}
        living = {side: team_score(state, side)[0] for side in ("red", "blue")}
        pending = [name for side in ("red", "blue") for name, plane in state.get(side, {}).items()
                   if plane.get("IfPresenceHitting", 0)]
        return RuntimeError(f"Platform clock error: {message}; ports={ports}, previous={self.last_time}, "
                            f"current={current}, alive={living}, in_flight={in_flight_missiles(state)}, "
                            f"pending_missile_owners={pending}, raw_clocks={self._clock_snapshot}")

    def _read_clock(self, own_groups):
        self._clock_snapshot = {side: {name: plane.get("simulation/sim-time-sec") if isinstance(plane, dict) else None
                                      for name, plane in planes.items()} for side, planes in own_groups.items()}
        try:
            self._frame_time = frame_time(own_groups, self.profile["simulation"]["timestamp_tolerance_seconds"])
        except ValueError as error:
            raise self._clock_error(str(error), raw=own_groups) from error

    def _reset_platform(self, request):
        if not self.dual_socket:
            raw = self.client.reset(request)
            if not isinstance(raw, dict):
                raise ConnectionError("Platform reset returned no valid state")
            self._read_clock({side: raw.get(side, {}) for side in ("red", "blue")})
            return raw
        self.red_client._send_data(request)
        self.blue_client._send_data(request)
        red_payload = self.red_client._receive_data()
        blue_payload = self.blue_client._receive_data()
        if not isinstance(red_payload, dict) or not isinstance(blue_payload, dict):
            raise ConnectionError("Both platform sockets must return a valid state")
        self._read_clock({"red": red_payload.get("red", {}), "blue": blue_payload.get("blue", {})})
        self._side_states = {"red": self._side_view(red_payload, "red"),
                             "blue": self._side_view(blue_payload, "blue")}
        return self._merge_payloads(red_payload, blue_payload)

    def _step_platform(self, red_command, blue_command):
        if not self.dual_socket:
            raw = self.client.step({"red": red_command, "blue": blue_command})
            if not isinstance(raw, dict):
                raise ConnectionError("Platform step returned no valid state")
            self._read_clock({side: raw.get(side, {}) for side in ("red", "blue")})
            return raw
        # Both writes must happen before either read; the platform advances after both sides respond.
        self.red_client._send_data({"red": red_command})
        self.blue_client._send_data({"blue": blue_command})
        red_payload = self.red_client._receive_data()
        blue_payload = self.blue_client._receive_data()
        if not isinstance(red_payload, dict) or not isinstance(blue_payload, dict):
            raise ConnectionError("Both platform sockets must return a valid state")
        self._read_clock({"red": red_payload.get("red", {}), "blue": blue_payload.get("blue", {})})
        self._side_states = {
            "red": self._side_view(red_payload, "red", self._side_states["red"]),
            "blue": self._side_view(blue_payload, "blue", self._side_states["blue"]),
        }
        return self._merge_payloads(red_payload, blue_payload, self.state)

    def _initial_attributes(self, initialization=None, initial=False, learner_spawn_side=None):
        if learner_spawn_side not in (None, "left", "right"):
            raise ValueError("learner_spawn_side must be left or right")
        if initialization is not None and learner_spawn_side is not None:
            raise ValueError("Explicit initialization cannot be combined with a training spawn side")
        if initialization is not None:
            if "red" not in initialization or "blue" not in initialization:
                raise ValueError("Initialization must contain red and blue aircraft")
            return {"flag": {"init" if initial else "reset": {"save": int(self.scenario["initialization"].get("save_replay", False)),
                                                              "SplitScreen": 0}},
                    "red": deepcopy(initialization["red"]), "blue": deepcopy(initialization["blue"])}
        settings = self.scenario["initialization"]
        extent = float(settings["longitude_extent_deg"])
        scale_min = float(settings["longitude_scale_min"])
        center_lon = float(self.profile["battlefield"]["center_longitude_deg"])
        red_lon = center_lon-extent * self.rng.uniform(scale_min, 1.0)
        blue_lon = center_lon+extent * self.rng.uniform(scale_min, 1.0)
        mirror = learner_spawn_side is not None and ((self.side == "red") != (learner_spawn_side == "left"))
        red_lat = self.rng.uniform(settings["latitude_min_deg"], settings["latitude_max_deg"])
        blue_lat = self.rng.uniform(settings["latitude_min_deg"], settings["latitude_max_deg"])
        altitude = self.rng.uniform(settings["altitude_min_ft"], settings["altitude_max_ft"])
        result = {"flag": {"init" if initial else "reset": {
            "save": int(settings.get("save_replay", False)), "SplitScreen": 0}},
            "red": {}, "blue": {}}
        for side, lon, base_lat, heading in (
            ("red", red_lon, red_lat, settings["red_heading_deg"]),
            ("blue", blue_lon, blue_lat, settings["blue_heading_deg"]),
        ):
            if mirror:
                lon = 2*center_lon-lon
                heading = (-float(heading)) % 360
            team_speed = self.rng.uniform(settings["speed_min_fps"], settings["speed_max_fps"])
            for index in range(self.num_agents):
                result[side][f"{side}_{index}"] = {
                    "ic/h-sl-ft": altitude,
                    "ic/terrain-elevation-ft": 1e-8,
                    "ic/long-gc-deg": lon,
                    "ic/lat-geod-deg": base_lat + index * settings["formation_spacing_deg"],
                    # Official reset_variable gives every member the same team speed.
                    "ic/u-fps": team_speed,
                    "ic/v-fps": 0.0, "ic/w-fps": 0.0,
                    "ic/p-rad_sec": 0.0, "ic/q-rad_sec": 0.0, "ic/r-rad_sec": 0.0,
                    "ic/roc-fpm": 0.0, "ic/psi-true-deg": heading,
                    "ic/phi-deg": 0.0, "ic/theta-deg": 0.0,
                }
        return result

    def _combat_start_initialization(self, learner_spawn_side, formations):
        """Project official random spawns into a training-only combat state."""
        request = self._initial_attributes(None, initial=self.state is None,
                                           learner_spawn_side=learner_spawn_side)
        field, units = self.profile["battlefield"], self.profile["units"]
        scale = float(units["earth_meters_per_degree"])
        cos_lat = math.cos(math.radians(float(field["center_latitude_deg"])))
        positions, centers, forwards, speeds = {}, {}, {}, {}
        for side in ("red", "blue"):
            rows = request[side]
            values = [np.array([
                (float(plane["ic/lat-geod-deg"])-field["center_latitude_deg"])*scale,
                (float(plane["ic/long-gc-deg"])-field["center_longitude_deg"])*scale*cos_lat,
            ]) for plane in rows.values()]
            start_center = np.mean(values, axis=0)
            distance = float(np.linalg.norm(start_center))
            forward = -start_center/max(distance, 1e-9)
            speed = float(next(iter(rows.values()))["ic/u-fps"])*float(units["feet_to_meters"])
            centers[side] = start_center + forward*speed*self.combat_start_seconds
            forwards[side] = forward
            speeds[side] = speed
            positions[side] = values

        for side in ("red", "blue"):
            rows = request[side]
            start_center = np.mean(positions[side], axis=0)
            relative = [value-start_center for value in positions[side]]
            template = FORMATION_SCHEMA["templates"].get(formations.get(side))
            if template is not None:
                params = FORMATION_SCHEMA["parameters"]
                right = np.array([-forwards[side][1], forwards[side][0]])
                local = np.asarray(template, dtype=np.float64)
                relative = [local[index, 0]*params["depth_m"]*forwards[side]+
                            local[index, 1]*params["lateral_span_m"]*right
                            for index in range(self.num_agents)]
            # `forward` already points from the projected spawn toward the
            # map center; use it directly so the nose follows the travel path.
            heading = math.degrees(math.atan2(forwards[side][1], forwards[side][0])) % 360.0
            for index, plane in enumerate(rows.values()):
                point = centers[side]+relative[index]
                plane["ic/lat-geod-deg"] = field["center_latitude_deg"]+point[0]/scale
                plane["ic/long-gc-deg"] = field["center_longitude_deg"]+point[1]/(scale*cos_lat)
                plane["ic/psi-true-deg"] = heading
                plane["ic/u-fps"] = speeds[side]/float(units["feet_to_meters"])
        return request

    def _validate_state(self, state):
        if not isinstance(state, dict) or "red" not in state or "blue" not in state:
            raise RuntimeError("Platform returned no valid 4v4 state")
        for side in ("red", "blue"):
            if not all(f"{side}_{i}" in state[side] for i in range(self.num_agents)):
                raise RuntimeError(f"Platform state is missing {side} aircraft")

    def _info(self, reward_detail=None):
        sim_time = self.last_time
        info = {
            "sim_time": sim_time,
            "episode_done": self.episode_done,
            "outcome": outcome(self.state) if self.episode_done else None,
            "red_score": team_score(self.state, "red"),
            "blue_score": team_score(self.state, "blue"),
            "side_observations": {},
            "side_action_masks": {},
            "critic_observations": None,
            "raw_state": self.state,
            "formations": dict(self.formations),
        }
        for side in ("red", "blue"):
            view = self._side_states[side] if self.dual_socket else self.state
            info["side_observations"][side] = self.encoder.encode_actor(view, side, self.tracker, sim_time)
            info["side_action_masks"][side] = masks_for_side(
                view, side, self.encoder, self.profile, self.adapter.weapons.pending)
            self._frames[side] = [decode_actor(
                row, self.encoder, alive(view.get(side, {}).get(f"{side}_{index}", {})),
                self.encoder.selected_missiles.get(f"{side}_{index}"),
                plane=view.get(side, {}).get(f"{side}_{index}"),
                missile=view.get("missile", {}).get(self.encoder.selected_missiles.get(f"{side}_{index}")))
                for index, row in enumerate(info["side_observations"][side])]
        info["critic_observations"] = self.encoder.encode_critic(
            np.concatenate((info["side_observations"]["red"], info["side_observations"]["blue"]), axis=0),
            sim_time)
        if reward_detail is not None:
            info["reward_detail"] = reward_detail
        return info

    def reset(self, initialization=None, learner_spawn_side=None, formations="random"):
        selected_formations = resolve_formations(formations, self.formation_rng)
        self.tracker.reset()
        self.encoder.selected_missiles.clear()
        self._pending_routing = None
        self.adapter.reset()
        raw = self._reset_platform(self._initial_attributes(
            initialization, initial=self.state is None, learner_spawn_side=learner_spawn_side))
        self._validate_state(raw)
        self.state = raw
        if not self.dual_socket:
            self._side_states = {"red": raw, "blue": raw}
        self.episode_done = False
        self.steps = 0
        if self._frame_time is None:
            raise self._clock_error("Reset did not supply a current timestamp", raw)
        self.last_time = self._frame_time
        self.formations = selected_formations
        own_states = {side: (self._side_states[side][side] if self.dual_socket else raw[side]) for side in ("red", "blue")}
        self.rules.reset(own_states, self.formations, self.last_time)
        self.tracker.update(raw, self.last_time)
        info = self._info()
        self.reward.reset(self.last_time, raw)
        return info["side_observations"][self.side], info["critic_observations"], info

    def reset_training(self, learner_spawn_side):
        """Reset one sampled battle at its assigned map side, keeping team IDs."""
        if isinstance(learner_spawn_side, dict):
            spawn_side = learner_spawn_side["spawn_side"]
            formations = learner_spawn_side["formations"]
            if self.training_start_mode == "combat_start":
                result = self.reset(initialization=self._combat_start_initialization(spawn_side, formations),
                                    formations=formations)
                result[2]["training_start_mode"] = "combat_start"
                return result
            result = self.reset(learner_spawn_side=spawn_side, formations=formations)
            result[2]["training_start_mode"] = "official"
            return result
        return self.reset(learner_spawn_side=learner_spawn_side)

    def route_actions(self, actions):
        if self.state is None or self.episode_done:
            raise RuntimeError("Reset before routing actions")
        if set(actions) != {"red", "blue"}:
            raise ValueError("Actions must include both sides")
        routing = {"actions": {}, "masks": {}, "modes": {}}
        for side in ("red", "blue"):
            view = self._side_states[side] if self.dual_socket else self.state
            routed, mask, modes = self.rules.route(view, side, self._frames[side], actions[side])
            routing["actions"][side], routing["masks"][side], routing["modes"][side] = routed, mask, modes
        self._pending_routing = routing
        return routing

    def flight_requests(self, actions):
        """Read each side's current view; no recurrent state advances here."""
        if self.state is None or self.episode_done:
            raise RuntimeError("Call reset before step, including after a finished round")
        if set(actions) != {"red", "blue"}:
            raise ValueError("Actions must include red and blue arrays")
        action_state = self._side_states if self.dual_socket else {"red": self.state, "blue": self.state}
        return [(self.adapter.flight_controller, *self.adapter.flight_inputs(action_state[side], side, actions[side]))
                for side in ("red", "blue")]

    def prepare_step(self, actions, flight_controls=None):
        if self.state is None or self.episode_done:
            raise RuntimeError("Call reset before step, including after a finished round")
        if set(actions) != {"red", "blue"}:
            raise ValueError("Actions must include red and blue arrays")
        action_state = self._side_states if self.dual_socket else {"red": self.state, "blue": self.state}
        return {side: self.adapter.side_commands(
                    action_state[side], side, actions[side], self.tracker,
                    flight_controls=None if flight_controls is None else flight_controls[side],
                    weapon_enabled=self._pending_routing["masks"][side][:, 0])
                for side in ("red", "blue")}

    def step(self, actions):
        routed = self.route_actions(actions)
        return self.step_commands(self.prepare_step(routed["actions"]))

    def step_commands(self, command):
        """Execute prepared commands; platform I/O and rewards stay in workers."""
        if self.state is None or self.episode_done:
            raise RuntimeError("Call reset before step, including after a finished round")
        if self._pending_routing is None:
            raise RuntimeError("Route policy actions before preparing platform commands")
        routing, self._pending_routing = self._pending_routing, None
        simulation = self.profile["simulation"]
        for stalled in range(int(simulation["max_stalled_steps"])+1):
            raw = self._step_platform(command["red"], command["blue"])
            self._validate_state(raw)
            sim_time = self._frame_time
            both_dead = all(team_score(raw, side)[0] == 0 for side in ("red", "blue"))
            if sim_time is None:
                if not both_dead:
                    raise self._clock_error("No fresh timestamp while aircraft remain alive", raw)
                sim_time = self.last_time  # Event-only terminal frame, no invented elapsed time.
            elif both_dead:
                # Old dead-aircraft timestamps are not a new running clock.
                sim_time = max(self.last_time, sim_time)
            if sim_time < self.last_time:
                raise self._clock_error("Time moved backward", raw, sim_time)
            done = round_done(raw, self.profile["limits"]["battle_seconds"],
                              simulation["max_missile_overtime_seconds"], sim_time)
            if sim_time > self.last_time or done:
                break
            if stalled == int(simulation["max_stalled_steps"]):
                raise self._clock_error(f"Time did not advance after {stalled} hold exchanges", raw, sim_time)
            # Stay inside this transition: no new Actor/baseline inference,
            # deferred request consumption, GRU update or buffer row. Never
            # resend a launch or a mode toggle when asking the platform to move.
            command = {side: {name: {**fields, "fcs/weapon-launch": 0, "switch-missile": 0,
                                    "switch-acmType": 0, "change-target": 9}
                              for name, fields in planes.items()} for side, planes in command.items()}
        self.state = raw
        self.steps += 1
        self.last_time = sim_time
        self.episode_done = done
        self.tracker.update(raw, sim_time)
        info = self._info()
        rewards, reward_detail = self.reward.calculate(self._frames, sim_time, raw, self.episode_done,
                                                      routing["masks"], routing["modes"])
        info["reward_detail"] = reward_detail
        info["policy_control_masks"] = routing["masks"]
        info["control_modes"] = routing["modes"]
        info["executed_actions"] = routing["actions"]
        info["stalled_responses"] = stalled
        dones = np.asarray([self.episode_done or not alive(raw[self.side][f"{self.side}_{i}"])
                           for i in range(self.num_agents)], dtype=bool)
        return (info["side_observations"][self.side],
                info["critic_observations"],
                np.asarray(rewards[self.side], dtype=np.float32).reshape(self.num_agents, 1), dones, info)

    def close(self):
        if self.dual_socket:
            self.red_client.close()
            self.blue_client.close()
        else:
            self.client.close()


class ParallelEnvironments:
    """One I/O worker per platform; inference and optimization stay in the main thread."""

    def __init__(self, config, profile, scenario, clients=None):
        self.config = config
        self.num_envs = min(int(config["parallel_envs"]), int(config["n_rollout_threads"]))
        self.num_agents = int(scenario["aircraft_per_side"])
        self.side = config["control_side"]
        self.processes = []
        self.envs = []
        self.closed = False
        if clients is not None and len(clients) != self.num_envs:
            raise ValueError("Provide exactly one test client per configured environment")
        self.executor = ThreadPoolExecutor(max_workers=self.num_envs, thread_name_prefix="platform")
        futures = []
        try:
            # Construct the fixed network once. Every environment gets its own
            # controller/hidden-state dictionary referring to these weights.
            self.flight_controller_name = config.get("flight_controller", "baseline")
            self.baseline_actor = (FrozenFlightController(profile["units"]["feet_to_meters"]).actor
                                   if self.flight_controller_name == "baseline" else None)
            if clients is None and config["platform_launch"]["enabled"]:
                self._launch()
            launch = config["platform_launch"]
            for index in range(self.num_envs):
                futures.append(self.executor.submit(
                    CompetitionEnv, profile, scenario,
                    side=self.side, host=config["host"],
                    red_port=config["red_port"]+index*config["port_stride"],
                    blue_port=config["blue_port"]+index*config["port_stride"],
                    seed=config["seed"]+index,
                    client=None if clients is None else clients[index],
                    connect_timeout=launch["connect_timeout_seconds"],
                    socket_timeout=launch["socket_timeout_seconds"],
                    flight_actor=self.baseline_actor,
                    flight_controller=self.flight_controller_name,
                    training_start_mode=config.get("training_start_mode", "official"),
                    combat_start_seconds=config.get("combat_start_seconds", 200.0),
                ))
            self.envs = [future.result() for future in futures]
            self.encoder = self.envs[0].encoder
        except BaseException:
            # Constructor futures have bounded connection times. Close every successful one.
            for future in futures:
                future.cancel()
            self.executor.shutdown(wait=True)
            for future in futures:
                if not future.cancelled() and future.exception() is None:
                    future.result().close()
            self.close()
            raise

    def _launch(self):
        if self.config["host"] not in ("127.0.0.1", "localhost"):
            raise ValueError("Automatic platform launch only supports localhost")
        launch = self.config["platform_launch"]
        executable = Path(launch["executable"])
        if not executable.is_absolute():
            executable = Path(__file__).resolve().parents[1] / executable
        if not executable.is_file():
            raise FileNotFoundError(executable)
        # Refuse collisions; never stop a user-owned platform process.
        for index in range(self.num_envs):
            for key in ("red_port", "blue_port"):
                port = self.config[key]+index*self.config["port_stride"]
                with socket.socket() as probe:
                    if os.name == "nt":
                        probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
                    probe.bind((self.config["host"], port))
        for index in range(self.num_envs):
            port = self.config["red_port"]+index*self.config["port_stride"]
            command = [str(executable), f"Ip={self.config['host']}", f"Port={port}",
                       "Red=0", "Blue=1", f"RedNum={self.num_agents}", f"BlueNum={self.num_agents}",
                       f"PlayMode={launch['play_mode']}", f"Scenes={launch['scene']}", "-logFile", os.devnull]
            options = {}
            if os.name == "nt":
                startup = subprocess.STARTUPINFO()
                startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startup.wShowWindow = subprocess.SW_HIDE
                options = {"startupinfo": startup, "creationflags": subprocess.CREATE_NO_WINDOW}
            self.processes.append(subprocess.Popen(command, cwd=executable.parent,
                                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **options))

    def call(self, indices, method, payloads=None):
        indices = list(indices)
        if len(set(indices)) != len(indices):
            raise ValueError("An environment cannot appear twice in one parallel call")
        if payloads is not None and len(payloads) != len(indices):
            raise ValueError("One payload is required per active environment")
        if method == "step":
            if payloads is None:
                raise ValueError("A step requires actions for each active environment")
            return self._batched_step(indices, payloads)
        if not indices:
            return []
        futures = [self.executor.submit(getattr(self.envs[index], method),
                                       *(() if payloads is None else (payloads[slot],)))
                   for slot, index in enumerate(indices)]
        completed, _ = wait(futures, return_when=FIRST_EXCEPTION)
        failure = next(((index, future.exception()) for index, future in zip(indices, futures)
                        if future in completed and future.exception() is not None), None)
        if failure is not None:
            index, error = failure
            for future in futures:
                future.cancel()
            self.close()
            raise RuntimeError(f"Parallel platform {method} failed in environment {index}; "
                               f"collection aborted: {error}") from error
        return [future.result() for future in futures]

    def _batched_step(self, indices, actions):
        # No workers are running at this barrier. Gather and infer on the calling
        # thread, then dispatch prepared commands to the existing socket workers.
        routed = [self.envs[index].route_actions(action)["actions"] for index, action in zip(indices, actions)]
        requests = [request for index, action in zip(indices, routed)
                    for request in self.envs[index].flight_requests(action)]
        if self.flight_controller_name == "baseline":
            controls = FrozenFlightController.controls_many(requests)
        else:
            controls = type(requests[0][0]).controls_many(requests)
        commands = [self.envs[index].prepare_step(action, {"red": controls[2*slot], "blue": controls[2*slot+1]})
                    for slot, (index, action) in enumerate(zip(indices, routed))]
        return self.call(indices, "step_commands", commands)

    def close(self):
        if self.closed:
            return
        self.closed = True
        for env in self.envs:
            try:
                env.close()
            except OSError:
                pass
        self.executor.shutdown(wait=True)
        for process in self.processes:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
