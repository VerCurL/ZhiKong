"""Compact relative Actor observations and own-team concatenated Critic input."""

import math
from pathlib import Path

import numpy as np
import yaml

from .utils.utils import decode_aircraft_id
from .model.flight_controller import calibrated_airspeed_mps

_ROOT = Path(__file__).resolve().parents[1]
with (_ROOT / "data" / "schemas" / "observation_schema.yaml").open(encoding="utf-8") as _stream:
    OBSERVATION_SCHEMA = yaml.safe_load(_stream)
with (_ROOT / "configs" / "platform_profile.yaml").open(encoding="utf-8") as _stream:
    PLATFORM_PROFILE = yaml.safe_load(_stream)
with (_ROOT / "configs" / "scenario_4v4.yaml").open(encoding="utf-8") as _stream:
    SCENARIO = yaml.safe_load(_stream)

SELF_DIM = int(OBSERVATION_SCHEMA["self_dim"])
TEAM_DIM = int(OBSERVATION_SCHEMA["teammate_dim"])
ENEMY_DIM = int(OBSERVATION_SCHEMA["enemy_dim"])
MISSILE_DIM = int(OBSERVATION_SCHEMA["missile_dim"])
MISSILE_SLOTS = int(OBSERVATION_SCHEMA["missile_slots"])
FT = float(PLATFORM_PROFILE["units"]["feet_to_meters"])


def alive(plane):
    return bool(plane) and int(plane.get("DeathEvent", 99)) == 99 and float(plane.get("LifeCurrent", 200)) > 0


def digits_mask(value, count):
    try:
        return [c == "1" for c in str(abs(int(value))).zfill(count)[::-1][:count]]
    except (TypeError, ValueError):
        return [False] * count


def position(plane, feet_to_meters=FT):
    return (float(plane.get("position/lat-geod-deg", 0.0)),
            float(plane.get("position/long-gc-deg", 0.0)),
            float(plane.get("position/h-sl-ft", 0.0)) * feet_to_meters)


def offset(own, other, feet_to_meters=FT,
           earth_meters_per_degree=PLATFORM_PROFILE["units"]["earth_meters_per_degree"]):
    lat0, lon0, h0 = position(own, feet_to_meters)
    lat1, lon1, h1 = position(other, feet_to_meters)
    north = (lat1-lat0)*earth_meters_per_degree
    east = (lon1-lon0)*earth_meters_per_degree*math.cos(math.radians(lat0))
    up = h1-h0
    return north, east, up, math.sqrt(north*north+east*east+up*up)


def velocity(plane, feet_to_meters=FT):
    return np.array([float(plane.get("velocities/v-north-fps", 0.0))*feet_to_meters,
                     float(plane.get("velocities/v-east-fps", 0.0))*feet_to_meters,
                     -float(plane.get("velocities/v-down-fps", 0.0))*feet_to_meters])


def _motion_available(plane):
    fields = ("position/lat-geod-deg", "position/long-gc-deg", "position/h-sl-ft",
              "velocities/v-north-fps", "velocities/v-east-fps", "velocities/v-down-fps")
    try:
        return all(key in plane and math.isfinite(float(plane[key])) for key in fields)
    except (TypeError, ValueError):
        return False


def _relative_features(relative, own_velocity, target_velocity, normalization, relative_speed=None):
    r = np.asarray(relative, dtype=np.float64)
    own_v, target_v = np.asarray(own_velocity), np.asarray(target_velocity)
    distance = float(np.linalg.norm(r))
    denominator = max(distance, 1e-8)
    ao = math.acos(float(np.clip(np.dot(r, own_v)/max(distance*np.linalg.norm(own_v), 1e-8), -1, 1)))
    ta = math.acos(float(np.clip(np.dot(r, target_v)/max(distance*np.linalg.norm(target_v), 1e-8), -1, 1)))
    if relative_speed is None:
        relative_speed = float(np.linalg.norm(target_v)-np.linalg.norm(own_v))
    side = float(np.sign(own_v[0]*r[1]-own_v[1]*r[0]))
    return [relative_speed/normalization["speed_mps"], float(r[2])/normalization["relative_altitude_m"],
            ao/math.pi, ta/math.pi, distance/normalization["distance_m"], side]


def angle_features(own, other, feet_to_meters=FT,
                   earth_meters_per_degree=PLATFORM_PROFILE["units"]["earth_meters_per_degree"]):
    relative = offset(own, other, feet_to_meters, earth_meters_per_degree)[:3]
    own_velocity = velocity(own, feet_to_meters)
    other_velocity = velocity(other, feet_to_meters)
    relative_speed = (float(other.get("velocities/u-fps", 0.0))-
                      float(own.get("velocities/u-fps", 0.0)))*feet_to_meters
    values = _relative_features(relative, own_velocity, other_velocity,
                                OBSERVATION_SCHEMA["normalization"], relative_speed)
    return values[2]*math.pi, values[3]*math.pi, values[0]*OBSERVATION_SCHEMA["normalization"]["speed_mps"]


def _self_features(name, plane, tracker, sim_time, short_mode, feet_to_meters=FT,
                   normalization=None, life_max=None, sraam_capacity=None, amraam_capacity=None):
    if not alive(plane):
        return [0.0]*SELF_DIM
    n = normalization or OBSERVATION_SCHEMA["normalization"]
    sraam_capacity = PLATFORM_PROFILE["aircraft"]["sraam_capacity"] if sraam_capacity is None else sraam_capacity
    amraam_capacity = PLATFORM_PROFILE["aircraft"]["amraam_capacity"] if amraam_capacity is None else amraam_capacity
    roll, pitch = float(plane.get("attitude/roll-rad", 0)), float(plane.get("attitude/pitch-rad", 0))
    return [position(plane, feet_to_meters)[2]/n["altitude_m"],
            math.sin(roll), math.cos(roll), math.sin(pitch), math.cos(pitch),
            *(float(plane.get("velocities/"+axis+"-fps", 0))*feet_to_meters/n["speed_mps"] for axis in ("u", "v", "w")),
            calibrated_airspeed_mps(plane, feet_to_meters)/n["speed_mps"]]


def _missile_position(missile):
    return {"position/lat-geod-deg": missile["Lattitude"],
            "position/long-gc-deg": missile["Longitude"],
            "position/h-sl-ft": missile["Altitude"]}


def _missile_velocity(missile, feet_to_meters):
    speed = float(missile["Speed"])*feet_to_meters
    yaw, pitch = math.radians(float(missile["Yaw"])), math.radians(float(missile["Pitch"]))
    return np.array([speed*math.cos(pitch)*math.cos(yaw), speed*math.cos(pitch)*math.sin(yaw), speed*math.sin(pitch)])


class ObservationEncoder:
    def __init__(self, short_mode=None, detection_m=None, missile_detection_m=None,
                 num_agents=None, schema=None, feet_to_meters=None, earth_meters_per_degree=None,
                 life_max=None, sraam_capacity=None, amraam_capacity=None,
                 filter_visibility=True, battle_seconds=None):
        self.schema = schema or OBSERVATION_SCHEMA
        self.schema_version = int(self.schema["version"])
        self.selected_missiles = {}
        self.num_agents = int(num_agents if num_agents is not None else SCENARIO["aircraft_per_side"])
        self.teammate_slots, self.enemy_slots = self.num_agents-1, self.num_agents
        self.self_dim, self.teammate_dim = int(self.schema["self_dim"]), int(self.schema["teammate_dim"])
        self.enemy_dim, self.missile_dim = int(self.schema["enemy_dim"]), int(self.schema["missile_dim"])
        self.missile_slots = int(self.schema["missile_slots"])
        if self.schema_version != 6 or (self.self_dim, self.teammate_dim, self.enemy_dim, self.missile_slots, self.missile_dim) != (9, 6, 6, 1, 6):
            raise ValueError("Observation version 6 requires the LAGCFC 9/6/6/1x6 feature layout")
        self.actor_dim = self.self_dim+self.teammate_slots*self.teammate_dim+self.enemy_slots*self.enemy_dim+self.missile_slots*self.missile_dim
        self.critic_time = bool(self.schema["critic"]["include_remaining_time"])
        self.critic_dim = self.num_agents*2*self.actor_dim+int(self.critic_time)
        self.ft = float(feet_to_meters if feet_to_meters is not None else PLATFORM_PROFILE["units"]["feet_to_meters"])
        self.earth_meters_per_degree = float(earth_meters_per_degree if earth_meters_per_degree is not None else PLATFORM_PROFILE["units"]["earth_meters_per_degree"])
        self.life_max = float(life_max if life_max is not None else PLATFORM_PROFILE["aircraft"]["initial_life"])
        self.sraam_capacity = float(sraam_capacity if sraam_capacity is not None else PLATFORM_PROFILE["aircraft"]["sraam_capacity"])
        self.amraam_capacity = float(amraam_capacity if amraam_capacity is not None else PLATFORM_PROFILE["aircraft"]["amraam_capacity"])
        self.normalization = self.schema["normalization"]
        self.filter_visibility = bool(filter_visibility)
        self.short_mode = PLATFORM_PROFILE["weapon"]["aim_mode_short"] if short_mode is None else short_mode
        self.detection_m = PLATFORM_PROFILE["limits"]["enemy_detection_m"] if detection_m is None else detection_m
        self.missile_detection_m = PLATFORM_PROFILE["limits"]["missile_detection_m"] if missile_detection_m is None else missile_detection_m
        self.battle_seconds = float(PLATFORM_PROFILE["limits"]["battle_seconds"] if battle_seconds is None else battle_seconds)

    def visible_enemies(self, state, side, own_name, filter_visibility=None):
        own = state.get(side, {}).get(own_name, {})
        enemy_side = "blue" if side == "red" else "red"
        seen = digits_mask(own.get("TargetIntoView", 0), self.num_agents)
        filtering = self.filter_visibility if filter_visibility is None else filter_visibility
        result = {}
        for i in range(self.num_agents):
            name = f"{enemy_side}_{i}"
            enemy = state.get(enemy_side, {}).get(name)
            if enemy and (not filtering or (seen[i] and offset(own, enemy, self.ft, self.earth_meters_per_degree)[3] <= self.detection_m)):
                result[name] = enemy
        return result

    def _geometry(self, own, target):
        return _relative_features(
            offset(own, target, self.ft, self.earth_meters_per_degree)[:3],
            velocity(own, self.ft), velocity(target, self.ft), self.normalization,
            (float(target.get("velocities/u-fps", 0.0))-
             float(own.get("velocities/u-fps", 0.0)))*self.ft)

    def _missile_features(self, state, side, own_name, own):
        self.selected_missiles[own_name] = None
        enemy_side = "blue" if side == "red" else "red"
        candidates = []
        for identifier, missile in state.get("missile", {}).items():
            if not isinstance(missile, dict) or int(missile.get("Status", -1)) != 0:
                continue
            owner = decode_aircraft_id(missile.get("Owner"))
            if not owner or not owner.startswith(enemy_side+"_") or decode_aircraft_id(missile.get("Target")) != own_name:
                continue
            fields = ("Longitude", "Lattitude", "Altitude", "Speed", "Yaw", "Pitch")
            try:
                if not all(key in missile and math.isfinite(float(missile[key])) for key in fields):
                    continue
                if float(missile["Speed"]) < 0:
                    continue
                relative = offset(own, _missile_position(missile), self.ft, self.earth_meters_per_degree)[:3]
                missile_velocity = _missile_velocity(missile, self.ft)
                own_velocity = velocity(own, self.ft)
                relative_speed = float(np.linalg.norm(missile_velocity)-
                                       float(own.get("velocities/u-fps", 0.0))*self.ft)
                geometry = _relative_features(relative, own_velocity, missile_velocity,
                                              self.normalization, relative_speed)
            except (TypeError, ValueError):
                continue
            distance, closing = geometry[4]*self.normalization["distance_m"], -geometry[0]*self.normalization["speed_mps"]
            if self.filter_visibility and distance > self.missile_detection_m:
                continue
            # LAGCFC keeps one most dangerous incoming missile by time to go.
            time_to_go = distance/closing if closing > 0 else math.inf
            candidates.append(((time_to_go, distance, str(identifier)), geometry))
        if not candidates:
            return [0.0]*self.missile_dim
        selected = min(candidates, key=lambda item: item[0])
        self.selected_missiles[own_name] = selected[0][2]
        return selected[1]

    def encode_actor(self, state, side, tracker, sim_time):
        if side not in ("red", "blue"):
            raise ValueError("side must be red or blue")
        own_team, enemy_side = state.get(side, {}), "blue" if side == "red" else "red"
        outputs, n = [], self.normalization
        for index in range(self.num_agents):
            name = f"{side}_{index}"
            own = own_team.get(name, {})
            if not alive(own):
                self.selected_missiles[name] = None
                outputs.append(np.zeros(self.actor_dim, dtype=np.float32))
                continue
            if not _motion_available(own):
                raise ValueError(f"Live aircraft {name} is missing finite position/velocity inputs")
            vector = _self_features(name, own, tracker, sim_time, self.short_mode, self.ft, n,
                                    self.life_max, self.sraam_capacity, self.amraam_capacity)
            for j in range(self.num_agents):
                if j == index:
                    continue
                ally = own_team.get(f"{side}_{j}", {})
                living = alive(ally)
                geometry = self._geometry(own, ally) if living and _motion_available(ally) else [0.0]*self.teammate_dim
                vector.extend(geometry)
            visible = self.visible_enemies(state, side, name)
            for j in range(self.num_agents):
                target_name = f"{enemy_side}_{j}"
                target = visible.get(target_name, {})
                if alive(target) and _motion_available(target):
                    geometry = self._geometry(own, target)
                else:
                    geometry = [0.0]*self.enemy_dim
                vector.extend(geometry)
            vector.extend(self._missile_features(state, side, name, own))
            if len(vector) != self.actor_dim or not np.all(np.isfinite(vector)):
                raise ValueError("Actor observation violates the compact finite feature contract")
            encoded = np.asarray(vector, dtype=np.float32)
            # LAGCFC clips the aircraft/team/enemy prefix (51 values) before
            # appending the missile slot; the missile geometry is left raw.
            encoded[:self.actor_dim-self.missile_dim] = np.clip(
                encoded[:self.actor_dim-self.missile_dim], -n["clip"], n["clip"])
            outputs.append(encoded)
        return np.stack(outputs)

    def encode_critic(self, actor_observations, sim_time):
        """Concatenate all eight LAGCFC local Actor rows for centralized V(s)."""
        observations = np.asarray(actor_observations, dtype=np.float32)
        expected = self.num_agents*2
        if observations.shape != (expected, self.actor_dim) or not np.all(np.isfinite(observations)):
            raise ValueError("Critic requires all eight finite Actor observation rows")
        vector = observations.reshape(-1)
        return np.repeat(vector[None, :], self.num_agents, axis=0).astype(np.float32)
