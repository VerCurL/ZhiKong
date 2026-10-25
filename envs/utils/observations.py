"""Decode a compact Actor row once; no hidden coordinates or motion reconstruction."""

import math
import numpy as np


def _digits(value, slots):
    try:
        return [int(digit) for digit in str(int(value)).zfill(slots)[::-1][:slots]]
    except (TypeError, ValueError, OverflowError):
        return [9]*slots


def decode_actor(observation, encoder, living=True, missile_id=None, plane=None, missile=None):
    obs = np.asarray(observation, dtype=np.float64)
    if obs.shape != (encoder.actor_dim,) or not np.all(np.isfinite(obs)):
        raise ValueError("Expected one finite unstandardized compact Actor observation")
    n = encoder.normalization
    plane = plane or {}
    frame = {"alive": bool(living), "height": float(obs[0]*n["altitude_m"]),
             "body_speed": float(np.linalg.norm(obs[5:8])*n["speed_mps"]),
             "cas": float(obs[8]*n["speed_mps"]),
             "short_ammo": float(plane.get("SRAAMCurrentNum", 0)),
             "medium_ammo": float(plane.get("AMRAAMCurrentNum", 0)),
             "alert": bool(plane.get("MissileAlert", 0)),
             "warning_number": int(plane.get("WarningNumber", 0)),
             "aim_mode": int(plane.get("AimMode", -1)),
             "enemies": {}, "missile": None, "missile_id": missile_id}
    in_range = _digits(plane.get("TargetEnterAttackRange", 0), encoder.num_agents)
    lock_value = int(float(plane.get("AMRAAMlockedTarget", 9999)))
    locks = _digits(lock_value, 4)
    start = encoder.self_dim+encoder.teammate_slots*encoder.teammate_dim
    for index in range(encoder.enemy_slots):
        slot = obs[start+index*encoder.enemy_dim:start+(index+1)*encoder.enemy_dim]
        if living and slot[4] > 0:
            frame["enemies"][index] = {"id": index, "closing": float(slot[0]*n["speed_mps"]),
                                       "distance": float(slot[4]*n["distance_m"]),
                                       "ao": float(slot[2]*math.pi), "ta": float(slot[3]*math.pi),
                                       "in_attack_range": bool(in_range[index]),
                                       "radar_lock": index in locks,
                                       "stable_radar_lock": index in locks and lock_value < 1000}
    start += encoder.enemy_slots*encoder.enemy_dim
    slot = obs[start:start+encoder.missile_dim]
    if living and slot[4] > 0:
        frame["missile"] = {"closing": float(slot[0]*n["speed_mps"]),
                             "height": float(slot[1]*n["relative_altitude_m"]),
                             "ao": float(slot[2]*math.pi), "ta": float(slot[3]*math.pi),
                             "distance": float(slot[4]*n["distance_m"]), "side": float(slot[5]),
                             "kind": 0, "age": 0.0}
        if isinstance(missile, dict):
            kind = str(missile_id or "").split("_", 1)[0].upper()
            frame["missile"]["kind"] = int(kind == "AMRAAM")
            frame["missile"]["age"] = float(missile.get("LaunchedTime", 0.0))
    frame["enemy_signature"] = tuple(frame["enemies"])
    return frame
