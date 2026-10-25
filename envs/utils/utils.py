"""Pure numeric helpers. These functions never read hidden platform state."""

import math
import numpy as np


def clip(value, low=-1.0, high=1.0):
    return float(np.clip(value, low, high))


def sigmoid(value):
    return 1.0/(1.0+math.exp(-clip(value, -60.0, 60.0)))


def softmax_score(scores, beta):
    values = np.asarray(scores, dtype=np.float64)
    if not values.size:
        return 0.0
    weights = np.exp(beta*(values-values.max()))
    return float(np.dot(weights, values)/weights.sum())


def ema(previous, current, dt, tau):
    decay = math.exp(-dt/tau)
    return decay*previous+(1-decay)*current


def interval_distance(value, low, high):
    return max(low-value, 0.0, value-high)


def decode_aircraft_id(value):
    """ZK 00..03 are red; 10..13 are blue, including numeric JSON values."""
    try:
        number = float(value)
        if not math.isfinite(number) or number != int(number):
            return None
        code = str(int(number)).zfill(2)
    except (TypeError, ValueError, OverflowError):
        return None
    if len(code) == 2 and code[0] in "01" and code[1] in "0123":
        return ("red" if code[0] == "0" else "blue")+"_"+code[1]
    return None


def has_locked_target(value, agents):
    try:
        parsed = float(value)
        if not math.isfinite(parsed) or parsed != int(parsed):
            return False
        number = int(parsed)
    except (TypeError, ValueError, OverflowError):
        return False
    return number >= 0 and any(character.isdigit() and int(character) < agents for character in str(number))


def local_position(latitude, longitude, altitude, profile):
    field = profile["battlefield"]
    scale = profile["units"]["earth_meters_per_degree"]
    return np.array([(latitude-field["center_latitude_deg"])*scale,
                     (longitude-field["center_longitude_deg"])*scale*math.cos(math.radians(field["center_latitude_deg"])),
                     altitude], dtype=np.float64)


def body_velocity_neu(body, yaw, pitch, roll):
    cy, sy, ct, st, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch), math.cos(roll), math.sin(roll)
    matrix = np.array([[ct*cy, sp*st*cy-cp*sy, cp*st*cy+sp*sy],
                       [ct*sy, sp*st*sy+cp*cy, cp*st*sy-sp*cy],
                       [st, -sp*ct, -cp*ct]])
    return matrix.dot(body)


def missile_relative_position(slot, normalization):
    distance = float(slot[1])*normalization["missile_distance_m"]
    height = float(slot[3])*normalization["altitude_m"]
    if distance < 0 or abs(height) > distance+1e-3:
        return None
    horizontal = math.sqrt(max(0.0, distance*distance-height*height))
    bearing = float(slot[2])*math.pi
    return np.array([horizontal*math.cos(bearing), horizontal*math.sin(bearing), height])


def closest_approach(relative, velocity, horizon):
    distance = float(np.linalg.norm(relative))
    square = float(np.dot(velocity, velocity))
    time = clip(-float(np.dot(relative, velocity))/max(square, 1e-9), 0.0, horizon)
    miss = float(np.linalg.norm(relative+time*velocity))
    closing = -float(np.dot(relative, velocity))/max(distance, 1e-9)
    return closing, time, miss
