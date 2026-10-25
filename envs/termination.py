"""Competition round end and ranking rules."""

import math

from .observation_encoder import alive


def simulation_time(state):
    times, living_times = [], []
    for side in ("red", "blue"):
        for plane in state.get(side, {}).values():
            if "simulation/sim-time-sec" in plane:
                value = float(plane["simulation/sim-time-sec"])
                if math.isfinite(value):
                    times.append(value)
                    if alive(plane):
                        living_times.append(value)
    # Destroyed aircraft retain sparse/cached telemetry; their old timestamp
    # must not freeze the clock while other aircraft continue the battle.
    return max(living_times or times, default=0.0)


def frame_time(own_groups, tolerance):
    """Read only fresh owner telemetry, before any historical fields are merged.

    Dead aircraft may omit their timestamp. Every live aircraft must supply a
    finite timestamp and all live timestamps must describe the same frame.
    None means no aircraft timestamp is available (normally an all-dead frame).
    """
    living, dead = [], []
    for side, planes in own_groups.items():
        for name, plane in planes.items():
            if not isinstance(plane, dict):
                raise ValueError(f"Invalid current telemetry for {side}/{name}")
            value = plane.get("simulation/sim-time-sec")
            if value is None:
                if alive(plane):
                    raise ValueError(f"Live aircraft {name} has no current timestamp")
                continue
            try:
                value = float(value)
            except (TypeError, ValueError, OverflowError):
                raise ValueError(f"Invalid current timestamp for {name}: {value!r}")
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid current timestamp for {name}: {value!r}")
            (living if alive(plane) else dead).append(value)
    if living and max(living)-min(living) > tolerance:
        raise ValueError(f"Current aircraft/Socket clocks disagree: {min(living)} .. {max(living)}")
    return max(living or dead) if living or dead else None


def in_flight_missiles(state):
    return sum(int(missile.get("Status", 0)) == 0 for missile in state.get("missile", {}).values()
               if isinstance(missile, dict))


def team_score(state, side):
    planes = list(state.get(side, {}).values())
    survivors = [plane for plane in planes if alive(plane)]
    ammunition = sum(int(p.get("SRAAMCurrentNum", 0)) + int(p.get("AMRAAMCurrentNum", 0)) for p in survivors)
    return len(survivors), ammunition


def outcome(state):
    red = team_score(state, "red")
    blue = team_score(state, "blue")
    if red > blue:
        return "red"
    if blue > red:
        return "blue"
    return "tie"


def round_done(state, time_limit_seconds, max_missile_overtime_seconds=None, sim_time=None):
    red_alive = team_score(state, "red")[0]
    blue_alive = team_score(state, "blue")[0]
    # No remaining live target can change the result once both teams are dead.
    if red_alive == 0 and blue_alive == 0:
        return True
    current_time = simulation_time(state) if sim_time is None else sim_time
    time_up = current_time >= time_limit_seconds
    if max_missile_overtime_seconds is not None and current_time >= time_limit_seconds+max_missile_overtime_seconds:
        return True
    pending_launches = any(bool(plane.get("IfPresenceHitting", 0))
                           for side in ("red", "blue") for plane in state.get(side, {}).values())
    return (red_alive == 0 or blue_alive == 0 or time_up) and not pending_launches and in_flight_missiles(state) == 0
