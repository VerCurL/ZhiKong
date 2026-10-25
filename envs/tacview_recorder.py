"""Explicitly enabled ACMI writer; importing this module never creates a file.

Record complete ZK state dictionaries (red/blue/missile), independently of the
environment, policy and sockets. Coordinates follow Tacview's public ACMI 2.2
specification: https://www.tacview.net/documentation/acmi/en/
"""

from datetime import datetime, timezone
import math
from pathlib import Path


def _text(value):
    return str(value).replace("\\", "\\\\").replace(",", "\\,").replace("\r", "").replace("\n", "\\n")


def _number(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _owner_name(value):
    text = str(value).strip()
    if text.startswith(("red_", "blue_")) and text.split("_")[-1].isdigit():
        return text
    try:
        code = str(int(text)).zfill(2)
    except (ValueError, TypeError):
        return None
    if len(code) == 2 and code[0] in "01":
        return f"{'red' if code[0] == '0' else 'blue'}_{int(code[1])}"
    return None


class TacviewRecorder:
    """One buffered file per battle; call record() for each returned state.

    Missing objects are removed as LeftArea, without claiming a kill. Terminal
    missile status and aircraft deaths end their tracks. Start a new recorder
    for a new battle: backwards simulation time is rejected.
    """

    def __init__(self, path, feet_to_meters, reference_time=None, title="ZK platform flight"):
        self.path = Path(path)
        self.ft = float(feet_to_meters)
        if not math.isfinite(self.ft) or self.ft <= 0:
            raise ValueError("feet_to_meters must be positive and finite")
        reference = reference_time or datetime.now(timezone.utc)
        if reference.tzinfo is None:
            raise ValueError("reference_time must include a timezone")
        self._reference = reference.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        self._ids = {}
        self._active = set()
        self._start_time = None
        self._last_time = None
        self.frames = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Unique default names and exclusive creation avoid overwriting a replay.
        self._stream = self.path.open("x", encoding="utf-8-sig", newline="\n", buffering=65536)
        self._stream.write("FileType=text/acmi/tacview\nFileVersion=2.2\n"
                           f"0,ReferenceTime={self._reference}\n"
                           "0,DataSource=ZK,DataRecorder=IntelligentAirCombatChallenge\n"
                           f"0,Title={_text(title)}\n")

    def _id(self, key):
        if key not in self._ids:
            self._ids[key] = format(len(self._ids)+1, "x")
        return self._ids[key]

    def _transform(self, fields, missile=False):
        if missile:
            position = (fields.get("Longitude"), fields.get("Lattitude", fields.get("Latitude")), fields.get("Altitude"))
            angles = (fields.get("Roll"), fields.get("Pitch"), fields.get("Yaw"))
        else:
            position = tuple(fields.get(key) for key in ("position/long-gc-deg", "position/lat-geod-deg", "position/h-sl-ft"))
            angles = (fields.get("attitude/roll-rad"), fields.get("attitude/pitch-rad"), fields.get("attitude/psi-deg"))
        position = [_number(value) for value in position]
        if any(value is None for value in position):
            return None
        if not -180 <= position[0] <= 180 or not -90 <= position[1] <= 90:
            return None
        result = [f"{position[0]:.9f}", f"{position[1]:.9f}", f"{position[2]*self.ft:.3f}"]
        for index, value in enumerate(angles):
            value = _number(value)
            if value is not None and not missile and index < 2:
                value = math.degrees(value)
            result.append("" if value is None else f"{value:.6f}")
        return "|".join(result)

    def record(self, state, sim_time=None):
        if self._stream.closed:
            raise ValueError("Tacview recorder is closed")
        if sim_time is None:
            times = [_number(plane.get("simulation/sim-time-sec"))
                     for side in ("red", "blue") for plane in state.get(side, {}).values()
                     if isinstance(plane, dict)]
            times = [value for value in times if value is not None]
            # Sparse terminal aircraft packets can omit time as well as position.
            sim_time = max(times) if times else self._last_time
        sim_time = _number(sim_time)
        if sim_time is None or sim_time < 0:
            raise ValueError("A finite nonnegative simulation time is required")
        if self._last_time is not None and sim_time < self._last_time:
            raise ValueError("Simulation time moved backward; use a new Tacview file for each battle")
        if self._start_time is None:
            self._start_time = sim_time
        self._last_time = sim_time
        lines = [f"#{sim_time-self._start_time:.6f}"]
        present = set()
        for side in ("red", "blue"):
            for name, plane in state.get(side, {}).items():
                if not isinstance(plane, dict):
                    continue
                key = ("aircraft", str(name))
                present.add(key)
                death, life = _number(plane.get("DeathEvent", 99)), _number(plane.get("LifeCurrent", 1))
                dead = (death is not None and death != 99) or (life is not None and life <= 0)
                transform = self._transform(plane)
                if transform is not None and (not dead or key in self._active):
                    line = f"{self._id(key)},T={transform}"
                    if key not in self._active:
                        color = "Red" if side == "red" else "Blue"
                        line += f",Type=Air+FixedWing,Name=F-16C,CallSign={_text(name)},Color={color},Coalition={color}"
                        self._active.add(key)
                    lines.append(line)
                if dead:
                    if key in self._active:
                        object_id = self._id(key)
                        lines.extend((f"0,Event=Destroyed|{object_id}|Aircraft destroyed", f"-{object_id}"))
                        self._active.remove(key)
        for name, missile in state.get("missile", {}).items():
            if not isinstance(missile, dict):
                continue
            key = ("missile", str(name))
            present.add(key)
            status = _number(missile.get("Status", 0))
            terminal = status is not None and status != 0
            transform = self._transform(missile, missile=True)
            if transform is not None and (not terminal or key in self._active):
                line = f"{self._id(key)},T={transform}"
                if key not in self._active:
                    owner = _owner_name(missile.get("Owner", ""))
                    color = "Red" if owner and owner.startswith("red_") else "Blue" if owner else "Orange"
                    line += f",Type=Weapon+Missile,Name=Missile,CallSign={_text(name)},Color={color}"
                    if owner:
                        line += f",Coalition={color},Parent={self._id(('aircraft', owner))}"
                    self._active.add(key)
                lines.append(line)
            if terminal:
                if key in self._active:
                    object_id = self._id(key)
                    lines.extend((f"0,Event=Message|{object_id}|Missile ended (Status={status:g})", f"-{object_id}"))
                    self._active.remove(key)
        for key in sorted(self._active-present):
            object_id = self._id(key)
            lines.extend((f"0,Event=LeftArea|{object_id}|Object no longer present in platform data", f"-{object_id}"))
            self._active.remove(key)
        self._stream.write("\n".join(lines)+"\n")
        self.frames += 1

    def flush(self):
        self._stream.flush()

    def close(self):
        if not self._stream.closed:
            self._stream.close()

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        self.close()
