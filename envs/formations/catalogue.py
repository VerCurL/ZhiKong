"""Post-initialization formation targets and balanced randomized opening plans."""

from itertools import product
from .wall import OFFSETS as WALL
from .box import OFFSETS as BOX
from .pairs import OFFSETS as PAIRS
from .wedge import OFFSETS as WEDGE
from .center import OFFSETS as CENTER

FORMATION_SCHEMA = {
    "version": 2,
    "templates": {"wall": WALL, "box": BOX, "pairs": PAIRS, "wedge": WEDGE, "center": CENTER},
    "parameters": {"lateral_span_m": 16000.0, "depth_m": 6000.0, "height_scale_m": 1000.0,
                   "blend_seconds": 30.0, "position_gain_per_s": .02, "max_correction_mps": 30.0,
                   "heading_deadband_deg": 3.0, "speed_deadband_mps": 3.0,
                   "height_deadband_m": 50.0, "center_release_radius_m": 10000.0},
}
FORMATION_NAMES = tuple(FORMATION_SCHEMA["templates"])


def resolve_formations(selection, rng):
    requested = {side: selection for side in ("red", "blue")} if isinstance(selection, str) else dict(selection)
    if set(requested) != {"red", "blue"}:
        raise ValueError("Formation selection must specify red and blue")
    result = {}
    for side, name in requested.items():
        if name != "random" and name not in FORMATION_NAMES:
            raise ValueError(f"Unknown formation {name!r}; choose random or {FORMATION_NAMES}")
        result[side] = rng.choice(FORMATION_NAMES) if name == "random" else name
    return result


def formation_schedule(count, learner_side, rng):
    """Balanced marginal probabilities and spawn sides; full coverage when divisible."""
    if count <= 0 or count % 2:
        raise ValueError("Balanced openings require a positive even battle count")
    if learner_side not in ("red", "blue"):
        raise ValueError("learner_side must be red or blue")
    combinations = list(product(("left", "right"), FORMATION_NAMES, FORMATION_NAMES))
    if count % len(combinations) == 0:
        plan = combinations*(count//len(combinations))
        rng.shuffle(plan)
    else:
        def balanced_names():
            names = list(FORMATION_NAMES)
            rng.shuffle(names)
            values = [names[i % len(names)] for i in range(count)]
            rng.shuffle(values)
            return values
        spawns = ["left"]*(count//2)+["right"]*(count//2)
        rng.shuffle(spawns)
        plan = list(zip(spawns, balanced_names(), balanced_names()))
    other = "blue" if learner_side == "red" else "red"
    return [{"spawn_side": spawn, "formations": {learner_side: own, other: opponent}}
            for spawn, own, opponent in plan]
