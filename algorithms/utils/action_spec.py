"""Single source for the discrete action-head schema."""

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
with (ROOT / "data" / "schemas" / "action_schema.yaml").open(encoding="utf-8") as _stream:
    ACTION_SCHEMA = yaml.safe_load(_stream)

ACTION_NAMES = tuple(ACTION_SCHEMA["heads"].keys())
ACTION_DIMS = tuple(len(values) for values in ACTION_SCHEMA["heads"].values())
ACTION_WIDTH = sum(ACTION_DIMS)
NEUTRAL_ACTION = tuple(int(value) for value in ACTION_SCHEMA["neutral_action"])
ACTION_OFFSETS = tuple(sum(ACTION_DIMS[:index]) for index in range(len(ACTION_DIMS)))
WEAPON_HEAD = ACTION_NAMES.index("weapon")
WEAPON_OFFSET = ACTION_OFFSETS[WEAPON_HEAD]
WEAPON_CHOICES = tuple(ACTION_SCHEMA["heads"]["weapon"])
FLIGHT_CONTROLLER = ACTION_SCHEMA["flight_controller"]


def validate_actor_action_space(state_dict):
    """Explain incompatible old policy heads before loading a checkpoint."""
    dimensions = tuple(state_dict[f"action.heads.{i}.weight"].shape[0] for i in range(len(ACTION_DIMS)))
    if dimensions != ACTION_DIMS:
        raise ValueError(f"Checkpoint action space {dimensions} differs from {ACTION_DIMS}; "
                         "the action contract changed; train a new high-level policy")


def validate_checkpoint_action_schema(folder):
    path = Path(folder)/"action_schema.yaml"
    if path.is_file():
        with path.open(encoding="utf-8") as stream:
            if yaml.safe_load(stream) != ACTION_SCHEMA:
                raise ValueError("Checkpoint action schema differs from the current flight-controller contract")
