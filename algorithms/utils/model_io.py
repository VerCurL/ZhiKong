"""Model-only persistence shared by training, evaluation and submission export."""

from pathlib import Path

import torch

from .action_spec import ACTION_SCHEMA, validate_actor_action_space
from envs.observation_encoder import OBSERVATION_SCHEMA
from envs.control_rules import EXECUTION_SCHEMA
from envs.formations import FORMATION_SCHEMA
from envs.model import FLIGHT_CONTROLLER_NAMES


def _action_schema_compatible(saved, current):
    """Compare action semantics while allowing the baseline asset path to move."""
    if not isinstance(saved, dict) or not isinstance(current, dict):
        return False
    saved = {**saved, "flight_controller": dict(saved.get("flight_controller", {}))}
    current = {**current, "flight_controller": dict(current.get("flight_controller", {}))}
    saved["flight_controller"].pop("model_path", None)
    current["flight_controller"].pop("model_path", None)
    return saved == current


def actor_file(path):
    path = Path(path)
    if path.is_dir():
        for name in ("actor_latest.pt", "actor.pt"):
            if (path/name).is_file():
                return path/name
        raise FileNotFoundError(f"No actor_latest.pt or actor.pt in {path}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_actor_model(path):
    """Read a self-contained model matching action and observation semantics."""
    path = actor_file(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if "state_dict" in payload:
        state = payload["state_dict"]
        if not _action_schema_compatible(payload.get("action_schema"), ACTION_SCHEMA):
            raise ValueError("Actor action schema differs from the current weapon/action contract; train a new policy")
        if payload.get("observation_schema") != OBSERVATION_SCHEMA:
            raise ValueError(f"Actor observation schema differs from version {OBSERVATION_SCHEMA['version']}; train a new policy")
        if payload.get("execution_schema") != EXECUTION_SCHEMA:
            raise ValueError("Actor execution rules differ from the current control contract; train a new policy")
        stats = payload["obs_normalizer"]
        update = int(payload["update"])
    else:
        raise ValueError(f"Legacy actor has no observation schema for version {OBSERVATION_SCHEMA['version']}; train a new policy")
    validate_actor_action_space(state)
    if payload.get("flight_controller", "baseline") not in FLIGHT_CONTROLLER_NAMES:
        raise ValueError("Actor checkpoint contains an unknown flight controller")
    hidden_dim, obs_dim = state["base.net.0.weight"].shape
    return {"path": path, "state_dict": state, "obs_normalizer": stats,
            "update": update, "hidden_dim": hidden_dim, "obs_dim": obs_dim,
            "formation_schema": payload.get("formation_schema"),
            "flight_controller": payload.get("flight_controller", "baseline")}


def load_critic_model(folder):
    folder = Path(folder)
    path = folder/"critic_latest.pt"
    if not path.is_file():
        path = folder/"critic.pt"
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if "state_dict" in payload:
        return payload["state_dict"], int(payload["update"])
    return payload, None


def actor_payload(actor, normalizer, update, flight_controller="baseline"):
    stats = normalizer.state_dict()
    return {"state_dict": {key: tensor.detach().cpu() for key, tensor in actor.state_dict().items()},
            "obs_normalizer": {"mean": torch.as_tensor(stats["mean"]).clone(),
                               "var": torch.as_tensor(stats["var"]).clone(), "count": float(stats["count"])},
            "update": int(update), "action_schema": ACTION_SCHEMA, "observation_schema": OBSERVATION_SCHEMA,
            "execution_schema": EXECUTION_SCHEMA, "formation_schema": FORMATION_SCHEMA,
            "flight_controller": flight_controller}
