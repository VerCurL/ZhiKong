"""Load the fixed project configuration layout and CLI overrides."""

from pathlib import Path
import argparse
import math

import yaml


ROOT = Path(__file__).resolve().parent
TRAIN_OUTPUT_ROOT = ROOT / "results"


def load_yaml(path):
    with Path(path).open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return value


def project_configs():
    folder = ROOT / "configs"
    configs = {name: load_yaml(folder / f"{name}.yaml") for name in
               ("platform_profile", "scenario_4v4", "train_mappo", "selfplay")}
    validate_configs(configs)
    return configs


def validate_configs(configs):
    scenario = configs["scenario_4v4"]
    train = configs["train_mappo"]
    if train.get("flight_controller", "baseline") not in ("baseline", "pid"):
        raise ValueError("flight_controller must be baseline or pid")
    if train.get("training_start_mode", "official") not in ("official", "combat_start"):
        raise ValueError("training_start_mode must be official or combat_start")
    if not math.isfinite(float(train.get("combat_start_seconds", 200.0))) or float(train.get("combat_start_seconds", 200.0)) < 0:
        raise ValueError("combat_start_seconds must be nonnegative")
    for phase in ("sampling", "training"):
        threads = train["torch_threads"][phase]
        if isinstance(threads, bool) or not isinstance(threads, int) or threads <= 0:
            raise ValueError(f"torch_threads.{phase} must be a positive integer")
    if not isinstance(train["reward_recorder"]["enabled"], bool):
        raise ValueError("reward_recorder.enabled must be a boolean")
    profile = configs["platform_profile"]
    simulation = profile["simulation"]
    if not 0 < float(simulation["timestamp_tolerance_seconds"]) < float(simulation["decision_seconds_fallback"]):
        raise ValueError("timestamp_tolerance_seconds must be positive and smaller than the nominal decision interval")
    stalled = simulation["max_stalled_steps"]
    if isinstance(stalled, bool) or not isinstance(stalled, int) or stalled < 0:
        raise ValueError("max_stalled_steps must be a nonnegative integer")
    if not 0 <= float(simulation["max_missile_overtime_seconds"]) < float("inf"):
        raise ValueError("max_missile_overtime_seconds must be finite and nonnegative")
    with (ROOT / "data" / "schemas" / "action_schema.yaml").open(encoding="utf-8") as stream:
        action_schema = yaml.safe_load(stream)
    with (ROOT / "data" / "schemas" / "observation_schema.yaml").open(encoding="utf-8") as stream:
        observation_schema = yaml.safe_load(stream)
    agents = int(scenario["aircraft_per_side"])
    if len(scenario["red_ids"]) != agents or len(scenario["blue_ids"]) != agents:
        raise ValueError("scenario aircraft_per_side does not match red_ids/blue_ids")
    if train["red_port"] == train["blue_port"]:
        raise ValueError("red_port and blue_port must be different for dual communication")
    if "rollout_steps" in train:
        raise ValueError("rollout_steps is obsolete; use n_rollout_threads for complete-battle counts")
    if "episodes_per_env" in train:
        raise ValueError("episodes_per_env is obsolete; use n_rollout_threads for total battles and parallel_envs for concurrency")
    for key in ("n_rollout_threads", "parallel_envs", "collection_max_episode_steps",
                "sequence_length", "mini_batches", "ppo_epochs", "max_updates", "log_interval"):
        if int(train[key]) <= 0:
            raise ValueError(f"{key} must be positive")
    if int(train["n_rollout_threads"]) % 2:
        raise ValueError("n_rollout_threads must be even so the learner starts half its battles on each map side")
    initialization = scenario["initialization"]
    if float(initialization["longitude_extent_deg"]) <= 0 or not 0 < float(initialization["longitude_scale_min"]) <= 1:
        raise ValueError("Spawn longitude extent must be positive and longitude_scale_min must be in (0,1]")
    ports = [int(train[key])+index*int(train["port_stride"])
             for index in range(min(int(train["parallel_envs"]), int(train["n_rollout_threads"])))
             for key in ("red_port", "blue_port")]
    if len(set(ports)) != len(ports) or any(not 1 <= port <= 65535 for port in ports):
        raise ValueError("Parallel environment port pairs must be unique and in 1..65535")
    if train["platform_launch"]["enabled"] and train["blue_port"] != train["red_port"]+1:
        raise ValueError("Automatically launched Red=0 Blue=1 platforms require adjacent port pairs")
    if int(train["max_minibatch_samples"]) < int(train["sequence_length"]):
        raise ValueError("max_minibatch_samples must hold at least one recurrent sequence")
    if configs["scenario_4v4"]["initialization"]["save_replay"]:
        raise ValueError("Training keeps trajectories in RAM; save_replay must be false")
    if float(train["sampling_memory_limit_mb"]) <= 0:
        raise ValueError("sampling_memory_limit_mb must be positive")
    action_dims = tuple(len(values) for values in action_schema["heads"].values())
    expected_actor_dim = (observation_schema["self_dim"] +
                          (agents - 1) * observation_schema["teammate_dim"] +
                          agents * observation_schema["enemy_dim"] +
                          observation_schema["missile_slots"] * observation_schema["missile_dim"])
    if observation_schema["critic"]["source"] != "all_eight_actor_observations":
        raise ValueError("Critic must concatenate all eight LAGCFC Actor observations")
    if not isinstance(observation_schema["critic"]["include_remaining_time"], bool):
        raise ValueError("critic.include_remaining_time must be a boolean")
    expected_critic_dim = agents * 2 * expected_actor_dim + int(observation_schema["critic"]["include_remaining_time"])
    if tuple(action_schema["neutral_action"]).__len__() != len(action_dims):
        raise ValueError("neutral_action does not match action heads")
    controller = action_schema["flight_controller"]
    for name, delta in (("altitude", "delta_altitude_m"), ("turn", "delta_heading_deg"), ("speed", "delta_speed_mps")):
        if len(action_schema["heads"][name]) != len(controller[delta]):
            raise ValueError(f"Flight-controller {delta} must match the {name} action head")
    if not 0 <= float(controller["throttle_min"]) < float(controller["throttle_max"]) <= 1:
        raise ValueError("Flight-controller throttle limits must be within 0..1")
    if any(float(scale) <= 0 for scale in controller["observation_scales"].values()):
        raise ValueError("Flight-controller observation scales must be positive")
    if any(not 0 <= int(choice) < dimension for choice, dimension in zip(action_schema["neutral_action"], action_dims)):
        raise ValueError("neutral_action is outside the configured action heads")
    if profile["aircraft"]["initial_life"] <= 0 or profile["aircraft"]["amraam_capacity"] <= 0:
        raise ValueError("platform aircraft capacities must be positive")
    return {"actor_dim": expected_actor_dim, "critic_dim": expected_critic_dim,
            "action_dims": action_dims}


def training_arguments():
    parser = argparse.ArgumentParser(description="Train 4v4 MAPPO on the official platform")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--red-port", type=int)
    parser.add_argument("--blue-port", type=int)
    parser.add_argument("--updates", type=int)
    parser.add_argument("--side", choices=("red", "blue"))
    parser.add_argument("--flight-controller", choices=("baseline", "pid"),
                        help="Fixed lower-level flight controller used after Actor maneuver deltas")
    parser.add_argument("--output", default=str(TRAIN_OUTPUT_ROOT), help="Parent output root; a unique 4v4/mappo/run-* directory is created")
    parser.add_argument("--resume", help="Continue in this run directory (or actor file's directory); append logs, load models/normalization with a fresh optimizer")
    parser.add_argument("--envs", "--parallel-envs", type=int, help="Maximum simultaneous platform environments")
    parser.add_argument("--rollout-episodes", type=int, help="Total complete battles collected before each PPO update")
    parser.add_argument("--no-launch-platforms", action="store_true", help="Connect to already running platform instances")
    return parser.parse_args()
