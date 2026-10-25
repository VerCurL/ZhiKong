"""Run one training-style combat-start battle without PPO updates.

Examples:
    python tests/test_training_battle.py --resume results/4v4/mappo/run-xxx
    python tests/test_training_battle.py --test-seconds 120

The platform uses PlayMode=0 and remains visible.
"""

import argparse
from contextlib import contextmanager
import math
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from algorithms.mappo.ppo_actor import PPOActor
from algorithms.utils.model_io import load_actor_model
from algorithms.utils.normalization import RunningMoments
from config import ROOT, project_configs
from envs.competition_env import CompetitionEnv
from envs.observation_encoder import ObservationEncoder, alive

class TestSidePolicy:
    """Frozen Actor used only by the one-battle training diagnostic."""

    def __init__(self, path, obs_dim, agents, config, device):
        self.device = device
        if path:
            model = load_actor_model(path)
            self.actor = PPOActor(obs_dim, model["hidden_dim"]).to(device).eval()
            self.actor.load_state_dict(model["state_dict"])
            self.normalizer = RunningMoments(obs_dim)
            self.normalizer.load_state_dict(model["obs_normalizer"])
            self.path = str(model["path"])
        else:
            self.actor = PPOActor(obs_dim, int(config["actor_hidden"])).to(device).eval()
            self.normalizer = RunningMoments(obs_dim)
            self.path = "random_actor"
        self.actor.requires_grad_(False)
        self.agents = agents
        self.hidden = np.zeros((agents, self.actor.hidden_dim), dtype=np.float32)
        self.reset_masks = np.zeros((agents, 1), dtype=np.float32)

    @torch.no_grad()
    def actions(self, observations, legal, active):
        active = np.asarray(active, dtype=np.float32).reshape(self.agents, 1)
        self.hidden *= active
        action, _, _, hidden = self.actor(
            torch.as_tensor(self.normalizer.normalize(observations), dtype=torch.float32, device=self.device),
            torch.as_tensor(self.hidden, dtype=torch.float32, device=self.device),
            torch.as_tensor(self.reset_masks*active, dtype=torch.float32, device=self.device),
            torch.as_tensor(legal, dtype=torch.bool, device=self.device),
            deterministic=True,
        )
        self.hidden = hidden.cpu().numpy()*active
        self.reset_masks = active
        return action.cpu().numpy()


@contextmanager
def visible_test_platform(config, scenario, no_launch=False):
    """Start PlayMode=0 visibly for the one-battle diagnostic."""
    launch = config["platform_launch"]
    executable = Path(launch["executable"])
    if not executable.is_absolute():
        executable = ROOT/executable
    processes = []
    try:
        if no_launch:
            yield
            return
        if not executable.is_file():
            raise FileNotFoundError(executable)
        env_count = 1
        for index in range(env_count):
            for key in ("red_port", "blue_port"):
                port = int(config[key])+index*int(config["port_stride"])
                with socket.socket() as probe:
                    if os.name == "nt":
                        probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
                    probe.bind((config["host"], port))
        port = int(config["red_port"])
        command = [str(executable), f"Ip={config['host']}", f"Port={port}",
                   "Red=0", "Blue=1", f"RedNum={int(scenario['aircraft_per_side'])}",
                   f"BlueNum={int(scenario['aircraft_per_side'])}", "PlayMode=0",
                   f"Scenes={launch['scene']}", "-logFile", os.devnull]
        # Deliberately keep the platform window visible in test mode.
        processes.append(subprocess.Popen(command, cwd=executable.parent,
                                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        yield
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def run_test_mode(args, configs):
    train, scenario, profile = configs["train_mappo"], configs["scenario_4v4"], configs["platform_profile"]
    if args.test_seconds <= 0 or not math.isfinite(args.test_seconds):
        raise ValueError("--test-seconds must be a positive finite number")
    model_source = args.test_model or args.resume
    if model_source:
        red_path = (ROOT/model_source).resolve()
        opponent_source = args.test_opponent or model_source
        blue_path = (ROOT/opponent_source).resolve()
    else:
        red_path = blue_path = None
    device = "cuda" if train["device"] == "cuda" and torch.cuda.is_available() else "cpu"
    agents = int(scenario["aircraft_per_side"])
    obs_dim = ObservationEncoder(num_agents=agents).actor_dim
    policies = {"red": TestSidePolicy(red_path, obs_dim, agents, train, device),
                "blue": TestSidePolicy(blue_path, obs_dim, agents, train, device)}
    test_config = dict(train)
    test_config["platform_launch"] = dict(train["platform_launch"], enabled=False)
    print(f"训练测试模式：PlayMode=0，combat_start={train.get('training_start_mode')}，只运行一局，不更新网络。", flush=True)
    print(f"红方 Actor：{policies['red'].path}", flush=True)
    print(f"蓝方 Actor：{policies['blue'].path}", flush=True)
    summaries = {side: {"center": 0, "policy": 0, "boundary": 0, "dead": 0} for side in ("red", "blue")}
    first_policy = {}
    with visible_test_platform(test_config, scenario, args.no_launch_platforms):
        env = CompetitionEnv(profile, scenario, host=test_config["host"],
                             red_port=test_config["red_port"], blue_port=test_config["blue_port"],
                             seed=int(test_config["seed"]),
                             connect_timeout=test_config["platform_launch"]["connect_timeout_seconds"],
                             socket_timeout=test_config["platform_launch"]["socket_timeout_seconds"],
                             flight_controller=test_config.get("flight_controller", "baseline"),
                             training_start_mode=test_config.get("training_start_mode", "combat_start"),
                             combat_start_seconds=test_config.get("combat_start_seconds", 200.0))
        try:
            opening = {"spawn_side": "left",
                       "formations": {"red": args.test_formation, "blue": args.test_formation}}
            _, _, info = env.reset_training(opening)
            print(f"开局模式={info.get('training_start_mode')} formations={info['formations']}", flush=True)
            start_time = float(info["sim_time"])
            next_print = start_time
            steps = 0
            while not info["episode_done"] and steps < int(train["collection_max_episode_steps"]):
                now = float(info["sim_time"])
                if now-start_time >= args.test_seconds:
                    break
                actions = {}
                for side, policy in policies.items():
                    view = env._side_states[side] if env.dual_socket else env.state
                    active = [float(alive(view[side].get(f"{side}_{index}", {}))) for index in range(agents)]
                    actions[side] = policy.actions(info["side_observations"][side],
                                                   info["side_action_masks"][side], active)
                _, _, _, _, info = env.step(actions)
                steps += 1
                for side in ("red", "blue"):
                    for index, mode in enumerate(info["control_modes"][side]):
                        summaries[side][mode] += 1
                        if mode == "policy" and f"{side}_{index}" not in first_policy:
                            frame = env._frames[side][index]
                            first_policy[f"{side}_{index}"] = {
                                "sim_time": float(info["sim_time"]),
                                "visible_enemies": len(frame["enemies"]),
                                "missile": frame["missile"] is not None,
                            }
                if float(info["sim_time"]) >= next_print:
                    print(f"t={float(info['sim_time'])-start_time:.1f}s steps={steps} "
                          f"alive=red:{info['red_score'][0]} blue:{info['blue_score'][0]} "
                          f"modes={summaries}", flush=True)
                    next_print += 10.0
            print(f"测试结束：sim_time={info['sim_time']:.2f}s steps={steps} "
                  f"outcome={info['outcome']} first_policy={first_policy} modes={summaries}", flush=True)
        finally:
            env.close()


def main():
    configs=project_configs()
    train=configs["train_mappo"]
    launch=train["platform_launch"]
    parser=argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--resume", help="Actor run directory or actor checkpoint used by both sides")
    parser.add_argument("--test-model", help="Red Actor run directory or actor checkpoint")
    parser.add_argument("--test-opponent", help="Blue Actor run directory or actor checkpoint")
    parser.add_argument("--test-seconds", type=float, default=900.0)
    parser.add_argument("--test-formation", choices=("random", "center", "wall", "box", "pairs", "wedge"), default="center")
    parser.add_argument("--flight-controller", choices=("baseline", "pid"), default=train.get("flight_controller", "baseline"))
    parser.add_argument("--host", default=train["host"])
    parser.add_argument("--red-port", type=int, default=38888)
    parser.add_argument("--blue-port", type=int)
    parser.add_argument("--scene", type=int, default=launch["scene"])
    parser.add_argument("--executable", type=Path, default=ROOT/launch["executable"])
    parser.add_argument("--no-launch-platforms", action="store_true")
    args=parser.parse_args()
    args.blue_port=args.red_port+1 if args.blue_port is None else args.blue_port
    args.executable=(ROOT/args.executable).resolve()
    if args.red_port == args.blue_port:
        parser.error("红蓝端口必须不同")
    if args.no_launch_platforms:
        train["platform_launch"]["enabled"]=False
    train["host"],train["red_port"],train["blue_port"]=args.host,args.red_port,args.blue_port
    train["flight_controller"]=args.flight_controller
    train["platform_launch"]["executable"]=str(args.executable)
    train["platform_launch"]["scene"]=args.scene
    run_test_mode(args, configs)


if __name__ == "__main__":
    main()
