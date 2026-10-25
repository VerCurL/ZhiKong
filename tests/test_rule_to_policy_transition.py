"""Measure when the current opening rules hand aircraft to the Actor policy.

Examples:
    python tests/test_rule_to_policy_transition.py --max-seconds 180
    python tests/test_rule_to_policy_transition.py --formation random --red-model <actor.pt> --blue-model <actor.pt>
    python tests/test_rule_to_policy_transition.py --no-launch-platform --red-port 19888 --blue-port 19889

The diagnostic does not train or write trajectories. It writes one compact JSON
summary and prints the first ``center -> policy`` transition for each aircraft.
"""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from algorithms.utils.action_spec import NEUTRAL_ACTION
from config import ROOT, project_configs
from envs.competition_env import CompetitionEnv
from envs.formations import FORMATION_NAMES
from envs.model import FLIGHT_CONTROLLER_NAMES
from envs.observation_encoder import ObservationEncoder, alive
from test_policy_battle import SidePolicy


def arguments():
    configs = project_configs()
    train = configs["train_mappo"]
    launch = train["platform_launch"]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--red-model", type=Path, help="Optional trained Actor; omit to send neutral actions")
    parser.add_argument("--blue-model", type=Path, help="Optional trained Actor; omit to send neutral actions")
    parser.add_argument("--host", default=train["host"])
    parser.add_argument("--red-port", type=int, default=20888)
    parser.add_argument("--blue-port", type=int)
    parser.add_argument("--no-launch-platform", action="store_true")
    parser.add_argument("--executable", type=Path, default=ROOT/launch["executable"])
    parser.add_argument("--scene", type=int, default=launch["scene"])
    parser.add_argument("--play-mode", type=int, choices=(0, 1), default=0,
                        help="Platform mode; 0 is headless simulation, 1 shows the platform window")
    parser.add_argument("--formation", choices=tuple(dict.fromkeys(("center", "random")+FORMATION_NAMES)), default="center")
    parser.add_argument("--flight-controller", choices=FLIGHT_CONTROLLER_NAMES,
                        default=train.get("flight_controller", "baseline"))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-seconds", type=float, default=180.0,
                        help="Stop after this simulation time if all aircraft have not entered policy control")
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--print-interval", type=float, default=5.0)
    parser.add_argument("--connect-timeout", type=float, default=launch["connect_timeout_seconds"])
    parser.add_argument("--socket-timeout", type=float, default=launch["socket_timeout_seconds"])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if (args.red_model is None) != (args.blue_model is None):
        parser.error("请同时提供 --red-model 和 --blue-model，或都不提供")
    if args.red_model is not None:
        args.red_model = (ROOT/args.red_model).resolve()
        args.blue_model = (ROOT/args.blue_model).resolve()
    args.executable = (ROOT/args.executable).resolve()
    args.blue_port = args.red_port+1 if args.blue_port is None else args.blue_port
    if args.red_port == args.blue_port or any(not 1 <= p <= 65535 for p in (args.red_port, args.blue_port)):
        parser.error("红蓝端口必须不同，且位于 1..65535")
    if not args.no_launch_platform and (args.host not in ("127.0.0.1", "localhost") or args.blue_port != args.red_port+1):
        parser.error("自动启动要求 localhost 和相邻红蓝端口")
    for name in ("max_seconds", "print_interval", "connect_timeout", "socket_timeout"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"{name} 必须为正数且有限")
    if args.max_steps <= 0:
        parser.error("max-steps 必须为正整数")
    if args.seed is None:
        args.seed = secrets.randbits(32)
    if args.output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        args.output = ROOT/"results"/"diagnostics"/f"rule_to_policy_{stamp}.json"
    else:
        args.output = (ROOT/args.output).resolve()
    return args, configs


@contextmanager
def platform(args, agents):
    process = None
    try:
        if not args.no_launch_platform:
            if not args.executable.is_file():
                raise FileNotFoundError(args.executable)
            for port in (args.red_port, args.blue_port):
                with socket.socket() as probe:
                    if os.name == "nt":
                        probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
                    probe.bind((args.host, port))
            command = [str(args.executable), f"Ip={args.host}", f"Port={args.red_port}",
                       "Red=0", "Blue=1", f"RedNum={agents}", f"BlueNum={agents}",
                       f"PlayMode={args.play_mode}", f"Scenes={args.scene}", "-logFile", os.devnull]
            options = {}
            if os.name == "nt" and args.play_mode == 0:
                startup = subprocess.STARTUPINFO()
                startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startup.wShowWindow = subprocess.SW_HIDE
                options = {"startupinfo": startup, "creationflags": subprocess.CREATE_NO_WINDOW}
            process = subprocess.Popen(command, cwd=args.executable.parent,
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **options)
        yield
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def position(plane, profile):
    field, units = profile["battlefield"], profile["units"]
    north = (float(plane["position/lat-geod-deg"])-field["center_latitude_deg"])*units["earth_meters_per_degree"]
    east = ((float(plane["position/long-gc-deg"])-field["center_longitude_deg"])*
            units["earth_meters_per_degree"]*math.cos(math.radians(field["center_latitude_deg"])))
    altitude = float(plane.get("position/h-sl-ft", 0.0))*units["feet_to_meters"]
    return {"latitude_deg": float(plane["position/lat-geod-deg"]),
            "longitude_deg": float(plane["position/long-gc-deg"]),
            "altitude_m": altitude, "north_m": north, "east_m": east,
            "distance_to_center_m": math.hypot(north, east)}


def aircraft_snapshot(env, side, index, profile):
    view = env._side_states[side] if env.dual_socket else env.state
    plane = view.get(side, {}).get(f"{side}_{index}", {})
    return position(plane, profile) if plane else None


def policy_actions(args, env, info, policies):
    if policies is None:
        neutral = np.tile(np.asarray(NEUTRAL_ACTION, dtype=np.int64), (env.num_agents, 1))
        return {"red": neutral.copy(), "blue": neutral.copy()}
    actions = {}
    for side, policy in policies.items():
        view = env._side_states[side] if env.dual_socket else env.state
        active = [float(alive(view.get(side, {}).get(f"{side}_{i}", {})))
                  for i in range(env.num_agents)]
        actions[side] = policy.actions(info["side_observations"][side],
                                       info["side_action_masks"][side], active,
                                       stochastic=False)
    return actions


def run(args, configs):
    profile, scenario = configs["platform_profile"], configs["scenario_4v4"]
    agents = int(scenario["aircraft_per_side"])
    obs_dim = ObservationEncoder(num_agents=agents).actor_dim
    policies = None
    if args.red_model is not None:
        policies = {side: SidePolicy(getattr(args, side+"_model"), obs_dim, agents)
                    for side in ("red", "blue")}
    summary = {"seed": args.seed, "formation_request": args.formation,
               "flight_controller": args.flight_controller, "play_mode": args.play_mode,
               "model_actions": policies is not None, "output": str(args.output),
               "initial": {}, "transitions": {}, "mode_counts": {},
               "steps": 0, "sim_seconds": 0.0, "stop_reason": None}
    with platform(args, agents):
        env = CompetitionEnv(profile, scenario, host=args.host,
                             red_port=args.red_port, blue_port=args.blue_port,
                             seed=args.seed, connect_timeout=args.connect_timeout,
                             socket_timeout=args.socket_timeout,
                             flight_controller=args.flight_controller)
        try:
            for policy in (policies or {}).values():
                policy.reset()
            _, _, info = env.reset(formations=args.formation)
            for side in ("red", "blue"):
                summary["initial"][side] = {f"{side}_{i}": aircraft_snapshot(env, side, i, profile)
                                             for i in range(agents)}
                summary["transitions"][side] = {}
                summary["mode_counts"][side] = {"center": 0, "policy": 0, "boundary": 0, "dead": 0}
            print(f"seed={args.seed} formation={info['formations']} controller={args.flight_controller}", flush=True)
            print("初始化位置：", json.dumps(summary["initial"], ensure_ascii=False), flush=True)
            required = {(side, i) for side in ("red", "blue") for i in range(agents)
                        if alive((env._side_states[side] if env.dual_socket else env.state)[side][f"{side}_{i}"])}
            last_print = -args.print_interval
            while not info["episode_done"] and summary["steps"] < args.max_steps:
                if float(info["sim_time"]) >= args.max_seconds:
                    summary["stop_reason"] = "max_seconds"
                    break
                actions = policy_actions(args, env, info, policies)
                # Route separately so we can inspect the exact pre-step mode and
                # position before the platform advances.
                routing = env.route_actions(actions)
                sim_time = float(info["sim_time"])
                for side in ("red", "blue"):
                    for index, mode in enumerate(routing["modes"][side]):
                        summary["mode_counts"][side][mode] = summary["mode_counts"][side].get(mode, 0)+1
                        key = (side, index)
                        aircraft_name = f"{side}_{index}"
                        if mode == "policy" and aircraft_name not in summary["transitions"][side]:
                            frame = env._frames[side][index]
                            event = {"sim_time": sim_time, "step": summary["steps"],
                                     "position": aircraft_snapshot(env, side, index, profile),
                                     "visible_enemy_count": len(frame["enemies"]),
                                     "missile_threat": frame["missile"] is not None,
                                     "enemy_signature": list(frame.get("enemy_signature", ())),
                                     "formation": info["formations"][side]}
                            summary["transitions"][side][aircraft_name] = event
                            print(f"首次策略控制 {side}_{index}: t={sim_time:.2f}s "
                                  f"step={summary['steps']} position={json.dumps(event['position'], ensure_ascii=False)} "
                                  f"visible_enemies={event['visible_enemy_count']} missile={event['missile_threat']}", flush=True)
                command = env.prepare_step(routing["actions"])
                _, _, _, _, info = env.step_commands(command)
                summary["steps"] += 1
                summary["sim_seconds"] = float(info["sim_time"])
                if summary["sim_seconds"] >= last_print+args.print_interval:
                    print(f"t={summary['sim_seconds']:.1f}s "
                          f"policy={sum(len(v) for v in summary['transitions'].values())}/8 "
                          f"modes={summary['mode_counts']}", flush=True)
                    last_print = summary["sim_seconds"]
                if required and all(f"{side}_{i}" in summary["transitions"][side] for side, i in required):
                    summary["stop_reason"] = "all_aircraft_entered_policy"
                    break
            if summary["stop_reason"] is None:
                summary["stop_reason"] = "episode_done" if info["episode_done"] else "max_steps"
        finally:
            env.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"诊断结果已写入：{args.output}", flush=True)
    return summary


def main():
    args, configs = arguments()
    run(args, configs)


if __name__ == "__main__":
    main()
