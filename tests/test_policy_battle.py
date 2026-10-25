"""4v4 策略模型可视化对战：PlayMode=1，默认逐帧记录 Tacview。

先填写下方 RED_MODEL_PATH / BLUE_MODEL_PATH，然后直接运行：
    python tests/test_policy_battle.py
也可通过命令行覆盖路径（支持 run 目录或某个 actor 文件）：
    python tests/test_policy_battle.py --red-model result/4v4/mappo/run-红方/actor_latest.pt --blue-model result/4v4/mappo/run-蓝方/actor_10.pt

默认运行完整一局，使用独立端口 19888/19889，记录保存到 results/tacview/。
--duration 30 可提前结束短时测试；--rounds 可连续测试多局，每局单独录制。
默认每次启动自动生成随机种子；传入 --seed 1000 可复现对应的初始化与阵型安排。
"""

import argparse
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
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
import torch

from algorithms.mappo.ppo_actor import PPOActor
from algorithms.utils.model_io import load_actor_model
from algorithms.utils.normalization import RunningMoments
from config import ROOT, project_configs
from envs.competition_env import CompetitionEnv
from envs.observation_encoder import ObservationEncoder, alive
from envs.tacview_recorder import TacviewRecorder
from envs.formations import FORMATION_NAMES
from envs.model import FLIGHT_CONTROLLER_NAMES


# ======================= 模型路径配置 =======================
# 相对路径以项目根目录为准；也可以填写绝对路径。
# 例如：r"result/4v4/mappo/run-20261021-153000-a1b2c3d4/actor_latest.pt"
RED_MODEL_PATH = "results/4v4/mappo/run-20261025-005445/actor_3.pt"
BLUE_MODEL_PATH = "results/4v4/mappo/run-20261025-005445/actor_3.pt"
# ===========================================================


class SidePolicy:
    """One frozen actor, normalizer and independent per-aircraft recurrent state."""

    def __init__(self, path, obs_dim, agents):
        model = load_actor_model(path)
        if model["obs_dim"] != obs_dim:
            raise ValueError(f"{path}: Actor 观测维度为 {model['obs_dim']}，当前场景需要 {obs_dim}")
        self.path = model["path"]
        self.update = model["update"]
        self.actor = PPOActor(obs_dim, model["hidden_dim"]).eval()
        self.actor.load_state_dict(model["state_dict"])
        self.actor.requires_grad_(False)
        self.normalizer = RunningMoments(obs_dim)
        self.normalizer.load_state_dict(model["obs_normalizer"])
        if self.normalizer.mean.shape != (obs_dim,) or self.normalizer.var.shape != (obs_dim,):
            raise ValueError(f"{path}: Actor 归一化统计维度与观测不一致")
        self.agents = agents
        self.reset()

    def reset(self):
        self.hidden = np.zeros((self.agents, self.actor.hidden_dim), dtype=np.float32)
        self.masks = np.zeros((self.agents, 1), dtype=np.float32)

    @torch.no_grad()
    def actions(self, observations, legal, active, stochastic=False):
        active = np.asarray(active, dtype=np.float32).reshape(self.agents, 1)
        self.hidden *= active
        action, _, _, hidden = self.actor(
            torch.from_numpy(self.normalizer.normalize(observations)), torch.from_numpy(self.hidden),
            torch.from_numpy(self.masks*active), torch.as_tensor(legal, dtype=torch.bool),
            deterministic=not stochastic)
        self.hidden = hidden.cpu().numpy()*active
        self.masks = active.copy()
        return action.cpu().numpy()


def arguments():
    configs = project_configs()
    train = configs["train_mappo"]
    launch = train["platform_launch"]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--red-model", type=Path, default=RED_MODEL_PATH or None)
    parser.add_argument("--blue-model", type=Path, default=BLUE_MODEL_PATH or None)
    parser.add_argument("--host", default=train["host"])
    parser.add_argument("--red-port", type=int, default=19888)
    parser.add_argument("--blue-port", type=int, help="Default: red port + 1")
    parser.add_argument("--no-launch-platform", action="store_true", help="Connect to an existing PlayMode=1 4v4 platform")
    parser.add_argument("--executable", type=Path, default=ROOT/launch["executable"])
    parser.add_argument("--scene", type=int, default=launch["scene"])
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--formation", choices=("random",)+FORMATION_NAMES, default="random",
                        help="Opening for both sides; center uses original center navigation, random draws independently (official init is unchanged)")
    parser.add_argument("--flight-controller", choices=FLIGHT_CONTROLLER_NAMES, default=None,
                        help="Override train_mappo.yaml; otherwise use its flight_controller value")
    parser.add_argument("--duration", type=float, help="Optional simulation seconds per round; default: complete battle")
    parser.add_argument("--sim-speed", type=float, default=0, help="Simulation / wall-clock speed; 0 disables pacing")
    parser.add_argument("--stochastic", action="store_true", help="Sample policy actions; default uses deterministic actions")
    parser.add_argument("--seed", type=int, default=None,
                        help="Base seed for reproducible initialization/formations; default: generate a fresh random seed")
    parser.add_argument("--print-interval", type=float, default=5, help="Telemetry interval in simulation seconds")
    parser.add_argument("--tacview-dir", type=Path, default=ROOT/"results"/"tacview")
    parser.add_argument("--connect-timeout", type=float, default=launch["connect_timeout_seconds"])
    parser.add_argument("--socket-timeout", type=float, default=launch["socket_timeout_seconds"])
    parser.add_argument("--max-steps", type=int, default=configs["platform_profile"]["simulation"]["evaluation_safety_steps"])
    args = parser.parse_args()
    if args.flight_controller is None:
        args.flight_controller = train["flight_controller"]
    for side in ("red", "blue"):
        path = getattr(args, side+"_model")
        if path is None:
            parser.error(f"请配置脚本顶部 {side.upper()}_MODEL_PATH，或传入 --{side}-model")
        setattr(args, side+"_model", (ROOT/path).resolve())
    args.executable = (ROOT/args.executable).resolve()
    args.tacview_dir = (ROOT/args.tacview_dir).resolve()
    args.blue_port = args.red_port+1 if args.blue_port is None else args.blue_port
    if args.red_port == args.blue_port or any(not 1 <= port <= 65535 for port in (args.red_port, args.blue_port)):
        parser.error("红蓝端口必须不同，且位于 1..65535")
    if not args.no_launch_platform and (args.host not in ("127.0.0.1", "localhost") or args.blue_port != args.red_port+1):
        parser.error("自动启动要求 localhost 和相邻红蓝端口")
    for name in ("rounds", "max_steps", "print_interval", "connect_timeout", "socket_timeout"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"{name} 必须为正数且有限")
    if not math.isfinite(args.sim_speed) or args.sim_speed < 0:
        parser.error("sim-speed 必须为非负有限数")
    if args.duration is not None and (not math.isfinite(args.duration) or args.duration <= 0):
        parser.error("duration 必须为正数且有限")
    if args.seed is None:
        args.seed = secrets.randbits(32)
    return args, configs


@contextmanager
def visible_platform(args, agents):
    process = None
    try:
        if not args.no_launch_platform:
            executable = args.executable.resolve()
            if not executable.is_file():
                raise FileNotFoundError(executable)
            for port in (args.red_port, args.blue_port):
                with socket.socket() as probe:
                    if os.name == "nt":
                        probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
                    probe.bind((args.host, port))
            command = [str(executable), f"Ip={args.host}", f"Port={args.red_port}", "Red=0", "Blue=1",
                       f"RedNum={agents}", f"BlueNum={agents}", "PlayMode=1", f"Scenes={args.scene}", "-logFile", os.devnull]
            # Rendering is explicitly enabled for this test; leave its window visible.
            process = subprocess.Popen(command, cwd=executable.parent,
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            print(f"连接已有平台：Red=0 Blue=1 RedNum={agents} BlueNum={agents} PlayMode=1，"
                  f"端口 {args.red_port}/{args.blue_port}", flush=True)
        yield
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def recording_state(env, info):
    """Prefer each owner's full aircraft packets over sparse enemy reports."""
    if not env.dual_socket:
        return info["raw_state"]
    state = {side: env._side_states[side][side] for side in ("red", "blue")}
    state["missile"] = {}
    for side in ("red", "blue"):
        for name, fields in env._side_states[side]["missile"].items():
            if isinstance(fields, dict):
                state["missile"].setdefault(name, {}).update(fields)
    return state


def run(args, configs):
    scenario = configs["scenario_4v4"]
    profile = configs["platform_profile"]
    agents = int(scenario["aircraft_per_side"])
    obs_dim = ObservationEncoder(num_agents=agents).actor_dim
    print(f"本次随机种子：{args.seed}（可通过 --seed {args.seed} 复现初始化与阵型安排；多局按序加 1）", flush=True)
    torch.manual_seed(args.seed)
    # Load and validate both models before launching any native platform.
    policies = {side: SidePolicy(getattr(args, side+"_model"), obs_dim, agents) for side in ("red", "blue")}
    for side, policy in policies.items():
        print(f"{side.upper()} 模型：{policy.path}（第 {policy.update} 轮，hidden={policy.actor.hidden_dim}）", flush=True)
    print(f"{agents}v{agents} 策略对战，PlayMode=1，{'采样' if args.stochastic else '确定性'}动作。Ctrl+C 可停止。", flush=True)
    summaries = []
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    with ExitStack() as stack:
        stack.enter_context(visible_platform(args, agents))
        env = CompetitionEnv(profile, scenario, host=args.host,
                             red_port=args.red_port, blue_port=args.blue_port, seed=args.seed,
                             connect_timeout=args.connect_timeout, socket_timeout=args.socket_timeout,
                             flight_controller=args.flight_controller)
        stack.callback(env.close)
        for round_index in range(args.rounds):
            env.rng.seed(args.seed+round_index)
            env.seed_formations(args.seed+round_index)
            for policy in policies.values():
                policy.reset()
            _, _, info = env.reset(formations=args.formation)
            print(f"进场阵型：红方 {info['formations']['red']}，蓝方 {info['formations']['blue']}（初始化使用比赛想定）", flush=True)
            start_sim = previous_sim = float(info["sim_time"])
            start_wall = time.monotonic()
            elapsed, next_print, steps = 0.0, 0.0, 0
            path = args.tacview_dir/f"policy_battle_{stamp}_round_{round_index+1:03d}.acmi"
            with TacviewRecorder(path, profile["units"]["feet_to_meters"], title=f"ZK policy battle round {round_index+1}") as recorder:
                print(f"第 {round_index+1}/{args.rounds} 局，Tacview：{path}", flush=True)
                recorder.record(recording_state(env, info), sim_time=info["sim_time"])
                while not info["episode_done"] and steps < args.max_steps and (args.duration is None or elapsed < args.duration):
                    if elapsed+1e-6 >= next_print:
                        print(f"t={elapsed:.1f}s 红方存活/余弹={info['red_score']} 蓝方存活/余弹={info['blue_score']}", flush=True)
                        next_print += args.print_interval
                    commands = {}
                    for side, policy in policies.items():
                        own = env._side_states[side][side] if env.dual_socket else info["raw_state"][side]
                        active = [float(alive(own.get(f"{side}_{index}", {}))) for index in range(agents)]
                        commands[side] = policy.actions(info["side_observations"][side], info["side_action_masks"][side],
                                                        active, stochastic=args.stochastic)
                    _, _, _, _, info = env.step(commands)
                    steps += 1
                    now = float(info["sim_time"])
                    recorder.record(recording_state(env, info), sim_time=now)
                    if not math.isfinite(now) or (now <= previous_sim and not info["episode_done"]):
                        raise RuntimeError("平台仿真时间无效或未推进，请检查平台是否处于运行状态")
                    previous_sim, elapsed = now, now-start_sim
                    if args.sim_speed > 0 and not info["episode_done"]:
                        deadline = start_wall+elapsed/args.sim_speed
                        while time.monotonic() < deadline:
                            time.sleep(min(0.1, max(0.0, deadline-time.monotonic())))
                reason = "finished" if info["episode_done"] else "step_limit" if steps >= args.max_steps else "duration_limit"
                summary = {"round": round_index+1, "steps": steps, "sim_seconds": elapsed,
                           "outcome": info["outcome"] if info["episode_done"] else None,
                           "formations": dict(info["formations"]),
                           "stop_reason": reason, "tacview": path, "frames": recorder.frames}
                summaries.append(summary)
                result = info["outcome"] if info["episode_done"] else f"提前结束：{reason}"
                print(f"第 {round_index+1} 局结束：{result}，{steps} 步，{elapsed:.2f} 仿真秒，"
                      f"已记录 {recorder.frames} 帧。", flush=True)
    return summaries


def main():
    args, configs = arguments()
    try:
        run(args, configs)
    except KeyboardInterrupt:
        print("\n对战测试已停止，Tacview 已保存已采集的帧，连接和本次启动的平台已关闭。", flush=True)


if __name__ == "__main__":
    main()
