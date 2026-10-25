"""1v1 baseline 飞控可视化测试；红蓝双方跟踪固定飞行目标，不加载 MAPPO。

启动可见平台窗口并运行演示：
    python tests/test_baseline_flight.py
仅测试右转，或连接手动启动的 1v1 平台：
    python tests/test_baseline_flight.py --maneuver right --duration 90
    python tests/test_baseline_flight.py --no-launch-platform --red-port 18888 --blue-port 18889

默认独立使用 18888/18889，红蓝双方均使用固定 baseline，关闭武器和平台内置回放保存。
本测试默认生成 results/tacview/ 下的 ACMI 文件；--tacview 指定路径，--no-tacview 关闭。
窗口由官方平台渲染，可在平台中选择跟随视角。Ctrl+C 退出；只清理本脚本启动的进程。
"""

import argparse
from contextlib import ExitStack, contextmanager
import csv
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from config import ROOT, load_yaml
from comm_interface import env as PlatformClient
from algorithms.utils.action_spec import FLIGHT_CONTROLLER, NEUTRAL_ACTION
from envs.model.flight_controller import FrozenFlightController
from envs.model import FLIGHT_CONTROLLER_NAMES
from envs.model.pid.controller import PIDFlightController
from envs.observation_encoder import alive
from envs.tacview_recorder import TacviewRecorder


class TargetFlightController(FrozenFlightController):
    """Only replace the three target errors; reuse baseline features and decoding."""

    def observation(self, plane, target):
        observation = super().observation(plane, NEUTRAL_ACTION)
        height, heading, speed = target
        scales = self.settings["observation_scales"]
        observation[:3] = (
            (height-float(plane["position/h-sl-ft"])*self.ft)/scales["altitude_error_m"],
            math.radians((heading-float(plane["attitude/psi-deg"])+180) % 360-180),
            (speed-float(plane["velocities/u-fps"])*self.ft)/scales["speed_mps"],
        )
        return observation


def arguments():
    profile = load_yaml(ROOT/"configs/platform_profile.yaml")
    train = load_yaml(ROOT/"configs/train_mappo.yaml")
    launch = train["platform_launch"]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default=train["host"])
    parser.add_argument("--red-port", type=int, default=18888, help="Independent test red port (default: 18888)")
    parser.add_argument("--blue-port", type=int, help="Default: red port + 1")
    startup = parser.add_mutually_exclusive_group()
    startup.add_argument("--launch-platform", dest="launch_platform", action="store_true",
                         help="Launch a visible 1v1 platform window (default)")
    startup.add_argument("--no-launch-platform", dest="launch_platform", action="store_false",
                         help="Connect to an already running 1v1 platform")
    parser.set_defaults(launch_platform=True)
    parser.add_argument("--executable", type=Path, default=ROOT/launch["executable"])
    parser.add_argument("--scene", type=int, default=launch["scene"])
    parser.add_argument("--split-screen", action="store_true")
    parser.add_argument("--model", type=Path, help="Override the fixed baseline checkpoint")
    parser.add_argument("--flight-controller", choices=FLIGHT_CONTROLLER_NAMES,
                        default=train["flight_controller"],
                        help="Override train_mappo.yaml flight_controller")
    parser.add_argument("--maneuver", choices=("demo", "hold", "left", "right", "climb", "descend", "accelerate", "decelerate"), default="demo")
    parser.add_argument("--phase-seconds", type=float, default=20, help="Simulation seconds per demonstration phase")
    parser.add_argument("--duration", type=float, help="Simulation seconds; default: one complete maneuver sequence")
    parser.add_argument("--altitude-m", type=float, default=6000)
    parser.add_argument("--speed-mps", type=float, default=240)
    parser.add_argument("--heading-deg", type=float, default=90, help="Initial heading for both aircraft")
    parser.add_argument("--height-delta-m", type=float, default=max(FLIGHT_CONTROLLER["delta_altitude_m"]))
    parser.add_argument("--heading-delta-deg", type=float, default=max(FLIGHT_CONTROLLER["delta_heading_deg"]))
    parser.add_argument("--speed-delta-mps", type=float, default=max(FLIGHT_CONTROLLER["delta_speed_mps"]))
    parser.add_argument("--sim-speed", type=float, default=1, help="Simulation / wall-clock speed; 0 runs without pacing")
    parser.add_argument("--print-interval", type=float, default=2, help="Telemetry interval in simulation seconds")
    parser.add_argument("--csv", type=Path, help="Optional telemetry CSV, sampled at print-interval; no file by default")
    recording = parser.add_mutually_exclusive_group()
    recording.add_argument("--tacview", "--acmi", type=Path, help="ACMI output path; default: results/tacview/baseline_flight_<UTC time>.acmi")
    recording.add_argument("--no-tacview", action="store_true", help="Disable Tacview recording for this test")
    parser.add_argument("--connect-timeout", type=float, default=launch["connect_timeout_seconds"])
    parser.add_argument("--socket-timeout", type=float, default=launch["socket_timeout_seconds"])
    parser.add_argument("--max-steps", type=int, default=profile["simulation"]["evaluation_safety_steps"])
    args = parser.parse_args()
    if not args.no_tacview and args.tacview is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        args.tacview = ROOT/"results"/"tacview"/f"baseline_flight_{stamp}.acmi"
    args.blue_port = args.red_port+1 if args.blue_port is None else args.blue_port
    positive = ("phase_seconds", "altitude_m", "speed_mps", "height_delta_m", "heading_delta_deg",
                "speed_delta_mps", "print_interval", "connect_timeout", "socket_timeout", "max_steps")
    for key in positive:
        if not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            parser.error(f"{key} must be positive and finite")
    if not math.isfinite(args.sim_speed) or args.sim_speed < 0 or not math.isfinite(args.heading_deg):
        parser.error("sim-speed must be nonnegative and heading-deg must be finite")
    if args.duration is not None and (not math.isfinite(args.duration) or args.duration <= 0):
        parser.error("duration must be positive and finite")
    if args.red_port == args.blue_port or any(not 1 <= p <= 65535 for p in (args.red_port, args.blue_port)):
        parser.error("Red and blue ports must be distinct and within 1..65535")
    if args.launch_platform and (args.host not in ("localhost", "127.0.0.1") or args.blue_port != args.red_port+1):
        parser.error("Automatic launch requires localhost and adjacent port pairs")
    return args, profile, launch


def maneuver_sequence(args):
    if args.maneuver == "demo":
        return ["hold", "right", "left", "climb", "descend", "accelerate", "decelerate", "hold"]
    return ["hold"] if args.maneuver == "hold" else ["hold", args.maneuver]


def flight_target(args, phase):
    # Every phase refers to the original fixed targets, not to the last frame.
    height, heading, speed = args.altitude_m, args.heading_deg, args.speed_mps
    if phase in ("left", "right"):
        heading += args.heading_delta_deg*(-1 if phase == "left" else 1)
    elif phase in ("climb", "descend"):
        height += args.height_delta_m*(-1 if phase == "descend" else 1)
    elif phase in ("accelerate", "decelerate"):
        speed += args.speed_delta_mps*(-1 if phase == "decelerate" else 1)
    return height, heading % 360, speed


def initialization(args, ft):
    request = {"flag": {"init": {"save": 0, "SplitScreen": int(args.split_screen)}}}
    for side, latitude in (("red", -0.03), ("blue", 0.03)):
        # Parallel, separated flight paths make camera observation easier.
        request[side] = {f"{side}_0": {
            "ic/h-sl-ft": args.altitude_m/ft, "ic/terrain-elevation-ft": 1e-8,
            "ic/long-gc-deg": -0.05, "ic/lat-geod-deg": latitude,
            "ic/u-fps": args.speed_mps/ft, "ic/v-fps": 0, "ic/w-fps": 0,
            "ic/p-rad_sec": 0, "ic/q-rad_sec": 0, "ic/r-rad_sec": 0,
            "ic/roc-fpm": 0, "ic/psi-true-deg": args.heading_deg % 360,
            "ic/phi-deg": 0, "ic/theta-deg": 0,
        }}
    return request


@contextmanager
def visible_platform(args, launch):
    process = None
    try:
        if args.launch_platform:
            executable = args.executable.resolve()
            if not executable.is_file():
                raise FileNotFoundError(executable)
            for port in (args.red_port, args.blue_port):
                with socket.socket() as probe:
                    if os.name == "nt":
                        probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
                    probe.bind((args.host, port))
            command = [str(executable), f"Ip={args.host}", f"Port={args.red_port}",
                       "Red=0", "Blue=1", "RedNum=1", "BlueNum=1",
                       f"PlayMode=1", f"Scenes={args.scene}", "-logFile", os.devnull]
            # No hidden-window or batch/headless flags: rendering is the purpose of this script.
            process = subprocess.Popen(command, cwd=executable.parent,
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            print(f"连接已启动的 1v1 平台：红方 {args.host}:{args.red_port}，蓝方 {args.host}:{args.blue_port}", flush=True)
            print("平台启动参数需要 Red=0 Blue=1 RedNum=1 BlueNum=1。", flush=True)
        yield
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def exchange(clients, requests, recorder=None):
    # Advancing the simulator requires both sides; write both before reading either.
    for side in ("red", "blue"):
        clients[side]._send_data(requests[side])
    planes = {}
    missiles = {} if recorder is not None else None
    for side in ("red", "blue"):
        payload = clients[side]._receive_data()
        if not isinstance(payload, dict):
            raise ConnectionError(f"{side} Socket 未返回有效状态")
        group = payload.get(side, payload)
        names = [name for name in group if str(name).startswith(f"{side}_")]
        if names != [f"{side}_0"]:
            raise RuntimeError(f"需要 1v1 平台，{side} 实际返回 {names}；请检查 RedNum=1 BlueNum=1")
        planes[side] = group[f"{side}_0"]
        if recorder is not None:
            for name, fields in payload.get("missile", {}).items():
                if isinstance(fields, dict):
                    missiles.setdefault(name, {}).update(fields)
    if recorder is not None:
        recorder.record({"red": {"red_0": planes["red"]}, "blue": {"blue_0": planes["blue"]}, "missile": missiles})
    return planes


def simulation_time(planes):
    times = [float(plane["simulation/sim-time-sec"]) for plane in planes.values()]
    if not all(math.isfinite(value) for value in times) or max(times)-min(times) > 0.001:
        raise RuntimeError(f"红蓝方仿真时间无效或不同步：{times}")
    return max(times)


def control_command(values):
    return {"mode": 0, **{f"fcs/{key}-cmd-norm": float(value)
                          for key, value in zip(("aileron", "elevator", "rudder", "throttle"), values)},
            "fcs/weapon-launch": 0, "switch-missile": 0, "switch-acmType": 0, "change-target": 9}


def run(args, profile, launch):
    ft = profile["units"]["feet_to_meters"]
    controller = TargetFlightController(ft, args.model) if args.flight_controller == "baseline" else PIDFlightController(ft)
    sequence = maneuver_sequence(args)
    duration = args.duration if args.duration is not None else len(sequence)*args.phase_seconds
    for phase in sequence:
        height, _, speed = flight_target(args, phase)
        if not 0 < height <= profile["limits"]["maximum_altitude_m"] or speed <= 0:
            raise ValueError(f"{phase} 的目标高度或速度超出有效范围")
    print(f"1v1 {args.flight_controller} 飞控测试：双方独立循环状态；模式 {args.maneuver}，时长 {duration:g} 仿真秒。", flush=True)
    print("每阶段使用固定目标；初始速度和目标速度均为机体前向速度。武器关闭，Ctrl+C 退出。", flush=True)
    with ExitStack() as stack:
        stack.enter_context(visible_platform(args, launch))
        clients = {}
        for side, port in (("red", args.red_port), ("blue", args.blue_port)):
            clients[side] = PlatformClient(args.host, port, args.connect_timeout, args.socket_timeout)
            stack.callback(clients[side].close)
        recorder = None
        if args.tacview is not None:
            recorder = stack.enter_context(TacviewRecorder(args.tacview, ft, title=f"ZK baseline flight: {args.maneuver}"))
            print(f"Tacview 记录：{recorder.path.resolve()}", flush=True)
        request = initialization(args, ft)
        planes = exchange(clients, {side: request for side in clients}, recorder=recorder)
        controller.reset()
        if not all(alive(plane) for plane in planes.values()):
            raise RuntimeError("初始化时飞机已死亡，请检查平台场景")
        if any(not any(key in plane for key in ("velocities/vc-fps", "velocities/vc-kts")) for plane in planes.values()):
            print("校准空速未直接返回，使用马赫数、高度与标准大气估算。", flush=True)
        writer = None
        if args.csv is not None:
            args.csv.parent.mkdir(parents=True, exist_ok=True)
            stream = stack.enter_context(args.csv.open("w", newline="", encoding="utf-8"))
            writer = csv.writer(stream)
            writer.writerow(("sim_seconds", "phase", "aircraft", "altitude_m", "target_altitude_m",
                             "heading_deg", "target_heading_deg", "body_u_mps", "target_body_u_mps",
                             "roll_deg", "pitch_deg", "aileron", "elevator", "rudder", "throttle"))
        start_sim = previous_sim = simulation_time(planes)
        start_wall = time.monotonic()
        elapsed, next_print, last_phase, steps = 0.0, 0.0, None, 0
        while elapsed < duration:
            if steps >= args.max_steps:
                raise RuntimeError("达到最大步数，停止测试；可检查平台时间推进或调整 --max-steps")
            phase = sequence[min(int((elapsed+1e-6)/args.phase_seconds), len(sequence)-1)]
            target = flight_target(args, phase)
            if phase != last_phase:
                print(f"\n阶段 {phase}：高度 {target[0]:.0f} m，航向 {target[1]:.0f}°，速度 {target[2]:.0f} m/s", flush=True)
                last_phase = phase
            if args.flight_controller == "baseline":
                values = controller.controls(["red_0", "blue_0"], [planes["red"], planes["blue"]], [target, target])
            else:
                def target_action(plane):
                    heading_error = (target[1]-float(plane["attitude/psi-deg"])+180) % 360-180
                    height_error = target[0]-float(plane["position/h-sl-ft"])*ft
                    speed_error = target[2]-float(plane["velocities/ve-fps"])*ft
                    choices = [int(np.argmin(np.abs(np.asarray(FLIGHT_CONTROLLER["delta_altitude_m"])-height_error))),
                               int(np.argmin(np.abs(np.asarray(FLIGHT_CONTROLLER["delta_heading_deg"])-heading_error))),
                               int(np.argmin(np.abs(np.asarray(FLIGHT_CONTROLLER["delta_speed_mps"])-speed_error))), 0]
                    return np.asarray(choices, dtype=np.int64)
                values = controller.controls(["red_0", "blue_0"], [planes["red"], planes["blue"]],
                                             [target_action(planes["red"]), target_action(planes["blue"])])
            if elapsed+1e-6 >= next_print:
                for side, controls in zip(("red", "blue"), values):
                    plane = planes[side]
                    height = float(plane["position/h-sl-ft"])*ft
                    heading = float(plane["attitude/psi-deg"])
                    speed = float(plane["velocities/u-fps"])*ft
                    roll, pitch = (math.degrees(float(plane[key])) for key in ("attitude/roll-rad", "attitude/pitch-rad"))
                    print(f"t={elapsed:6.1f}s {side:4s} 高度 {height:7.1f}/{target[0]:.0f}m "
                          f"航向 {heading:6.1f}/{target[1]:.0f}° 速度 {speed:6.1f}/{target[2]:.0f}m/s "
                          f"滚转 {roll:+6.1f}° 俯仰 {pitch:+6.1f}° "
                          f"舵面 {controls[0]:+.2f},{controls[1]:+.2f},{controls[2]:+.2f} 油门 {controls[3]:.2f}", flush=True)
                    if writer is not None:
                        writer.writerow((elapsed, phase, f"{side}_0", height, target[0], heading, target[1],
                                         speed, target[2], roll, pitch, *controls))
                next_print += args.print_interval
            planes = exchange(clients, {side: {side: {f"{side}_0": control_command(control)}}
                                        for side, control in zip(("red", "blue"), values)}, recorder=recorder)
            steps += 1
            dead = [side for side, plane in planes.items() if not alive(plane)]
            if dead:
                print(f"停止测试：{dead} 飞机已死亡，DeathEvent="
                      f"{[planes[side].get('DeathEvent') for side in dead]}", flush=True)
                break
            now = simulation_time(planes)
            if now <= previous_sim:
                raise RuntimeError("平台仿真时间未推进，请确认平台处于运行状态")
            previous_sim = now
            elapsed = now-start_sim
            if args.sim_speed > 0:
                deadline = start_wall+elapsed/args.sim_speed
                while time.monotonic() < deadline:
                    time.sleep(min(0.1, max(0.0, deadline-time.monotonic())))
        print(f"测试结束：{steps} 步，{elapsed:.2f} 仿真秒，实际耗时 {time.monotonic()-start_wall:.2f} 秒。", flush=True)
        if args.csv is not None:
            print(f"低频遥测已保存：{args.csv}", flush=True)
        if recorder is not None:
            print(f"Tacview 已记录 {recorder.frames} 帧，退出后可直接打开 ACMI 文件。", flush=True)


def main():
    args, profile, launch = arguments()
    try:
        run(args, profile, launch)
    except KeyboardInterrupt:
        print("\n测试已停止，连接已关闭。", flush=True)


if __name__ == "__main__":
    main()
