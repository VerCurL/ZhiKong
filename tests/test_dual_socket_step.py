"""4v4 双 Socket 实测：多蓝机/多枚蓝方来袭弹准备完成后，只抓取一个正式步。

运行：python tests/test_dual_socket_step.py
默认使用独立端口 20888/20889，自动启动 PlayMode=0 平台。
结果保存到 results/socket_step_4v4/ 下的新目录，原始响应不经过环境合并或观测编码。

飞机可以指定初始化位置；在飞导弹不能由初始化接口直接创建，因此需要真实锁定、
发射的准备阶段。准备步与最后一个正式观测步分别计数，绝不隐瞒额外仿真步。
默认要求红 Socket 可见全部 4 架蓝机和至少 4 枚指向红方的蓝方在飞导弹。
"""

import argparse
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from comm_interface import env as PlatformClient
from config import ROOT, load_yaml
from algorithms.utils.action_spec import NEUTRAL_ACTION, WEAPON_CHOICES
from envs.model.flight_controller import FrozenFlightController
from envs.model import FLIGHT_CONTROLLER_NAMES
from envs.model.pid.controller import PIDFlightController
from envs.observation_encoder import ObservationEncoder, alive
from envs.utils.utils import decode_aircraft_id
from envs.weapon_control import weapon_plans


SIDES = ("red", "blue")
AGENTS = 4


def arguments():
    profile = load_yaml(ROOT/"configs/platform_profile.yaml")
    train = load_yaml(ROOT/"configs/train_mappo.yaml")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--red-port", type=int, default=20888)
    parser.add_argument("--no-launch-platform", action="store_true")
    parser.add_argument("--executable", type=Path, default=ROOT/train["platform_launch"]["executable"])
    parser.add_argument("--scene", type=int, default=train["platform_launch"]["scene"])
    parser.add_argument("--separation-m", type=float, default=15000)
    parser.add_argument("--spacing-m", type=float, default=500)
    parser.add_argument("--altitude-m", type=float, default=7000)
    parser.add_argument("--speed-mps", type=float, default=200)
    parser.add_argument("--incoming-missiles", type=int, choices=(2, 3, 4), default=4)
    parser.add_argument("--max-prepare-seconds", type=float, default=30)
    parser.add_argument("--max-prepare-steps", type=int, default=300)
    parser.add_argument("--connect-timeout", type=float, default=60)
    parser.add_argument("--socket-timeout", type=float, default=30)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--flight-controller", choices=FLIGHT_CONTROLLER_NAMES,
                        default=train["flight_controller"],
                        help="Override train_mappo.yaml flight_controller")
    args = parser.parse_args()
    args.blue_port = args.red_port+1
    if not 1 <= args.red_port < 65535:
        parser.error("red-port must allow two adjacent ports in 1..65535")
    if not args.no_launch_platform and args.host not in ("127.0.0.1", "localhost"):
        parser.error("Automatic launch requires localhost")
    for key in ("separation_m", "spacing_m", "altitude_m", "speed_mps", "max_prepare_seconds",
                "connect_timeout", "socket_timeout"):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            parser.error(key+" must be finite and positive")
    if args.max_prepare_steps <= 0 or args.altitude_m >= profile["limits"]["maximum_altitude_m"]:
        parser.error("Invalid step limit or altitude")
    longest = math.hypot(args.separation_m, (AGENTS-1)*args.spacing_m)
    if longest >= min(profile["limits"]["enemy_detection_m"], profile["limits"]["missile_detection_m"]):
        parser.error("Every initial aircraft pair must fit inside aircraft and missile detection ranges")
    if args.output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        args.output = ROOT/"results"/"socket_step_4v4"/stamp
    return args, profile


def save(folder, name, value):
    (folder/name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def initial_request(args, profile):
    field, units = profile["battlefield"], profile["units"]
    latitude = field["center_latitude_deg"]
    scale = units["earth_meters_per_degree"]
    half_lon = args.separation_m/2/(scale*math.cos(math.radians(latitude)))
    request = {"flag": {"init": {"save": 0, "SplitScreen": 0}}}
    for side, sign, heading in (("red", -1, 90), ("blue", 1, 270)):
        request[side] = {f"{side}_{i}": {
            "ic/h-sl-ft": args.altitude_m/units["feet_to_meters"], "ic/terrain-elevation-ft": 1e-8,
            "ic/long-gc-deg": field["center_longitude_deg"]+sign*half_lon,
            "ic/lat-geod-deg": latitude+(i-(AGENTS-1)/2)*args.spacing_m/scale,
            "ic/u-fps": args.speed_mps/units["feet_to_meters"], "ic/v-fps": 0, "ic/w-fps": 0,
            "ic/p-rad_sec": 0, "ic/q-rad_sec": 0, "ic/r-rad_sec": 0, "ic/roc-fpm": 0,
            "ic/psi-true-deg": heading, "ic/phi-deg": 0, "ic/theta-deg": 0,
        } for i in range(AGENTS)}
    return request


@contextmanager
def platform(args):
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
            options = {}
            if os.name == "nt":
                startup = subprocess.STARTUPINFO()
                startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startup.wShowWindow = subprocess.SW_HIDE
                options.update(startupinfo=startup, creationflags=subprocess.CREATE_NO_WINDOW)
            command = [str(executable), f"Ip={args.host}", f"Port={args.red_port}", "Red=0", "Blue=1",
                       "RedNum=4", "BlueNum=4", "PlayMode=0", f"Scenes={args.scene}",
                       "-logFile", str(args.output/"platform.log")]
            process = subprocess.Popen(command, cwd=executable.parent, stdin=subprocess.DEVNULL,
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


def exchange(clients, requests):
    for side in SIDES:
        clients[side]._send_data(requests[side])
    response = {side: clients[side]._receive_data() for side in SIDES}
    for side in SIDES:
        own = response[side].get(side, {}) if isinstance(response[side], dict) else {}
        if set(own) != {f"{side}_{i}" for i in range(AGENTS)}:
            raise RuntimeError(f"{side} Socket did not return the expected four own-aircraft entries")
    return response


def sim_time(payload, side):
    times = [float(p["simulation/sim-time-sec"]) for p in payload[side].values()
             if "simulation/sim-time-sec" in p]
    if not times or not all(math.isfinite(t) for t in times):
        raise RuntimeError(f"{side} Socket has no finite simulation time")
    return max(times)


def visible_targets(payload):
    coordinates = ("position/h-sl-ft", "position/lat-geod-deg", "position/long-gc-deg")
    return [name for name, value in payload.get("blue", {}).items()
            if isinstance(value, dict) and all(k in value for k in coordinates) and alive(value)]


def incoming(payload):
    return {name: missile for name, missile in payload.get("missile", {}).items()
            if isinstance(missile, dict) and int(missile.get("Status", -1)) == 0
            and (decode_aircraft_id(missile.get("Owner")) or "").startswith("blue_")
            and (decode_aircraft_id(missile.get("Target")) or "").startswith("red_")}


def snapshot_summary(responses):
    result = {}
    for side, payload in responses.items():
        result[side+"_socket"] = {
            "top_level_keys": list(payload), "sim_time": sim_time(payload, side),
            "field_inventory": {group: {name: {"field_count": len(value), "fields": sorted(value)}
                                         for name, value in payload.get(group, {}).items() if isinstance(value, dict)}
                                for group in ("red", "blue", "missile")},
            "target_into_view": {name: value.get("TargetIntoView") for name, value in payload[side].items()},
        }
    result["red_visible_blue_aircraft"] = visible_targets(responses["red"])
    result["red_visible_blue_incoming_missiles"] = list(incoming(responses["red"]))
    return result


def commands(responses, controller, encoder, profile, fired, prepare, missile_count):
    names, planes = [], []
    for side in SIDES:
        for i in range(AGENTS):
            name = f"{side}_{i}"
            plane = responses[side][side][name]
            if not alive(plane):
                raise RuntimeError("An aircraft died before the single-step capture was completed")
            names.append(name)
            planes.append(plane)
    controls = controller.controls(names, planes, np.tile(NEUTRAL_ACTION, (len(names), 1)))
    requests = {side: {side: {}} for side in SIDES}
    for row, name in enumerate(names):
        side, index = name.split("_")
        control = {"mode": 0, **dict(zip(("fcs/aileron-cmd-norm", "fcs/elevator-cmd-norm",
                                          "fcs/rudder-cmd-norm", "fcs/throttle-cmd-norm"), map(float, controls[row]))),
                   "fcs/weapon-launch": 0, "switch-missile": 0, "switch-acmType": 0, "change-target": 9}
        if prepare and side == "blue" and int(index) < missile_count and name not in fired:
            plan = weapon_plans(responses[side], side, name, encoder, profile)[WEAPON_CHOICES.index("amraam")]
            control.update({key: plan[key] for key in ("fcs/weapon-launch", "switch-missile", "change-target")})
        requests[side][side][name] = control
    return requests


def run(args, profile):
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    print(f"4v4 dual Socket {args.red_port}/{args.blue_port}; output: {args.output}", flush=True)
    print("Aircraft init -> real missile preparation -> exactly one captured control step", flush=True)
    torch.set_num_threads(1)
    controller_class = FrozenFlightController if args.flight_controller == "baseline" else PIDFlightController
    controller = controller_class(profile["units"]["feet_to_meters"])
    encoder = ObservationEncoder(filter_visibility=False)
    initialization = initial_request(args, profile)
    save(args.output, "initialization_request.json", initialization)
    trace, metadata = [], {"prepared_steps": 0, "captured_steps": 0, "status": "started"}
    try:
        with ExitStack() as stack:
            stack.enter_context(platform(args))
            clients = {}
            for side, port in (("red", args.red_port), ("blue", args.blue_port)):
                clients[side] = PlatformClient(args.host, port, args.connect_timeout, args.socket_timeout)
                stack.callback(clients[side].close)
            responses = exchange(clients, {side: initialization for side in SIDES})
            for side in SIDES:
                save(args.output, side+"_initial_response.json", responses[side])
            metadata["initial"] = snapshot_summary(responses)
            started = sim_time(responses["red"], "red")
            initial_ammo = {f"blue_{i}": int(responses["blue"]["blue"][f"blue_{i}"].get("AMRAAMCurrentNum", 0))
                            for i in range(AGENTS)}
            fired = set()
            for step in range(args.max_prepare_steps+1):
                now = sim_time(responses["red"], "red")
                fired.update(name for name, ammo in initial_ammo.items()
                             if int(responses["blue"]["blue"][name].get("AMRAAMCurrentNum", ammo)) < ammo)
                trace.append({"step": step, "sim_time": now, "blue_aircraft": visible_targets(responses["red"]),
                              "incoming_missiles": list(incoming(responses["red"])), "fired_blue_aircraft": sorted(fired),
                              "blue_fire_control": {name: {key: value.get(key) for key in
                                  ("AimMode", "AMRAAMCurrentNum", "AMRAAMlockedTarget", "TargetIntoView", "TargetEnterAttackRange")}
                                  for name, value in responses["blue"]["blue"].items()}})
                metadata["prepared_steps"] = step
                if len(visible_targets(responses["red"])) == AGENTS and len(incoming(responses["red"])) >= args.incoming_missiles:
                    break
                if step == args.max_prepare_steps or now-started >= args.max_prepare_seconds:
                    for side in SIDES:
                        save(args.output, side+"_preparation_failed_response.json", responses[side])
                    raise RuntimeError("Preparation did not produce four visible Blue aircraft and the required incoming missiles; see preparation_trace.json")
                requests = commands(responses, controller, encoder, profile, fired, True, args.incoming_missiles)
                previous = now
                responses = exchange(clients, requests)
                if sim_time(responses["red"], "red") <= previous:
                    raise RuntimeError("Platform time did not advance during preparation")
            before = responses
            metadata["before_step"] = snapshot_summary(before)
            for side in SIDES:
                save(args.output, side+"_before_step_response.json", before[side])
            requests = commands(before, controller, encoder, profile, fired, False, args.incoming_missiles)
            for side in SIDES:
                save(args.output, side+"_step_request.json", requests[side])
            after = exchange(clients, requests)
            metadata["captured_steps"] = 1
            for side in SIDES:
                save(args.output, side+"_step_response.json", after[side])
            metadata["after_step"] = snapshot_summary(after)
            metadata["dt"] = {side: sim_time(after[side], side)-sim_time(before[side], side) for side in SIDES}
            if min(metadata["dt"].values()) <= 0:
                raise RuntimeError("Captured step did not advance both Socket clocks")
            if len(visible_targets(after["red"])) != AGENTS or len(incoming(after["red"])) < args.incoming_missiles:
                raise RuntimeError("Captured step no longer contains the required multiple aircraft/missiles")
            metadata["status"] = "passed"
            print(f"PASS: prepared {metadata['prepared_steps']} steps; captured 1 step; dt={metadata['dt']}", flush=True)
            print(f"Red Socket: Blue aircraft={visible_targets(after['red'])}; incoming Blue missiles={list(incoming(after['red']))}", flush=True)
            for side in SIDES:
                inventory = metadata["after_step"][side+"_socket"]["field_inventory"]
                print(side+" Socket fields: "+str({group: {name: item["field_count"] for name, item in entries.items()}
                                                    for group, entries in inventory.items()}), flush=True)
    except BaseException as error:
        metadata.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save(args.output, "preparation_trace.json", trace)
        save(args.output, "summary.json", metadata)


if __name__ == "__main__":
    arguments_, profile_ = arguments()
    run(arguments_, profile_)
