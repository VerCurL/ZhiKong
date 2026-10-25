"""
训练模型的比赛智能体导出入口。

文件作用：
    将指定检查点中的 Actor 转换成部署使用的 TorchScript 模型，并生成比赛智能体目录。
    部署推理使用确定性动作，保留循环网络隐藏状态；不需要 Critic 和训练优化器。

主要流程：
    1. 读取检查点中的 Actor 权重、观测归一化统计及相关配置。
    2. 导出 actor.ts，并生成观测编码、合法动作掩码、动作转换和推理运行代码。
    3. 根据官方样例生成红、蓝方智能体入口及通信相关文件。
    4. 写入部署使用的归一化统计、平台参数、依赖清单和安装脚本。
    5. 复制自测模板，输出生成的队伍目录路径。

输出与使用说明：
    默认生成 submission/队伍名称/，其中包含 agent/、environment/ 和 self_check/。
    --team-name 使用 ASCII 字母、数字或下划线；--output 可指定输出父目录。
    本脚本只生成目录，不自动压缩、上传或运行自测；自测模板需按实际环境配置路径。

运行示例（在项目根目录执行）：
    python scripts/export_submission.py --checkpoint result/4v4/mappo/run-时间/actor_latest.pt --team-name my_team
"""

import argparse
import ast
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch import nn
import yaml

from algorithms.mappo.ppo_actor import PPOActor
from config import ROOT, project_configs
from algorithms.utils.model_io import load_actor_model
from envs.control_rules import CONTROL_SCHEMA
from envs.reward.fkr.missile_avoid_reward import THREAT_POLICY
from envs.formations import FORMATION_SCHEMA
from envs.model import FLIGHT_CONTROLLER_NAMES
from envs.observation_encoder import OBSERVATION_SCHEMA, PLATFORM_PROFILE, SELF_DIM, TEAM_DIM, ENEMY_DIM, MISSILE_DIM
from algorithms.utils.action_spec import (ACTION_DIMS, ACTION_WIDTH, ACTION_NAMES, ACTION_OFFSETS,
                                         NEUTRAL_ACTION, WEAPON_HEAD, WEAPON_OFFSET, WEAPON_CHOICES, FLIGHT_CONTROLLER)


SAMPLE = (ROOT / "files" / "2026-智能空中博弈算法挑战赛" /
          "2026-智能空中博弈算法挑战赛-科目一二智能体提交教学说明" /
          "样例代码" / "智能体样例代码" / "team_name")


class DeterministicActor(nn.Module):
    def __init__(self, actor):
        super().__init__()
        self.actor = actor

    def forward(self, observation, hidden, reset_mask, action_mask):
        features = self.actor.base(observation)
        recurrent, next_hidden = self.actor.rnn(features, hidden, reset_mask)
        features = self.actor.action.base(recurrent)
        pieces = torch.split(action_mask, ACTION_DIMS, dim=-1)
        action = torch.stack([head(features).masked_fill(~valid, -1e9).argmax(-1)
                              for head, valid in zip(self.actor.action.heads, pieces)], dim=-1)
        return action, next_hidden


def _definitions(path, names):
    source = Path(path).read_text(encoding="utf-8")
    tree = ast.parse(source)
    found = {node.name: ast.get_source_segment(source, node) for node in tree.body
             if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names}
    missing = set(names)-set(found)
    if missing:
        raise ValueError(f"Missing runtime definitions in {path}: {sorted(missing)}")
    return "\n\n".join(found[name] for name in names)


def runtime_source(hidden_dim, actor_dim, critic_dim, num_agents, feet_to_meters,
                   earth_meters_per_degree, action_dims, action_width, flight_controller,
                   aircraft_config, platform_profile=None, formation_schema=None,
                   flight_controller_name="baseline"):
    utilities = _definitions(ROOT / "envs" / "utils" / "utils.py", ["decode_aircraft_id", "has_locked_target"])
    obs = _definitions(ROOT / "envs" / "observation_encoder.py",
                       ["alive", "digits_mask", "position", "offset", "velocity", "angle_features",
                        "_motion_available", "_relative_features", "_self_features", "_missile_position",
                        "_missile_velocity", "ObservationEncoder"])
    tracker = _definitions(ROOT / "envs" / "state_tracker.py", ["StateTracker"])
    weapons = _definitions(ROOT / "envs" / "weapon_control.py",
                           ["_ready_slots", "_locked_indices", "_amraam_targets", "weapon_plans", "WeaponController"])
    mask = _definitions(ROOT / "envs" / "action_mask.py", ["masks_for_side"])
    baseline = _definitions(ROOT / "envs" / "model" / "baseline" / "baseline_actor.py",
                            ["check", "MLPLayer", "MLPBase", "GRULayer", "Categorical", "ACTLayer", "BaselineActor"])
    controller = _definitions(ROOT / "envs" / "model" / "flight_controller.py",
                              ["calibrated_airspeed_mps", "FrozenFlightController"])
    adapter = _definitions(ROOT / "envs" / "action_adapter.py", ["ActionAdapter"])
    decoding = _definitions(ROOT / "envs" / "utils" / "observations.py", ["decode_actor"])
    rules = _definitions(ROOT / "envs" / "control_rules.py", ["ControlRules"])
    threat = _definitions(ROOT / "envs" / "reward" / "fkr" / "missile_avoid_reward.py", ["threat_active"])
    formation = _definitions(ROOT / "envs" / "formations" / "controller.py", ["FormationController"])
    header = '''import json
import math
import random
from itertools import permutations
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from .pid.controller import PIDFlightController
except (ImportError, KeyError):
    from envs.model.pid.controller import PIDFlightController

FT = __FEET_TO_METERS__
SCENARIO = {"aircraft_per_side": __NUM_AGENTS__, "red_ids": list(range(__NUM_AGENTS__))}
SELF_DIM = __SELF_DIM__
TEAM_DIM = __TEAM_DIM__
ENEMY_DIM = __ENEMY_DIM__
MISSILE_DIM = __MISSILE_DIM__
ACTOR_DIM = __ACTOR_DIM__
CRITIC_DIM = __CRITIC_DIM__
ACTION_DIMS = __ACTION_DIMS__
ACTION_WIDTH = __ACTION_WIDTH__
ACTION_NAMES = __ACTION_NAMES__
ACTION_OFFSETS = __ACTION_OFFSETS__
NEUTRAL_ACTION = __NEUTRAL_ACTION__
WEAPON_HEAD = __WEAPON_HEAD__
WEAPON_OFFSET = __WEAPON_OFFSET__
WEAPON_CHOICES = __WEAPON_CHOICES__
FLIGHT_CONTROLLER = __FLIGHT_CONTROLLER__
OBSERVATION_SCHEMA = __OBSERVATION_SCHEMA__
PLATFORM_PROFILE = __PLATFORM_PROFILE__
CONTROL_SCHEMA = __CONTROL_SCHEMA__
THREAT_POLICY = __THREAT_POLICY__
FORMATION_SCHEMA = __FORMATION_SCHEMA__
FLIGHT_CONTROLLER_NAME = __FLIGHT_CONTROLLER_NAME__

def validate_flight_controller(name):
    if name not in ("baseline", "pid"):
        raise ValueError("Unknown flight controller")
    return name
'''
    footer = '''

class DeployedPolicy:
    def __init__(self, agent_dir, formation="random", seed=None):
        if formation != "random" and formation not in FORMATION_SCHEMA["templates"]:
            raise ValueError("Unknown opening formation")
        self.requested_formation = formation
        self.formation_rng = random.Random(seed)
        agent_dir = Path(agent_dir)
        self.model = torch.jit.load(str(agent_dir / "actor.ts"), map_location="cpu").eval()
        stats = json.loads((agent_dir / "obs_normalizer.json").read_text(encoding="utf-8"))
        self.mean = np.asarray(stats["mean"], dtype=np.float64)
        self.var = np.asarray(stats["var"], dtype=np.float64)
        self.profile = json.loads((agent_dir / "platform_profile.json").read_text(encoding="utf-8"))
        self.encoder = ObservationEncoder(
            self.profile["weapon"]["aim_mode_short"],
            self.profile["limits"]["enemy_detection_m"],
            self.profile["limits"]["missile_detection_m"],
            num_agents=__NUM_AGENTS__,
            feet_to_meters=self.profile["units"]["feet_to_meters"],
            earth_meters_per_degree=self.profile["units"]["earth_meters_per_degree"],
            life_max=self.profile["aircraft"]["initial_life"],
            sraam_capacity=self.profile["aircraft"]["sraam_capacity"],
            amraam_capacity=self.profile["aircraft"]["amraam_capacity"],
            filter_visibility=False,
        )
        controller_model = agent_dir / ("baseline_model.pt" if FLIGHT_CONTROLLER_NAME == "baseline" else "pid/model.json")
        self.adapter = ActionAdapter(self.encoder, self.profile, model_path=controller_model,
                                     flight_controller=FLIGHT_CONTROLLER_NAME)
        self.rules = ControlRules(self.profile, FORMATION_SCHEMA)
        self.reset()

    def reset(self):
        self.tracker = StateTracker()
        self.encoder.selected_missiles.clear()
        self.adapter.reset()
        self.rules.formation.clear()
        self.formations = {}
        self.hidden = np.zeros((__NUM_AGENTS__, __HIDDEN_DIM__), dtype=np.float32)
        self.reset_mask = np.zeros((__NUM_AGENTS__, 1), dtype=np.float32)

    def act(self, state, side, count):
        if side not in state:
            state = {"red": {name: data for name, data in state.items() if name.startswith("red_")},
                     "blue": {name: data for name, data in state.items() if name.startswith("blue_")},
                     "missile": state.get("missile", {})}
        sim_time = max((float(p.get("simulation/sim-time-sec", 0)) for p in state.get(side, {}).values()), default=0.0)
        if side not in self.formations:
            name = self.formation_rng.choice(tuple(FORMATION_SCHEMA["templates"])) if self.requested_formation == "random" else self.requested_formation
            self.rules.formation.reset(state, side, name, sim_time)
            self.formations[side] = name
        obs = self.encoder.encode_actor(state, side, self.tracker, sim_time)
        legal = masks_for_side(state, side, self.encoder, self.profile, self.adapter.weapons.pending)
        normalized = np.clip((obs-self.mean)/np.sqrt(self.var+1e-6), -10, 10).astype(np.float32)
        with torch.no_grad():
            action, hidden = self.model(
                torch.from_numpy(normalized), torch.from_numpy(self.hidden),
                torch.from_numpy(self.reset_mask), torch.from_numpy(legal.astype(np.bool_)))
        self.hidden = hidden.numpy()
        self.reset_mask = np.array([[float(alive(state.get(side, {}).get(f"{side}_{i}", {})))]
                                    for i in range(__NUM_AGENTS__)], dtype=np.float32)
        frames = [decode_actor(row, self.encoder, alive(state.get(side, {}).get(f"{side}_{i}", {})),
                               self.encoder.selected_missiles.get(f"{side}_{i}")) for i, row in enumerate(obs)]
        executed, self.last_policy_control_mask, self.last_control_modes = self.rules.route(state, side, frames, action.numpy())
        commands = self.adapter.side_commands(state, side, executed, self.tracker,
                                              weapon_enabled=self.last_policy_control_mask[:, 0])
        self.tracker.update(state, sim_time)
        return {side: {f"{side}_{i}": commands[f"{side}_{i}"] for i in range(count)}}
'''
    source = "\n\n".join((header, utilities, threat, obs, tracker, weapons, mask, baseline, controller, adapter, decoding, formation, rules,
                          footer.replace("np.zeros((4, 128)", f"np.zeros(({num_agents}, {hidden_dim})")))
    profile = platform_profile or PLATFORM_PROFILE
    runtime_profile = {**profile, "units": {**profile["units"], "feet_to_meters": feet_to_meters,
                       "earth_meters_per_degree": earth_meters_per_degree}, "aircraft": dict(aircraft_config)}
    source = source.replace("__OBSERVATION_SCHEMA__", repr(OBSERVATION_SCHEMA))
    source = source.replace("__PLATFORM_PROFILE__", repr(runtime_profile))
    source = source.replace("__CONTROL_SCHEMA__", repr(CONTROL_SCHEMA))
    source = source.replace("__THREAT_POLICY__", repr(THREAT_POLICY))
    source = source.replace("__FORMATION_SCHEMA__", repr(formation_schema or FORMATION_SCHEMA))
    source = source.replace("__FLIGHT_CONTROLLER_NAME__", repr(flight_controller_name))
    source = source.replace("__FEET_TO_METERS__", repr(feet_to_meters))
    source = source.replace("__SELF_DIM__", str(SELF_DIM))
    source = source.replace("__TEAM_DIM__", str(TEAM_DIM))
    source = source.replace("__ENEMY_DIM__", str(ENEMY_DIM))
    source = source.replace("__MISSILE_DIM__", str(MISSILE_DIM))
    source = source.replace("__ACTOR_DIM__", str(actor_dim))
    source = source.replace("__CRITIC_DIM__", str(critic_dim))
    source = source.replace("__ACTION_DIMS__", repr(tuple(action_dims)))
    source = source.replace("__ACTION_WIDTH__", str(action_width))
    source = source.replace("__ACTION_NAMES__", repr(ACTION_NAMES))
    source = source.replace("__ACTION_OFFSETS__", repr(ACTION_OFFSETS))
    source = source.replace("__NEUTRAL_ACTION__", repr(NEUTRAL_ACTION))
    source = source.replace("__WEAPON_HEAD__", repr(WEAPON_HEAD))
    source = source.replace("__WEAPON_OFFSET__", repr(WEAPON_OFFSET))
    source = source.replace("__WEAPON_CHOICES__", repr(WEAPON_CHOICES))
    source = source.replace("__MISSILE_SLOTS__", str(OBSERVATION_SCHEMA["missile_slots"]))
    source = source.replace("__OBS_NORMALIZATION__", repr(OBSERVATION_SCHEMA["normalization"]))
    source = source.replace("__OBS_VERSION__", str(OBSERVATION_SCHEMA["version"]))
    source = source.replace("__NUM_AGENTS__", str(num_agents))
    source = source.replace("__HIDDEN_DIM__", str(hidden_dim))
    source = source.replace("__FLIGHT_CONTROLLER__", repr(flight_controller))
    source = source.replace("__EARTH_METERS_PER_DEGREE__", repr(earth_meters_per_degree))
    source = source.replace("__INITIAL_LIFE__", repr(aircraft_config["initial_life"]))
    source = source.replace("__SRAAM_CAPACITY__", repr(aircraft_config["sraam_capacity"]))
    source = source.replace("__AMRAAM_CAPACITY__", repr(aircraft_config["amraam_capacity"]))
    source = source.replace("__NUM_AGENTS__", str(num_agents))
    return source


RED_AGENT = '''import json
from pathlib import Path

from agent.base_agent import Base_agent
from agent.runtime import DeployedPolicy


class Example_agent(Base_agent):
    def __init__(self, side, project_name, ip, port, log_txt_name):
        super().__init__(side, project_name, ip, port, log_txt_name)
        self.policy = DeployedPolicy(Path(__file__).resolve().parent)

    def reset(self):
        self.policy.reset()

    def agent_step(self, is_reset=False):
        try:
            packet = self.socket.recv(8192 * 16)
            if not packet:
                return False
            observation = json.loads(packet.decode("utf-8"))
            if observation.get("is_reset", is_reset):
                self.reset()
            action = self.policy.act(observation, self.side, self.control_num)
            self.socket.send(json.dumps(action).encode("utf-8"))
            return True
        except Exception as error:
            self.log_txt(str(error), "a")
            return False
'''


class Exporter:
    def __init__(self, checkpoint, team_name, output, flight_controller=None):
        self.actor_model = load_actor_model(checkpoint)
        self.checkpoint = self.actor_model["path"].parent
        self.team_name = team_name
        self.folder = Path(output) / team_name
        self.configs = project_configs()
        config_file = self.checkpoint / "config.yaml"
        profile_file = self.checkpoint / "platform_profile.yaml"
        if config_file.is_file():
            self.configs["train_mappo"] = yaml.safe_load(config_file.read_text(encoding="utf-8"))
        if profile_file.is_file():
            self.configs["platform_profile"] = yaml.safe_load(profile_file.read_text(encoding="utf-8"))
        if flight_controller is None:
            flight_controller = self.actor_model.get("flight_controller")
        if flight_controller is not None:
            if flight_controller not in FLIGHT_CONTROLLER_NAMES:
                raise ValueError(f"Unknown flight controller: {flight_controller}")
            self.configs["train_mappo"]["flight_controller"] = flight_controller
        self.num_agents = int(self.configs["scenario_4v4"]["aircraft_per_side"])
        self.actor_dim = int(OBSERVATION_SCHEMA["self_dim"] +
                            (self.num_agents - 1) * OBSERVATION_SCHEMA["teammate_dim"] +
                            self.num_agents * OBSERVATION_SCHEMA["enemy_dim"] +
                            OBSERVATION_SCHEMA["missile_slots"] * OBSERVATION_SCHEMA["missile_dim"])
        self.critic_dim = self.num_agents*2*self.actor_dim+int(OBSERVATION_SCHEMA["critic"]["include_remaining_time"])

    def export(self):
        agent = self.folder / "agent"
        environment = self.folder / "environment"
        check = self.folder / "self_check"
        for folder in (agent, environment, check):
            folder.mkdir(parents=True, exist_ok=True)
        if self.actor_model["obs_dim"] != self.actor_dim:
            raise ValueError("Actor observation dimension differs from the configured scenario")
        actor = PPOActor(self.actor_dim, self.actor_model["hidden_dim"])
        actor.load_state_dict(self.actor_model["state_dict"])
        actor.eval()
        wrapper = DeterministicActor(actor).eval()
        sample = (torch.zeros((self.num_agents, self.actor_dim)), torch.zeros((self.num_agents, actor.hidden_dim)),
                  torch.ones((self.num_agents, 1)), torch.ones((self.num_agents, ACTION_WIDTH), dtype=torch.bool))
        traced = torch.jit.trace(wrapper, sample, check_trace=False)
        traced.save(str(agent / "actor.ts"))
        shutil.copy2(SAMPLE / "agent" / "base_agent.py", agent / "base_agent.py")
        main_source = (SAMPLE / "main_versus.py").read_text(encoding="utf-8")
        main_source = "import sys\nfrom pathlib import Path\nsys.path.insert(0, str(Path(__file__).resolve().parents[1]))\n" + main_source
        (agent / "main_versus.py").write_text(main_source, encoding="utf-8")
        (agent / "runtime.py").write_text(
            runtime_source(actor.hidden_dim, self.actor_dim, self.critic_dim, self.num_agents,
                           self.configs["platform_profile"]["units"]["feet_to_meters"],
                           self.configs["platform_profile"]["units"]["earth_meters_per_degree"],
                           ACTION_DIMS, ACTION_WIDTH, FLIGHT_CONTROLLER,
                           self.configs["platform_profile"]["aircraft"], platform_profile=self.configs["platform_profile"],
                           formation_schema=self.actor_model.get("formation_schema"),
                           flight_controller_name=self.configs["train_mappo"].get("flight_controller", "baseline")),
            encoding="utf-8")
        baseline = self.checkpoint / "baseline_model.pt"
        if not baseline.is_file():
            baseline = ROOT / FLIGHT_CONTROLLER["model_path"]
        shutil.copy2(baseline, agent / "baseline_model.pt")
        pid_package = agent / "pid"
        pid_package.mkdir(exist_ok=True)
        for name in ("__init__.py", "controller.py", "contracts.py", "pid.py", "pid_config.json",
                     "pid_speed_proportional.py", "runtime.py", "model.json"):
            shutil.copy2(ROOT / "envs" / "model" / "pid" / name, pid_package / name)
        shutil.copy2(ROOT / "envs" / "model" / "LICENSE", agent / "baseline_LICENSE")
        (agent / "red_agent.py").write_text(RED_AGENT, encoding="utf-8")
        (agent / "blue_agent.py").write_text("from agent.red_agent import Example_agent\n", encoding="utf-8")
        (agent / "__init__.py").write_text("", encoding="utf-8")
        stats = self.actor_model["obs_normalizer"]
        (agent / "obs_normalizer.json").write_text(
            json.dumps({key: np.asarray(stats[key]).tolist() for key in ("mean", "var")}), encoding="utf-8")
        (agent / "platform_profile.json").write_text(
            json.dumps(self.configs["platform_profile"], indent=2), encoding="utf-8")
        install = (ROOT / "submission" / "team_name" / "environment" / "install.sh").read_text(encoding="utf-8")
        (environment / "install.sh").write_text(install.replace("team_name", self.team_name), encoding="utf-8")
        shutil.copy2(ROOT / "submission" / "team_name" / "environment" / "requirements.txt",
                     environment / "requirements.txt")
        for name in ("main_competition.py", "env_competition.py", "worker_competition.py"):
            shutil.copy2(ROOT / "submission" / "team_name" / "self_check" / name, check / name)
        return self.folder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Run directory, actor_latest.pt or a historical actor_<update>.pt")
    parser.add_argument("--team-name", required=True)
    parser.add_argument("--output", default=str(ROOT / "submission"))
    parser.add_argument("--flight-controller", choices=FLIGHT_CONTROLLER_NAMES,
                        help="Override the controller recorded in the training config")
    args = parser.parse_args()
    if not args.team_name.isascii() or not args.team_name.replace("_", "").isalnum():
        raise ValueError("team-name must be an ASCII identifier")
    print(Exporter(args.checkpoint, args.team_name, args.output, args.flight_controller).export())


if __name__ == "__main__":
    main()
