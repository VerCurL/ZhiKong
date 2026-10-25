"""
已训练模型的成对对战评估入口。

文件作用：
    加载待评估模型和对手模型，让双方在比赛平台上对战，比较模型的实际表现。
    评估只使用双方 Actor 和各自的观测归一化统计，不更新网络参数。

主要流程：
    1. 从 --checkpoint 和 --opponent 指定的检查点目录加载模型及相关配置。
    2. 连接已经启动的平台，通过红、蓝两个 Socket 控制双方飞机。
    3. 对每组初始场景执行两场对战，待评估模型分别控制红方和蓝方。
    4. 使用确定性动作推理，记录胜负、存活数、弹药、发射与命中统计和仿真时间。
    5. 输出各场评估结果，并打印胜场、平局和综合得分。

使用说明：
    本脚本不会自动启动平台，需要提前准备好可连接的平台实例。
    --rounds 表示初始场景组数，默认 50 组，即实际执行 100 场对战。
    默认结果文件为 results/evaluations/paired_eval.csv。

运行示例（在项目根目录执行）：
    python scripts/evaluate_mappo.py --checkpoint result/4v4/mappo/run-时间/actor_20.pt --opponent result/4v4/mappo/run-时间/actor_10.pt --rounds 50
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import yaml

from algorithms.mappo.ppo_actor import PPOActor
from algorithms.utils.normalization import RunningMoments
from config import ROOT, project_configs
from envs.competition_env import CompetitionEnv
from envs.model import FLIGHT_CONTROLLER_NAMES
from runner.evaluation_runner import EvaluationRunner
from algorithms.utils.model_io import load_actor_model


def load_actor(model, obs_dim):
    if model["obs_dim"] != obs_dim:
        raise ValueError("Actor observation dimension differs from the configured scenario")
    actor = PPOActor(obs_dim, model["hidden_dim"])
    actor.load_state_dict(model["state_dict"])
    return actor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Run directory or actor model file")
    parser.add_argument("--opponent", required=True, help="Run directory or actor model file")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--red-port", type=int)
    parser.add_argument("--blue-port", type=int)
    parser.add_argument("--rounds", type=int, default=50, help="Number of paired initializations")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--flight-controller", choices=FLIGHT_CONTROLLER_NAMES,
                        help="Fixed lower-level flight controller; defaults to saved training config")
    parser.add_argument("--output", default=str(ROOT / "results" / "evaluations" / "paired_eval.csv"))
    args = parser.parse_args()
    model = load_actor_model(args.checkpoint)
    opponent_model = load_actor_model(args.opponent)
    checkpoint_folder = model["path"].parent
    configs = project_configs()
    saved_config = checkpoint_folder / "config.yaml"
    saved_profile = checkpoint_folder / "platform_profile.yaml"
    saved_scenario = checkpoint_folder / "scenario_4v4.yaml"
    if saved_config.is_file():
        configs["train_mappo"] = yaml.safe_load(saved_config.read_text(encoding="utf-8"))
    if saved_profile.is_file():
        configs["platform_profile"] = yaml.safe_load(saved_profile.read_text(encoding="utf-8"))
    if saved_scenario.is_file():
        configs["scenario_4v4"] = yaml.safe_load(saved_scenario.read_text(encoding="utf-8"))
    host = args.host or configs["train_mappo"]["host"]
    eval_seed = args.seed if args.seed is not None else configs["train_mappo"]["evaluation_seed"]
    base_port = args.port or configs["train_mappo"]["red_port"]
    baseline = checkpoint_folder / "baseline_model.pt"
    flight_controller = args.flight_controller or model.get("flight_controller") or configs["train_mappo"].get("flight_controller", "baseline")
    env = CompetitionEnv(configs["platform_profile"], configs["scenario_4v4"],
                         host=host, port=base_port,
                         red_port=args.red_port or base_port,
                         blue_port=args.blue_port or base_port + 1,
                         seed=eval_seed, flight_controller_path=baseline if flight_controller == "baseline" and baseline.is_file() else None,
                         flight_controller=flight_controller)
    actor = load_actor(model, env.encoder.actor_dim)
    opponent = load_actor(opponent_model, env.encoder.actor_dim)
    normalizer = RunningMoments(env.encoder.actor_dim)
    normalizer.load_state_dict(model["obs_normalizer"])
    opponent_normalizer = RunningMoments(env.encoder.actor_dim)
    opponent_normalizer.load_state_dict(opponent_model["obs_normalizer"])
    evaluator = EvaluationRunner(env, actor, opponent, normalizer, opponent_normalizer)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    try:
        for index in range(args.rounds):
            env.rng.seed(eval_seed+index)
            request = env._initial_attributes()
            initialization = {side: request[side] for side in ("red", "blue")}
            for side in ("red", "blue"):
                result = evaluator.run_round(initialization, side)
                result["pair"] = index
                rows.append(result)
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        wins = sum(row["result"] == "win" for row in rows)
        ties = sum(row["result"] == "tie" for row in rows)
        print(f"rounds={len(rows)} wins={wins} ties={ties} score={(wins+0.5*ties)/len(rows):.3f}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
