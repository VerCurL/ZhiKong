"""
4v4 MAPPO 训练入口。

读取训练、场景、平台、奖励和自博弈配置，启动完整的采样与训练流程。
单局训练版对战诊断位于 tests/test_training_battle.py。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import project_configs, training_arguments, validate_configs
from runner.competition_runner import CompetitionRunner


def main():
    args = training_arguments()
    configs = project_configs()
    train = configs["train_mappo"]
    if args.host:
        train["host"] = args.host
    if args.port:
        train["red_port"] = args.port
        train["blue_port"] = args.port + 1
    if args.red_port:
        train["red_port"] = args.red_port
    if args.blue_port:
        train["blue_port"] = args.blue_port
    if args.updates:
        train["max_updates"] = args.updates
    if args.side:
        train["control_side"] = args.side
    if args.flight_controller:
        train["flight_controller"] = args.flight_controller
    if args.envs is not None:
        train["parallel_envs"] = args.envs
    if args.rollout_episodes is not None:
        train["n_rollout_threads"] = args.rollout_episodes
    if args.no_launch_platforms:
        train["platform_launch"]["enabled"] = False
    validate_configs(configs)
    runner = CompetitionRunner(train, configs["scenario_4v4"], configs["platform_profile"],
                               configs["selfplay"], args.output, resume=args.resume)
    try:
        start_update = runner.load(args.resume) if args.resume else 0
        runner.run(start_update=start_update)
    finally:
        runner.close()


if __name__ == "__main__":
    main()
