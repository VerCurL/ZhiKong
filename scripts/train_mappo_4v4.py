"""
4v4 MAPPO 训练入口。

文件作用：
    读取训练、场景、平台、奖励和自博弈配置，启动完整的采样与训练流程。
    具体的环境管理、轨迹采样和网络更新由 CompetitionRunner 及其调用的模块完成。

主要流程：
    1. 读取 configs/ 下的配置，并应用命令行指定的端口、并行环境数、采样总局数等参数。
    2. 新训练创建 run 目录；--resume 在原目录加载模型和归一化，使用新的优化器继续训练。
    3. 每轮分批并行对战，等待每批全部结束后开始下一批；完成总局数后才进行 PPO 更新。
    4. 每轮保存模型，写训练统计和独立奖励统计 CSV。
    5. 训练结束或异常退出时关闭连接，并清理本次训练启动的平台进程。

输出内容：
    默认保存到 results/4v4/mappo/run-时间/，log/training.csv 记录训练统计，
    log/rewards.csv 记录各类奖励的每局均值、标准差、极值和本轮总和。

运行示例（在项目根目录执行）：
    python scripts/train_mappo_4v4.py
    # 选择 C50-M1 PID 作为下层固定飞控
    python scripts/train_mappo_4v4.py --flight-controller pid
    python scripts/train_mappo_4v4.py --envs 2 --rollout-episodes 8 --updates 10
    # 中断后，从已有 run 中最后完整保存的一轮继续（将目录替换为自己的 run）。
    python scripts/train_mappo_4v4.py --resume results/4v4/mappo/run-20261022-141515

    # 继续训练到总计第 1000 轮：每轮采样 32 局，同时最多运行 8 个环境。
    python scripts/train_mappo_4v4.py --resume results/4v4/mappo/run-20261022-141515 --envs 8 --rollout-episodes 32 --updates 1000

恢复说明：
    --resume 指向包含 actor_latest.pt 和 critic_latest.pt 的旧 run 目录，二者轮数必须一致。
    若最后保存为第 7 轮，则从第 8 轮重新采样；未保存的采样和训练进度不会恢复。
    --updates 是目标总轮数，不是本次新增轮数，必须大于检查点轮数。
    恢复 Actor、Critic 及观测归一化；优化器、历史对手池和随机状态重新开始。
    模型结构、观测、动作和执行规则须与当前代码兼容；原训练所用的非默认配置需自行保持一致。
    模型继续写入指定 resume 所在的 run 目录，CSV 追加记录；缺少 log 目录或 CSV 时自动创建。
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
