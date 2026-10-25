"""
比赛平台时间步长、探测和武器状态的测量入口。

文件作用：
    通过固定动作驱动平台，记录平台实际返回的状态，辅助核实训练使用的平台参数。
    本脚本不加载训练模型，也不会根据测量结果自动修改配置文件。

主要流程：
    1. 读取平台、场景和奖励配置，连接已经启动的平台并初始化一场对战。
    2. 给双方发送中性动作；指定 --fire 时，让 red_0 尝试发射导弹。
    3. 逐步记录仿真时间、时间差 dt、最近敌机距离、探测和攻击范围状态。
    4. 同时记录锁定、剩余弹药、告警、飞行中导弹、越界、生命值和死亡事件。
    5. 达到指定步数或对战结束后，保存测量表格和简要汇总。

使用说明：
    本脚本不会自动启动平台，需要提前准备好可连接的平台实例。
    --steps 默认 5000；--fire 表示请求发射，实际能否发射仍取决于合法动作和平台条件。
    默认输出 data/calibration/platform_probe.csv 及同名 JSON 汇总文件。

运行示例（在项目根目录执行）：
    python scripts/calibrate_platform.py --steps 5000
    python scripts/calibrate_platform.py --steps 5000 --fire
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from config import ROOT, project_configs
from envs.competition_env import CompetitionEnv
from envs.observation_encoder import offset
from envs.termination import simulation_time
from algorithms.utils.action_spec import NEUTRAL_ACTION, WEAPON_HEAD


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--red-port", type=int)
    parser.add_argument("--blue-port", type=int)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--fire", action="store_true", help="Request red_0 missile launches during the probe")
    parser.add_argument("--output", default=str(ROOT / "data" / "calibration" / "platform_probe.csv"))
    args = parser.parse_args()
    configs = project_configs()
    host = args.host or configs["train_mappo"]["host"]
    base_port = args.port or configs["train_mappo"]["red_port"]
    env = CompetitionEnv(configs["platform_profile"], configs["scenario_4v4"], host=host, port=base_port,
                         red_port=args.red_port or base_port,
                         blue_port=args.blue_port or base_port + 1)
    neutral = np.tile(np.array(NEUTRAL_ACTION, dtype=np.int64), (env.num_agents, 1))
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["step", "sim_time", "dt", "aircraft", "nearest_enemy_m", "target_visible", "attack_range",
              "aim_mode", "sraam_lock", "amraam_lock", "sraam_left", "amraam_left",
              "warning", "missile_alert", "missiles_flying", "out_of_bounds", "life", "death"]
    try:
        _, _, info = env.reset()
        previous_time = simulation_time(info["raw_state"])
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for step in range(args.steps):
                red_action = neutral.copy()
                if args.fire:
                    red_action[0, WEAPON_HEAD] = 1
                _, _, _, _, info = env.step({"red": red_action, "blue": neutral})
                now = info["sim_time"]
                for side in ("red", "blue"):
                    other = "blue" if side == "red" else "red"
                    for index in range(env.num_agents):
                        name = f"{side}_{index}"
                        plane = info["raw_state"][side][name]
                        distances = [offset(plane, target,
                                            configs["platform_profile"]["units"]["feet_to_meters"],
                                            configs["platform_profile"]["units"]["earth_meters_per_degree"])[3]
                                      for target in info["raw_state"][other].values()]
                        writer.writerow({"step": step, "sim_time": now, "dt": now-previous_time,
                                         "aircraft": name, "nearest_enemy_m": min(distances),
                                         "target_visible": plane.get("TargetIntoView"),
                                         "attack_range": plane.get("TargetEnterAttackRange"),
                                         "aim_mode": plane.get("AimMode"),
                                         "sraam_lock": plane.get("SRAAMTargetLocked"),
                                         "amraam_lock": plane.get("AMRAAMlockedTarget"),
                                         "sraam_left": plane.get("SRAAMCurrentNum"),
                                         "amraam_left": plane.get("AMRAAMCurrentNum"),
                                         "warning": plane.get("WarningNumber"),
                                         "missile_alert": plane.get("MissileAlert"),
                                         "missiles_flying": sum(int(m.get("Status", 0)) == 0 for m in info["raw_state"].get("missile", {}).values()),
                                         "out_of_bounds": plane.get("IsOutOfValidBattleArea"),
                                         "life": plane.get("LifeCurrent"),
                                         "death": plane.get("DeathEvent")})
                previous_time = now
                if info["episode_done"]:
                    break
        summary = {"rows": step+1, "last_sim_time": previous_time, "csv": str(path)}
        path.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary))
    finally:
        env.close()


if __name__ == "__main__":
    main()
