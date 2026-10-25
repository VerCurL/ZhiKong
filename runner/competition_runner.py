"""Training lifecycle, per-run CSV logs and model-only saves."""

from datetime import datetime, timedelta, timezone
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from algorithms.mappo.ppo_actor import PPOActor
from algorithms.mappo.ppo_policy import PPOPolicy
from algorithms.mappo.ppo_trainer import PPOTrainer
from algorithms.utils.normalization import RunningMoments
from algorithms.utils.torch_execution import torch_thread_count
from algorithms.utils.selfplay import HistoricalActorPool
from algorithms.utils.model_io import actor_file, actor_payload, load_actor_model, load_critic_model
from envs.competition_env import ParallelEnvironments
from envs.observation_encoder import ObservationEncoder, OBSERVATION_SCHEMA
from envs.formations import FORMATION_NAMES
from envs.reward.fkr.multiplecombat_reward import REWARD_METRIC_COLUMNS, REWARD_RATIO_COLUMNS, REWARD_VERSION
from config import TRAIN_OUTPUT_ROOT, validate_configs
from .rollout_runner import RolloutRunner
from .training_logger import TrainingLogger
from .reward_recorder import RewardRecorder, append_csv


def select_device(requested):
    if requested == "cpu":
        return "cpu"
    if requested not in ("auto", "cuda"):
        return requested
    if not torch.cuda.is_available():
        if requested == "cuda":
            raise RuntimeError("CUDA was requested but is unavailable")
        return "cpu"
    major, minor = torch.cuda.get_device_capability()
    architecture = f"sm_{major}{minor}"
    supported = torch.cuda.get_arch_list()
    if architecture not in supported:
        if requested == "cuda":
            raise RuntimeError(f"Installed PyTorch does not support GPU architecture {architecture}")
        print(f"PyTorch does not support {architecture}; using CPU", flush=True)
        return "cpu"
    return "cuda"


class CompetitionRunner:
    def __init__(self, config, scenario, profile, selfplay, output_root=None, client=None, clients=None, resume=None):
        validate_configs({"train_mappo": config, "scenario_4v4": scenario,
                          "platform_profile": profile, "selfplay": selfplay})
        if client is not None:
            if clients is not None or min(config["parallel_envs"], config["n_rollout_threads"]) != 1:
                raise ValueError("Single injected client requires one actual parallel environment")
            clients = [client]
        self.config = config
        self.scenario = scenario
        self.profile = profile
        if resume is not None:
            self.output_root = actor_file(resume).resolve().parent
        else:
            root = Path(output_root) if output_root is not None else TRAIN_OUTPUT_ROOT
            stamp = datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=8))).strftime("%Y%m%d-%H%M%S")
            agents = int(scenario["aircraft_per_side"])
            self.output_root = root/f"{agents}v{agents}"/"mappo"/f"run-{stamp}"
            self.output_root.mkdir(parents=True, exist_ok=False)
        (self.output_root/"log").mkdir(parents=True, exist_ok=True)
        print(f"训练输出目录：{self.output_root}", flush=True)
        self.device = select_device(config["device"])
        random.seed(config["seed"])
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        encoder = ObservationEncoder(num_agents=scenario["aircraft_per_side"])
        self.policy = PPOPolicy(encoder.actor_dim, encoder.critic_dim, config, self.device)
        self.opponent = PPOActor(encoder.actor_dim, config["actor_hidden"]).to(self.policy.device)
        self.opponent.eval()
        self.pool = HistoricalActorPool(selfplay["history_limit"], selfplay["latest_probability"], config["seed"])
        self.normalizer = RunningMoments(encoder.actor_dim)
        self.trainer = PPOTrainer(self.policy, config)
        self.rng = np.random.default_rng(config["seed"])
        self.envs = ParallelEnvironments(config, profile, scenario, clients=clients)
        self.rollout = RolloutRunner(self.envs, self.policy, self.opponent, self.pool, self.normalizer, config)

    def save(self, update):
        self.output_root.mkdir(parents=True, exist_ok=True)
        path = self.output_root/f"actor_{update}.pt"
        torch.save(actor_payload(self.policy.actor, self.normalizer, update,
                                 flight_controller=self.config.get("flight_controller", "baseline")), path)
        # Publish latest models atomically so readers never see a partial file.
        for name in ("actor_latest", "critic_latest"):
            temporary = self.output_root/f".{name}.tmp"
            try:
                if name == "actor_latest":
                    shutil.copyfile(path, temporary)
                else:
                    torch.save({"state_dict": {key: tensor.detach().cpu() for key, tensor in self.policy.critic.state_dict().items()},
                                "update": int(update)}, temporary)
                temporary.replace(self.output_root/f"{name}.pt")
            finally:
                if temporary.exists():
                    temporary.unlink()
        return path

    def load(self, checkpoint_folder):
        actor = load_actor_model(checkpoint_folder)
        critic, update = load_critic_model(actor["path"].parent)
        if update is not None and update != actor["update"]:
            raise ValueError("Actor and critic updates differ; resume using the matching latest models")
        self.policy.actor.load_state_dict(actor["state_dict"])
        self.policy.critic.load_state_dict(critic)
        self.normalizer.load_state_dict(actor["obs_normalizer"])
        self.output_root = actor["path"].resolve().parent
        print(f"已加载第 {actor['update']} 轮模型；优化器、对手池和随机状态从本次运行重新开始。", flush=True)
        return actor["update"]

    def close(self):
        self.envs.close()

    def run(self, start_update=0):
        log_file = self.output_root / "log" / "training.csv"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        columns = ["update", "reward_sum", "finished_rounds", "wins", "ties", "losses",
                   "left_spawn_rounds", "right_spawn_rounds",
                   "environment_steps", "sample_slots", "training_chunks", "buffer_mb", "opponent",
                   "collect_seconds", "train_seconds", "policy_loss", "value_loss", "entropy",
                   "ratio", "optimizer_updates", "actor_optimizer_updates", "fps", "total_environment_steps", "elapsed_seconds",
                   "average_episode_reward", "win_rate"]
        columns += list(REWARD_METRIC_COLUMNS)+list(REWARD_RATIO_COLUMNS)
        columns += ["reward_version", "observation_version", "sampling_torch_threads", "training_torch_threads"]
        columns += [prefix+name+"_rounds" for prefix in ("formation_", "opponent_formation_") for name in FORMATION_NAMES]
        logger = TrainingLogger(self.config, self.envs.num_envs, self.envs.num_agents, self.device)
        try:
            with append_csv(log_file, columns) as (stream, writer), \
                    RewardRecorder(log_file.parent, self.config["reward_recorder"]["enabled"]) as reward_recorder:
                logger.header()
                # 与 LAGCFC 一致，统计整个训练循环的累计环境步吞吐量。
                # 恢复训练时分子和计时均从本次运行重新开始，不计历史步数。
                training_started = time.monotonic()
                total_environment_steps = 0
                for update in range(start_update+1, int(self.config["max_updates"])+1):
                    logger.start_collection(update)
                    started = time.monotonic()
                    buffer, rollout_info = self.rollout.collect(progress=logger.collection_progress)
                    collect_seconds = time.monotonic()-started
                    logger.collection_complete(rollout_info, collect_seconds)
                    logger.start_training(update)
                    started = time.monotonic()
                    with torch_thread_count(self.config["torch_threads"]["training"]):
                        train_info = self.trainer.update(buffer, self.rng, progress=logger.training_progress)
                    train_seconds = time.monotonic()-started
                    # Commit statistics only after every PPO pass has consumed the
                    # observations encoded under the frozen behavior statistics.
                    self.normalizer.merge(rollout_info.pop("pending_normalizer"))
                    self.pool.add(self.policy.actor, self.normalizer)
                    del buffer
                    results = [episode["outcome"] for episode in rollout_info["episodes"]]
                    checkpoint = self.save(update)
                    total_environment_steps += rollout_info["environment_steps"]
                    elapsed_seconds = time.monotonic()-training_started
                    row = {"update": update, "reward_sum": rollout_info["reward_sum"],
                           "finished_rounds": len(results), "wins": results.count(self.envs.side),
                           "ties": results.count("tie"), "losses": len(results)-results.count(self.envs.side)-results.count("tie"),
                           "collect_seconds": collect_seconds, "train_seconds": train_seconds,
                           "fps": int(total_environment_steps/max(elapsed_seconds, 1e-9)),
                           "total_environment_steps": total_environment_steps, "elapsed_seconds": elapsed_seconds,
                           "average_episode_reward": rollout_info["reward_sum"]/len(results),
                           "win_rate": results.count(self.envs.side)/len(results),
                           **{key: rollout_info[key] for key in ("environment_steps", "sample_slots", "training_chunks", "buffer_mb", "opponent",
                                                                "left_spawn_rounds", "right_spawn_rounds")},
                           **train_info, **rollout_info["reward_metrics"], **rollout_info["formation_counts"],
                           "reward_version": REWARD_VERSION, "observation_version": OBSERVATION_SCHEMA["version"],
                           "sampling_torch_threads": self.config["torch_threads"]["sampling"],
                           "training_torch_threads": self.config["torch_threads"]["training"]}
                    writer.writerow(row)
                    stream.flush()
                    reward_recorder.record(update, rollout_info, self.envs.side)
                    if update % int(self.config["log_interval"]) == 0:
                        logger.summary(row, checkpoint)
                    elif checkpoint is not None:
                        logger.checkpoint_saved(checkpoint)
        finally:
            logger.finish_progress()
            self.close()
