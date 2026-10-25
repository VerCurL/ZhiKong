"""训练阶段进度和每轮统计的终端展示；不依赖额外的终端界面库。"""

import sys


class TrainingLogger:
    def __init__(self, config, num_envs, num_agents, device, stream=None):
        self.config = config
        self.num_envs = num_envs
        self.num_agents = num_agents
        self.device = str(device)
        self.stream = stream if stream is not None else sys.stdout
        self.interactive = bool(getattr(self.stream, "isatty", lambda: False)())
        self.progress_width = 0
        self.opponent = None

    def _write(self, text, end="\n"):
        # Windows 的旧终端编码可能无法表示装饰符号，保留可表示的中文和数值。
        encoding = getattr(self.stream, "encoding", None) or "utf-8"
        try:
            text.encode(encoding)
        except UnicodeEncodeError:
            for symbol, replacement in (("▶", ">"), ("✓", "+"), ("░", "."), ("█", "#")):
                text = text.replace(symbol, replacement)
            text = text.encode(encoding, errors="replace").decode(encoding)
        self.stream.write(text + end)
        self.stream.flush()

    def finish_progress(self):
        if self.progress_width:
            self._write("")
            self.progress_width = 0

    def _line(self, text=""):
        self.finish_progress()
        self._write(text)

    def _progress(self, label, completed, total):
        filled = min(16, 16 * completed // max(total, 1))
        text = f"  {label}  [{'█' * filled}{'░' * (16-filled)}]  {completed} / {total}"
        if self.interactive:
            self._write("\r" + text.ljust(self.progress_width), end="")
            self.progress_width = max(self.progress_width, len(text))
            if completed == total:
                self.finish_progress()
        else:
            self._write(text)

    def header(self):
        self._line("═" * 64)
        self._line(f"  MAPPO · {self.num_agents}v{self.num_agents} 自博弈训练  │  机动模型 {self.config['flight_controller']}")
        self._line(f"  设备 {self.device.upper()}  │  并行环境 {self.num_envs}  │  "
                   f"每轮对战 {self.config['n_rollout_threads']} 场")
        batches = (int(self.config["n_rollout_threads"])+self.num_envs-1)//self.num_envs
        self._line(f"  每轮分 {batches} 批采样  │  每批全部结束后开始下一批")
        # self._line(f"  CPU 算子线程：采样 {self.config['torch_threads']['sampling']} / "
        #            f"PPO {self.config['torch_threads']['training']}  │  baseline 跨环境合批")
        self._line(f"  PPO {self.config['ppo_epochs']} epochs  │  Chunk {self.config['sequence_length']} 步  │  "
                   f"单批最多 {self.config['max_minibatch_samples']} 样本位置")
        self._line("═" * 64)

    def start_collection(self, update):
        self.opponent = None
        self._line(f"\n▶ 第 {update:04d} / {self.config['max_updates']} 轮  ·  采样阶段")

    def collection_progress(self, completed, total, opponent):
        if self.opponent is None:
            self.opponent = opponent
            self._line(f"  对手 {opponent}  │  训练方 {self.config['control_side'].upper()}")
        self._progress("对战进度", completed, total)

    def collection_complete(self, info, seconds):
        self._line("\n✓ 采样完成")
        self._line(f"  环境步数 {info['environment_steps']:,}  │  样本位置 {info['sample_slots']:,}")
        self._line(f"  训练 Chunk {info['training_chunks']:,}  │  Buffer {info['buffer_mb']:.1f} MiB  │  "
                   f"耗时 {seconds:.2f} s")

    def start_training(self, update):
        self._line(f"\n▶ 第 {update:04d} / {self.config['max_updates']} 轮  ·  训练阶段")
        self.training_progress(0, int(self.config["ppo_epochs"]))

    def training_progress(self, completed, total):
        self._progress("Epoch", completed, total)

    @staticmethod
    def _duration(seconds):
        hours, remainder = divmod(int(seconds), 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def summary(self, row, checkpoint=None):
        self._line(f"\n┌────────────────── 第 {row['update']:04d} 轮训练完成 ──────────────────")
        self._line("│")
        self._line(f"│  速度    FPS {row['fps']}")
        self._line(f"│          累计环境步 {row['total_environment_steps']:,}  ·  "
                   f"累计耗时 {self._duration(row['elapsed_seconds'])}")
        self._line(f"│          本轮采样 {row['collect_seconds']:.2f} s  ·  "
                   f"本轮训练 {row['train_seconds']:.2f} s")
        self._line("│")
        self._line(f"│  战绩    胜 {row['wins']}  /  平 {row['ties']}  /  负 {row['losses']}  ·  "
                   f"胜率 {row['win_rate']:.2%}")
        self._line(f"│  奖励    平均每场 {row['average_episode_reward']:+.2f}  ·  "
                   f"本轮合计 {row['reward_sum']:+.2f}")
        self._line("│")
        self._line(f"│  损失    Policy {row['policy_loss']:.4f}  ·  Value {row['value_loss']:.4f}")
        self._line(f"│  策略    Entropy {row['entropy']:.4f}  ·  Ratio {row['ratio']:.4f}")
        self._line(f"│  更新    {self.config['ppo_epochs']} epochs  ·  {row['optimizer_updates']} 次优化器更新"
                   f"（其中 Actor {row['actor_optimizer_updates']} 次）")
        if checkpoint is not None:
            self._line("│")
            self._line(f"│  保存    {checkpoint}")
        self._line("└" + "─" * 63)

    def checkpoint_saved(self, checkpoint):
        self._line(f"  ✓ 已保存模型：{checkpoint}")
