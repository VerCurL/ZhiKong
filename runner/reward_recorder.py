"""One CSV row per update, summarizing complete battles from every worker."""

import csv
from contextlib import contextmanager
import math
from pathlib import Path
import statistics
import tempfile

from envs.reward.fkr.multiplecombat_reward import (
    DENSE_NAMES, REWARD_METRIC_COLUMNS, REWARD_RATIO_COLUMNS, summarize_reward_metrics,
)


REWARD_STAT_METRICS = tuple(key for key in REWARD_METRIC_COLUMNS
                            if key.startswith(("reward_", "event_", "signal_"))) + ("reward_clip_adjustment",)
STAT_NAMES = ("sum", "mean", "std", "min", "max")
COUNTER_METRICS = tuple(key for key in REWARD_METRIC_COLUMNS if key not in REWARD_STAT_METRICS)
REWARD_CSV_COLUMNS = ("update", "policy_update", "side", "episodes", "environment_steps", "opponent") + tuple(
    metric+"_"+stat for metric in REWARD_STAT_METRICS for stat in STAT_NAMES
) + COUNTER_METRICS + REWARD_RATIO_COLUMNS


def _csv_columns(path, columns):
    """Extend old logs atomically; retain historical columns and cell values."""
    if not path.exists() or path.stat().st_size == 0:
        return None
    temporary = None
    try:
        with path.open("r", newline="", encoding="utf-8-sig") as source:
            reader = csv.DictReader(source)
            header = reader.fieldnames
            if not header:
                return None
            if len(header) != len(set(header)):
                raise ValueError(f"Duplicate CSV column names: {path}; existing records were preserved")
            missing = [column for column in columns if column not in header]
            if not missing:
                return header
            header = header+missing
            with tempfile.NamedTemporaryFile(mode="w", newline="", encoding="utf-8",
                                             dir=path.parent, prefix="."+path.name+".",
                                             suffix=".tmp", delete=False) as target:
                temporary = Path(target.name)
                writer = csv.DictWriter(target, fieldnames=header)
                writer.writeheader()
                for row in reader:
                    if None in row:
                        raise ValueError(f"CSV row has more values than columns: {path}; existing records were preserved")
                    writer.writerow(row)
        # Close both files before replacing on Windows. Failed migration leaves
        # the original file intact; old rows have blank newly introduced cells.
        temporary.replace(path)
        return header
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


@contextmanager
def append_csv(path, columns):
    """Create or append logs, extending older schemas without dropping data."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = _csv_columns(path, columns)
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=header if header is not None else columns)
        if header is None:
            writer.writeheader()
            stream.flush()
        yield stream, writer


class RewardRecorder:
    """Log team returns once per battle, avoiding four copies of shared rewards."""

    def __init__(self, log_dir, enabled):
        self.path = Path(log_dir)/"rewards.csv"
        self.enabled = enabled
        self.stream = None
        self.last_update = None

    def __enter__(self):
        if self.enabled:
            self._csv_context = append_csv(self.path, REWARD_CSV_COLUMNS)
            self.stream, self.writer = self._csv_context.__enter__()
        return self

    def record(self, update, rollout, side):
        if not self.enabled:
            return
        if self.stream is None:
            raise RuntimeError("Open RewardRecorder with a context manager before recording")
        if self.last_update is not None and update <= self.last_update:
            raise ValueError("Reward updates must increase; refusing duplicate or reordered records")
        episodes = rollout["episodes"]
        if not episodes:
            raise ValueError("Reward statistics require complete battles")
        samples = {metric: [] for metric in REWARD_STAT_METRICS}
        totals = dict.fromkeys(REWARD_METRIC_COLUMNS, 0.0)
        for episode in episodes:
            values = {key: float(episode["reward_metrics"][key]) for key in REWARD_METRIC_COLUMNS}
            if not all(math.isfinite(value) for value in values.values()):
                raise ValueError("Reward statistics must be finite")
            for key, value in values.items():
                totals[key] += value
            # Signed correction makes the contribution sum reconcile with the
            # actual clipped dense return without assigning clipping to a class.
            values["reward_clip_adjustment"] = values["reward_dense"]-sum(values["reward_"+name] for name in DENSE_NAMES)
            for metric in samples:
                samples[metric].append(values[metric])
        row = {"update": int(update), "policy_update": int(update)-1, "side": side,
               "episodes": len(episodes), "environment_steps": rollout["environment_steps"], "opponent": rollout["opponent"]}
        for metric, values in samples.items():
            stats = (math.fsum(values), statistics.mean(values), statistics.pstdev(values), min(values), max(values))
            row.update({metric+"_"+name: value for name, value in zip(STAT_NAMES, stats)})
        summary = summarize_reward_metrics(totals)
        row.update({key: summary[key] for key in COUNTER_METRICS+REWARD_RATIO_COLUMNS})
        self.writer.writerow(row)
        self.stream.flush()
        self.last_update = update

    def __exit__(self, exc_type, exc_value, traceback):
        if self.stream is not None:
            self._csv_context.__exit__(exc_type, exc_value, traceback)
            self.stream = None
