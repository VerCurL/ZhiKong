"""Plot per-update reward trends after training, using log/rewards.csv.

    python scripts/plot_rewards.py results/4v4/mappo/run-时间 --window 20

The window is a trailing mean over recorded updates. Shading shows per-update
battle return standard deviation, not a confidence interval for the mean.
"""

import argparse
import csv
import math
from pathlib import Path


def read_rewards(path):
    path = Path(path)
    if path.is_dir():
        path = path/"rewards.csv" if (path/"rewards.csv").is_file() else path/"log"/"rewards.csv"
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("Reward CSV has no completed training updates")
    updates = [int(row["update"]) for row in rows]
    if any(b <= a for a, b in zip(updates, updates[1:])):
        raise ValueError("Reward CSV updates must be strictly increasing")
    return path, rows


def trailing_mean(values, window):
    if window < 1:
        raise ValueError("Smoothing window must be positive")
    return [math.fsum(values[max(0, i-window+1):i+1])/min(i+1, window) for i in range(len(values))]


def plot_rewards(path, output=None, window=1):
    if window < 1:
        raise ValueError("Smoothing window must be positive")
    path, rows = read_rewards(path)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("Plotting needs matplotlib; install the project requirements first") from error
    output = Path(output) if output is not None else path.parent/"reward_plots"
    output.mkdir(parents=True, exist_ok=True)
    updates = [int(row["update"]) for row in rows]
    groups = {"reward_trends": ["reward_"+name for name in (
        "total", "terminal", "event", "dense", "distance", "attack", "defense", "energy", "safety", "clip_adjustment")],
              "event_trends": [key[:-5] for key in rows[0] if key.startswith("event_") and key.endswith("_mean")],
              "signal_trends": [key[:-5] for key in rows[0] if key.startswith("signal_") and key.endswith("_mean")]}
    paths = []
    for name, metrics in groups.items():
        if not metrics:
            continue
        figure, axes = plt.subplots(math.ceil(len(metrics)/3), 3, squeeze=False, figsize=(15, 3.2*math.ceil(len(metrics)/3)))
        try:
            for axis, metric in zip(axes.flat, metrics):
                mean = [float(row[metric+"_mean"]) for row in rows]
                std = [float(row[metric+"_std"]) for row in rows]
                if not all(math.isfinite(value) for value in mean+std):
                    raise ValueError("Reward plots require finite statistics")
                axis.fill_between(updates, [m-s for m, s in zip(mean, std)], [m+s for m, s in zip(mean, std)],
                                  color="steelblue", alpha=.15, label="Battle mean +/- std")
                axis.plot(updates, mean, color="steelblue", alpha=.5, linewidth=.8, label="Battle mean")
                if window > 1:
                    axis.plot(updates, trailing_mean(mean, window), color="navy", linewidth=1.4, label="Trailing mean ({})".format(window))
                axis.axhline(0, color="gray", linewidth=.6)
                axis.set_title(metric)
                axis.set_xlabel("Training update (rollout before this update)")
                axis.set_ylabel("Mean return per battle" if name != "signal_trends" else "Time-integrated signal per battle")
                axis.grid(alpha=.25)
            for axis in list(axes.flat)[len(metrics):]:
                axis.set_visible(False)
            axes.flat[0].legend(fontsize=7)
            figure.suptitle("{} | {} | {}".format(name, rows[0]["side"], path.parent.parent.name))
            figure.tight_layout(rect=(0, 0, 1, .97))
            target = output/(name+".png")
            figure.savefig(str(target), dpi=150)
            paths.append(target)
        finally:
            plt.close(figure)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="Run directory, log directory, or rewards.csv")
    parser.add_argument("--output", help="Plot directory; default: log/reward_plots")
    parser.add_argument("--window", type=int, default=1, help="Trailing mean window over updates")
    args = parser.parse_args()
    for path in plot_rewards(args.path, args.output, args.window):
        print(path)


if __name__ == "__main__":
    main()
