# 4v4 MAPPO Training

The trainer alternates complete-battle collection and PPO optimization. Network weights, the opponent and inference normalization are fixed throughout collection. Battles run in batches: every battle in a batch must end before the next batch starts. All configured battles finish before any optimizer update. The next collection uses the newly trained policy.

## Start

```powershell
conda activate jsbsim
python .\scripts\train_mappo_4v4.py
```

By default `n_rollout_threads: 32` specifies the total battles per update, while `parallel_envs: 8` limits concurrent battles. The trainer launches eight independent ZK processes and reuses them for four consecutive batches of eight battles. Each process uses `Red=0 Blue=1`, four aircraft on each side, and two sockets:

| Environment | Red port | Blue port |
| --- | ---: | ---: |
| 0 | 8888 | 8889 |
| 1 | 8890 | 8891 |
| ... | ... | ... |
| 7 | 8902 | 8903 |

One I/O worker thread handles each independent simulator; the main thread batches neural inference across active environments. Two sockets belong to one battle. Only `min(parallel_envs, n_rollout_threads)` platform processes and port pairs are created. The trainer refuses occupied ports when launching platforms and closes only processes it started.

For a smaller initial run:

```powershell
python .\scripts\train_mappo_4v4.py --envs 2 --rollout-episodes 4 --updates 1
```

To use already running simulators, start one per configured port pair, then pass `--no-launch-platforms`. For one manually started platform:

```powershell
.\platform\Windows\ZK.exe Ip=127.0.0.1 Port=8888 Red=0 Blue=1 RedNum=4 BlueNum=4 PlayMode=0 Scenes=3
python .\scripts\train_mappo_4v4.py --envs 1 --rollout-episodes 32 --no-launch-platforms
```

## Fixed baseline flight controller

The control chain is `MAPPO Actor -> maneuver deltas + weapon choice -> selected fixed flight controller -> platform controls -> simulator`. `baseline` uses the frozen LAGCFC controller and `pid` uses the C50-M1 EAS/PID controller; see `docs/flight_controllers.md`. The previous proportional flight-control rules have been removed.

`data/schemas/action_schema.yaml` is the single source for maneuver choices, input scales and throttle limits:

| High-level head | Choices in index order |
| --- | --- |
| altitude | +100 m, hold, -100 m |
| turn | -30°, -15°, hold, +15°, +30° |
| speed | +17 m/s, hold, -17 m/s |
| weapon | none, AIM-9M (short-range), AIM-120B (medium-range) |

The policy outputs four discrete values with dimensions `[3, 5, 3, 3]`; action masks and logits have width 14. The neutral action is `[1, 2, 1, 0]`. The baseline receives 12 features: altitude delta / 1000, heading delta in radians, body-forward speed delta / 340, altitude / 5000, sine/cosine of roll, sine/cosine of pitch, body u/v/w speeds / 340, and calibrated airspeed / 340. Platform feet are converted to meters first. LAGCFC's clipping of the nine own-aircraft features to [-10, 10] is retained, independently of the high-level observation normalizer.

The official platform's return-field table lists Mach and altitude but does not list calibrated airspeed. The adapter prefers `velocities/vc-fps` or `velocities/vc-kts` if available; otherwise it estimates CAS using Mach, an ISA pressure profile and the compressible pitot relation (including supersonic flow). This fallback assumes standard atmosphere. Live aircraft missing required attitude/body-speed fields raise an error instead of feeding zero-filled flight states to the model.

Baseline outputs use LAGCFC's `[41, 41, 41, 30]` quantization: the three control surfaces map to [-1, 1] and throttle to [0.4, 0.9]. Each environment maintains independent baseline GRU states for every red and blue aircraft. States reset before every battle and clear on death. Baseline weights are frozen on CPU and excluded from PPO; high-level actor and critic remain trainable. Target selection, missile switching and weapon commands continue through the competition adapter.

Policies using an earlier weapon schema containing gun require new training. This includes the old three-choice `none/missile/gun` schema even though its head dimensions match: its index 2 must not be reinterpreted as AIM-120B. Resume, evaluation and export reject incompatible policy heads/schemas. Actor model files embed their action schema and observation normalization. The fixed baseline stays in `envs/model/`; exports carry the baseline and its runtime, so deployment uses the same control chain.

## Weapon requests and platform commands

The competition scenario specifies AIM-9M infrared missiles (two per aircraft) and AIM-120B radar missiles (four per aircraft). The policy supports these two missile types only. Gun actions and their firing-distance configuration have been removed.

The weapon head uses these high-level indices, defined in `data/schemas/action_schema.yaml`:

| Policy choice | Meaning | Platform command when ready |
| ---: | --- | --- |
| 0 | No shot | `fcs/weapon-launch=0`, no missile-mode switch |
| 1 | AIM-9M / SRAAM request | Confirm `AimMode=0` (ACM), then `fcs/weapon-launch=1` |
| 2 | AIM-120B / AMRAAM request | Confirm `AimMode=1` (CRM), then `fcs/weapon-launch=1` |

`envs/weapon_control.py` produces the same plans for action masks and command conversion. Selecting another missile type with remaining ammunition is a preparatory action: send `switch-missile=1` and no shot. Wait for returned state confirming the selected mode; the policy must request that type again when ready. No shot is queued across steps. Choosing none does not automatically switch missile modes.

In the current missile mode, firing requires ammunition, an available rail (`*_CanReload`), a visible living target, `TargetEnterAttackRange`, the configured distance gate and the matching target's lock. AIM-9M uses `SRAAMTargetLocked`; AIM-120B follows the documented next-ready-rail lock index and fallback through lower rail indices in `AMRAAMlockedTarget`. A lock on a different target does not authorize a shot. Missing rail flags in legacy telemetry fall back to the next sequential rail inferred from remaining ammunition. Wrong-mode requests can prepare a missile before contact; current-mode requests with no legal shot are masked out.

Sources: the official competition scenario and software/platform manual under `files/2026-智能空中博弈算法挑战赛/`. Both policy missile requests map to platform command 1 after mode selection and readiness checks. The adapter emits only 0 (none) or 1 (selected missile), never platform gun command 2. Training, visual tests and exported inference use the same mapping.

## Collection and optimization

- `n_rollout_threads` is the total completed-battle count per update; `parallel_envs` is the concurrent environment limit. CLI `--rollout-episodes` overrides the total and `--envs` (alias `--parallel-envs`) overrides concurrency. The old `episodes_per_env` and `rollout_steps` settings have been removed.
- With 32 total battles and eight environments, batches complete as `8 -> 8 -> 8 -> 8 -> PPO`. For 10 total battles and three environments, batches are `3 -> 3 -> 3 -> 1 -> PPO`; no extra battles are sampled.
- One frozen opponent is selected at the start of each collection phase and used by all environments throughout that phase, following the one-opponent LAGCFC configuration.
- An environment that finishes early waits for every remaining battle in its batch. Hidden states and episode counters reset before each new batch. Data from all batches accumulates in one RAM buffer; GAE and chunk boundaries stay within each battle. Failure or a safety-limit breach aborts the whole collection phase without any PPO update.
- Trajectories and GAE targets live only in RAM. GRU state is stored at chunk starts. Each episode is split independently into 8-step sequences; short tails are padded and masked out of normalization and losses.
- Each PPO epoch shuffles complete chunks, preserving time order inside each chunk. Data is divided into 5 groups, then split further when needed to respect `max_minibatch_samples: 4096`. Four epochs consume every active sample four times. The number of optimizer steps depends on the collected battle lengths and is logged.
- `sampling_memory_limit_mb: 4096` bounds the trajectory buffer; exceeding it aborts collection instead of spilling to disk. This limit does not include the simulators, network activations or optimizer memory.

## Saved outputs

Each training invocation creates an independent directory, following LAGCFC's model naming. The default output root is `results`; `--output` changes that root. The run name contains only Asia/Shanghai time, with second precision:

```text
results/
└── 4v4/
    └── mappo/
        └── run-20261021-153000/
            ├── actor_1.pt
            ├── actor_2.pt
            ├── ...
            ├── actor_latest.pt
            ├── critic_latest.pt
            └── log/
                ├── training.csv
                └── rewards.csv
```

Every completed PPO update saves `actor_<update>.pt` and replaces `actor_latest.pt` and `critic_latest.pt`. Actor history, the two latest model files and the training/reward CSV logs are written. Optimizer state, historical critics, RNG state, opponent pools, configuration snapshots and duplicate baseline files are not saved. The old `checkpoint_interval` setting has been removed. `save_replay` must remain false; automatically launched Unity logs and console streams are discarded.

The independent reward recorder is enabled by `reward_recorder.enabled` in `configs/train_mappo.yaml`. It writes one row per update to `log/rewards.csv`, using all completed battles from every collection batch. Each reward contribution, event and unweighted signal includes the sum, per-battle mean, population standard deviation, minimum and maximum. Team rewards are counted once per battle. Plot trends after training with `python scripts/plot_rewards.py results/4v4/mappo/run-时间 --window 20`; Matplotlib is required only for plotting. See [reward recording details](docs/reward_design.md#153-独立奖励记录器与训练趋势).

Actor files contain a dictionary with `state_dict`, `obs_normalizer`, `update` and `action_schema`; normalization is required for consistent inference and stays inside the actor file. Critic files contain `state_dict` and `update`. Tensor data can be loaded with `torch.load(..., weights_only=True)`. The CSV receives one summary row per update: completed battles, wins/ties/losses, rewards, environment steps, sample/chunk counts, buffer size, opponent, phase durations, losses, throughput and actual optimizer update count.

`--resume` loads the latest actor/critic and actor normalization, then continues with a new optimizer, opponent pool and random state. It creates a new run directory and continues numbering from the saved update; it is a model warm start rather than exact restoration of the previous training process. The parallel environment count may change. Use matching latest models; an older historical actor cannot resume alongside a newer latest critic. Existing legacy checkpoint folders remain readable when their normalization sidecar exists.

```powershell
python .\scripts\train_mappo_4v4.py --resume result/4v4/mappo/run-时间 --updates 1000
python .\scripts\evaluate_mappo.py --checkpoint result/4v4/mappo/run-时间/actor_20.pt --opponent result/4v4/mappo/run-时间/actor_10.pt
python .\scripts\export_submission.py --checkpoint result/4v4/mappo/run-时间/actor_latest.pt --team-name my_team
```

## Visual 1v1 baseline test

```powershell
python .\tests\test_baseline_flight.py
python .\tests\test_baseline_flight.py --maneuver right --duration 90
```

This independent script launches a visible 1v1 window by default. Both aircraft use the frozen baseline with separate recurrent states. Weapons and the platform's built-in replay saving are disabled. Fixed target errors are recalculated every step. The default demonstration switches between hold, right/left turns, climb/descent, acceleration/deceleration and recovery, with 20 simulation seconds per phase. Watch the aircraft using the platform camera controls; `--split-screen` requests the platform's split view.

The script uses separate ports 18888/18889 by default. To connect to a manually started platform:

```powershell
.\platform\Windows\ZK.exe Ip=127.0.0.1 Port=18888 Red=0 Blue=1 RedNum=1 BlueNum=1 PlayMode=0 Scenes=3
python .\tests\test_baseline_flight.py --no-launch-platform --maneuver hold --duration 120
```

Playback follows returned simulation time at real-time speed; `--sim-speed 2` doubles playback and `--sim-speed 0` runs without pacing. Terminal telemetry includes target/actual altitude, heading, body-forward speed, attitude and control commands. Optional `--csv results/baseline_flight.csv` saves telemetry at `--print-interval` simulation-second intervals. Ctrl+C exits and only the platform process launched by the script is terminated. Initial altitude/speed/heading, maneuver deltas, duration and phase length are available as CLI options (`--help`).

## Visual policy battle test

`tests/test_policy_battle.py` runs two trained high-level policies in a visible `PlayMode=1` 4v4 platform and records Tacview data. Set the two paths near the top of that script:

```python
RED_MODEL_PATH = r"result/4v4/mappo/run-红方/actor_latest.pt"
BLUE_MODEL_PATH = r"result/4v4/mappo/run-蓝方/actor_10.pt"
```

Paths can point to a run directory or a specific actor file. Relative paths resolve against the project root. Run the script directly, or override the paths on the command line:

```powershell
python .\tests\test_policy_battle.py
python .\tests\test_policy_battle.py --red-model "result/4v4/mappo/run-红方/actor_latest.pt" --blue-model "result/4v4/mappo/run-蓝方/actor_10.pt"
```

Both models load and validate before the platform launches. Each side uses its own actor, saved observation normalization, hidden width and per-aircraft recurrent states. Inference runs on CPU with deterministic actions by default; `--stochastic` enables action sampling with `--seed`. The same dual-socket visibility, action masks and frozen baseline flight controller used in training are applied. Weapons are controlled by the policies. Every new battle resets both policies and the baseline states.

The test uses ports 19888/19889 by default. It launches one platform with `Red=0 Blue=1 RedNum=4 BlueNum=4 PlayMode=1` and closes only that owned process on exit. `--no-launch-platform` connects to an already running platform with those settings. `--red-port`, `--blue-port`, `--scene` and `--executable` can be overridden. Occupied ports are rejected during automatic launch.

One complete battle runs by default. `--rounds 3` runs three battles, and `--duration 30` limits each battle to 30 simulation seconds. `--max-steps` is an additional safety limit. Battles cut short by either limit are reported as incomplete, without declaring a winner. `--sim-speed 1` follows simulation time in real time; `--sim-speed 0` disables pacing. This script performs no training or model saves.

Each battle writes `results/tacview/policy_battle_<UTC timestamp>_round_001.acmi` by default; `--tacview-dir` changes the destination. The initial state and every response are recorded, including terminal states. Recording combines each side's full own-aircraft packets and deduplicates visible missile data from the two sockets. Only returned missile data can be included. Ctrl+C and exceptions close the recorder and connections so completed frames remain readable.

## Tacview recording

The baseline test enables the independent `envs/tacview_recorder.py` recorder by default. Each run creates a unique `results/tacview/baseline_flight_<UTC timestamp>.acmi` file; open it in Tacview after the test exits. These are UTF-8 ACMI 2.2 recordings following the [official format](https://www.tacview.net/documentation/acmi/en/).

```powershell
# Specify a new output file (existing files are never overwritten).
python .\tests\test_baseline_flight.py --maneuver right --duration 90 --tacview results/tacview/right_turn.acmi
# Run the test with recording disabled.
python .\tests\test_baseline_flight.py --no-tacview
```

The recorder saves the initial state and every returned simulation step, independently of the terminal/CSV print interval. Aircraft positions, roll/pitch/yaw, side colors, missile positions/orientations and launch-parent associations are included. Aircraft heights and missile `Altitude` values are converted from feet to meters; aircraft roll/pitch are converted from radians to degrees, while missile attitude is already in degrees. Time starts at zero relative to the first recorded simulation state. If both terminal aircraft packets omit time, terminal events use the last known timestamp. A new battle requires a new file.

The test merges each side's own aircraft data and deduplicates the missiles returned by its two sockets. Only received missile data can be recorded; missiles outside both sides' visibility cannot be reconstructed. Explicit aircraft deaths and missile terminal states remove their objects. Missing objects generate a `LeftArea` event without asserting a kill. Missing/invalid positions never become artificial coordinates at zero.

The baseline test keeps weapons disabled, so its normal recordings contain the two aircraft; missile support applies when missile state is supplied. The file stays open with buffered writes during the test and closes on normal completion, exceptions or Ctrl+C, preserving completed frames. `scripts/train_mappo_4v4.py` and its environment/runner do not activate or import the recorder, so training produces no ACMI files or recording writes.

To reuse the recorder in a future explicitly enabled entry point:

```python
from envs.tacview_recorder import TacviewRecorder

with TacviewRecorder("battle.acmi", feet_to_meters=0.3048) as recorder:
    recorder.record(state)  # {"red": {...}, "blue": {...}, "missile": {...}}
    # Or supply sim_time explicitly when the state has no timestamp.
```
