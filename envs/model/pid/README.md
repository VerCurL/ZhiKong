# PID flight controller

This directory contains the copied C50-M1 PID implementation, audited `model.json` and `pid_config.json` from `files/PIDPlatform`. `controller.py` adapts the current ZK aircraft telemetry and high-level maneuver deltas to the PID EAS/5 Hz interface. It does not alter the four platform output fields.
