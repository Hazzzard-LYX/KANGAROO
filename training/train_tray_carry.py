"""Fine-tune the walking policy to carry the 100 g tray in the saved pose."""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import tray_carry_task
from mjlab.scripts.train import TrainConfig, launch_training


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    checkpoint = root / "logs/rsl_rl/kangaroo_walking/2026-10-07_14-42-40_refine_stand_lateral/model_10000.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    cfg = TrainConfig.from_task(tray_carry_task.TASK_ID)
    cfg.env.scene.num_envs = 1024
    cfg.env.episode_length_s = 40.0
    cfg.agent.resume = True
    cfg.agent.load_run = "2026-10-07_14-42-40_refine_stand_lateral"
    cfg.agent.load_checkpoint = "model_10000.pt"
    cfg.agent.max_iterations = 10000
    cfg.agent.save_interval = 250
    cfg.agent.run_name = "tray_carry_010kg"
    cfg.agent.seed = 47

    record_path = root / "logs/active_training.json"
    record = {
        "pid": os.getpid(),
        "phase": "tray_carry_010kg",
        "started": datetime.now().astimezone().isoformat(),
        "num_envs": cfg.env.scene.num_envs,
        "iterations": cfg.agent.max_iterations,
        "seed": cfg.agent.seed,
        "resume_checkpoint": str(checkpoint),
        "automation_id": "kangaroo",
        "task_id": tray_carry_task.TASK_ID,
        "tray_mass_kg": tray_carry_task.TRAY_MASS,
        "pose_file": str(tray_carry_task.POSE_PATH),
        "run_directory_prefix": "logs/rsl_rl/kangaroo_walking/*_tray_carry_010kg",
    }
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    launch_training(tray_carry_task.TASK_ID, cfg)


if __name__ == "__main__":
    main()
