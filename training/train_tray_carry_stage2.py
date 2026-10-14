"""Stage 2: continue the standing carry policy with velocity commands enabled."""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import tray_carry_task
from mjlab.scripts.train import TrainConfig, launch_training


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path,
                        help="Passing Stage 1 checkpoint to fine-tune.")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    cfg = TrainConfig.from_task(tray_carry_task.TASK_ID)
    cfg.env.scene.num_envs = 1024
    cfg.env.episode_length_s = 40.0
    # The normal walking command curriculum starts in the easy range, then widens
    # the speed envelope as samples accumulate.
    cfg.agent.resume = True
    cfg.agent.load_run = checkpoint.parent.name
    cfg.agent.load_checkpoint = checkpoint.name
    cfg.agent.max_iterations = 7500
    cfg.agent.save_interval = 250
    cfg.agent.run_name = "tray_carry_stage2_velocity"
    cfg.agent.seed = 48

    record_path = root / "logs/active_training.json"
    history_path = root / "logs/training_history.jsonl"
    record = {
        "pid": os.getpid(), "phase": "tray_carry_stage2_velocity",
        "started": datetime.now().astimezone().isoformat(),
        "num_envs": cfg.env.scene.num_envs, "iterations": cfg.agent.max_iterations,
        "seed": cfg.agent.seed, "resume_checkpoint": str(checkpoint),
        "task_id": tray_carry_task.TASK_ID, "command": "velocity_tracking_curriculum",
        "tray_mass_kg": tray_carry_task.TRAY_MASS,
        "grasp_model": "bilateral no-slip site constraints; 100 g free-body dynamics",
        "automation_id": "kangaroo",
    }
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    launch_training(tray_carry_task.TASK_ID, cfg)
    record["status"] = "stage2_complete"
    record["finished"] = datetime.now().astimezone().isoformat()
    record["pid"] = None
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    with history_path.open("a") as stream:
        stream.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
