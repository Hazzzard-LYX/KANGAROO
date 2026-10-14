"""Stage 1: fine-tune the walking policy to stand while carrying the tray."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
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
    command = cfg.env.commands["twist"]
    command.heading_command = False
    command.ranges.heading = None
    command.ranges.lin_vel_x = (0.0, 0.0)
    command.ranges.lin_vel_y = (0.0, 0.0)
    command.ranges.ang_vel_z = (0.0, 0.0)
    command.rel_standing_envs = 1.0
    command.rel_heading_envs = 0.0
    command.rel_forward_envs = 0.0
    command.resampling_time_range = (1.0e8, 1.0e8)
    cfg.env.curriculum.pop("command_vel", None)

    cfg.agent.resume = True
    cfg.agent.load_run = "2026-10-07_14-42-40_refine_stand_lateral"
    cfg.agent.load_checkpoint = "model_10000.pt"
    cfg.agent.max_iterations = 2500
    cfg.agent.save_interval = 250
    cfg.agent.run_name = "tray_carry_stage1_stand"
    cfg.agent.seed = 47

    record_path = root / "logs/active_training.json"
    history_path = root / "logs/training_history.jsonl"
    if record_path.exists():
        old = json.loads(record_path.read_text())
        if old.get("phase", "").startswith("tray_carry_"):
            old["status"] = "stopped_for_initialization_or_reward_correction; logs and checkpoints preserved"
            with history_path.open("a") as stream:
                stream.write(json.dumps(old) + "\n")
    record = {
        "pid": os.getpid(), "phase": "tray_carry_stage1_stand",
        "started": datetime.now().astimezone().isoformat(),
        "num_envs": cfg.env.scene.num_envs, "iterations": cfg.agent.max_iterations,
        "seed": cfg.agent.seed, "resume_checkpoint": str(checkpoint),
        "task_id": tray_carry_task.TASK_ID, "command": "standing_only",
        "tray_mass_kg": tray_carry_task.TRAY_MASS,
        "grasp_model": "bilateral no-slip site constraints; 100 g free-body dynamics",
        "automation_id": "kangaroo",
    }
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    launch_training(tray_carry_task.TASK_ID, cfg)

    run_dirs = sorted((root / "logs/rsl_rl/kangaroo_walking").glob("*_tray_carry_stage1_stand"))
    if not run_dirs:
        raise FileNotFoundError("Stage 1 run directory was not created")
    checkpoints = list(run_dirs[-1].glob("model_*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No Stage 1 checkpoints in {run_dirs[-1]}")
    checkpoint = max(checkpoints, key=lambda p: int(re.search(r"model_(\d+)\.pt", p.name).group(1)))

    report_path = root.parent.parent / "Documents/Codex/2026-10-07/new-chat-3/outputs/tray_carry_stage1_evaluation.json"
    record["status"] = "evaluating_stage1_standing"
    record["run_directory"] = str(run_dirs[-1])
    record["evaluation_checkpoint"] = str(checkpoint)
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    subprocess.run([
        sys.executable, str(root / "training/evaluate_tray_carry.py"),
        "--checkpoint", str(checkpoint), "--output", str(report_path),
        "--envs", "64", "--seconds", "20", "--seed", "2026",
    ], check=True)
    evaluation = json.loads(report_path.read_text())
    record["stage1_evaluation"] = evaluation
    record["finished"] = datetime.now().astimezone().isoformat()
    if evaluation["pass"]:
        record["status"] = "stage1_passed_starting_stage2"
        record_path.write_text(json.dumps(record, indent=2) + "\n")
        subprocess.run([
            sys.executable, str(root / "training/train_tray_carry_stage2.py"),
            "--checkpoint", str(checkpoint),
        ], check=True)
        record["status"] = "curriculum_complete"
    else:
        record["status"] = "stage1_did_not_pass; stage2_not_started"
    record["pid"] = None
    record["finished"] = datetime.now().astimezone().isoformat()
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    with history_path.open("a") as stream:
        stream.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
