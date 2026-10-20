"""Train loaded walking (tray 1 -> 6 kg curriculum) from scratch: no checkpoint is loaded."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

import loaded_carry_task
from mjlab.scripts.train import TrainConfig, launch_training


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--iterations", type=int, default=loaded_carry_task.agent.max_iterations)
    parser.add_argument("--save-interval", type=int, default=250)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--run-name", default="scratch")
    args = parser.parse_args()

    cfg = TrainConfig.from_task(loaded_carry_task.TASK_ID)
    cfg.env.scene.num_envs = args.num_envs
    cfg.agent.resume = False
    cfg.agent.max_iterations = args.iterations
    cfg.agent.save_interval = args.save_interval
    cfg.agent.run_name = args.run_name
    cfg.agent.seed = args.seed

    commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    record = {
        "pid": os.getpid(),
        "phase": "loaded_carry_scratch",
        "started": datetime.now().astimezone().isoformat(),
        "task_id": loaded_carry_task.TASK_ID,
        "num_envs": args.num_envs,
        "iterations": args.iterations,
        "seed": args.seed,
        "resume_checkpoint": None,
        "run_name": args.run_name,
        "run_directory_prefix": f"logs/rsl_rl/{cfg.agent.experiment_name}/*_{args.run_name}",
        "commit": commit,
        "uncommitted_changes": True,
        "pose_file": str(loaded_carry_task.tray.POSE_PATH),
        "tray_mesh": str(loaded_carry_task.tray.TRAY_STL_PATH),
        "curriculum": {
            "levels": loaded_carry_task.N_LEVELS,
            "mass_kg": [loaded_carry_task.MASS_MIN, loaded_carry_task.MASS_MAX],
            "speed_scale": [loaded_carry_task.SPEED_SCALE_MIN, 1.0],
            "promote_survival": loaded_carry_task.PROMOTE_SURVIVAL,
            "demote_survival": loaded_carry_task.DEMOTE_SURVIVAL,
            "replay_prob": loaded_carry_task.REPLAY_PROB,
        },
    }
    (root / "logs/active_training.json").write_text(json.dumps(record, indent=2) + "\n")
    with (root / "logs/training_history.jsonl").open("a") as f:
        f.write(json.dumps(record) + "\n")
    launch_training(loaded_carry_task.TASK_ID, cfg)


if __name__ == "__main__":
    main()
