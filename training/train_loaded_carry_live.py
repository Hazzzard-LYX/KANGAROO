"""Train loaded carry from scratch while viewing a training environment (native viewer)."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from datetime import datetime
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

import torch

import loaded_carry_task
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.scripts.train import TrainConfig
from mjlab.tasks.registry import load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.os import dump_yaml
from mjlab.viewer import NativeMujocoViewer


class TrainingViewer(NativeMujocoViewer):
    """Let PPO own every step/reset so UI commands cannot corrupt its rollout."""

    def reset_environment(self) -> None:
        print("[VIEWER] Manual reset disabled during PPO training.", flush=True)

    def _single_step(self) -> None:
        print("[VIEWER] Press Space to resume PPO training.", flush=True)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=30000)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--save-interval", type=int, default=250)
    parser.add_argument("--viewer-fps", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if min(args.num_envs, args.iterations, args.save_interval, args.viewer_fps) <= 0:
        parser.error("Environment count, iterations, save interval and viewer FPS must be positive")

    record_path = root / "logs/active_training.json"
    history_path = root / "logs/training_history.jsonl"

    def interrupt_training(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt_training)

    configure_torch_backends()
    cfg = TrainConfig.from_task(loaded_carry_task.TASK_ID)
    cfg.env.scene.num_envs = args.num_envs
    cfg.env.viewer.max_extra_envs = 0
    cfg.env.viewer.entity_name = "robot"
    cfg.env.viewer.distance = 2.5
    cfg.env.viewer.elevation = -15.0
    cfg.env.seed = cfg.agent.seed = args.seed
    cfg.agent.resume = False
    cfg.agent.max_iterations = args.iterations
    cfg.agent.save_interval = args.save_interval
    cfg.agent.run_name = f"grip_v2_live_{args.num_envs}env"

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = root / "logs/rsl_rl" / cfg.agent.experiment_name / f"{stamp}_{cfg.agent.run_name}"
    log_dir.mkdir(parents=True, exist_ok=False)
    source_dir = log_dir / "source"
    source_dir.mkdir()
    for name in ("train_loaded_carry_live.py", "loaded_carry_task.py", "tray_carry_task.py",
                 "walking_task.py", "gripper_walking_model.py"):
        shutil.copy2(root / "training" / name, source_dir / name)
    shutil.copy2(loaded_carry_task.POSE_PATH, source_dir / loaded_carry_task.POSE_PATH.name)
    dump_yaml(log_dir / "params/env.yaml", asdict(cfg.env))
    dump_yaml(log_dir / "params/agent.yaml", asdict(cfg.agent))
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=root, text=True).strip()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    record = {
        "pid": os.getpid(),
        "phase": "loaded_carry_grip_v2_live",
        "started": datetime.now().astimezone().isoformat(),
        "num_envs": args.num_envs,
        "iterations": args.iterations,
        "seed": args.seed,
        "resume_checkpoint": None,
        "task_id": loaded_carry_task.TASK_ID,
        "pose_file": str(loaded_carry_task.POSE_PATH),
        "tray_mesh": str(loaded_carry_task.TRAY_STL_PATH),
        "rewards_added": ["grip_pinch", "grip_slip", "arm7_support", "torque_limit"],
        "branch": branch,
        "commit": commit,
        "uncommitted_changes": True,
        "save_interval": args.save_interval,
        "viewer": "native; training environment 0",
        "run_directory": str(log_dir),
        "status": "initializing",
    }
    record_path.write_text(json.dumps(record, indent=2) + "\n")

    raw = env = runner = viewer = None
    try:
        print(f"[INFO] {args.num_envs} environment(s), from scratch, live viewer", flush=True)
        raw = ManagerBasedRlEnv(cfg=cfg.env, device=args.device)
        env = RslRlVecEnvWrapper(raw, clip_actions=cfg.agent.clip_actions)
        runner_cls = load_runner_cls(loaded_carry_task.TASK_ID)
        runner = runner_cls(env, asdict(cfg.agent), str(log_dir), args.device)
        runner.add_git_repo_to_log(__file__)
        # Render the actual PPO rollout. Do not run a separate inference/play environment.
        viewer = TrainingViewer(env, policy=lambda _obs: None, frame_rate=args.viewer_fps)
        viewer.setup()
        # This mjlab version otherwise renders every other environment on the CPU.
        viewer.vd = None
        viewer._show_debug_vis = False
        env_step = env.step
        next_render = 0.0

        def live_step(action):
            nonlocal next_render
            if viewer.is_running():
                viewer._process_actions()
                while viewer._is_paused and viewer.is_running():
                    viewer.sync_env_to_viewer()
                    time.sleep(1.0 / args.viewer_fps)
                    viewer._process_actions()
                viewer.sync_viewer_to_env()
            result = env_step(action)
            viewer._step_count += 1
            now = time.monotonic()
            if viewer.is_running() and now >= next_render:
                viewer.sync_env_to_viewer()
                next_render = time.monotonic() + 1.0 / args.viewer_fps
            return result

        env.step = live_step
        record["status"] = "running"
        record["start_iteration"] = runner.current_learning_iteration
        record_path.write_text(json.dumps(record, indent=2) + "\n")
        print("[INFO] Native training viewer opened. Space: pause/resume; comma/period: change environment.", flush=True)
        runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=False)
        record["status"] = "completed"
    except KeyboardInterrupt:
        record["status"] = "interrupted_by_user"
        if runner is not None:
            saved = log_dir / f"model_{runner.current_learning_iteration}_interrupted.pt"
            runner.save(str(saved))
            record["interrupted_checkpoint"] = str(saved)
    except Exception as exc:
        record["status"] = "failed"
        record["error"] = repr(exc)
        raise
    finally:
        if viewer is not None:
            viewer.close()
        if env is not None:
            env.close()
        elif raw is not None:
            raw.close()
        record["pid"] = None
        record["finished"] = datetime.now().astimezone().isoformat()
        record_path.write_text(json.dumps(record, indent=2) + "\n")
        with history_path.open("a") as stream:
            stream.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
