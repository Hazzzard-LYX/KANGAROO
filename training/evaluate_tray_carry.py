"""Measure standing survival and tray retention independently of training reward."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch

import tray_carry_task
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_rl_cfg, load_runner_cls
from mjlab.utils.lab_api.math import quat_apply_inverse, quat_mul
from mjlab.utils.torch import configure_torch_backends


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--envs", type=int, default=64)
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    configure_torch_backends()

    cfg = tray_carry_task.carry_env_cfg(play=True)
    cfg.scene.num_envs = args.envs
    cfg.seed = args.seed
    cfg.episode_length_s = args.seconds + 5.0
    command_cfg = cfg.commands["twist"]
    command_cfg.heading_command = False
    command_cfg.ranges.heading = None
    command_cfg.ranges.lin_vel_x = (0.0, 0.0)
    command_cfg.ranges.lin_vel_y = (0.0, 0.0)
    command_cfg.ranges.ang_vel_z = (0.0, 0.0)
    command_cfg.rel_standing_envs = 1.0
    command_cfg.rel_heading_envs = 0.0
    command_cfg.rel_forward_envs = 0.0
    command_cfg.resampling_time_range = (1.0e8, 1.0e8)

    raw = ManagerBasedRlEnv(cfg=cfg, device=args.device)
    agent_cfg = load_rl_cfg(tray_carry_task.TASK_ID)
    env = RslRlVecEnvWrapper(raw, clip_actions=agent_cfg.clip_actions)
    runner = load_runner_cls(tray_carry_task.TASK_ID)(
        env, asdict(agent_cfg), device=args.device
    )
    runner.load(str(args.checkpoint.resolve()), load_cfg={"actor": True}, strict=True,
                map_location=args.device)
    policy = runner.get_inference_policy(device=args.device)
    env.reset()
    command = raw.command_manager.get_term("twist")
    command.vel_command_b.zero_()

    alive = torch.ones(args.envs, dtype=torch.bool, device=args.device)
    failed_at = torch.full((args.envs,), args.seconds, device=args.device)
    pos_error_sq = torch.zeros(args.envs, device=args.device)
    angle_error = torch.zeros(args.envs, device=args.device)
    sample_count = torch.zeros(args.envs, device=args.device)
    target_pos = tray_carry_task.TRAY_REL_POS_TORSO.to(device=args.device)
    target_quat = tray_carry_task.TRAY_REL_QUAT.to(device=args.device)
    steps = round(args.seconds / raw.step_dt)

    for step in range(steps):
        command.vel_command_b.zero_()
        obs = env.get_observations()
        with torch.inference_mode():
            action = policy(obs)
        obs, _, done, _ = env.step(action)
        newly_failed = alive & done.bool()
        failed_at[newly_failed] = (step + 1) * raw.step_dt
        alive &= ~done.bool()

        if (step + 1) * raw.step_dt < 2.0:
            continue
        robot, tray = raw.scene["robot"].data, raw.scene["tray"].data
        torso_pos, torso_quat, _, _ = tray_carry_task._torso_state(raw.scene["robot"])
        rel_pos = quat_apply_inverse(torso_quat, tray.root_link_pos_w - torso_pos)
        rel_quat = quat_mul(
            torch.cat((torso_quat[:, :1], -torso_quat[:, 1:]), dim=-1),
            tray.root_link_quat_w,
        )
        dot = torch.abs(torch.sum(rel_quat * target_quat, dim=-1)).clamp(0.0, 1.0)
        errors = torch.sum((rel_pos - target_pos) ** 2, dim=-1)
        live = alive
        pos_error_sq[live] += errors[live]
        angle_error[live] += (2.0 * torch.acos(dot[live]))
        sample_count[live] += 1

    rate = alive.float().mean().item()
    valid = sample_count > 0
    position_rmse = (pos_error_sq[valid] / sample_count[valid]).mean().sqrt().item() if valid.any() else None
    mean_angle = (angle_error[valid] / sample_count[valid]).mean().item() if valid.any() else None
    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "seconds": args.seconds, "envs": args.envs, "seed": args.seed,
        "grasp_assumption": "bilateral no-slip site constraints; tray remains a 0.1 kg free body",
        "standing_survival_rate": rate,
        "mean_time_to_failure_s": failed_at.mean().item(),
        "tray_relative_position_rmse_m": position_rmse,
        "tray_relative_orientation_error_deg": (mean_angle * 180.0 / torch.pi) if mean_angle is not None else None,
        "pass": bool(rate >= 0.95 and position_rmse is not None and position_rmse <= 0.03
                     and mean_angle is not None and mean_angle <= 15.0 * torch.pi / 180.0),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    env.close()


if __name__ == "__main__":
    main()
