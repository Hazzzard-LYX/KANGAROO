"""Evaluate a loaded-carry policy at fixed tray masses and fixed velocity commands.

Every (mass, command) pair runs in its own block of environments for one episode. The
command is written directly every control step, so the curriculum's per-level speed
scaling does not apply here. Reported per pair: survival, tray drops, velocity tracking
error after a settling time, tray drift relative to the torso, tray tilt, jaw angle
change, hand-tray contact force and arm joint torque saturation.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch

import loaded_carry_task as L
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends

COMMANDS = {
    "stand": (0.0, 0.0, 0.0), "fwd_0.3": (0.3, 0.0, 0.0), "fwd_0.6": (0.6, 0.0, 0.0),
    "fwd_1.0": (1.0, 0.0, 0.0), "fwd_1.2": (1.2, 0.0, 0.0), "back_0.5": (-0.5, 0.0, 0.0),
    "side_0.2": (0.0, 0.2, 0.0), "yaw_0.5": (0.0, 0.0, 0.5),
}
LEVELS = {1.0: 0, 4.0: 3, 6.0: 5, 8.0: 7}   # V5: total kg -> number of 1 kg blocks on the 1 kg tray


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--envs-per-case", type=int, default=16)
    p.add_argument("--seconds", type=float, default=20.0)
    p.add_argument("--settle", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    configure_torch_backends()

    cases = [(m, c) for m in LEVELS for c in COMMANDS]
    n = len(cases) * a.envs_per_case
    case_of = torch.arange(n, device=a.device) // a.envs_per_case
    level = torch.tensor([LEVELS[m] for m, _ in cases], device=a.device)[case_of]
    cmd = torch.tensor([COMMANDS[c] for _, c in cases], device=a.device)[case_of]

    cfg = L.loaded_env_cfg(play=True)
    cfg.scene.num_envs = n
    cfg.seed = a.seed
    cfg.episode_length_s = a.seconds + 5.0
    c = cfg.commands["twist"]
    c.heading_command, c.ranges.heading = False, None
    c.rel_standing_envs = c.rel_heading_envs = c.rel_forward_envs = c.rel_world_envs = 0.0
    c.init_velocity_prob = 0.0
    c.resampling_time_range = (1.0e8, 1.0e8)

    raw = ManagerBasedRlEnv(cfg=cfg, device=a.device)
    cur = L.get_curriculum(raw)
    cur.sample = lambda env_ids, step: cur.level.__setitem__(env_ids, level[env_ids])
    agent_cfg = load_rl_cfg(L.TASK_ID)
    env = RslRlVecEnvWrapper(raw, clip_actions=agent_cfg.clip_actions)
    runner = load_runner_cls(L.TASK_ID)(env, asdict(agent_cfg), device=a.device)
    runner.load(str(a.checkpoint.resolve()), load_cfg={"actor": True}, strict=True,
                map_location=a.device)
    policy = runner.get_inference_policy(device=a.device)
    env.reset()
    command = raw.command_manager.get_term("twist")

    body = int(raw.scene["tray"].indexing.body_ids[0])
    raw.sim.forward()
    masses = raw.sim.model.body_subtreemass[:, body]   # tray + blocks
    assert torch.allclose(masses, L.level_mass(level)), "tray masses not applied"

    mj = raw.sim.mj_model
    act = {mj.joint(int(mj.actuator_trnid[i, 0])).name.removeprefix("robot/"): i for i in range(mj.nu)}
    arm_ids = [act[f"arm_{s}_{j}_joint"] for s in ("left", "right") for j in range(1, 8)]
    arm_cap = torch.tensor(mj.actuator_forcerange[arm_ids, 1], device=a.device, dtype=torch.float32)
    jaw_ids = [raw.scene["robot"].joint_names.index(f"gripper_{s}_finger_joint") for s in ("left", "right")]
    robot = raw.scene["robot"]
    jaw0 = robot.data.joint_pos[:, jaw_ids].clone()
    target = L.TRAY_REL_POS_TORSO.to(a.device)

    alive = torch.ones(n, dtype=torch.bool, device=a.device)
    dropped = torch.zeros_like(alive)
    reason = {}
    acc = {k: torch.zeros(n, device=a.device) for k in
           ("cnt", "evx", "evy", "ewz", "vx", "z", "drift", "tilt", "djaw", "force")}
    sat = torch.zeros(n, len(arm_ids), device=a.device)
    peak = torch.zeros(n, len(arm_ids), device=a.device)
    tm = raw.termination_manager
    steps = round(a.seconds / raw.step_dt)
    for k in range(steps):
        command.vel_command_b[:] = cmd
        with torch.inference_mode():
            action = policy(env.get_observations())
        _, _, done, _ = env.step(action)
        done = done.bool()
        newly = alive & done
        if newly.any():
            for name in tm.active_terms:
                hit = newly & tm.get_term(name).bool()
                if hit.any():
                    reason[name] = reason.get(name, 0) + int(hit.sum())
            dropped |= newly & tm.get_term("tray_fallen").bool()
        alive &= ~done
        if (k + 1) * raw.step_dt < a.settle:
            continue
        live = alive
        d = robot.data
        st = L.tray_state_torso(raw, L.SceneEntityCfg("tray"), L.SceneEntityCfg("robot"))
        tilt = torch.rad2deg(2 * torch.acos(
            torch.abs(torch.sum(st[:, 3:7] * L.TRAY_REL_QUAT.to(a.device), -1)).clamp(max=1)))
        tau = raw.sim.data.actuator_force[:, arm_ids].abs() / arm_cap
        f = raw.scene["tray_support"].data.force.norm(dim=-1).sum(-1)
        vals = dict(cnt=torch.ones(n, device=a.device),
                    evx=(d.root_link_lin_vel_b[:, 0] - cmd[:, 0]).abs(),
                    evy=(d.root_link_lin_vel_b[:, 1] - cmd[:, 1]).abs(),
                    ewz=(d.root_link_ang_vel_b[:, 2] - cmd[:, 2]).abs(),
                    vx=d.root_link_lin_vel_b[:, 0], z=d.root_link_pos_w[:, 2],
                    drift=(st[:, :3] - target).norm(dim=-1), tilt=tilt,
                    djaw=(d.joint_pos[:, jaw_ids] - jaw0).mean(-1), force=f)
        for key, v in vals.items():
            acc[key][live] += v[live]
        sat[live] += (tau[live] > 0.95).float()
        peak[live] = torch.maximum(peak[live], tau[live])

    rows = []
    names = [f"arm_{s}_{j}" for s in ("left", "right") for j in range(1, 8)]
    for i, (m, cname) in enumerate(cases):
        sel = case_of == i
        cnt = acc["cnt"][sel].clamp(min=1)
        mean = {k: float((acc[k][sel] / cnt).mean()) for k in acc if k != "cnt"}
        sat_frac = (sat[sel] / cnt.unsqueeze(-1)).mean(0)
        worst = int(peak[sel].mean(0).argmax())
        rows.append(dict(
            mass_kg=m, command=cname, cmd=list(COMMANDS[cname]),
            survival=float(alive[sel].float().mean()), tray_dropped=float(dropped[sel].float().mean()),
            vx_mean=round(mean["vx"], 3), base_z=round(mean["z"], 3), err_vx=round(mean["evx"], 3), err_vy=round(mean["evy"], 3),
            err_yaw=round(mean["ewz"], 3), tray_drift_cm=round(100 * mean["drift"], 2),
            tray_tilt_deg=round(mean["tilt"], 2), jaw_change_deg=round(57.2958 * mean["djaw"], 2),
            hand_tray_force_N=round(mean["force"], 1),
            arm_peak_torque_frac=round(float(peak[sel].mean(0).max()), 3), arm_peak_joint=names[worst],
            arm_time_above_95pct=round(float(sat_frac.max()), 4),
        ))
    report = dict(checkpoint=str(a.checkpoint.resolve()), seconds=a.seconds, settle_s=a.settle,
                  envs_per_case=a.envs_per_case, seed=a.seed, termination_counts=reason, cases=rows)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + "\n")
    hdr = f"{'kg':>4} {'command':9} {'surv':>5} {'drop':>5} {'vx':>6} {'z':>5} {'e_vx':>5} {'e_vy':>5} {'e_yaw':>5} {'drift':>6} {'tilt':>5} {'jaw°':>6} {'F_N':>6} {'arm_pk':>6} joint"
    print(hdr)
    for r in rows:
        print(f"{r['mass_kg']:4.1f} {r['command']:9} {r['survival']:5.2f} {r['tray_dropped']:5.2f} {r['vx_mean']:6.2f} {r['base_z']:5.2f} "
              f"{r['err_vx']:5.2f} {r['err_vy']:5.2f} {r['err_yaw']:5.2f} {r['tray_drift_cm']:6.2f} {r['tray_tilt_deg']:5.1f} "
              f"{r['jaw_change_deg']:6.1f} {r['hand_tray_force_N']:6.1f} {r['arm_peak_torque_frac']:6.2f} {r['arm_peak_joint']}")
    print("terminations:", reason)
    env.close()


if __name__ == "__main__":
    main()
