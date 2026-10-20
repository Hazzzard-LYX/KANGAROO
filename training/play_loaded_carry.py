"""Play a loaded-carry policy at a fixed tray mass and show commanded vs actual velocity.

Default: a scripted command sequence that loops (5 s per segment). --keyboard: hold
arrows to walk, Ctrl+Left/Right to turn (same keys as keyboard_play.py).
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import mujoco
import torch

import loaded_carry_task as L
from keyboard_play import X11Keyboard, velocity_for_keys
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT / "logs/rsl_rl/kangaroo_loaded_carry/2026-10-15_15-44-18_scratch/model_29999.pt"
SEQUENCE = [
    ("stand", (0.0, 0.0, 0.0)), ("forward 0.3", (0.3, 0.0, 0.0)), ("forward 0.6", (0.6, 0.0, 0.0)),
    ("forward 1.0", (1.0, 0.0, 0.0)), ("forward 1.2", (1.2, 0.0, 0.0)), ("stand", (0.0, 0.0, 0.0)),
    ("backward 0.5", (-0.5, 0.0, 0.0)), ("left 0.2", (0.0, 0.2, 0.0)), ("right 0.2", (0.0, -0.2, 0.0)),
    ("turn left 0.5", (0.0, 0.0, 0.5)), ("turn right 0.5", (0.0, 0.0, -0.5)),
]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--mass", type=float, default=6.0, help="Tray mass (kg), 1.0-6.0 in 0.5 steps.")
    p.add_argument("--segment", type=float, default=5.0, help="Seconds per scripted command.")
    p.add_argument("--keyboard", action="store_true")
    a = p.parse_args()
    level = round((a.mass - L.MASS_MIN) / (L.MASS_MAX - L.MASS_MIN) * (L.N_LEVELS - 1))
    if not 0 <= level < L.N_LEVELS or abs(float(L.level_mass(torch.tensor(level))) - a.mass) > 1e-6:
        p.error(f"--mass must be one of the curriculum masses {L.MASS_MIN}..{L.MASS_MAX} step 0.5")
    configure_torch_backends()

    cfg = L.loaded_env_cfg(play=True)
    cfg.scene.num_envs = 1
    cfg.episode_length_s = 3600.0
    c = cfg.commands["twist"]
    c.heading_command, c.ranges.heading = False, None
    c.rel_standing_envs = c.rel_heading_envs = c.rel_forward_envs = c.rel_world_envs = 0.0
    c.init_velocity_prob = 0.0
    c.resampling_time_range = (1.0e8, 1.0e8)
    raw = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
    cur = L.get_curriculum(raw)
    cur.sample = lambda env_ids, step: cur.level.__setitem__(env_ids, level)
    agent_cfg = load_rl_cfg(L.TASK_ID)
    env = RslRlVecEnvWrapper(raw, clip_actions=agent_cfg.clip_actions)
    runner = load_runner_cls(L.TASK_ID)(env, asdict(agent_cfg), device="cuda:0")
    runner.load(str(a.checkpoint.resolve()), load_cfg={"actor": True}, strict=True, map_location="cuda:0")
    policy = runner.get_inference_policy(device="cuda:0")
    env.reset()

    cmd_term = raw.command_manager.get_term("twist")
    robot = raw.scene["robot"]
    mj = raw.sim.mj_model
    act = {mj.joint(int(mj.actuator_trnid[i, 0])).name.removeprefix("robot/"): i for i in range(mj.nu)}
    arm4 = [act["arm_left_4_joint"], act["arm_right_4_joint"]]
    arm4_cap = float(mj.actuator_forcerange[arm4[0], 1])
    keyboard = X11Keyboard() if a.keyboard else None
    state = {"step": 0, "label": "", "cmd": (0.0, 0.0, 0.0), "v": torch.zeros(3), "tau": (0.0, 0.0)}

    def controlled_policy(_obs):
        if keyboard:
            keys, window = keyboard.read()
            label, cmd = "keyboard", velocity_for_keys(keys)
            keyboard.title(window, cmd)
        else:
            seg = int(state["step"] * raw.step_dt / a.segment) % len(SEQUENCE)
            label, cmd = SEQUENCE[seg]
        cmd_term.vel_command_b[:] = torch.tensor(cmd, device=raw.device)
        d = robot.data
        v = torch.stack((d.root_link_lin_vel_b[0, 0], d.root_link_lin_vel_b[0, 1],
                         d.root_link_ang_vel_b[0, 2])).cpu()
        state["v"] = 0.9 * state["v"] + 0.1 * v  # ~0.2 s smoothing
        tau = raw.sim.data.actuator_force[0, arm4].abs().cpu() / arm4_cap
        state.update(step=state["step"] + 1, label=label, cmd=cmd, tau=tuple(tau.tolist()))
        return policy(env.get_observations())

    class TrackingViewer(NativeMujocoViewer):
        def _safe_key_callback(self, key):
            if keyboard is None or key not in (262, 263, 264, 265, 341, 345):
                super()._safe_key_callback(key)

        def _set_status_overlay(self, viewer: mujoco.viewer.Handle) -> None:
            cmd, v, tau = state["cmd"], state["v"], state["tau"]
            left = "Tray mass\nCommand\n\nvx  (m/s)\nvy  (m/s)\nyaw (rad/s)\n\narm4 torque L/R"
            right = (f"{a.mass:.1f} kg\n{state['label']}\n   cmd   actual\n"
                     f"{cmd[0]:+6.2f}  {float(v[0]):+6.2f}\n{cmd[1]:+6.2f}  {float(v[1]):+6.2f}\n"
                     f"{cmd[2]:+6.2f}  {float(v[2]):+6.2f}\n\n{100 * tau[0]:3.0f}% / {100 * tau[1]:3.0f}%")
            viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150.value,
                              mujoco.mjtGridPos.mjGRID_TOPLEFT.value, left, right))

    print(f"LOADED_CARRY_PLAY_READY checkpoint={a.checkpoint} mass={a.mass} kg "
          f"mode={'keyboard' if keyboard else 'scripted'}", flush=True)
    try:
        TrackingViewer(env, controlled_policy).run()
    finally:
        if keyboard:
            keyboard.close()
        env.close()


if __name__ == "__main__":
    main()
