"""Velocity tracking with a 100 g tray held in the user's saved two-hand pose."""
from __future__ import annotations

import json
import math
import re
import copy
from pathlib import Path

import mujoco
import torch
from mjlab.entity import EntityCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.registry import register_mjlab_task
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse, quat_mul
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

import walking_task
from gripper_walking_model import ROOT, apply_gripper_model, gripper_spec

TASK_ID = "Mjlab-Velocity-Flat-Kangaroo-Tray-Carry"
POSE_PATH = ROOT / "assets/tray/carry_pose.json"
TRAY_MASS = 0.1
# Saved tray pose relative to the base link and pelvis/arm frame respectively.
TRAY_REL_POS_BASE = torch.tensor([0.2502660259, -0.2, -0.023], dtype=torch.float32)
TRAY_REL_POS_TORSO = torch.tensor([0.34206578, -0.2, -0.0886], dtype=torch.float32)
TRAY_REL_QUAT = torch.tensor([0.5, 0.5, 0.5, 0.5], dtype=torch.float32)


def tray_spec() -> mujoco.MjSpec:
    """Contact model: thin tray floor and two rounded handle posts, total mass 100 g."""
    spec = mujoco.MjSpec()
    src = ROOT / "assets/tray/tray_original.stl"
    spec.add_mesh(name="tray_visual", file=str(src), scale=[0.001] * 3)
    body = spec.worldbody.add_body(name="tray")
    body.add_freejoint(name="tray_free")
    # The original part is 400 x 300 mm. Its CAD origin has the same rotation and
    # translation convention as the static editor. Collisions are decomposed into a
    # shallow pan and two elliptical posts so the concave pan is not convex-hull filled.
    body.add_geom(
        name="tray_visual_geom", type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname="tray_visual", density=0.0, contype=0, conaffinity=0,
        rgba=(0.22, 0.55, 0.82, 1.0),
    )
    contact = dict(condim=4, friction=(0.9, 0.02, 0.0001),
    solref=(0.01, 1.0), solimp=(0.98, 0.995, 0.001, 0.5, 2.0))
    body.add_geom(
        name="tray_pan_collision", type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=(0.2, 0.125, 0.15), size=(0.2, 0.025, 0.15), mass=0.07,
        rgba=(0.22, 0.55, 0.82, 0.0), **contact,
    )
    post_quat = (math.sqrt(0.5), -math.sqrt(0.5), 0.0, 0.0)
    for idx, x in enumerate((0.05, 0.35)):
        body.add_geom(
            name=f"tray_handle_{idx}_collision", type=mujoco.mjtGeom.mjGEOM_ELLIPSOID,
            pos=(x, 0.05, 0.225), quat=post_quat, size=(0.018, 0.05, 0.03),
            mass=0.015, contype=0, conaffinity=0,
            rgba=(0.22, 0.55, 0.82, 0.0), **contact,
        )
    body.add_site(name="carry_anchor_right", pos=(0.05, 0.05, 0.225), size=(0.004,))
    body.add_site(name="carry_anchor_left", pos=(0.35, 0.05, 0.225), size=(0.004,))
    return spec


def add_grasp_constraints(spec: mujoco.MjSpec):
    """Latch each modeled handle to its matching gripper at the saved fit pose."""
    # The coordinates were derived from the saved pose's arm kinematics. Sites
    # follow the hand links; site-to-site constraints work across every reset yaw.
    hand_anchors = {
        "left": ((-0.00067423, -0.01055860, 0.18386543),
                 (0.96378140, -0.00578730, 0.26620066, -0.01513695)),
        "right": ((0.00066353, -0.01051582, 0.18386666),
                  (-0.96377914, 0.00581892, 0.26620957, -0.01511172)),
    }
    for side, (local_pos, local_quat) in hand_anchors.items():
        hand = spec.body(f"robot/arm_{side}_7_link")
        hand.add_site(name=f"robot_carry_anchor_{side}", pos=local_pos,
                      quat=local_quat, size=(0.004,))
        eq = spec.add_equality(name=f"tray_grasp_{side}")
        eq.type = mujoco.mjtEq.mjEQ_WELD if side == "left" else mujoco.mjtEq.mjEQ_CONNECT
        eq.objtype = mujoco.mjtObj.mjOBJ_SITE
        eq.name1 = f"tray/carry_anchor_{side}"
        eq.name2 = f"robot_carry_anchor_{side}"
        eq.solref = (0.01, 1.0)


def tray_reset(env, env_ids: torch.Tensor, tray_cfg: SceneEntityCfg, robot_cfg: SceneEntityCfg):
    tray, robot = env.scene[tray_cfg.name], env.scene[robot_cfg.name]
    # Reset events run before the environment's observation refresh, so the robot's
    # cached root pose can still be from the previous episode. Refresh MuJoCo state
    # after reset_base/reset_robot_joints before deriving the tray's world pose.
    env.sim.forward()
    root_pos = robot.data.root_link_pos_w[env_ids]
    root_quat = robot.data.root_link_quat_w[env_ids]
    rel_pos = TRAY_REL_POS_BASE.to(device=env.device).expand(len(env_ids), -1)
    rel_quat = TRAY_REL_QUAT.to(device=env.device).expand(len(env_ids), -1)
    pos = root_pos + quat_apply(root_quat, rel_pos)
    quat = quat_mul(root_quat, rel_quat)
    pose = torch.cat((pos, quat), dim=-1)
    tray.write_root_link_pose_to_sim(pose, env_ids=env_ids)
    tray.write_root_link_velocity_to_sim(torch.zeros(len(env_ids), 6, device=env.device), env_ids=env_ids)


def hold_nominal_targets(env, env_ids: torch.Tensor, robot_cfg: SceneEntityCfg):
    robot = env.scene[robot_cfg.name]
    robot.set_joint_position_target(robot.data.default_joint_pos[env_ids], env_ids=env_ids)


def _torso_state(robot):
    body_ids, _ = robot.find_bodies("pelvis_2_link")
    idx = int(body_ids[0])
    data = robot.data
    return (
        data.body_link_pos_w[:, idx], data.body_link_quat_w[:, idx],
        data.body_link_lin_vel_w[:, idx], data.body_link_ang_vel_w[:, idx],
    )


def tray_position_reward(env, tray_cfg: SceneEntityCfg, robot_cfg: SceneEntityCfg, std: float = 0.08):
    tray, robot = env.scene[tray_cfg.name], env.scene[robot_cfg.name]
    torso_pos, torso_quat, _, _ = _torso_state(robot)
    rel = quat_apply_inverse(torso_quat, tray.data.root_link_pos_w - torso_pos)
    target = TRAY_REL_POS_TORSO.to(device=env.device)
    return torch.exp(-torch.sum((rel - target) ** 2, dim=-1) / (std * std))


def tray_upright_reward(env, tray_cfg: SceneEntityCfg):
    tray = env.scene[tray_cfg.name]
    local_up = torch.tensor([0.0, 1.0, 0.0], device=env.device).expand(len(tray.data.root_link_pos_w), -1)
    world_up = quat_apply(tray.data.root_link_quat_w, local_up)
    return torch.clamp(world_up[:, 2], min=0.0)


def tray_orientation_reward(env, tray_cfg: SceneEntityCfg, robot_cfg: SceneEntityCfg):
    """Keep the tray's full grasp orientation relative to the torso."""
    tray, robot = env.scene[tray_cfg.name], env.scene[robot_cfg.name]
    _, torso_quat, _, _ = _torso_state(robot)
    rel = quat_mul(
        torch.cat((torso_quat[:, :1], -torso_quat[:, 1:]), dim=-1),
        tray.data.root_link_quat_w,
    )
    target = TRAY_REL_QUAT.to(device=env.device).expand_as(rel)
    # q and -q represent the same rotation; abs(dot) makes the loss sign invariant.
    alignment = torch.abs(torch.sum(rel * target, dim=-1)).clamp(0.0, 1.0)
    return alignment.square()


def tray_relative_motion_penalty(env, tray_cfg: SceneEntityCfg, robot_cfg: SceneEntityCfg):
    tray, robot = env.scene[tray_cfg.name], env.scene[robot_cfg.name]
    torso_pos, _, torso_lin_vel, torso_ang_vel = _torso_state(robot)
    pos_delta = tray.data.root_link_pos_w - torso_pos
    expected_v = torso_lin_vel + torch.cross(torso_ang_vel, pos_delta, dim=-1)
    dv = tray.data.root_link_lin_vel_w - expected_v
    dw = tray.data.root_link_ang_vel_w - torso_ang_vel
    return torch.sum(dv * dv, dim=-1) + 0.1 * torch.sum(dw * dw, dim=-1)


def tray_fallen(env, tray_cfg: SceneEntityCfg, robot_cfg: SceneEntityCfg):
    tray, robot = env.scene[tray_cfg.name], env.scene[robot_cfg.name]
    torso_pos, _, _, _ = _torso_state(robot)
    rel = tray.data.root_link_pos_w - torso_pos
    up = quat_apply(tray.data.root_link_quat_w,
                    torch.tensor([0.0, 1.0, 0.0], device=env.device).expand(len(rel), -1))
    return (rel[:, 2] < -0.30) | (up[:, 2] < 0.45) | (torch.linalg.norm(rel[:, :2], dim=-1) > 0.65)


def carry_env_cfg(play: bool = False):
    cfg = walking_task.walking_env_cfg(play=play)
    # Preserve the actor/critic dimensions of the unloaded walking policy for a warm start.
    # Keep the seven arm joints fixed at the grasp pose while leaving the waist
    # available to the walking policy for balance.
    robot = cfg.scene.entities["robot"]
    pose = json.loads(POSE_PATH.read_text())
    default = {k: v for k, v in robot.init_state.joint_pos.items()
               if k.startswith("leg_") or k.startswith("pelvis_")}
    default.update({k: v for k, v in pose["joint_positions_rad"].items()
                    if k.startswith(("arm_", "gripper_"))})
    robot.init_state.joint_pos = default

    action = cfg.actions["joint_pos"]
    for name in list(action.scale):
        if name.startswith("arm_"):
            action.scale[name] = 0.0
    cfg.scene.entities["tray"] = EntityCfg(
        spec_fn=tray_spec,
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.2502660259, -0.2, 0.877), rot=(0.5, 0.5, 0.5, 0.5),
            joint_pos={}, joint_vel={},
        ),
    )
    cfg.scene.spec_fn = add_grasp_constraints
    cfg.events["hold_nominal_targets"] = EventTermCfg(
        func=hold_nominal_targets, mode="reset", params={"robot_cfg": SceneEntityCfg("robot")}
    )
    cfg.events["reset_tray"] = EventTermCfg(
        func=tray_reset, mode="reset",
        params={"tray_cfg": SceneEntityCfg("tray"), "robot_cfg": SceneEntityCfg("robot")},
    )
    cfg.rewards["tray_position"] = RewardTermCfg(
        func=tray_position_reward, weight=2.0,
        params={"tray_cfg": SceneEntityCfg("tray"), "robot_cfg": SceneEntityCfg("robot"), "std": 0.08},
    )
    cfg.rewards["tray_upright"] = RewardTermCfg(
        func=tray_upright_reward, weight=1.0, params={"tray_cfg": SceneEntityCfg("tray")}
    )
    cfg.rewards["tray_orientation"] = RewardTermCfg(
        func=tray_orientation_reward, weight=1.0,
        params={"tray_cfg": SceneEntityCfg("tray"), "robot_cfg": SceneEntityCfg("robot")},
    )
    cfg.rewards["tray_relative_motion"] = RewardTermCfg(
        func=tray_relative_motion_penalty, weight=-0.08,
        params={"tray_cfg": SceneEntityCfg("tray"), "robot_cfg": SceneEntityCfg("robot")},
    )
    cfg.terminations["tray_fallen"] = TerminationTermCfg(
        func=tray_fallen, params={"tray_cfg": SceneEntityCfg("tray"), "robot_cfg": SceneEntityCfg("robot")}
    )
    cfg.scene.num_envs = 1 if play else 1024
    cfg.episode_length_s = 40.0
    cfg.viewer.body_name = "pelvis_2_link"
    return cfg


agent = copy.deepcopy(walking_task.agent)
agent.experiment_name = "kangaroo_walking"
agent.run_name = "tray_carry_010kg"
agent.max_iterations = 10000
agent.save_interval = 250
register_mjlab_task(
    task_id=TASK_ID, env_cfg=carry_env_cfg(), play_env_cfg=carry_env_cfg(play=True),
    rl_cfg=agent, runner_cls=VelocityOnPolicyRunner,
)
