"""Loaded walking with a two-hand tray, trained from scratch.

Full action space (legs, waist, both 7-DoF arms, both grippers) and full proprioception,
plus the tray state. An adaptive curriculum raises the tray mass (1 -> 6 kg) and the
commanded speed range together: every reset samples a difficulty level per environment,
and the sampling distribution drifts towards harder levels only while the survival rate
at the current frontier stays high (no hard switch; easier levels are still replayed).
"""
from __future__ import annotations

import copy
import json
import math
import re

import torch
from mjlab.entity import EntityCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg, RecomputeLevel, requires_model_fields
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.mdp.velocity_command import (
    UniformVelocityCommand,
    UniformVelocityCommandCfg,
)
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner
from mjlab.utils.lab_api.math import quat_apply_inverse, quat_mul
from pal_mjlab.robots.pal_kangaroo.kangaroo_constants import KANGAROO_GRIPPERS_ACTION_SCALE
from pal_mjlab.tasks.velocity.kangaroo.env_cfgs import pal_kangaroo_flat_env_cfg

import tray_carry_task as tray
import walking_task

TASK_ID = "Mjlab-Velocity-Flat-Kangaroo-Loaded-Carry"

# Curriculum: level k in 0..N_LEVELS-1. Mass and speed scale rise together.
N_LEVELS = 11
MASS_MIN, MASS_MAX = 1.0, 6.0
SPEED_SCALE_MIN = 0.3
FINAL_RANGES = dict(lin_vel_x=(-0.6, 1.2), lin_vel_y=(-0.2, 0.2), ang_vel_z=(-0.6, 0.6))
FRONTIER_SIGMA = 0.7      # width of the sampling bump around the frontier (levels)
REPLAY_PROB = 0.2         # share of resets drawn uniformly from the levels below the frontier
PROMOTE_SURVIVAL = 0.85   # frontier survival rate that moves the frontier up
DEMOTE_SURVIVAL = 0.50    # ... and below which it moves down
FRONTIER_STEP = 0.1       # levels per decision
WINDOW_EPISODES = 256     # frontier episodes per decision

ARM_SCALE_FACTOR = 1.0 / 3.0  # of PAL's official arm action scales
GRIPPER_SCALE = 0.15          # rad per unit action, around the reset jaw angle

PASSIVE_LEG = ("leg_left_femur_joint", "leg_left_knee_joint",
               "leg_right_femur_joint", "leg_right_knee_joint")


def level_mass(level: torch.Tensor) -> torch.Tensor:
    return MASS_MIN + (MASS_MAX - MASS_MIN) * level.float() / (N_LEVELS - 1)


def level_speed_scale(level: torch.Tensor) -> torch.Tensor:
    return SPEED_SCALE_MIN + (1.0 - SPEED_SCALE_MIN) * level.float() / (N_LEVELS - 1)


class LoadCurriculum:
    """Per-environment difficulty levels and the adaptive frontier."""

    def __init__(self, num_envs: int, device: str):
        self.level = torch.zeros(num_envs, dtype=torch.long, device=device)
        # Step at which each env's current episode was drawn; -1 = not drawn by the
        # curriculum (e.g. episodes shortened by the runner's random initial lengths).
        self.start_step = torch.full((num_envs,), -1, dtype=torch.long, device=device)
        self.frontier = 0.0
        self.window_survived = 0
        self.window_total = 0
        self.last_rate = float("nan")
        self.survived_by_level = torch.zeros(N_LEVELS, device=device)
        self.total_by_level = torch.zeros(N_LEVELS, device=device)

    def probabilities(self) -> torch.Tensor:
        top = min(N_LEVELS - 1, math.floor(self.frontier) + 1)
        k = torch.arange(N_LEVELS, device=self.level.device, dtype=torch.float32)
        bump = torch.exp(-0.5 * ((k - self.frontier) / FRONTIER_SIGMA) ** 2)
        bump[top + 1:] = 0.0
        bump /= bump.sum()
        replay = torch.zeros_like(bump)
        replay[: math.floor(self.frontier) + 1] = 1.0
        replay /= replay.sum()
        return (1.0 - REPLAY_PROB) * bump + REPLAY_PROB * replay

    def record(self, env_ids: torch.Tensor, survived: torch.Tensor) -> None:
        lv = self.level[env_ids]
        self.total_by_level.index_add_(0, lv, torch.ones_like(lv, dtype=torch.float32))
        self.survived_by_level.index_add_(0, lv, survived.float())
        hard = lv >= math.floor(self.frontier)
        self.window_total += int(hard.sum())
        self.window_survived += int((survived & hard).sum())
        if self.window_total >= WINDOW_EPISODES:
            rate = self.window_survived / self.window_total
            self.last_rate = rate
            if rate >= PROMOTE_SURVIVAL:
                self.frontier = min(N_LEVELS - 1.0, self.frontier + FRONTIER_STEP)
            elif rate < DEMOTE_SURVIVAL:
                self.frontier = max(0.0, self.frontier - FRONTIER_STEP)
            self.window_survived = self.window_total = 0

    def sample(self, env_ids: torch.Tensor, step: int) -> None:
        p = self.probabilities()
        self.level[env_ids] = torch.multinomial(p, len(env_ids), replacement=True)
        self.start_step[env_ids] = step


def get_curriculum(env) -> LoadCurriculum:
    if not hasattr(env, "_load_curriculum"):
        env._load_curriculum = LoadCurriculum(env.num_envs, env.device)
    return env._load_curriculum


def load_curriculum(env, env_ids: torch.Tensor) -> dict[str, torch.Tensor]:
    """Record how the ending episodes finished, update the frontier, draw new levels."""
    cur = get_curriculum(env)
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    step = int(env.common_step_counter)
    if step > 0 and hasattr(env, "reset_time_outs"):
        # Count only complete episodes: the runner randomizes initial episode lengths,
        # so early time-outs can end after a few steps; those are neither success nor
        # failure. A failure is any termination; a success is a time-out after the
        # full episode length.
        length = step - cur.start_step[env_ids]
        valid = cur.start_step[env_ids] >= 0
        failed = env.reset_terminated[env_ids] & valid
        survived = env.reset_time_outs[env_ids] & ~env.reset_terminated[env_ids] \
            & valid & (length >= env.max_episode_length - 1)
        counted = failed | survived
        if counted.any():
            cur.record(env_ids[counted], survived[counted])
    cur.sample(env_ids, step if step > 0 else -1)
    rate_by_level = cur.survived_by_level / cur.total_by_level.clamp(min=1)
    out = {
        "frontier": torch.tensor(cur.frontier),
        "frontier_survival": torch.tensor(cur.last_rate),
        "mean_level": cur.level.float().mean(),
        "mean_mass_kg": level_mass(cur.level).mean(),
    }
    for k in range(N_LEVELS):
        out[f"survival_L{k:02d}"] = rate_by_level[k]
    return out


_TRAY_DEFAULTS: dict[str, torch.Tensor] = {}


@requires_model_fields("body_mass", "body_inertia", recompute=RecomputeLevel.set_const)
def apply_tray_mass(env, env_ids: torch.Tensor, tray_cfg: SceneEntityCfg):
    """Set the tray mass of each resetting env from its curriculum level (uniform density)."""
    tray_entity = env.scene[tray_cfg.name]
    body = int(tray_entity.indexing.body_ids[0])
    if "mass" not in _TRAY_DEFAULTS:
        mj = env.sim.mj_model
        _TRAY_DEFAULTS["mass"] = torch.tensor(float(mj.body_mass[body]), device=env.device)
        _TRAY_DEFAULTS["inertia"] = torch.tensor(mj.body_inertia[body], device=env.device,
                                                 dtype=torch.float32)
    mass = level_mass(get_curriculum(env).level[env_ids])
    ratio = (mass / _TRAY_DEFAULTS["mass"]).unsqueeze(-1)
    env.sim.model.body_mass[env_ids, body] = mass
    env.sim.model.body_inertia[env_ids, body] = _TRAY_DEFAULTS["inertia"] * ratio


class LevelVelocityCommand(UniformVelocityCommand):
    """Velocity command whose ranges are scaled per env by its curriculum level."""

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        super()._resample_command(env_ids)
        scale = level_speed_scale(get_curriculum(self._env).level[env_ids]).unsqueeze(-1)
        self.vel_command_b[env_ids] *= scale
        self.vel_command_w[env_ids] *= scale

    def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
        super()._update_command(env_ids)
        # Heading envs recompute yaw rate every step from the full range; cap it per level.
        if self.cfg.heading_command:
            cap = level_speed_scale(get_curriculum(self._env).level) * self.cfg.ranges.ang_vel_z[1]
            ids = self.is_heading_env
            self.vel_command_b[ids, 2] = torch.maximum(
                torch.minimum(self.vel_command_b[ids, 2], cap[ids]), -cap[ids])


class LevelVelocityCommandCfg(UniformVelocityCommandCfg):
    def build(self, env):
        return LevelVelocityCommand(self, env)


# -- Observations ----------------------------------------------------------------------

def tray_state_torso(env, tray_cfg: SceneEntityCfg, robot_cfg: SceneEntityCfg) -> torch.Tensor:
    """Tray position, orientation (w >= 0), linear and angular velocity in the torso frame."""
    tray_entity, robot = env.scene[tray_cfg.name], env.scene[robot_cfg.name]
    torso_pos, torso_quat, torso_v, torso_w = tray._torso_state(robot)
    d = tray_entity.data
    rel_pos = quat_apply_inverse(torso_quat, d.root_link_pos_w - torso_pos)
    conj = torch.cat((torso_quat[:, :1], -torso_quat[:, 1:]), dim=-1)
    rel_quat = quat_mul(conj, d.root_link_quat_w)
    rel_quat = rel_quat * torch.where(rel_quat[:, :1] < 0, -1.0, 1.0)
    expected_v = torso_v + torch.cross(torso_w, d.root_link_pos_w - torso_pos, dim=-1)
    rel_v = quat_apply_inverse(torso_quat, d.root_link_lin_vel_w - expected_v)
    rel_w = quat_apply_inverse(torso_quat, d.root_link_ang_vel_w - torso_w)
    return torch.cat((rel_pos, rel_quat, rel_v, rel_w), dim=-1)


def tray_mass_obs(env) -> torch.Tensor:
    return (level_mass(get_curriculum(env).level) / MASS_MAX).unsqueeze(-1)


def tray_contact_forces(env, sensor_name: str) -> torch.Tensor:
    data = env.scene[sensor_name].data
    f = data.force.flatten(start_dim=1)
    return torch.sign(f) * torch.log1p(torch.abs(f))


# -- Config ----------------------------------------------------------------------------

def _official_value(table: dict, name: str):
    return next(v for rx, v in table.items() if re.fullmatch(rx, name))


def loaded_env_cfg(play: bool = False):
    cfg = walking_task.walking_env_cfg(play=play)
    official = pal_kangaroo_flat_env_cfg(play=play)
    robot = cfg.scene.entities["robot"]
    pose = json.loads(tray.POSE_PATH.read_text())

    # Reset pose: official legs/waist, arms and jaws from the tray pose file.
    default = {k: v for k, v in robot.init_state.joint_pos.items()
               if k.startswith(("leg_", "pelvis_"))}
    default.update({k: v for k, v in pose["joint_positions_rad"].items()
                    if k.startswith(("arm_", "gripper_"))})
    robot.init_state.joint_pos = default

    # -- Actions: 12 legs + 2 waist + 14 arms + 2 grippers = 30.
    legs = [n for n in walking_task_legacy_names()
            if n.startswith(("leg_", "pelvis_")) and n not in PASSIVE_LEG]
    arms = [f"arm_{s}_{i}_joint" for s in ("left", "right") for i in range(1, 8)]
    grippers = ["gripper_left_finger_joint", "gripper_right_finger_joint"]
    official_scale = KANGAROO_GRIPPERS_ACTION_SCALE  # PAL's 7-DoF-arm grippers model
    scale = {n: _official_value(official_scale, n) for n in legs}
    scale.update({n: ARM_SCALE_FACTOR * _official_value(official_scale, n) for n in arms})
    scale.update({n: GRIPPER_SCALE for n in grippers})
    action = cfg.actions["joint_pos"]
    assert isinstance(action, JointPositionActionCfg)
    action.actuator_names = tuple(legs + arms + grippers)
    action.scale = scale

    # -- Observations: every actuated joint + passive leg joints, and the tray state.
    obs_joints = list(action.actuator_names) + list(PASSIVE_LEG)
    joint_cfg = SceneEntityCfg("robot", joint_names=obs_joints, preserve_order=True)
    tray_params = {"tray_cfg": SceneEntityCfg("tray"), "robot_cfg": SceneEntityCfg("robot")}
    for name, group in cfg.observations.items():
        for term in ("joint_pos", "joint_vel"):
            group.terms[term].params["asset_cfg"] = joint_cfg
        group.terms["tray_state"] = ObservationTermCfg(func=tray_state_torso, params=tray_params)
        if name == "critic":
            group.terms["tray_mass"] = ObservationTermCfg(func=tray_mass_obs)
            group.terms["tray_contact_forces"] = ObservationTermCfg(
                func=tray_contact_forces, params={"sensor_name": "hand_tray_contact"})

    cfg.scene.sensors = (cfg.scene.sensors or ()) + (ContactSensorCfg(
        name="hand_tray_contact",
        primary=ContactMatch(mode="subtree", pattern=("arm_left_7_link", "arm_right_7_link"),
                             entity="robot"),
        secondary=ContactMatch(mode="body", pattern="tray", entity="tray"),
        fields=("found", "force"), reduce="netforce", num_slots=1,
    ),)

    # -- Tray entity and reset.
    cfg.scene.entities["tray"] = EntityCfg(
        spec_fn=tray.tray_spec,
        init_state=EntityCfg.InitialStateCfg(
            pos=tuple(pose["tray_pos"]), rot=(0.5, 0.5, 0.5, 0.5), joint_pos={}, joint_vel={}),
    )
    cfg.events["hold_nominal_targets"] = EventTermCfg(
        func=tray.hold_nominal_targets, mode="reset", params={"robot_cfg": SceneEntityCfg("robot")})
    cfg.events["reset_tray"] = EventTermCfg(func=tray.tray_reset, mode="reset", params=tray_params)
    cfg.events["tray_mass"] = EventTermCfg(
        func=apply_tray_mass, mode="reset", params={"tray_cfg": SceneEntityCfg("tray")})

    # -- Command and curriculum: per-env speed scaling replaces the step-based stages.
    old = cfg.commands["twist"]
    cmd = LevelVelocityCommandCfg(**{f: getattr(old, f) for f in old.__dataclass_fields__})
    cmd.ranges = copy.deepcopy(old.ranges)
    cmd.ranges.lin_vel_x = FINAL_RANGES["lin_vel_x"]
    cmd.ranges.lin_vel_y = FINAL_RANGES["lin_vel_y"]
    cmd.ranges.ang_vel_z = FINAL_RANGES["ang_vel_z"]
    cfg.commands["twist"] = cmd
    cfg.curriculum.pop("command_vel", None)
    cfg.curriculum["load"] = CurriculumTermCfg(func=load_curriculum)

    # -- Rewards: posture over all non-gripper actuated joints (arms held near the tray pose).
    pose_params = cfg.rewards["pose"].params
    pose_joints = legs + arms
    pose_params["asset_cfg"] = SceneEntityCfg("robot", joint_names=pose_joints, preserve_order=True)
    off_pose = official.rewards["pose"].params
    for key in ("std_standing", "std_walking", "std_running"):
        pose_params[key] = {n: _official_value(off_pose[key], n) for n in pose_joints}
    w = {"tray_position": 2.0, "tray_upright": 1.0, "tray_orientation": 1.0}
    cfg.rewards["tray_position"] = RewardTermCfg(
        func=tray.tray_position_reward, weight=w["tray_position"], params={**tray_params, "std": 0.08})
    cfg.rewards["tray_upright"] = RewardTermCfg(
        func=tray.tray_upright_reward, weight=w["tray_upright"], params={"tray_cfg": SceneEntityCfg("tray")})
    cfg.rewards["tray_orientation"] = RewardTermCfg(
        func=tray.tray_orientation_reward, weight=w["tray_orientation"], params=tray_params)
    cfg.rewards["tray_relative_motion"] = RewardTermCfg(
        func=tray.tray_relative_motion_penalty, weight=-0.08, params=tray_params)
    cfg.terminations["tray_fallen"] = TerminationTermCfg(func=tray.tray_fallen, params=tray_params)

    cfg.sim.nconmax = 128
    cfg.sim.njmax = 512
    cfg.scene.num_envs = 1 if play else 1024
    cfg.episode_length_s = 20.0
    cfg.viewer.body_name = "pelvis_2_link"
    return cfg


def walking_task_legacy_names():
    from gripper_walking_model import LEGACY_JOINTS
    return LEGACY_JOINTS


agent = copy.deepcopy(walking_task.agent)
agent.experiment_name = "kangaroo_loaded_carry"
agent.run_name = "scratch"
agent.max_iterations = 30000
agent.save_interval = 250
register_mjlab_task(
    task_id=TASK_ID, env_cfg=loaded_env_cfg(), play_env_cfg=loaded_env_cfg(play=True),
    rl_cfg=agent, runner_cls=VelocityOnPolicyRunner,
)
