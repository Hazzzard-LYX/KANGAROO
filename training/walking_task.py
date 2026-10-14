"""Unloaded walking baseline derived from PAL Robotics' official flat task."""
from pathlib import Path
import os
from gripper_walking_model import apply_gripper_model
import pal_mjlab.tasks  # Register official tasks as well.
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner
from pal_mjlab.tasks.velocity.kangaroo.env_cfgs import pal_kangaroo_flat_env_cfg
from pal_mjlab.tasks.velocity.kangaroo.rl_cfg import pal_kangaroo_ppo_runner_cfg
import pal_mjlab.robots.pal_kangaroo.kangaroo_constants as robot_constants

# Use the assets downloaded for this project, retaining PAL's actuator setup.
ASSETS = Path(__file__).resolve().parents[1] / 'assets/kangaroo/xmls'
robot_constants.KANGAROO_XML = ASSETS / 'kangaroo.xml'
TASK_ID = 'Mjlab-Velocity-Flat-Kangaroo-Walking'


def walking_env_cfg(play=False):
    cfg = pal_kangaroo_flat_env_cfg(play=play)
    if os.environ.get("KANGAROO_WALKING_MODEL", "grippers") == "grippers":
        apply_gripper_model(cfg)
    command = cfg.commands['twist']
    command.ranges.lin_vel_x = (-0.3, 0.6)
    command.ranges.lin_vel_y = (-0.2, 0.2)
    command.ranges.ang_vel_z = (-0.4, 0.4)
    if not play:
        cfg.curriculum['command_vel'].params['velocity_stages'] = [
            {'step': 0, 'lin_vel_x': (-0.3, 0.6), 'ang_vel_z': (-0.4, 0.4)},
            {'step': 2000 * 24, 'lin_vel_x': (-0.5, 0.9), 'ang_vel_z': (-0.5, 0.5)},
            {'step': 5000 * 24, 'lin_vel_x': (-0.6, 1.2), 'ang_vel_z': (-0.6, 0.6)},
        ]
    else:
        command.ranges.lin_vel_x = (-0.6, 1.2)
        command.ranges.ang_vel_z = (-0.6, 0.6)
    return cfg


agent = pal_kangaroo_ppo_runner_cfg()
agent.experiment_name = 'kangaroo_walking'
agent.logger = 'tensorboard'
agent.max_iterations = 10000
agent.save_interval = 250
register_mjlab_task(
    task_id=TASK_ID,
    env_cfg=walking_env_cfg(),
    play_env_cfg=walking_env_cfg(play=True),
    rl_cfg=agent,
    runner_cls=VelocityOnPolicyRunner,
)
