"""Fine tune stable checkpoint with sustained standing and explicit lateral commands."""
from dataclasses import dataclass, fields
from pathlib import Path
import torch
import walking_task
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommand, UniformVelocityCommandCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.scripts.train import TrainConfig, launch_training

class BalancedCommand(UniformVelocityCommand):
    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        choice = torch.rand(len(env_ids), device=self.device)
        stand = env_ids[choice < .30]
        lateral = env_ids[(choice >= .30) & (choice < .65)]
        self.vel_command_b[stand] = 0.
        self.vel_command_b[lateral] = 0.
        sign = torch.where(torch.rand(len(lateral), device=self.device) < .5, -1., 1.)
        self.vel_command_b[lateral, 1] = sign * torch.empty(len(lateral), device=self.device).uniform_(.12, .25)
        self.is_standing_env[env_ids] = False
        self.is_standing_env[stand] = True
        self.is_forward_env[env_ids] = False
        self.is_heading_env[env_ids] = False
        self.is_world_env[env_ids] = False

@dataclass(kw_only=True)
class BalancedCommandCfg(UniformVelocityCommandCfg):
    def build(self, env):
        return BalancedCommand(self, env)

def lateral_tracking(env):
    cmd = env.command_manager.get_command('twist')
    vy = env.scene['robot'].data.root_link_lin_vel_b[:, 1]
    return torch.exp(-((vy-cmd[:,1])/.1).square()) * (cmd[:,1].abs() > .05)

def main():
    root = Path(__file__).resolve().parents[1]
    cfg = TrainConfig.from_task(walking_task.TASK_ID)
    cfg.env.scene.num_envs = 1024
    cfg.env.episode_length_s = 40.
    old = cfg.env.commands['twist']
    cmd = BalancedCommandCfg(**{f.name:getattr(old,f.name) for f in fields(old) if f.init})
    cmd.heading_command=False
    cmd.ranges.heading=None
    cmd.rel_standing_envs=0.
    cmd.rel_heading_envs=0.
    cmd.rel_forward_envs=0.
    cmd.resampling_time_range=(10.,30.)
    cmd.ranges.lin_vel_x=(-.6,1.2)
    cmd.ranges.lin_vel_y=(-.25,.25)
    cmd.ranges.ang_vel_z=(-.6,.6)
    cfg.env.commands['twist']=cmd
    cfg.env.curriculum.pop('command_vel',None)
    cfg.env.rewards['lateral_tracking']=RewardTermCfg(func=lateral_tracking,weight=1.)
    cfg.agent.resume=True
    cfg.agent.load_run='2026-10-07_11-10-40_full_walking'
    cfg.agent.load_checkpoint='model_7500.pt'
    cfg.agent.max_iterations=4000
    cfg.agent.run_name='refine_stand_lateral'
    cfg.agent.seed=43
    cfg.agent.save_interval=250
    launch_training(walking_task.TASK_ID,cfg)

if __name__=='__main__':main()
