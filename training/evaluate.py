"""Fixed-command survival and speed tracking evaluation, independent of reward."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import torch
import walking_task
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--envs-per-command', type=int, default=64)
    p.add_argument('--seconds', type=float, default=30)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    assert args.seconds > 0 and args.envs_per_command > 0
    configure_torch_backends()
    scenarios = {
        'stand': (0., 0., 0.), 'forward_0.3': (.3, 0., 0.),
        'forward_0.6': (.6, 0., 0.), 'forward_1.0': (1., 0., 0.),
        'forward_1.2': (1.2, 0., 0.), 'backward': (-.4, 0., 0.),
        'lateral_left': (0., .2, 0.), 'lateral_right': (0., -.2, 0.),
        'turn_left': (.4, 0., .4), 'turn_right': (.4, 0., -.4),
        'command_switch': (0., 0., 0.),
    }
    cfg = walking_task.walking_env_cfg(play=True)
    cfg.scene.num_envs = len(scenarios) * args.envs_per_command
    cfg.seed = args.seed
    cfg.episode_length_s = args.seconds + 10
    cmd_cfg = cfg.commands['twist']
    cmd_cfg.heading_command = False
    cmd_cfg.ranges.heading = None
    for field in ['rel_standing_envs', 'rel_heading_envs', 'rel_forward_envs', 'rel_world_envs', 'init_velocity_prob']:
        setattr(cmd_cfg, field, 0.)
    cmd_cfg.resampling_time_range = (1e8, 1e8)
    cmd_cfg.ranges.lin_vel_x = (0., 0.)
    cmd_cfg.ranges.lin_vel_y = (0., 0.)
    cmd_cfg.ranges.ang_vel_z = (0., 0.)
    raw = ManagerBasedRlEnv(cfg=cfg, device=args.device)
    agent_cfg = load_rl_cfg(walking_task.TASK_ID)
    env = RslRlVecEnvWrapper(raw, clip_actions=agent_cfg.clip_actions)
    runner = load_runner_cls(walking_task.TASK_ID)(env, asdict(agent_cfg), device=args.device)
    runner.load(str(args.checkpoint.resolve()), load_cfg={'actor': True}, strict=True, map_location=args.device)
    policy = runner.get_inference_policy(device=args.device)
    env.reset()
    command = raw.command_manager.get_term('twist')
    desired = torch.tensor(list(scenarios.values()), device=args.device).repeat_interleave(args.envs_per_command, dim=0)
    alive = torch.ones(cfg.scene.num_envs, device=args.device, dtype=torch.bool)
    first_failure = torch.full((cfg.scene.num_envs,), float(args.seconds), device=args.device, dtype=torch.float32)
    squared = torch.zeros(cfg.scene.num_envs, 3, device=args.device)
    failures = {name: torch.zeros(cfg.scene.num_envs, device=args.device, dtype=torch.bool) for name in raw.termination_manager.active_terms}
    counts = torch.zeros(cfg.scene.num_envs, device=args.device)
    sequence = [(0., 0., 0.), (.6, 0., 0.), (.4, 0., .4), (0., 0., 0.)]
    steps = round(args.seconds / raw.step_dt)
    for step in range(steps):
        elapsed = step * raw.step_dt
        phase = min(int(4 * elapsed / args.seconds), 3)
        desired[-args.envs_per_command:] = torch.tensor(sequence[phase], device=args.device)
        command.vel_command_b.copy_(desired)
        obs = env.get_observations()
        with torch.inference_mode():
            action = policy(obs)
        if not torch.isfinite(action).all():
            raise RuntimeError('Non-finite policy action')
        _, _, done, _ = env.step(action)
        newly_failed = alive & done.bool()
        for name in failures:
            failures[name] |= newly_failed & raw.termination_manager.get_term(name)
        first_failure[newly_failed] = (step + 1) * raw.step_dt
        alive &= ~done.bool()
        robot = raw.scene['robot']
        measured = torch.cat((robot.data.root_link_lin_vel_b[:, :2], robot.data.root_link_ang_vel_b[:, 2:3]), dim=1)
        valid = alive.clone()
        if elapsed < 2.:
            valid[:] = False
        if elapsed % (args.seconds / 4) < 2.:
            valid[-args.envs_per_command:] = False
        squared[valid] += (measured[valid] - desired[valid]).square()
        counts[valid] += 1
    results = {}
    for i, name in enumerate(scenarios):
        sl = slice(i * args.envs_per_command, (i + 1) * args.envs_per_command)
        n = counts[sl].sum().item()
        rmse = (squared[sl].sum(0) / n).sqrt().tolist() if n else None
        survival = alive[sl].float().mean().item()
        results[name] = {
            'survival_rate': survival,
            'failure_counts': {name: int(value[sl].sum().item()) for name, value in failures.items()},
            'mean_time_to_first_failure_s': first_failure[sl].mean().item(),
            'velocity_rmse_xyz': rmse,
            'valid_env_steps': int(n),
            'pass': bool(args.seconds >= 30 and args.envs_per_command >= 64 and survival >= .95 and rmse and all(x <= lim for x, lim in zip(rmse, [.2, .15, .2]))),
        }
    report = {'robot_model': str(__import__('os').environ.get('KANGAROO_WALKING_MODEL', 'grippers')), 'checkpoint': str(args.checkpoint.resolve()), 'seed': args.seed,
              'seconds': args.seconds, 'envs_per_command': args.envs_per_command,
              'conditions': 'No pushes or observation noise; official startup domain randomization retained. RMSE excludes first 2 seconds and 2 seconds after sequence switches, and all samples after first failure. Velocity units: m/s, m/s, rad/s.',
              'acceptance': 'Every scenario: 30s or longer, >=64 trials, >=95% survival, RMSE <= [0.2,0.15,0.2]. Repeat with additional seeds and visual inspection before final acceptance.',
              'results': results, 'all_pass': all(r['pass'] for r in results.values())}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    env.close()


if __name__ == '__main__':
    main()
