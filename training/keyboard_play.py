"""Hold arrows to walk; hold Ctrl + left/right to turn. X11 native demo."""
import argparse
import ctypes as C
import ctypes.util
from dataclasses import asdict
import json
import os
from pathlib import Path


def velocity_for_keys(keys, forward=.6, backward=.4, lateral=.2, yaw=.4):
    vx = forward * ('Up' in keys) - backward * ('Down' in keys)
    if 'Up' in keys and 'Down' in keys:
        vx = 0.
    side = int('Left' in keys) - int('Right' in keys)
    return (vx, 0., yaw * side) if 'Control' in keys else (vx, lateral * side, 0.)


class X11Keyboard:
    """Poll held state (including release) only while this viewer has focus."""
    def __init__(self):
        self.x = C.CDLL(ctypes.util.find_library('X11'))
        self.x.XOpenDisplay.argtypes = [C.c_char_p]
        self.x.XOpenDisplay.restype = C.c_void_p
        self.display = self.x.XOpenDisplay(None)
        if not self.display:
            raise RuntimeError('X11 display unavailable; run from the desktop session.')
        W = C.c_ulong
        self.x.XKeysymToKeycode.argtypes = [C.c_void_p, W]
        self.x.XKeysymToKeycode.restype = C.c_ubyte
        self.x.XQueryKeymap.argtypes = [C.c_void_p, C.c_void_p]
        self.x.XGetInputFocus.argtypes = [C.c_void_p, C.POINTER(W), C.POINTER(C.c_int)]
        self.x.XInternAtom.argtypes = [C.c_void_p, C.c_char_p, C.c_int]
        self.x.XInternAtom.restype = W
        self.x.XGetWindowProperty.argtypes = [C.c_void_p, W, W, C.c_long, C.c_long, C.c_int, W, C.POINTER(W), C.POINTER(C.c_int), C.POINTER(W), C.POINTER(W), C.POINTER(C.c_void_p)]
        self.x.XQueryTree.argtypes = [C.c_void_p, W, C.POINTER(W), C.POINTER(W), C.POINTER(C.c_void_p), C.POINTER(C.c_uint)]
        self.x.XFree.argtypes = [C.c_void_p]
        self.x.XStoreName.argtypes = [C.c_void_p, W, C.c_char_p]
        self.x.XFlush.argtypes = [C.c_void_p]
        self.x.XCloseDisplay.argtypes = [C.c_void_p]
        self.pid_atom = self.x.XInternAtom(self.display, b'_NET_WM_PID', 0)
        symbols = {'Up': 0xff52, 'Down': 0xff54, 'Left': 0xff51, 'Right': 0xff53, 'Control_L': 0xffe3, 'Control_R': 0xffe4}
        self.codes = {name: self.x.XKeysymToKeycode(self.display, value) for name, value in symbols.items()}
        self.last_title = None

    def focused_window(self):
        window, revert = C.c_ulong(), C.c_int()
        self.x.XGetInputFocus(self.display, C.byref(window), C.byref(revert))
        for _ in range(8):
            if window.value <= 1:
                break
            typ, count, remaining = C.c_ulong(), C.c_ulong(), C.c_ulong()
            fmt, data = C.c_int(), C.c_void_p()
            self.x.XGetWindowProperty(self.display, window, self.pid_atom, 0, 1, 0, 0, C.byref(typ), C.byref(fmt), C.byref(count), C.byref(remaining), C.byref(data))
            pid = C.cast(data, C.POINTER(C.c_ulong))[0] if data.value and count.value and fmt.value == 32 else None
            if data.value:
                self.x.XFree(data)
            if pid == os.getpid():
                return window.value
            root, parent, children, n = C.c_ulong(), C.c_ulong(), C.c_void_p(), C.c_uint()
            self.x.XQueryTree(self.display, window, C.byref(root), C.byref(parent), C.byref(children), C.byref(n))
            if children.value:
                self.x.XFree(children)
            window = parent
        return None

    def read(self):
        window = self.focused_window()
        if window is None:
            return set(), None
        buf = (C.c_ubyte * 32)()
        self.x.XQueryKeymap(self.display, buf)
        keys = {name for name, code in self.codes.items() if buf[code // 8] & (1 << (code % 8))}
        if keys & {'Control_L', 'Control_R'}:
            keys.add('Control')
        return keys, window

    def title(self, window, velocity):
        label = f'Kangaroo keyboard | vx={velocity[0]:+.2f} vy={velocity[1]:+.2f} yaw={velocity[2]:+.2f} | Arrows: walk | Ctrl+Left/Right: turn | Enter: reset'
        if window and label != self.last_title:
            self.x.XStoreName(self.display, window, label.encode())
            self.x.XFlush(self.display)
            self.last_title = label

    def close(self):
        self.x.XCloseDisplay(self.display)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--forward', type=float, default=.6)
    p.add_argument('--backward', type=float, default=.4)
    p.add_argument('--lateral', type=float, default=.2)
    p.add_argument('--yaw', type=float, default=.4)
    args = p.parse_args()
    import walking_task
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_rl_cfg, load_runner_cls
    from mjlab.viewer import NativeMujocoViewer
    from mjlab.utils.torch import configure_torch_backends
    root = Path(__file__).resolve().parents[1]
    if args.checkpoint is None:
        selected = root / 'logs/selected_policy.json'
        if selected.exists():
            args.checkpoint = Path(json.loads(selected.read_text())['checkpoint'])
        else:
            state = json.loads((root / 'logs/active_training.json').read_text())
            args.checkpoint = max(Path(state['run_directory']).glob('model_*.pt'), key=lambda p: p.stat().st_mtime)
    configure_torch_backends()
    cfg = walking_task.walking_env_cfg(play=True)
    cfg.scene.num_envs = 1
    command = cfg.commands['twist']
    command.heading_command = False
    command.ranges.heading = None
    command.resampling_time_range = (1e8, 1e8)
    for field in ['rel_standing_envs', 'rel_heading_envs', 'rel_forward_envs', 'rel_world_envs', 'init_velocity_prob']:
        setattr(command, field, 0.)
    command.ranges.lin_vel_x = (0., 0.)
    command.ranges.lin_vel_y = (0., 0.)
    command.ranges.ang_vel_z = (0., 0.)
    raw = ManagerBasedRlEnv(cfg=cfg, device='cuda:0')
    agent_cfg = load_rl_cfg(walking_task.TASK_ID)
    env = RslRlVecEnvWrapper(raw, clip_actions=agent_cfg.clip_actions)
    runner = load_runner_cls(walking_task.TASK_ID)(env, asdict(agent_cfg), device='cuda:0')
    runner.load(str(args.checkpoint), load_cfg={'actor': True}, strict=True, map_location='cuda:0')
    policy = runner.get_inference_policy(device='cuda:0')
    keyboard = X11Keyboard()
    cmd = raw.command_manager.get_term('twist')

    def controlled_policy(_obs):
        keys, window = keyboard.read()
        velocity = velocity_for_keys(keys, args.forward, args.backward, args.lateral, args.yaw)
        for i, value in enumerate(velocity):
            cmd.vel_command_b[:, i] = value
        keyboard.title(window, velocity)
        return policy(env.get_observations())

    class KeyboardViewer(NativeMujocoViewer):
        def _safe_key_callback(self, key):
            # Right arrow normally single-steps MuJoCo; reserve all arrows here.
            if key not in (262, 263, 264, 265, 341, 345):
                super()._safe_key_callback(key)

    print(f'KEYBOARD_DEMO_READY checkpoint={args.checkpoint}', flush=True)
    print('Arrows=body-frame walking; Ctrl+Left/Right=turn; release/focus loss=zero command; Enter=reset; Space=pause.', flush=True)
    try:
        KeyboardViewer(env, controlled_policy).run()
    finally:
        keyboard.close()
        env.close()


if __name__ == '__main__':
    main()
