"""CYCLO service interactive loop. Commands execute on the physics thread only.

The Agent owns the two spool paths and authenticates/fences every command before
appending. This module accepts no Python/Hydra/shell fragments from web clients.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from uuid import UUID


def timestamp():
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


class ControlState:
    def __init__(self):
        self.paused = False
        self.pending_steps = 0
        self.stepping_result = None
        self.results = []
        self.seen = set()

    def apply(self, item, step, reset):
        command_id = item.get('command_id') if isinstance(item, dict) else None
        try:
            if str(UUID(command_id)) != command_id or command_id in self.seen:
                return
        except (ValueError, TypeError, AttributeError):
            return
        self.seen.add(command_id)
        result = {'command_id': command_id, 'status': 'rejected',
                  'applied_physics_step': None, 'error': 'INVALID_COMMAND'}
        try:
            expires = datetime.fromisoformat(item['expires_at'].replace('Z', '+00:00'))
            if expires.utcoffset() is None or expires <= datetime.now(timezone.utc):
                raise ValueError('expired')
            command = item['command']
            action = command['action']
            if set(command) != ({'action', 'steps'} if action == 'step' else {'action', 'seed'} if action == 'reset' else {'action'}):
                raise ValueError('unknown field')
            if self.pending_steps:
                raise ValueError('step sequence in progress')
            if action == 'sim_play':
                self.paused = False
            elif action == 'sim_pause':
                self.paused = True
                self.pending_steps = 0
            elif action == 'reset':
                seed = command['seed']
                if type(seed) is not int or not 0 <= seed <= 2147483647:
                    raise ValueError('invalid seed')
                reset(seed=seed)
            elif action == 'step':
                steps = command['steps']
                if not self.paused or type(steps) is not int or not 1 <= steps <= 100:
                    raise ValueError('invalid step')
                self.pending_steps = steps
            else:
                raise ValueError('unsupported action')
            result.update(status='applied', applied_physics_step=step, error=None)
            if action == 'step':
                result.update(status='accepted', applied_physics_step=None)
                self.stepping_result = result
        except (ValueError, KeyError, TypeError, AttributeError):
            pass
        except Exception:
            import traceback
            traceback.print_exc()
            result.update(status='failed', error='PHYSICS_COMMAND_FAILED')
        self.results = (self.results + [result])[-1000:]


    def advance(self, step):
        if self.pending_steps:
            self.pending_steps -= 1
            if self.pending_steps == 0:
                self.stepping_result.update(status='applied', applied_physics_step=step)
                self.stepping_result = None


def write_state(path, state):
    temporary = path.with_suffix('.next')
    with temporary.open('w') as stream:
        json.dump(state, stream, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)



# Presentation only: these settings never modify USD, rewards, or physics state.
WEB_EYE = (-1.2, -1.2, 0.9)
WEB_TARGET = (0.1, 0.0, 0.2)


def configure_web_view(sim):
    """Use the pinned Newton/Viser camera lifecycle for initial and reconnecting clients."""
    import math
    direction = tuple(b - a for a, b in zip(WEB_EYE, WEB_TARGET))
    pitch = math.degrees(math.atan2(direction[2], math.hypot(*direction[:2])))
    yaw = math.degrees(math.atan2(direction[1], direction[0]))
    for visualizer in sim.visualizers:
        if type(visualizer).__name__ != 'ViserVisualizer':
            continue
        # Isaac Lab exposes visualizers; its pinned Newton backend owns this server.
        # Keep this small compatibility boundary in service code, not the upstream checkout.
        viewer = visualizer._viewer
        viewer.set_camera(WEB_EYE, pitch, yaw)
        server = viewer._server
        server.initial_camera.look_at = WEB_TARGET
        server.scene.add_light_directional('/cyclo/key', intensity=2.5,
                                          position=(3.0, 2.0, 4.0), cast_shadow=False)
        server.scene.add_light_directional('/cyclo/fill', intensity=1.2,
                                          position=(-3.0, -1.0, 2.0), cast_shadow=False)
        sim.set_camera_view(WEB_EYE, WEB_TARGET)
        return
    raise RuntimeError('The service requires the pinned Viser visualizer')


def main():
    os.environ['CYCLO_SERVICE_PROFILE'] = 'omy'
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=['Cyclo-Reach-OMY-v0'], required=True)
    parser.add_argument('--num_envs', type=int, choices=[1], default=1)
    parser.add_argument('--state_file', type=Path, required=True)
    parser.add_argument('--commands_file', type=Path, required=True)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.state_file.parent.mkdir(parents=True, exist_ok=True)
    launcher = AppLauncher(args)
    app = launcher.app
    env = None
    try:
        import gymnasium as gym
        import torch
        import cyclo_lab  # noqa: F401 — register actual robot task
        from isaaclab_tasks.utils import parse_env_cfg

        cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
        from isaaclab.visualizers import VisualizerCfg
        cfg.sim.default_visualizer_cfg = VisualizerCfg(eye=WEB_EYE, lookat=WEB_TARGET)
        env = gym.make(args.task, cfg=cfg)
        env.reset()
        configure_web_view(env.unwrapped.sim)
        control = ControlState()
        step = 0
        offset = 0
        publish_at = 0.0
        while app.is_running():
            if args.commands_file.exists():
                with args.commands_file.open('rb') as stream:
                    stream.seek(offset)
                    for _ in range(100):
                        line = stream.readline(8193)
                        if not line or not line.endswith(b'\n'):
                            break
                        offset += len(line)
                        if len(line) > 8192:
                            continue
                        try:
                            with torch.inference_mode():
                                control.apply(json.loads(line), step, env.reset)
                        except (ValueError, UnicodeDecodeError):
                            pass
            if not control.paused or control.pending_steps:
                with torch.inference_mode():
                    actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
                    env.step(actions)
                step += 1
                if control.paused and control.pending_steps:
                    control.advance(step)
            else:
                env.unwrapped.sim.render()
                time.sleep(0.01)
            if time.monotonic() >= publish_at:
                write_state(args.state_file, {'runtime_ready': step > 0,
                    'physics_step': step, 'paused': control.paused,
                    'observed_at': timestamp(), 'command_results': control.results})
                publish_at = time.monotonic() + 0.25
    except BaseException:
        import traceback
        traceback.print_exc()
        raise
    finally:
        if env is not None:
            env.close()
        app.close()


if __name__ == '__main__':
    main()
