"""World-model prediction and velocity controls for the SuperMimic playback viewer."""

import torch
from supermimic.tasks.task_spec import build_actor_obs_td


class PlaybackPrediction:
    def __init__(self, runner):
        self.runner = runner
        self.reset()

    def reset(self):
        self.steps = 0

    @torch.no_grad()
    def advance(self):
        r = self.runner
        rep = r.rep
        if self.steps == 0 or (r.reset_rows is not None and r.reset_rows[0]):
            state = r.bridge.state()[:1]
            self.aux = rep.extract_aux_state(state)
            self.state = rep.represent(state)
            self.history = None
            self.action = torch.zeros_like(r.bridge.env.action_manager.action[:1])
            self.context = r.bridge.context()[:1]
            self.steps = 0
        goals = {
            spec.name: spec.extract_training_goal(r.bridge.motion, r.reference_frames()[:1], self.aux)
            for spec in r.specs
        }
        if r.play_velocity is not None:
            goals["velocity"] = r.play_velocity.reshape(1, -1)
        sensor = rep.compute_actor_sensor(
            self.state, self.context, self.aux, rep.joint_state(self.state), self.action
        )
        if self.history is None:
            self.history = sensor[:, None].expand(-1, r.cfg.history_length, -1)
        else:
            self.history = torch.cat((self.history[:, 1:], sensor[:, None]), dim=1)
        self.action = r.actor(build_actor_obs_td(goals, self.history))
        self.aux, self.state = r.dynamics(self.state, self.action, self.aux, self.context)
        self.steps = (self.steps + 1) % r.cfg.policy_horizon
        state = rep.reconstruct_full_state(self.aux, self.state)
        if r.cfg.representation == "maxcoord":
            width = 3 * len(r.bridge.fk.body_names)
            return state[0, :width].reshape(-1, 3)
        return r.bridge.fk(
            state[:, :3], state[:, 3:7], state[:, 7:30], state[:, 30:33], state[:, 33:36], state[:, 36:]
        )[0][0]


def create_play_viewer(runner, motions):
    from cyclo_lab.manager_based.mimic.play_viewer import TrackingPlayViewer
    from .export import velocity_ranges

    def velocity_controls(ui):
        ranges = velocity_ranges()
        for axis, (label, bounds) in enumerate((
            ("Forward (m/s)", ranges["lin_vel_x"]),
            ("Lateral (m/s)", ranges["lin_vel_y"]),
            ("Yaw (rad/s)", ranges["ang_vel_z"]),
        )):
            with ui.HStack():
                ui.Label(label)
                slider = ui.FloatSlider(min=bounds[0], max=bounds[1])
                slider.model.set_value(float(runner.play_velocity[axis]))

                def set_velocity(model, axis=axis):
                    runner.play_velocity[axis] = model.as_float

                slider.model.add_value_changed_fn(set_velocity)

    return TrackingPlayViewer(
        runner, motions, predictor=PlaybackPrediction(runner),
        extra_controls=velocity_controls if runner.play_velocity is not None else None,
    )
