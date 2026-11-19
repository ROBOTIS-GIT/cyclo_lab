"""Deployment wrapper from Cyclo MJLab BlinkMimic."""

from torch import nn


DYNAMIC_BATCH = True


def commands(encoder):
    return {}


def dependency_notice(vendor):
    return vendor / "THIRD_PARTY_NOTICES.md", "BlinkMimic-THIRD_PARTY_NOTICES.md"


class FlatPolicy(nn.Module):
    """
    Shape adapter only: temporal raw history followed by an external raw goal.
    """

    def __init__(self, policy):
        super().__init__()
        self.policy = policy
        self.hist = policy.hist
        self.obs_dim = policy.obs_norm.mean.numel()
        self.history_size = self.hist * self.obs_dim
        self.input_size = self.history_size + policy.goal_norm.mean.numel()

    def forward(self, observations):
        history = observations[:, : self.history_size].reshape(-1, self.hist, self.obs_dim)
        return self.policy(history, observations[:, self.history_size :])


def actor_contract(runner, encoder):
    n = runner.bridge.num_actions
    cfg = runner.cfg
    from .observation_spec import observation_config

    return {
        "blinkmimic": dict(
            schema="cyclo.blinkmimic.csv.v1",
            **observation_config(cfg.history_length, cfg.goal_frames, n),
            input_name="observations",
            normalization_in_onnx=True,
            reference_skip_in_onnx=cfg.reference_skip,
            imu_frame="pelvis",
            motion_fps=1 / runner.bridge.step_dt,
            reference_velocity="central_difference",
        )
    }



def actor_module(runner, encoder):
    return FlatPolicy(runner.average.module), "observations"
