# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--mouse_drag",
    action="store_true",
    default=False,
    help="Apply Newton viewer right-click dragging forces during playback.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
cli_args.validate_task_exists(args_cli.task)
# Isaac Lab 3 defaults to headless when no visualizer is selected. Newton
# playback uses its native visualizer by default, while callers can still force
# headless with ``--viz none``.
if (
    args_cli.visualizer is None
    and not getattr(args_cli, "visualizer_explicit", False)
    and not args_cli.headless
):
    args_cli.visualizer = ["newton"]
    args_cli.visualizer_explicit = True
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# Launch Kit before importing task configurations. K1's URDF converter needs
# Kit's USD modules, while SimulationCfg.physics still selects Newton dynamics.
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for installed RSL-RL version."""

import importlib.metadata as metadata

from packaging import version

installed_version = metadata.version("rsl-rl-lib")

"""Rest everything follows."""

import os
import time

import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)
try:
    from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg
except ImportError:
    def handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version):
        return agent_cfg

try:
    from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
except ImportError:
    from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import cyclo_lab  # noqa: F401
from cyclo_lab.utils.onnx_action_clipping import clip_onnx_actions

# PLACEHOLDER: Extension template (do not remove this comment)


def _embed_onnx_external_data(model_path: str):
    """Rewrite an ONNX export as one self-contained file."""
    import onnx

    model_metadata = onnx.load(model_path, load_external_data=False)
    external_locations = {
        entry.value
        for tensor in model_metadata.graph.initializer
        for entry in tensor.external_data
        if entry.key == "location"
    }
    if not external_locations:
        return

    model = onnx.load(model_path, load_external_data=True)
    temporary_path = f"{model_path}.embedded.tmp"
    try:
        onnx.save_model(model, temporary_path, save_as_external_data=False)
        embedded_model = onnx.load(temporary_path, load_external_data=False)
        if any(tensor.external_data for tensor in embedded_model.graph.initializer):
            raise RuntimeError(f"Failed to embed all ONNX external data into: {model_path}")
        onnx.checker.check_model(embedded_model, full_check=True)
        os.replace(temporary_path, model_path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)

    model_dir = os.path.realpath(os.path.dirname(model_path))
    for location in external_locations:
        external_path = os.path.realpath(os.path.join(model_dir, location))
        if os.path.commonpath((model_dir, external_path)) == model_dir and os.path.isfile(external_path):
            os.remove(external_path)

    print(f"[INFO]: Embedded ONNX weights into a single file: {model_path}", flush=True)


def _load_checkpoint(runner: OnPolicyRunner | DistillationRunner, checkpoint_path: str):
    """Load current RSL-RL checkpoints and legacy combined actor-critic checkpoints."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model_state_dict" not in checkpoint:
        runner.load(checkpoint_path)
        return

    if not isinstance(runner, OnPolicyRunner):
        raise ValueError("Legacy model_state_dict checkpoints are only supported for OnPolicyRunner.")

    legacy_state = checkpoint["model_state_dict"]
    actor_state = {
        f"mlp.{name.removeprefix('actor.')}": value
        for name, value in legacy_state.items()
        if name.startswith("actor.")
    }
    critic_state = {
        f"mlp.{name.removeprefix('critic.')}": value
        for name, value in legacy_state.items()
        if name.startswith("critic.")
    }

    actor_keys = runner.alg.actor.state_dict().keys()
    if "std" in legacy_state and "distribution.std_param" in actor_keys:
        actor_state["distribution.std_param"] = legacy_state["std"]

    runner.alg.actor.load_state_dict(actor_state, strict=True)
    runner.alg.critic.load_state_dict(critic_state, strict=True)
    runner.current_learning_iteration = checkpoint.get("iter", 0)
    print("[INFO]: Loaded legacy combined actor-critic checkpoint.", flush=True)


def _resolve_newton_mouse_drag(env):
    """Return the active Newton viewer and state accessor used for mouse picking."""
    try:
        from isaaclab_newton.physics import NewtonManager
    except ImportError as exc:
        raise RuntimeError("--mouse_drag requires the Newton backend.") from exc

    for visualizer in env.unwrapped.sim.visualizers:
        viewer = getattr(visualizer, "_viewer", None)
        if viewer is not None and hasattr(viewer, "apply_forces"):
            viewer.picking_enabled = True
            return viewer, NewtonManager.get_state_0

    raise RuntimeError("--mouse_drag requires an active Newton visualizer. Run play with '--viz newton'.")


def _run(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # handle deprecated configurations
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    mouse_drag_viewer = None
    newton_state = None
    if args_cli.mouse_drag:
        mouse_drag_viewer, newton_state = _resolve_newton_mouse_drag(env)
        print(
            "[INFO]: Newton mouse dragging enabled: right-click a robot body and drag it; release to stop.",
            flush=True,
        )

    print(f"[INFO]: Loading model checkpoint from: {resume_path}", flush=True)
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    _load_checkpoint(runner, resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # export the trained policy to JIT and ONNX formats
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")

    if (
        version.parse(installed_version) >= version.parse("4.0.0")
        and hasattr(runner, "export_policy_to_jit")
        and hasattr(runner, "export_policy_to_onnx")
    ):
        # use the new export functions for rsl-rl >= 4.0.0
        runner.export_policy_to_jit(path=export_model_dir, filename="policy.pt")
        runner.export_policy_to_onnx(path=export_model_dir, filename="policy.onnx")
    else:
        # extract the neural network for rsl-rl < 4.0.0
        if version.parse(installed_version) >= version.parse("2.3.0"):
            policy_nn = runner.alg.policy
        else:
            policy_nn = runner.alg.actor_critic

        # extract the normalizer
        if hasattr(policy_nn, "actor_obs_normalizer"):
            normalizer = policy_nn.actor_obs_normalizer
        elif hasattr(policy_nn, "student_obs_normalizer"):
            normalizer = policy_nn.student_obs_normalizer
        else:
            normalizer = None

        # export to JIT and ONNX
        export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
        export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")
    _embed_onnx_external_data(os.path.join(export_model_dir, "policy.onnx"))
    print(f"[INFO]: Exported policy to: {export_model_dir}", flush=True)

    # The policy exporters omit the environment wrapper's action clipping.
    clip_onnx_actions(os.path.join(export_model_dir, "policy.onnx"), agent_cfg.clip_actions)

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    timestep = 0
    # simulate environment
    while True:
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            if mouse_drag_viewer is not None:
                mouse_drag_viewer.apply_forces(newton_state())
            # env stepping
            obs, _, dones, _ = env.step(actions)
            # reset recurrent states for episodes that have terminated
            if version.parse(installed_version) >= version.parse("4.0.0"):
                policy.reset(dones)
            else:
                policy_nn.reset(dones)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play using the physics backend selected by the task config."""
    _run(env_cfg, agent_cfg)


if __name__ == "__main__":
    # run the main function
    try:
        main()
    finally:
        simulation_app.close()
