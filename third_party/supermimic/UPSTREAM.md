# SuperMimic source provenance and Isaac port

Source: https://github.com/HybridRobotics/supermimic
Revision: abc124c1bd5976341034668af3d12c087f2f1f55.
Copyright 2026 The SuperMimic Authors. Apache-2.0; complete LICENSE retained.
The original README is retained as upstream reference, not Cyclo setup instructions.
No upstream NOTICE was present. Original acknowledgements: ControlVAE (BSD-3-Clause),
robotic_world_model (Apache-2.0), DreamerV3 (MIT).

This snapshot contains portable algorithm code only. No mjlab runtime, registry, XML robot, bootstrap,
uv project/lock, playback viewer or MuJoCo simulator adapter is delivered here.

Retained verbatim: models.py, replay_buffer.py, learners/*.py,
envs/motion_tracking_env.py and envs/multi_task_world_model_env.py.
These implement dynamics, replay sampling, autoregressive world-model loss,
imagined differentiable rollouts, survival masks, BPTT actor updates, critic
bootstrap/GAE and Muon/Adam optimization. The Isaac runner calls them directly.

Modified by ROBOTIS: WorldModelEnv.__init__ reads an explicit native environment
contract instead of a simulator-specific manager/registry. Its rollout/reset, reward
and termination computations are unchanged. representations.py, tasks/task_spec.py and utils math users
import the local simulator-independent wxyz math module; environment type hints
are generic. tasks/__init__.py no longer registers mjlab tasks. Modified files
carry prominent notices. No representation, reward or learner equations changed.
math.py is a new portable Torch implementation with numerical/gradient tests.

New Isaac-specific work lives under cyclo_lab/learning/supermimic, and task variants
under manager_based/mimic/config/supermimic_k1_rev1. Real Isaac states (xyzw) are
explicitly converted to the original algorithm's wxyz convention. Differentiable
URDF FK uses live articulation joint ordering; runtime verification compares its
link poses/twists with Isaac. See docs/supermimic.md for commands and support limits.

The corrected native adapter validates redundant motion poses/twists against K1
source profiles, converts reference COM velocity to link-origin velocity, and
marks command resamples in replay. These changes live outside this vendored core.
Only GenCoord, rep-state observations, history 1, one motion encoder and constant
zero dynamics context are wired into Isaac. Official MaxCoord, multi-task/velocity
encoders, latent BC, proprio/history modes and return sampling are not claimed.
Preserving the original substep integration assumption does not establish
numerical simulator parity. See the native checkpoint v2 contract and fresh
validation report for exact supported runtime behavior.
