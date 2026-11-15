# Choose BeyondMimic or SuperMimic in Isaac Lab

Both choices use this checkout's pinned Isaac Lab 3.0 / Isaac Sim 6.2 and Python
3.12 environment. The algorithm uses Isaac-collected states directly. Do not install the original
upstream mjlab project or a separate Python 3.13 environment in Cyclo Lab.

## Setup and selection

Build/start/enter the existing [Docker environment](../docker/README.md). The RSL-RL
extra already supplies the algorithm's Torch, RSL normalization and TensorDict
requirements. If that extra was omitted, install only the matching versions:

```bash
uv pip install --python /opt/isaaclab-venv/bin/python -r requirements/supermimic.txt
```

Existing commands and `Cyclo-Mimic-K1-Rev1-Dance1` / `Dance2` PPO registrations are
unchanged. `Cyclo-BeyondMimic-*` aliases use exactly their existing configurations.
The new `Cyclo-SuperMimic-*` registrations expose a distinct learner config key,
so accidentally selecting a SuperMimic task in the PPO launcher fails instead of
training PPO under the wrong name.

```bash
# Existing direct PPO command remains valid.
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task Cyclo-Mimic-K1-Rev1-Dance1 --num_envs 4096 --headless
# Explicit algorithm choice, same runtime.
python scripts/reinforcement_learning/mimic.py --method beyondmimic train \
  --task Cyclo-BeyondMimic-K1-Rev1-Dance1 --num_envs 4096 --headless
python scripts/reinforcement_learning/mimic.py --method supermimic train \
  --task Cyclo-SuperMimic-K1-Rev1-Dance1 --num-envs 32 --iterations 1000 --headless
```

The SuperMimic path selects Isaac Sim PhysX by default. The original PPO task's
physics presets remain unchanged. Existing underscore `--num_envs` and
`--max_iterations` spellings are also accepted by the new learner.

## Shared input, separate checkpoints

Use the existing Soma-retargeter/CSV-to-NPZ workflow. Both methods consume the
same Cyclo NPZ directly; no second conversion, MJCF joint order or duplicated
motion dataset is needed. For another named 23-joint K1 clip:

```bash
python scripts/reinforcement_learning/mimic.py --method supermimic train \
  --task Cyclo-SuperMimic-K1-Rev1-Dance1 --motion-file /absolute/path/my_motion.npz \
  --num-envs 32 --iterations 1000 --headless
```

The selector's common `--motion-file` validates input for either method. SuperMimic
also validates its default/resolved file before launch and checks names again
against the live articulation before collection. The existing motion loader then
reorders named columns to the actual Isaac articulation order and converts
quaternions to Isaac 3.0 xyzw. The bridge converts
root quaternions to the original algorithm's wxyz once. Explicit unique `joint_names`, `body_names` and consistent `quaternion_order`
(or `quat_order`) metadata are required. All 23 joint names must match; body names
must cover the live articulation (extra named merged bodies may be omitted by the
shared loader). Arrays must have matching frame counts and widths, finite SI
world-frame values and unit quaternions. Units and frame semantics remain a
producer contract; numerical validation does not guess mislabeled units. Missing required body names, unmatched/duplicate joint names, duplicate body names, conflicting or
missing conventions, nonunit/zero quaternions, malformed arrays and non-50-Hz
input fail clearly. No resampling or rewriting occurs in either selector path.
The unchanged direct legacy PPO launcher retains its historical unnamed-K1
fallback for compatibility; use the selector and `--motion-file` for strict input.
Resample explicitly with the existing motion tools before selecting a learner.

| Method | Output | Playback |
| --- | --- | --- |
| BeyondMimic PPO | `logs/rsl_rl/<experiment>/<run>/model_*.pt`, params and exported policy | existing RSL-RL play launcher |
| SuperMimic | `logs/supermimic/<run>/model_<iteration>.pt` | new local checkpoint play path |

```bash
python scripts/reinforcement_learning/mimic.py --method supermimic play \
  --checkpoint logs/supermimic/<run>/model_1.pt --num-envs 1 --play-steps 1000 --headless
# Resume training, refilling replay with fresh Isaac transitions.
python scripts/reinforcement_learning/mimic.py --method supermimic train \
  --checkpoint logs/supermimic/<run>/model_1.pt --iterations 100 --headless
```

SuperMimic checkpoints contain dynamics, actor, critic, normalizers, optimizer
states, learner config, joint order, action transform, timestep, iteration and
motion path/digest and task. Play/resume restores the saved task and motion path
unless explicitly overridden; provide `--motion-file` if the saved path is no
longer available. They are not interchangeable with PPO or original mjlab bundles.
Replay and simulator state are recollected on resume; exact mid-episode replay
continuation is not promised. Local playback needs no W&B account.

## Bounded verification

Inspect GPU occupancy before running. This checks live Isaac FK parity, collects
a small sequence, updates the real world model and imagined-rollout policy,
asserts both parameter sets changed and metrics are finite, then saves/reloads
a checkpoint:

```bash
python scripts/reinforcement_learning/mimic.py --method supermimic train \
  --num-envs 2 --iterations 1 --collection-steps 8 --batch-size 2 \
  --horizon 2 --policy-horizon 2 --hidden 32 32 --latent 8 \
  --eager --verify --headless --output logs/supermimic/isaac-smoke
python -m pytest tests/supermimic -q
```

`--eager` disables compilation/graph capture only, retaining the same losses,
rollout gradients and optimizers. Without it, first compilation may take several
minutes. There is no arbitrary compile timeout in the launcher. The new process
uses spawned compiler workers, and eager mode requests one compile thread before
Torch initialization so no forked worker inherits a live Kit/CUDA context.

## Algorithm and supported scope

`cyclo_lab/learning/supermimic` owns the Isaac bridge, URDF FK and collection loop.
The vendored original dynamics, replay buffer, WM learner, differentiable WM env
and BPTT/actor-critic policy learners are called directly. They are not a PPO
substitute. The new task config inherits the same K1 asset and motion command,
then selects the original SuperMimic root anchor/reward/termination contract.
Only this new config disables material/COM/encoder randomization and pushes; its
constant context is explicitly zero. Existing PPO config remains unchanged.

The first native port supports generalized-coordinate K1 motion tracking,
rep-state actor input and one motion encoder (the original default active task).
The actor therefore uses privileged simulator state. History/proprio deployment,
multiple encoder tasks, MaxCoord K1, multi-clip return sampling, full convergence
and robot policy quality are not validated features of this port.

Existing BeyondMimic deployment uses `exported/policy.onnx`, matched
`params/sim2real.yaml` and motion CSV. The new SuperMimic checkpoint has no
validated Sim2Real export/controller contract. Do not deploy it through the
existing robot workflow. See [observed validation](supermimic-validation.md)
for actual execution results and remaining limitations.
