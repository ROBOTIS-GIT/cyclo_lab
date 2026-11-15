# Choose BeyondMimic or SuperMimic in Isaac Lab

Both choices run in this checkout's pinned Isaac Lab / Isaac Sim 6.2 / Python
3.12 environment (exact runtime versions are in the
[validation guide](supermimic-validation.md#runtime)). SuperMimic uses the official differentiable world model and
BPTT actor-critic learners with Isaac-collected transitions. No mjlab runtime or
separate Python environment is required.

## Setup and selection

Initialize the shared core with `git submodule update --init third_party/supermimic`.
It is the [ROBOTIS shared fork](https://github.com/ROBOTIS-GIT/supermimic/tree/10dd5dd3f02601caf1991790952a25db63a6fb99)
at the repository gitlink, based on official `abc124c1`. Read its
[portable-core API/provenance](https://github.com/ROBOTIS-GIT/supermimic/blob/10dd5dd3f02601caf1991790952a25db63a6fb99/PATCHES.md). Do not install
its optional research extra or use its standalone lock to replace Lab dependencies.

Use the existing [Docker environment](../docker/README.md). The RSL-RL extra
provides the matching Torch, normalization and TensorDict packages. If omitted,
install `requirements/supermimic.txt` into that same environment:

```bash
uv pip install --python /opt/isaaclab-venv/bin/python -r requirements/supermimic.txt
python scripts/reinforcement_learning/train.py --algorithm beyondmimic \
  --task Cyclo-BeyondMimic-K1-Rev1-Dance1 --num_envs 4096 --headless
python scripts/reinforcement_learning/train.py --algorithm supermimic \
  --task Cyclo-SuperMimic-K1-Rev1-Dance1 --num-envs 32 --iterations 1000 --eager --headless
```

`--algorithm` is optional: recognized `Cyclo-SuperMimic-*` tasks select SuperMimic;
`Cyclo-BeyondMimic-*` and historical `Cyclo-Mimic-*` select BeyondMimic. Without a
recognized task or explicit selector, the default is BeyondMimic. A conflicting
selector/task fails. Dance1 and Dance2 have registrations for both methods.
Other clips use `--motion-file` with one of these task configurations.

Historical commands remain valid:

```bash
python scripts/reinforcement_learning/mimic.py --method supermimic train --eager --headless
python scripts/reinforcement_learning/rsl_rl/train.py --task Cyclo-Mimic-K1-Rev1-Dance1 --headless
```

`mimic.py` also accepts `--algorithm`. Direct RSL-RL commands, existing PPO task
configs, observations and export code remain unchanged. The shared selector
forwards backend-specific options. SuperMimic accepts both `--num-envs` /
`--num_envs` and `--iterations` / `--max_iterations`.

SuperMimic inherits **Newton/MJWarp by default**, including the original K1
self-collision, actuator and solver presets. Select PhysX explicitly with
`physics=isaacsim_physx` (or `--physics isaacsim_physx`). Conflicting spellings fail.
The launcher starts Kit for the supported URDF importer/USD schemas; this does
not change the chosen physics engine. There is no silent PhysX fallback.
Additional Newton solver substeps are rejected: the supported preset uses
`num_substeps=1`, independently of policy decimation.

Select `--representation gencoord` (default) or `--representation maxcoord` with
any existing SuperMimic Dance1/Dance2 task. The task/algorithm aliases stay intact.
Both representations collect real Isaac state and call the same shared official
learners. MaxCoord tracks 14 K1 bodies, initializes analytic IK from native URDF
topology and integrates independent body states without FK projection. Native
body and action-joint mapping is name-based, including K1 one-hinge chains.

Material/friction, COM, home-offset and push randomization events must remain
disabled: the port's dynamics context is constant zero. Overrides that activate
these events fail. Reference-reset perturbations and collection exploration noise
remain enabled; they do not change hidden dynamics parameters.

## Original 50 Hz motion input

Both selectors accept `--motion-file /absolute/path/clip.npz`. Input files are
read-only; no implicit resampling, conversion or source rewrite occurs.
The common validator requires all 23 unique K1 joint names, unique body names,
explicit consistent `quaternion_order` / `quat_order` (`xyzw` or `wxyz`), matching
frame counts/widths, unit quaternions and real arrays finite **after float32
conversion**. Complex arrays and float32 overflow fail. Policy rate must be
exactly 50 Hz. The direct legacy PPO launcher retains its historical unnamed-file
fallback; the shared selector does not guess unnamed orders.

SuperMimic additionally requires the complete redundant named moving-body pose
and twist arrays. It verifies **every frame and body** against the supported K1
URDF FK and source-defined velocity profiles. Pose means link origin, coordinates
are world-frame SI with z up. A link-origin velocity profile or a documented COM
profile must explain the full archive; conflicting/unknown physical profiles fail.
Matching multiple profiles is allowed only when their derived root-link velocity
is identical within 1e-7 m/s. No producer identity is inferred from numerical agreement.

The bundled Dance1/Dance2 NPZ files pass the historical K1 COM profile from
[ai_sapiens b0ef567](https://github.com/ROBOTIS-GIT/ai_sapiens/blob/b0ef567ca77db0f37089ed3b23e492baba60d6be/ai_sapiens_description/urdf/k1_rev1/k1.urdf),
including the fixed head merged into the torso. The current URDF profile is also
supported. The extracted historical inertial values and exact source hashes live
in `learning/supermimic/k1_motion_profile.json`; joint transforms match the current
URDF. Producer/backend provenance remains `unspecified` when absent.

Root COM velocity is converted with
`v_link = v_COM - omega_world × (R_world_link * r_COM_link)`.
The SuperMimic-only command uses canonical link twists for reference rewards and
writes root **link** velocity during resets, including reset noise. Live state
collection and differentiable FK use the same link convention. Existing PPO
reference behavior is unchanged. All conversions occur in memory.

Optional `producer`/`motion_producer` and
`backend`/`physics_backend`/`simulator_backend`/`producer_backend` declarations must
agree within each group. `coordinate_frame`/`world_frame`/`velocity_frame`/
`body_velocity_frame` must be `world`; `up_axis` must be `z`, `units` must be `SI`.
`velocity_reference`/`velocity_point`/`body_velocity_point` must consistently be
`com` or `link_origin` and pass the numerical checks. Unsupported declarations
are not silently ignored as permission to reinterpret a trajectory.

## Local checkpoints, play and continuation

```bash
python scripts/reinforcement_learning/play.py --algorithm supermimic \
  --checkpoint logs/supermimic/<run>/model_5.pt --num-envs 1 --play-steps 1000 --eager --headless
python scripts/reinforcement_learning/train.py --algorithm supermimic \
  --checkpoint logs/supermimic/<run>/model_5.pt --iterations 100 --eager --headless
```

Native checkpoints use `cyclo-supermimic-isaac-v3`, restricted Torch loading and
atomic replacement. Every completed iteration saves the current dynamics, actor,
critic, normalizers, optimizers and iteration/collection counters. There is no
remote lookup, W&B account, upload or sync. v1 checkpoints are rejected because
they lack the physical/reference contract. Published v2 PhysX/GenCoord files
remain loadable when all existing physical/objective fields match; pass
`physics=isaacsim_physx` because the current default is Newton. v2 is never
silently reinterpreted as Newton or MaxCoord. New saves use v3.

The versioned declarative contract validates task, motion digest, URDF/mesh and
motion-profile hashes, exact physics timestep **and decimation**, backend class, representation, solver config,
robot/actuator settings, action transform/order, tracked bodies, privileged
observations, reward/termination/reset settings and zero context. Learner/model
configuration and representation are restored from the file. Physical/environment configuration must
match; repeat any supported nondefault environment overrides on play/continuation
or loading fails **before simulation**. Equal aggregate policy timestep alone is
insufficient. The motion path may be relocated with `--motion-file`, but its digest
must match. A new-clip warm start is not supported implicitly.

Playback changes initial frame selection only; it does not apply an inherited
play preset that could change physical settings. Environment count and rendering
may differ. Training continuation restores optimizers but starts fresh simulator,
replay and RNG state; it is **not exact trajectory resume**. Inference does not
restore optimizers. These files are neither PPO checkpoints nor MJLab inference
bundles. Lab uses `--checkpoint`; MJLab's separate native runtime uses
`--checkpoint-dir`. Common workflow terminology does not imply format interchange.

## Official algorithm and supported subset

Source: [HybridRobotics/supermimic abc124c1](https://github.com/HybridRobotics/supermimic/tree/abc124c1bd5976341034668af3d12c087f2f1f55).
The original dynamics, replay, supervised multi-step WM learner, imagined rollout,
survival masking, BPTT actor update and bootstrap/GAE critic fitting are retained.
The policy update freezes WM parameters while gradients propagate through the
model to actions. Collection snapshots pre-step state/frame/action; command
resamples are explicit replay boundaries, even without environment done flags.
Previous actions come from the real action manager: command-only resamples do not
pretend its action history was cleared. Real timeouts and joint-velocity/nonfinite
safety terminations also delimit replay. Imagined termination uses the official
soft anchor-height/orientation rule, and imagined reference time clamps at clip
ends. Actual environment rewards are not WM targets or actor-update inputs; the
actor uses the original imagined reward. Exact per-step equality of inherited
simulator reward diagnostics and imagined rewards is not claimed.

This port supports **GenCoord and MaxCoord K1, a single clip, one motion encoder, privileged
rep-state actor observations, history length 1 and fixed dynamics/zero context**.
It retains the Cyclo command's failure-adaptive start sampling, not the official
return sampler. Proprio/history actors, velocity-task encoders,
motion-to-velocity latent behavior cloning, multi-clip training and return sampling
are not integrated. There is no validated SuperMimic ONNX/robot-controller export.
Existing BeyondMimic `policy.onnx` / `sim2real.yaml` deployment remains separate.

The official constant-acceleration substep integration coefficient is preserved
using the actual Isaac decimation with Newton solver substeps fixed to one. This does not establish numerical PhysX/MuJoCo
parity or validate long-training quality. `--eager` preserves the original losses
and optimizers while disabling compilation/CUDA graphs. Compiled execution is
not covered by the fresh validation. See [validation evidence](supermimic-validation.md)
for exact commands, results and limits.
