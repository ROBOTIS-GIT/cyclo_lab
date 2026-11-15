# Validate native SuperMimic

Use this guide to check motion input, the native learning path, local checkpoints
and BeyondMimic compatibility. Run commands from the repository root in the
[configured Isaac environment](supermimic.md#setup-and-selection).

The reference results below were recorded on 2026-11-15 with shared core
`10dd5dd3f02601caf1991790952a25db63a6fb99`. They are bounded correctness checks,
not training-quality or convergence measurements. Raw simulator logs are not
tracked; capture output locally when repeating these checks.

## Runtime

The reference runtime uses:

- Isaac Sim 6.2.0 and the base branch's pinned Isaac Lab at
  `e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4`. The base migration is labeled
  Isaac Lab 3.0; this exact source and installed distribution report version
  `28.0.0`. The SHA is the reproducible runtime identity.
- Python 3.12.14, Torch 2.11.0, RSL-RL 5.5.1, TensorDict 0.14.2 and
  toml 0.10.2 for Cyclo asset configuration imports; Newton package 1.6.1.
- K1 assets at `611a070fad83d2a9ef9c1bc82213988450cf471c`.
- Shared SuperMimic fork at `10dd5dd3f02601caf1991790952a25db63a6fb99`,
  derived from official `abc124c1bd5976341034668af3d12c087f2f1f55`.

The [validation Dockerfile](../docker/Dockerfile.supermimic-validation) provides a
reproducible environment recipe using the pinned submodules and dependency lock:

```bash
docker build -f docker/Dockerfile.supermimic-validation -t cyclo-lab-validation .
```

Its entrypoint selects the environment's Python interpreter. For example, run
the CPU suite without GPU access or networking:

```bash
docker run --rm --network none \
  --mount "type=bind,src=$(pwd),dst=/workspace/cyclo_lab" \
  cyclo-lab-validation -m pytest tests/supermimic -q -p no:cacheprovider
```

For the simulator commands below, replace `python` with this Docker prefix when
using the validation image:

```bash
docker run --rm --gpus all --network host --ipc host \
  -e ACCEPT_EULA=Y -e OMNI_KIT_ALLOW_ROOT=1 \
  --mount "type=bind,src=$(pwd),dst=/workspace/cyclo_lab" \
  cyclo-lab-validation
```

Check GPU/process occupancy first and serialize simulator runs. For a linked
Git worktree, also mount its common Git metadata read-only at the path referenced
by `.git`; the existing PPO logger reads that metadata. These checks use local
checkpoints and do not require hardware execution or W&B operations. Set
`WANDB_MODE=disabled WANDB_DISABLED=true` for validation and select
`--logger tensorboard` explicitly for the legacy PPO checks.

The launcher inherits **Newton/MJWarp** and supports explicit
`physics=isaacsim_physx`. It requests Kit for importer/schema initialization while
preserving physics selection. All learning checks use `--eager`, retaining the
original learner equations and optimizers without Torch compilation/CUDA graphs.
Newton still compiles its own Warp kernels; allow first compilation to complete.
Do not interpret a startup/importer failure as a solver acceptance result.

For constrained machines, use an already available image with `--pull=never`,
read-only root/source, and isolated size-limited tmpfs mounts for `/tmp`,
`/isaac-sim/kit/{cache,data,logs}`, `/root/.cache`, `/root/.nv`,
`/root/.nvidia-omniverse`, `/isaac-sim/.nvidia-omniverse`, `/root/.local/share/ov`,
`/root/Documents`, `/var/cache/hub` and `/run`. Route Warp/Torch/Triton caches into
`/tmp`; bind a task-owned output directory separately. Do not overwrite shared
caches. Disable Docker log persistence/healthchecks and capture stdout locally.
Read-only startup may expose additional write requirements; inspect them rather
than silently making the root writable.

## Unit and contract suite

```bash
python -m pytest tests/supermimic -q -p no:cacheprovider
```

Reference result: **75 passed, 67 upstream warnings, no skips, exit 0**.
The plain interpreter in the older prebuilt image initially lacked `toml` on its
path. The final CPU run used the real Kit-bundled toml package, appended after
installed dependencies; no package was installed during validation. The recipe
above now pins toml explicitly, so new builds can use the ordinary command. Coverage
includes numerical WM/loss tests, five official learner updates in a labeled
unit-double collector for both representations, WM frozen-state assertions,
full optimizer/model reload,
rotating-pelvis COM conversion, short-clip wraps with both env done flags false,
name/quaternion permutation invariance, complex/overflow rejection, resolved
Isaac configuration guards, K1 one-hinge IK/body/joint permutation tests,
MaxCoord action gradients and checkpoint incompatibility. The shared standalone
`third_party/supermimic/tests/test_tensor_core.py` also passed five tests, including
explicit updates of Muon-managed WM and actor matrices.

Unit doubles test collector contracts; they are not physical simulator evidence.
Real simulator collection and learning checks are separate below.

## Verify original motion files

The suite checks every frame/body against the historical K1 COM profile from
URDF revision `b0ef567ca77db0f37089ed3b23e492baba60d6be`, including fixed-head mass
aggregation into the torso. This is a source-defined profile, not fitted motion
data or an inferred producer identity. Both bundled clips retain 300 frames at
50 Hz, and their source bytes are unchanged.

```bash
sha256sum source/cyclo_lab/data/motions/K1_rev1/dance1/dance1.npz \
  source/cyclo_lab/data/motions/K1_rev1/dance2/dance2.npz
```

Expected SHA-256:

- Dance1: `9e46d14b500221c43fa3c7bb438de991a1af859f904cc16ca10d01047f8883f4`
- Dance2: `682b4a950a247d6a822e5c41b832100ca28b7a57ddd2f29cc323f384f5914f50`

Absolute acceptance limits are 2e-5 m, 2e-5 quaternion components, 5e-5 m/s and
5e-5 rad/s. Unknown profiles fail. The current-URDF COM profile differs from these
archives by up to 0.041 m/s and is not accepted by widening tolerances. Missing
producer/backend labels remain unspecified; these results do not establish
arbitrary cross-simulator archive compatibility.

## Exercise multi-step learning and final-state saving

Use an unused output directory so existing checkpoints are preserved:

```bash
python scripts/reinforcement_learning/train.py --algorithm supermimic \
  --representation gencoord --num-envs 2 --iterations 5 --collection-steps 16 --batch-size 4 \
  --horizon 4 --policy-horizon 4 --hidden 32 32 --latent 8 \
  --eager --verify --headless --output logs/supermimic/verified-five-updates
```

Repeat with `--representation maxcoord`, then repeat both with
`physics=isaacsim_physx`, choosing a separate unused output directory each time.
Five updates use 80 vector steps from two environments (160 transitions). WM
horizon 4 scores three successive autoregressive predictions. Each update checks
finite metrics, changed parameters and exact equality of all WM tensors across
the actor update. `--verify` compares every saved model, normalizer and optimizer
tensor and full reload state. It also checks a rotated, moving pelvis reset,
nonzero COM offset, live link FK and MaxCoord IK.

Inspect `ISAAC_RUNTIME_PHYSICS` for the instantiated manager and solver, not just
configuration labels. Expected managers are `NewtonMJWarpManager` with
`SolverMuJoCo`, or `PhysxManager` with its live scene's TGS/PGS solver attribute.

| Backend / representation | Five updates | Fresh continuation | Fresh play |
| --- | --- | --- | --- |
| Default Newton/MJWarp GenCoord | Pass: 160 transitions, 5 updates | Pass: update 6 | Pass: 8 steps |
| Default Newton/MJWarp MaxCoord | Pass: 160 transitions, 5 updates | Pass: update 6 | Pass: 8 steps |
| Explicit Isaac Sim PhysX GenCoord | Pass: 160 transitions, 5 updates | Pass: update 6 | Pass: 8 steps |
| Explicit Isaac Sim PhysX MaxCoord | Pass: 160 transitions, 5 updates | Pass: update 6 | Pass: 8 steps |

All 12 matrix processes exited 0. Newton commands had **no backend override**;
PhysX commands appended `physics=isaacsim_physx`. Each continuation collected
another 32 transitions (96 cumulative vector steps), restored all saved optimizer
and model tensors before collecting, and saved/reloaded update 6 exactly. All
20 initial updates and four continuation updates changed both models and left
every WM tensor unchanged during actor optimization. Restricted CPU inspection
of saved updates 1 and 5 additionally confirmed changes in all three WM MLP
weight matrices and all six actor encoder/decoder weight matrices in every cell.
Observed final initial-run
WM state losses were 0.883219 (Newton GenCoord), 0.589963 (Newton MaxCoord),
0.473580 (PhysX GenCoord), and 0.486637 (PhysX MaxCoord).
These stochastic values are observations, not convergence thresholds.

The actual runtime objects were:

- `isaaclab_newton.physics.mjwarp_manager.NewtonMJWarpManager`, using
  `newton._src.solvers.mujoco.solver_mujoco.SolverMuJoCo`;
- `isaaclab_physx.physics.physx_manager.PhysxManager`, with live scene solver
  attribute `TGS`.

Both used simulation dt 0.005 s and policy decimation 4; Newton used one solver
substep. The inherited collision/actuator presets were retained. Rotated-pelvis
reset checks used a measured 0.0504002 m COM offset and passed link/COM velocity,
FK and (for MaxCoord) IK comparisons at 2e-5 absolute tolerance. Moving FK/IK
checks also passed after every collection. The real final-frame command check
observed frame 299 resampling both environments while `terminated` and
`truncated` were false; the separate collector unit regression verifies that
this counter change marks replay truncation.

## Check fresh-process continuation and inference

```bash
python scripts/reinforcement_learning/train.py --algorithm supermimic \
  --checkpoint logs/supermimic/verified-five-updates/model_5.pt \
  --num-envs 2 --iterations 1 --eager --verify --headless \
  --output logs/supermimic/verified-continuation
python scripts/reinforcement_learning/play.py --algorithm supermimic \
  --checkpoint logs/supermimic/verified-continuation/model_6.pt \
  --num-envs 2 --play-steps 8 --eager --verify --headless
```

Repeat with each learned model and its saved backend selection. Continuation
must verify exact model/optimizer restoration before collection, refill fresh
replay, perform update 6 and verify the new checkpoint. Inference must restore
model tensors in a separate process and execute eight finite real steps.

Look for `ISAAC_FRESH_RESTORE_PASS`, `ISAAC_CHECKPOINT_RELOAD_PASS` during training,
and `ISAAC_PLAY_PASS 8` during playback. This is learned-state continuation, not
exact simulator/replay/RNG resume. Playback retains the physics timestep and
decimation; initial frame selection, environment count and rendering may differ.

## Check incompatible checkpoint rejection

The following commands must fail before Kit startup:

```bash
python scripts/reinforcement_learning/play.py --algorithm supermimic \
  --checkpoint logs/supermimic/verified-five-updates/model_5.pt \
  --eager --headless env.decimation=8 env.sim.dt=0.0025
python scripts/reinforcement_learning/play.py --algorithm supermimic \
  --checkpoint logs/supermimic/verified-five-updates/model_5.pt \
  --motion-file source/cyclo_lab/data/motions/K1_rev1/dance2/dance2.npz \
  --eager --headless
python scripts/reinforcement_learning/play.py --algorithm supermimic \
  --checkpoint logs/supermimic/verified-five-updates/model_5.pt \
  --eager --headless physics=isaacsim_physx
python scripts/reinforcement_learning/play.py --algorithm supermimic \
  --checkpoint logs/supermimic/verified-five-updates/model_5.pt \
  --eager --headless --representation maxcoord
```

Fresh v3 checks returned the expected **exit 1 before Kit startup** for changed
backend, representation, motion digest, and separate `decimation`/`sim_dt`,
despite an unchanged 50 Hz policy rate in the timing case. Unit tests also reject changed
task, actuation/model, observation, action, reward, termination and physics
contracts, old v1 files and unsupported pickle objects. Also reject a changed
`--representation` or backend. Matching v2 PhysX/GenCoord contracts remain
compatible; they do not become Newton/MaxCoord models. A real local checkpoint
produced by the earlier published v2 code also passed
fresh PhysX/GenCoord continuation through update 6, exact current-format
save/reload and a separate eight-step playback process. See the
[checkpoint contract](supermimic.md#local-checkpoints-play-and-continuation) for
supported relocation and continuation behavior.

## Check BeyondMimic compatibility

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task Cyclo-Mimic-K1-Rev1-Dance1 \
  --num_envs 2 --max_iterations 1 --logger tensorboard --headless physics=isaacsim_physx
```

Run once without the physics override to test the inherited Newton default,
and once with explicit PhysX. Each bounded run must finish one PPO iteration
(48 transitions for two environments). Keep checkpoints in separate output
folders. The original asset, PPO task/agent/MDP/loader/direct-launcher files must
remain unchanged from base `2980071bd35d970f38799638fd375da37356a299`.
Alias kwargs equality is covered by the CPU suite.

Fresh default-Newton and explicit-PhysX PPO validation each passed one iteration
and 48 transitions (exit 0). Reference runs invoked the unchanged direct
RSL-RL CLI through an observational harness that asserted the resolved logger
was TensorBoard and reported the actual manager/solver after `gym.make`.
It did not replace the environment, observations, rewards, runner or updates.
W&B was disabled in the validation process. Shared-selector/task aliases are
covered separately by the CPU suite.

## Static checks and supported limits

The reference revision passed Ruff `E4,E7,E9,F`, formatting, AST compilation,
source/license byte comparisons and `git diff --check`. The official models,
replay buffer and all learner files remained byte-identical to the pinned
SuperMimic source; the Apache LICENSE and original acknowledgements are retained.

Supported scope is GenCoord and MaxCoord, privileged rep-state observations,
one motion encoder/clip, history 1 and fixed dynamics with zero context. Proprio/
history deployment, motion-to-velocity latent BC, multi-task/multi-clip training,
official return sampling and robot export are not integrated. Additional Newton
solver substeps are rejected. Compiled Torch/CUDA-graph learning, convergence,
all repository tasks and PPO play/ONNX export are outside these bounded checks.
Preserving the official integration equations does not establish numerical
simulator parity. Existing BeyondMimic deployment remains separate.
