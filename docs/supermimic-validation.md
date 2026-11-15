# Validate native SuperMimic

Use this guide to check motion input, the native learning path, local checkpoints
and BeyondMimic compatibility. Run commands from the repository root in the
[configured Isaac environment](supermimic.md#setup-and-selection).

The reference results below were recorded on 2026-11-15 at code revision
`fb5c31b43249276ad4a29f694be3c7ab1f8e0c5e`. They are bounded correctness checks,
not training-quality or convergence measurements. Raw simulator logs are not
tracked; capture output locally when repeating these checks.

## Runtime

The reference runtime uses:

- Isaac Sim 6.2.0 and Isaac Lab 3.0.0 at
  `e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4`.
- Python 3.12.14, Torch 2.11.0, RSL-RL 5.5.1 and TensorDict 0.14.2.
- K1 assets at `611a070fad83d2a9ef9c1bc82213988450cf471c`.
- Official SuperMimic at `abc124c1bd5976341034668af3d12c087f2f1f55`.

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
checkpoints and do not require hardware execution or W&B operations.

All recorded simulator checks used **PhysX** and `--eager`, which retains the
original learner equations and optimizers without compilation/CUDA graphs.
The native launcher selects PhysX and rejects unsupported Newton overrides.

## Unit and contract suite

```bash
python -m pytest tests/supermimic -q -p no:cacheprovider
```

Reference result: **69 passed, 70 upstream warnings, no skips, exit 0**. Coverage
includes numerical WM/loss tests, five official learner updates in a labeled
unit-double collector, WM frozen-state assertions, full optimizer/model reload,
rotating-pelvis COM conversion, short-clip wraps with both env done flags false,
name/quaternion permutation invariance, complex/overflow rejection, resolved
Isaac configuration guards and checkpoint incompatibility.

Unit doubles test collector contracts; they are not physical simulator evidence.
The real PhysX collection and learning check is described separately below.

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

| Clip | Maximum position error | Maximum linear-velocity error | Maximum angular-velocity error |
| --- | --- | --- | --- |
| Dance1 | 5.91e-7 m | 3.10e-6 m/s | 1.36e-5 rad/s |
| Dance2 | 6.11e-7 m | 2.18e-6 m/s | 7.57e-6 rad/s |

Absolute acceptance limits are 2e-5 m, 2e-5 quaternion components, 5e-5 m/s and
5e-5 rad/s. Unknown profiles fail. The current-URDF COM profile differs from these
archives by up to 0.041 m/s and is not accepted by widening tolerances. Missing
producer/backend labels remain unspecified; these results do not establish
arbitrary cross-simulator archive compatibility.

## Exercise multi-step learning and final-state saving

Use an unused output directory so existing checkpoints are preserved:

```bash
python scripts/reinforcement_learning/train.py --algorithm supermimic \
  --num-envs 2 --iterations 5 --collection-steps 16 --batch-size 4 \
  --horizon 4 --policy-horizon 4 --hidden 32 32 --latent 8 \
  --eager --verify --headless --output logs/supermimic/verified-five-updates
```

Reference result: **PASS, exit 0**. Five official WM and BPTT actor-critic updates
used 80 vector steps from two real PhysX environments (160 transitions). WM
horizon 4 scores three successive autoregressive predictions. Each update checks
finite metrics, changes in both parameter sets, and exact equality of all WM
tensors across the actor update. `--verify` compares every final saved model,
normalizer and optimizer tensor, then compares full state after reload.

The final artifact is `logs/supermimic/verified-five-updates/model_5.pt`.
Reference WM state losses were 0.749698, 0.570511, 0.310953, 0.476564 and 0.526199;
these stochastic small-batch values are observations, not acceptance thresholds.

## Check fresh-process continuation and inference

```bash
python scripts/reinforcement_learning/train.py \
  --task Cyclo-SuperMimic-K1-Rev1-Dance1 \
  --checkpoint logs/supermimic/verified-five-updates/model_5.pt \
  --num-envs 2 --iterations 1 --eager --verify --headless \
  --output logs/supermimic/verified-continuation
python scripts/reinforcement_learning/play.py --algorithm supermimic \
  --checkpoint logs/supermimic/verified-continuation/model_6.pt \
  --num-envs 1 --play-steps 8 --eager --verify --headless
```

Reference result: **both passed, exit 0**. Continuation verified exact
model/optimizer restoration before collection, refilled replay in a fresh
simulator, performed update 6 and verified the new checkpoint. Its recorded state
loss was 0.477105. Inference verified all model tensors in a separate process and
stepped PhysX eight times with finite states.

Look for `ISAAC_FRESH_RESTORE_PASS`, `ISAAC_CHECKPOINT_RELOAD_PASS` during training,
and `ISAAC_PLAY_PASS 8` during playback. This is learned-state continuation, not
exact simulator/replay/RNG resume. Playback retains the physics timestep and
decimation; initial frame selection, environment count and rendering may differ.

## Check incompatible checkpoint rejection

Both commands must fail before Kit startup:

```bash
python scripts/reinforcement_learning/play.py --algorithm supermimic \
  --checkpoint logs/supermimic/verified-five-updates/model_5.pt \
  --eager --headless env.decimation=8 env.sim.dt=0.0025
python scripts/reinforcement_learning/play.py --algorithm supermimic \
  --checkpoint logs/supermimic/verified-five-updates/model_5.pt \
  --motion-file source/cyclo_lab/data/motions/K1_rev1/dance2/dance2.npz \
  --eager --headless
```

Recorded v2-checkpoint checks returned the expected **exit 1 before Kit startup**:
`decimation` and `sim_dt` mismatch in the first case, despite an unchanged 50 Hz
policy rate; `motion_sha256` mismatch in the second. Unit tests also reject changed
task, actuation/model, observation, action, reward, termination and physics
contracts, old v1 files and unsupported pickle objects. See the
[checkpoint contract](supermimic.md#local-checkpoints-play-and-continuation) for
supported relocation and continuation behavior.

## Check BeyondMimic compatibility

```bash
python scripts/reinforcement_learning/train.py --algorithm beyondmimic \
  --task Cyclo-Mimic-K1-Rev1-Dance1 \
  --motion-file source/cyclo_lab/data/motions/K1_rev1/dance1/dance1.npz \
  --num_envs 2 --max_iterations 1 --headless physics=isaacsim_physx
```

Reference result: **PASS, exit 0** using the unchanged RSL-RL backend: one
iteration, 48 transitions, reported training time 2.17 s. The output follows
`logs/rsl_rl/cyclo_mimic_k1_rev1_dance1/<run>/model_0.pt`. Existing asset, PPO
task/agent/MDP/loader/direct-launcher files were unchanged from base
`2980071bd35d970f38799638fd375da37356a299`; alias kwargs matched in the runtime tests.

## Static checks and supported limits

The reference revision passed Ruff `E4,E7,E9,F`, formatting, AST compilation,
source/license byte comparisons and `git diff --check`. The official models,
replay buffer and all learner files remained byte-identical to the pinned
SuperMimic source; the Apache LICENSE and original acknowledgements are retained.

Supported scope is GenCoord, privileged rep-state observations, one motion
encoder/clip, history 1 and fixed dynamics with zero context. MaxCoord,
proprio/history deployment, motion-to-velocity latent BC, multi-task/multi-clip
training, official return sampling and robot export are not integrated. Compiled
CUDA-graph learning, Newton, long convergence/quality, all repository tasks and
PPO play/ONNX export are outside the recorded validation. Preserving official
integration equations does not establish numerical simulator parity. OmniHub,
importer and upstream deprecation warnings did not prevent the recorded checks.
