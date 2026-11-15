# Native SuperMimic corrective validation — 2026-11-15

Repository: `/home/hc/Desktop/cyclo_lab` (`ROBOTIS-GIT/cyclo_lab`).
Worktree: `/home/hc/Documents/Codex/2026-11-15/task-2/feature-supermimic`.
Branch: `feature-supermimic`; base `2980071bd35d970f38799638fd375da37356a299`
(`feature-isaacsim-6.2.0`, PR #44). The initial native HEAD was
`3eaef84602b57252bddfd0f2b93b1eb8050662da`, clean before this corrective pass.

## Runtime and isolation

Fresh simulator checks use Isaac Sim 6.2.0, Isaac Lab 3.0.0 at
`e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4`, Python 3.12.14, Torch 2.11.0,
RSL-RL 5.5.1 and TensorDict 0.14.2. The K1 asset gitlink remains
`611a070fad83d2a9ef9c1bc82213988450cf471c`.

Image: `cyclo-supermimic-isaac62:task2-git`,
`sha256:01ee9aef4fed328f3616b8ebefc5e37289a1708209f9e7424fce005f28205162`.
Recipe: `docker/Dockerfile.supermimic-validation`.
GPU: ProArt RTX 5070 Laptop. All simulator checks use **PhysX**, with
`--eager` (original learner equations/optimizers, without compile/CUDA graphs).
GPU processes and containers were inspected before launches; simulator runs were
serialized and only task-owned ephemeral containers were used.

Commands below run from `/workspace/cyclo_lab`, with the worktree mounted there.
GPU containers use `--gpus all --network host --ipc host`, `ACCEPT_EULA=Y`,
`OMNI_KIT_ALLOW_ROOT=1`, and the image's existing Python launcher. CPU checks use
`--network none` and no GPU device. PPO additionally mounts the source repository's
Git metadata read-only for its existing logger. No hardware or W&B operations
were performed. Other containers, user changes and motion data were preserved.

## Corrections verified

- Complete reference arrays now establish the physical velocity convention before
  collection. COM-to-link conversion and link-velocity reset writes are isolated
  to the new SuperMimic command; the existing PPO command remains unchanged.
- Command resample counters delimit replay even when environment done flags are
  false. Previous actions are read from the real action manager, preserving its
  history on command-only wraps and its zeros on real environment resets.
- Checkpoint v2 saves final post-update state atomically and uses restricted local
  deserialization. A declarative physical/environment contract rejects drift.
- Input arrays must be real and finite in float32. Conflicting frame, backend and
  velocity-point metadata fail. PhysX/fixed-dynamics/objective guards reject
  unsupported overrides before simulation.
- Common train/play selectors infer recognized task families, accept explicit
  `--algorithm`, reject conflicts, and preserve historical `mimic.py --method`
  and direct RSL-RL entrypoints.

The official models, replay buffer and every learner file are byte-identical to
`HybridRobotics/supermimic` revision `abc124c1bd5976341034668af3d12c087f2f1f55`.
The complete Apache LICENSE and original acknowledgements remain intact. No
unofficial algorithm/reproduction source is used in the deliverable.

## Fresh checks

### Unit and contract suite

```bash
python -m pytest tests/supermimic -q -p no:cacheprovider
```

**69 passed, 70 upstream warnings, no skips, exit 0.** Includes original numerical
WM/loss tests, five official learner updates in the explicitly labeled unit-double
collector, WM frozen-state assertions, full optimizer/model reload equality,
rotating-pelvis COM conversion, short-clip wraps with both env done flags false,
name/quaternion permutation invariance, complex/overflow rejection, actual resolved
Isaac configuration guards and checkpoint incompatibility. Unit doubles are not
counted as physical simulator evidence.

### Original immutable motion files

Every frame/body passes the historical K1 COM profile extracted from repository
URDF revision `b0ef567ca77db0f37089ed3b23e492baba60d6be`, including fixed-head
mass aggregation into the torso. This is a source-defined model, not a fitted
profile or an inferred producer identity. Both files retain 300 frames at 50 Hz.

| Clip | Maximum position error | Maximum linear-velocity error | Maximum angular-velocity error |
| --- | --- | --- | --- |
| Dance1 | 5.91e-7 m | 3.10e-6 m/s | 1.36e-5 rad/s |
| Dance2 | 6.11e-7 m | 2.18e-6 m/s | 7.57e-6 rad/s |

Absolute acceptance limits are fixed at 2e-5 m, 2e-5 quaternion components,
5e-5 m/s and 5e-5 rad/s. Unknown profiles fail; tolerances were not widened to
accept the current-URDF COM mismatch (which measured up to 0.041 m/s).
The producer/backend labels remain unspecified when absent. This does not claim
arbitrary cross-simulator archive compatibility.

Unchanged SHA-256:

- Dance1: `9e46d14b500221c43fa3c7bb438de991a1af859f904cc16ca10d01047f8883f4`
- Dance2: `682b4a950a247d6a822e5c41b832100ca28b7a57ddd2f29cc323f384f5914f50`

### Native multi-step learning and final-state saving

```bash
python scripts/reinforcement_learning/train.py --algorithm supermimic \
  --num-envs 2 --iterations 5 --collection-steps 16 --batch-size 4 \
  --horizon 4 --policy-horizon 4 --hidden 32 32 --latent 8 \
  --eager --verify --headless --output logs/supermimic/verified-five-updates
```

**PASS, exit 0.** Five genuine official WM and BPTT actor-critic updates use 80 vector steps from
two real PhysX environments (160 transitions). WM horizon 4 scores **three
successive autoregressive predictions**, not a generic one-step MSE. Each update
checks finite metrics, changes in both parameter sets, and exact equality of all
WM tensors across the actor update. `--verify` compares every final saved model,
normalizer and optimizer tensor, then compares full state after reload. The final
artifact is `logs/supermimic/verified-five-updates/model_5.pt`. WM state losses were
0.749698, 0.570511, 0.310953, 0.476564 and 0.526199. These small batches test the
learning path, not convergence or policy quality.

### Fresh-process continuation and inference

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

**Continuation PASS, exit 0.** Continuation verifies exact model/optimizer restoration **before collection**,
refills replay in a fresh simulator, performs update 6 and verifies the new final
checkpoint (iteration 6 state loss 0.477105). **Inference PASS, exit 0:** verifies all model tensors on a separate process and steps
PhysX eight times with finite states. This is learned-state continuation, not
exact simulator/replay/RNG resume. Playback retains the same physics timestep and
decimation; only initial frame selection, environment count and rendering differ.

### Incompatible checkpoint rejection

Using a real saved v2 checkpoint, playback with
`env.decimation=8 env.sim.dt=0.0025` fails on `decimation` and `sim_dt` even though
policy rate remains 50 Hz. Supplying Dance2 as `--motion-file` for a Dance1
checkpoint fails on `motion_sha256`. Both return exit 1 **before Kit startup**;
the negative-check harness exits 0. Unit checks also reject changed task,
actuation/model, observation, action, reward, termination and physics contracts,
old v1 files and unsupported pickle objects.

### Existing BeyondMimic PPO regression

```bash
python scripts/reinforcement_learning/train.py --algorithm beyondmimic \
  --task Cyclo-Mimic-K1-Rev1-Dance1 \
  --motion-file /workspace/cyclo_lab/source/cyclo_lab/data/motions/K1_rev1/dance1/dance1.npz \
  --num_envs 2 --max_iterations 1 --headless physics=isaacsim_physx
```

**PASS, exit 0:** unchanged RSL-RL backend, one iteration, 48 transitions;
reported training time 2.17 s. Output:
`logs/rsl_rl/cyclo_mimic_k1_rev1_dance1/2026-11-15_05-02-28/model_0.pt`.
Existing asset, PPO task/agent/MDP/loader/direct-launcher files have zero diff from
the PR #44 base. Alias kwargs remain equal in the target-runtime tests.

Static checks passed: Ruff `E4,E7,E9,F`, Ruff formatting, AST compilation of new
Python sources/tests, official source/license byte comparisons and `git diff --check`.

## Evidence and limits

Fresh logs: [suite](evidence/isaac-corrected-suite.txt),
[learning](evidence/isaac-corrected-learning.txt),
[continuation](evidence/isaac-corrected-continuation.txt),
[inference](evidence/isaac-corrected-inference.txt),
[negative checkpoint checks](evidence/isaac-corrected-rejections.txt),
[PPO regression](evidence/beyondmimic-corrected-regression.txt).
Only terminal ANSI escapes and trailing whitespace are removed from captured logs.

The original `isaac-native-*`, `isaac-unit-tests.txt` and
`beyondmimic-ppo-regression.txt` logs are **historical, pre-correction evidence**.
Those short runs did not expose reference COM mismatch or command-only replay
boundaries; they are not acceptance evidence for the corrected contracts.

Supported scope remains GenCoord, privileged rep-state observations, one motion
encoder/clip, history 1 and fixed dynamics with zero context. MaxCoord,
proprio/history deployment, motion-to-velocity latent BC, multi-task/multi-clip
training, official return sampling and robot export are not integrated. Compiled
CUDA-graph learning, Newton, long convergence/quality, all repository tasks and
PPO play/ONNX export were not validated in this pass. Preserved official
integration equations do not establish numerical simulator parity. OmniHub,
importer and upstream deprecation warnings did not prevent successful checks.
