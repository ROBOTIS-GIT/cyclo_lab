# Isaac-native SuperMimic validation — ProArt, 2026-11-15

Repository: `/home/hc/Desktop/cyclo_lab` (ROBOTIS-GIT/cyclo_lab).
Isolated worktree: `/home/hc/Documents/Codex/2026-11-15/task-2/feature-supermimic`.
Branch: `feature-supermimic`.
Base: `2980071bd35d970f38799638fd375da37356a299`, confirmed equal to fetched
`origin/feature-isaacsim-6.2.0` and PR #44 head.
The native port supersedes the initial optional mjlab implementation. Its
bootstrap, Python 3.13 project, simulator adapter, duplicate motion conversion
and old simulator evidence are removed. No second mjlab runtime is required.

## Actual runtime

- Isaac Sim 6.2.0, base manifest `sha256:a025b65a7442d56806a1cc3f5eed081caf736c24b72d62dccf68ef66945d8690`.
- Isaac Lab 3.0.0 gitlink `e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4`.
- Python 3.12.14, locked Torch 2.11.0, RSL-RL 5.5.1, TensorDict 0.14.2.
- K1 assets: unchanged ai_sapiens gitlink `611a070fad83d2a9ef9c1bc82213988450cf471c`.
- Verification image: `cyclo-supermimic-isaac62:task2-git`, manifest
  `sha256:01ee9aef4fed328f3616b8ebefc5e37289a1708209f9e7424fce005f28205162`.
- Recipe: `docker/Dockerfile.supermimic-validation`; existing production Docker
  setup remains unchanged. The Git executable is required by the existing PPO logger.
- All physical simulation below used actual Isaac Sim PhysX on the ProArt RTX 5070
  Laptop GPU. GPU occupancy was inspected; GPU runs were serialized. No robot,
  existing training process, other repository or existing container was modified.

Commands below ran from `/workspace/cyclo_lab` with this worktree mounted there.
Each GPU container used `--gpus all --network host --ipc host`, `ACCEPT_EULA=Y`,
`OMNI_KIT_ALLOW_ROOT=1`. The PPO run additionally mounted the source Git common
metadata at its absolute path read-only, so its existing Git logger could read it.

## Observed results

### Target-environment tests: PASS, exit 0

```bash
python -m pytest tests/supermimic -q
```

**49 passed**, 70 upstream deprecation warnings, no skips/failures. Tests cover
original generalized-coordinate/loss equations, FK joint gradients and finite
velocity differences, quaternion conventions, episode-boundary collection,
actual checkpoint optimizer/state reload, no-mjlab import, old task aliases and
config isolation, and strict shared-motion validation. Unit environment doubles
are explicitly labeled and are not counted as simulator smoke evidence.

### Native Isaac learning: actual updates and checkpoint reload PASS

```bash
python scripts/reinforcement_learning/mimic.py --method supermimic train \
  --num-envs 2 --iterations 2 --collection-steps 8 --batch-size 2 \
  --horizon 2 --policy-horizon 2 --hidden 32 32 --latent 8 \
  --eager --verify --headless --output logs/supermimic/isaac-native-final
```

The log identifies `isaaclab.envs.manager_based_rl_env` and `PhysxCfg`.
16 vector steps from 2 environments supplied replay (32 real environment
transitions). Original WM and imagined BPTT actor-critic learners each updated
twice; both parameter sets changed each time and every reported metric was finite.
Observed state loss: `0.1421301`, then `0.1230792`; these tiny batches do not measure
policy quality or convergence. Live articulation/URDF FK position error was at
most `5.22e-7 m`, quaternion error `4.18e-7`, twist error `3.16e-6`.
Both `model_1.pt` and `model_2.pt` (616,481 bytes each) saved and reloaded exactly.

This first run finished its computation and cleared SimulationContext, but a
worker remained at process teardown across a host disconnection. Only that owned
verification container was stopped after inspecting its idle process tree.
The new launcher now sets spawned compiler workers before Torch/Kit imports and
single-thread compilation for eager execution. The following fresh-process play
and resumed-learning runs verified normal exit after this correction.

### Fresh-process local play: PASS, exit 0

```bash
python scripts/reinforcement_learning/mimic.py --method supermimic play \
  --checkpoint logs/supermimic/isaac-native-final/model_2.pt \
  --num-envs 1 --play-steps 4 --eager --verify --headless
```

The saved task/config/motion path and real checkpoint loaded in a new Isaac
process. Four simulator steps had finite state; `ISAAC_PLAY_PASS 4` was emitted.
Docker exit event confirmed code 0 (104 seconds including runtime startup).

### Fresh-process resume and update: PASS, exit 0

```bash
python scripts/reinforcement_learning/mimic.py --method supermimic train \
  --checkpoint logs/supermimic/isaac-native-final/model_2.pt \
  --num-envs 2 --iterations 1 --collection-steps 8 --horizon 2 \
  --eager --verify --headless --output logs/supermimic/isaac-native-resume
```

The saved optimizer/normalizer/model config was restored, fresh Isaac transitions
were collected and iteration 3 updated both WM and actor parameters. All metrics
were finite (`Loss/state=0.08807583`). `model_3.pt` saved and reloaded; Docker and
the command result confirmed exit 0 (73 seconds including startup). This run also
executed the final strict motion contract and worker-lifecycle code.

### Existing BeyondMimic PPO regression: PASS, exit 0

```bash
python scripts/reinforcement_learning/mimic.py --method beyondmimic train \
  --task Cyclo-Mimic-K1-Rev1-Dance1 \
  --motion-file /workspace/cyclo_lab/source/cyclo_lab/data/motions/K1_rev1/dance1/dance1.npz \
  --num_envs 2 --max_iterations 1 --headless physics=isaacsim_physx
```

The selector called the unchanged existing RSL-RL launcher, using the old task
ID, same input and explicit PhysX preset. One PPO iteration finished (reported
training time 2.46 s), producing:
`logs/rsl_rl/cyclo_mimic_k1_rev1_dance1/2026-11-15_02-51-38/model_0.pt`.
The Docker exit event confirmed code 0. Existing task/agent/observation/reward,
asset and direct RSL-RL launcher files have zero diff against the PR #44 base.
New BeyondMimic alias kwargs equal the old registration in the target tests.

### Shared original data contract: PASS

Both real algorithms used the same 300-frame, named 23-joint/24-body, 50 Hz
`dance1.npz`; no conversion, resampling or source rewrite was needed.
Its SHA-256 remained
`9e46d14b500221c43fa3c7bb438de991a1af859f904cc16ca10d01047f8883f4`.
Unit cases independently permuted source joint/body columns, mapped between the
archive's breadth-first and URDF depth-first name orders, and supplied xyzw/wxyz
quaternions. The shared existing loader produced exactly equal arrays at 50 Hz;
both source hashes stayed unchanged. Invalid/missing/duplicate metadata,
conflicting conventions, nonunit/zero quaternions, nonfinite/incorrectly shaped
arrays, unknown joint names and wrong FPS were rejected. These producer variants
were array-level tests; only the actual repository archive was physically run.

### Other checks: PASS

- Ruff `E4,E7,E9,F` for new source, task configs, launchers and tests.
- `git diff --check`.
- Original LICENSE content/hash unchanged. `models.py`, `replay_buffer.py` and all
  learner files remain verbatim from official SuperMimic `abc124c1...`.
- No `.agents/skills` or AGENTS.md was present in the target checkout.
- Original checkout remained on main with its pre-existing untracked
  `woman_running/` motion directory preserved.

Evidence logs (terminal color codes/trailing whitespace removed): [tests](evidence/isaac-unit-tests.txt),
[native updates](evidence/isaac-native-training.txt),
[fresh play](evidence/isaac-native-play.txt),
[resume](evidence/isaac-native-resume.txt),
[PPO regression](evidence/beyondmimic-ppo-regression.txt).

## Limits — not claimed as verified

No long training, motion-quality/convergence evaluation, real robot operation,
push, merge or PR publication was performed. Native learning was checked with
`--eager`; compiled/CUDA-graph learning, full-scale performance, default Newton
preset regression, all repository tasks and PPO play/ONNX export were not rerun.
Existing PPO export code remains unchanged. SuperMimic has no validated robot
export/controller contract, privileged rep-state actor input, and supports only
GenCoord K1 single-motion tracking in this port. Multi-clip return sampling,
proprio/history deployment, multi-task encoders and MaxCoord K1 remain outside
this validated path. Startup emitted upstream deprecation/OmniHub/importer
warnings; none prevented the final successful checks above.

## Narrow comparison with the supplied K1 reproduced-reference report

The user supplied the [2026-10-28 K1 sharing note](https://robotis.slack.com/archives/C0BJP0LA4TU/p1793152532240979)
and its actuator summary. That work used `CuboiLeo/SuperMimic-Reproduced`, not the
HybridRobotics source ported here. The author's settings are not treated as
independently established optimal settings. No settings were copied in response.

| Item | Current Isaac-native implementation | Supplied reproduced-reference report |
| --- | --- | --- |
| Actuation | Existing `K1_REV1_INERTIA_TUNED_CFG` `ImplicitActuatorCfg`; 10 Hz inertia-derived PD, damping ratio 2, armature and static joint effort/velocity limits | `TorqueSpeedTableActuator`, QC080 table `k1_qc080_20261007`, motoring/braking four-quadrant speed-dependent clamp |
| QC080 effort | Existing 96.864 Nm constant joint effort limit; QC060 remains 47.277 Nm | Reported QC080 force_limit 97.76 |
| Delay/strength | No reset-randomized delay or strength interface in this port | Reset-specific delay/strength randomization interface described in the shared actuator file |
| Collision | Existing PhysX preset sets `spawn.self_collision=False`; new task disables `undesired_contacts`; original WM reward has no self-collision term | Author reported self-collision model and penalty |
| Joint offset | Home/encoder-offset event disabled in the new task. Existing reference-reset joint-position perturbation remains ±0.1 rad, a different quantity | Author reported joint offset ±0.07 rad |
| Push | Disabled in the new task | Author reported no push |

Code evidence: `assets/robots/K1_rev1.py`, `utils/k1_physics.py`,
`manager_based/mimic/tracking_env_cfg.py`, and the new
`manager_based/mimic/config/supermimic_k1_rev1/env_cfg.py`.
The torque-speed model, four-quadrant clamp, actuator delay/strength randomization
and self-collision training terms are not implemented here. The supplied archive's
loader/export code was not available for this comparison and is not claimed to
match. These are explicit differences in robot modeling/deployment scope, not
reasons to claim the reported settings are optimal or to expand this port silently.
