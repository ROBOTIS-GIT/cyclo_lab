# SuperMimic on K1

This integration calls the official SuperMimic differentiable world-model trainer,
not the Isaac Lab PPO launcher. It registers `SuperMimic-GenCoord-K1` in the
separate MuJoCo / mjlab_research runtime. Isaac Sim 6.2 / Isaac Lab 3.0 tasks remain
available through their existing scripts. No cyclo_mjlab checkout is required.

## Reproducible setup

From the cyclo_lab repository root:

```bash
git submodule update --init third_party/ai_sapiens
python3 scripts/supermimic/bootstrap.py
uv sync --project third_party/supermimic --python 3.13
PY=third_party/supermimic/.venv/bin/python
```

Use a separate Python 3.13 environment. Do not install this into Isaac Sim's Python.
The source in `third_party/supermimic` is preserved from official commit
`abc124c1bd5976341034668af3d12c087f2f1f55`; bootstrap fetches the official
`mjlab_research` revision `dd51a9fcc98791a2f0b018b3c7f71a3eefd5ecdd` as its
sibling. Bootstrap refuses to reset an existing dependency at another revision.
The upstream uv lock is preserved; uv may resolve it against the pinned sibling's
current package requirements. The complete original LICENSE and acknowledgements
are retained. K1 assets come from this branch's ai_sapiens gitlink
`611a070fad83d2a9ef9c1bc82213988450cf471c`.

## Motion preparation

Use Cyclo's existing CSV retarget/conversion pipeline first, then:

```bash
$PY scripts/supermimic/convert_motion.py \
  source/cyclo_lab/data/motions/K1_rev1/dance1/dance1.npz \
  data/supermimic/dance1.npz
```

The converter reads named 23-joint Cyclo NPZ, selects the pelvis root world pose
and twist, reorders joints into MJCF depth-first order, normalizes and converts
quaternions to wxyz, and writes the six fields consumed by official MotionLoader.
It rejects missing/duplicate joints, missing pelvis, nonfinite arrays, zero
quaternions, wrong shapes and non-50-Hz clips. Legacy files without quaternion
metadata need `--quaternion-order xyzw` or `wxyz`; never guess the convention.
Output uses exclusive creation, so existing reference data cannot be overwritten.
Resampling belongs to the existing CSV tool (`--output_fps 50`). Body arrays are
not reused for training: the official differentiable FK recomputes bodies from
K1 joints and root state. Outputs live in ignored `data/supermimic/`.

## Simulation, training and playback

```bash
$PY scripts/supermimic/smoke.py --motion-file data/supermimic/dance1.npz --device cpu
$PY scripts/supermimic/smoke.py --motion-file data/supermimic/dance1.npz --device cpu --learn
$PY scripts/supermimic/run.py train --help
WANDB_MODE=offline $PY scripts/supermimic/run.py train \
  --motion-file "$PWD/data/supermimic/dance1.npz" \
  --num-envs 32 --runner.max-iterations 10 --runner.run-name k1-check
$PY scripts/supermimic/run.py play --wandb-path ENTITY/PROJECT/RUN \
  --motion-file "$PWD/data/supermimic/dance1.npz"
```

Inspect GPU occupancy before any GPU run. The smoke command uses one environment,
builds a small real world model and steps the simulator three times, without
starting a learner or W&B run. Adding `--learn` collects twelve small replay
transitions and runs one real world-model update and one actor-critic update
with tiny networks and batches. It asserts both sets of parameters changed and
all metrics are finite; it does not create a W&B run or save a trained policy. The training command above is an example, not an
automatically started job; the upstream defaults are large (4096 environments,
one million iterations), so set bounds deliberately. Upstream output paths are
relative to the launch directory (`logs/`); launch from the repository root.
Official playback loads model bundles and run config via W&B; an offline training
run needs to be synced before this playback command can retrieve it.

K1 uses the inertia-tuned Cyclo PD formula (10 Hz, damping ratio 2), motor armature
and effort limits, default pose and `0.25 * torque / stiffness` action scaling.
XML torque actuators and home keyframes are removed in memory before mjlab adds
position actuators; robot asset files are never edited. IMU sensors use the root
frame. G1 foot contact matching is replaced with K1 ankle collision geometry,
and tracked hands use K1's rubber-hand body names. The existing upstream world
model, replay buffer, learners, rewards and return sampler remain unmodified.
Generalized-coordinate state is 59-wide for K1 (13 + 2*23); actions are 23-wide.

MaxCoord K1 is deliberately not registered: its analytic IK path needs separate
robot-specific validation. No Isaac Lab backend port of the world model, trained
K1 policy quality, sim-to-real export compatibility or real-robot operation is
claimed. Existing Isaac checkpoints are not SuperMimic model bundles.

## Tests

```bash
$PY -m pytest tests/supermimic -q
$PY -m pytest third_party/supermimic/tests/test_gencoord.py \
  third_party/supermimic/tests/test_wm_loss_space.py -q
```

Tests require pytest and the full runtime (including NumPy and MuJoCo).
The asset test compiles the actual pinned K1 MJCF and performs ten CPU physics
steps. See `docs/supermimic-validation.md` for observed results and limitations.
