# Observed validation — ProArt, 2026-11-15

Repository: `/home/hc/Desktop/cyclo_lab` (ROBOTIS-GIT/cyclo_lab).
Isolated worktree: `/home/hc/Documents/Codex/2026-11-15/task-2/feature-supermimic`.
Branch: `feature-supermimic`.
Base: `2980071bd35d970f38799638fd375da37356a299`.
Fetch confirmed `origin/feature-isaacsim-6.2.0` and PR #44 head were identical.
The final commit is reported by `git rev-parse HEAD` in this worktree.

The native app worktree service did not respond; Git worktree creation succeeded.
No AGENTS.md or `.agents/skills` directory was present in the target checkout.
The original checkout stayed on main with its existing untracked
`source/cyclo_lab/data/motions/K1_rev1/woman_running/` preserved.
No cyclo_mjlab files, existing environments, or robot processes were changed.
No real robot run, long training, push, merge or PR publication was performed.

Verification used a new environment outside the checkout:
`/home/hc/Documents/Codex/2026-11-15/task-2/.verify-env`.
Python 3.13.18, PyTorch 2.15.0+cpu, MuJoCo 3.16.0, Warp 1.19.0,
NumPy 2.5.4, pytest 9.2.0; editable official mjlab_research at the documented pin.
GPU inspection before simulator work: RTX 5070 Laptop, 13 MiB allocated, 6%
utilization. Another task was installing its own runtime; verification stayed
on CPU and left that process untouched.

From the repository root, with `PY` set to the environment's `bin/python`:

| Command | Observed result |
| --- | --- |
| `$PY scripts/supermimic/convert_motion.py source/cyclo_lab/data/motions/K1_rev1/dance1/dance1.npz data/supermimic/dance1.npz` | PASS: actual 300-frame repository motion converted |
| `$PY -m pytest tests/supermimic -q` | PASS: 10 tests, including actual K1 MJCF ten physics steps and isolated task configs |
| `$PY -m pytest third_party/supermimic/tests/test_gencoord.py third_party/supermimic/tests/test_wm_loss_space.py -q` | PASS: 31 upstream representation/loss tests |
| `$PY scripts/supermimic/run.py train --help` | PASS: official training parser and K1 task import |
| `$PY scripts/supermimic/run.py play --help` | PASS: official playback parser and K1 task import |
| `$PY scripts/supermimic/smoke.py --motion-file data/supermimic/dance1.npz --device cpu` | PASS: one real mjlab environment, full-state 59, rep-state 56, prediction 29, actions 23, three simulator steps |
| `$PY scripts/supermimic/smoke.py --motion-file data/supermimic/dance1.npz --device cpu --learn` | PASS: twelve replay collection steps, one real WM update and one real policy update; both parameter sets changed, all returned metrics finite |
| `ruff check scripts/supermimic tests/supermimic --select E4,E7,E9,F` | PASS |
| `$PY -m compileall -q scripts/supermimic third_party/supermimic/src` | PASS |
| `$PY scripts/supermimic/bootstrap.py` (first run and repeat) | PASS: exact dependency fetched, existing matching checkout reused |
| `$PY -m pip check` | PASS: no broken requirements |
| `git diff --check` | PASS |

Raw logs: [integration tests](evidence/supermimic-integration-tests.txt),
[upstream algorithm tests](evidence/supermimic-algorithm-tests.txt),
[CPU simulator](evidence/supermimic-cpu-smoke.txt),
[CPU learner update](evidence/supermimic-cpu-learn.txt).
First CPU smoke required roughly two minutes of one-time Warp kernel compilation;
cached checks completed quickly. Deprecation warnings from upstream TorchScript
and an MJCF integrator attach warning occurred; no current failing check remains.
An initial launcher reflection error was fixed by importing the official module
normally instead of executing it under a replacement `__main__`. A first learner
smoke sampling-mode mismatch was fixed to match the play environment's `start`
mode. The final commands above exited successfully after those corrections.

Not run: full upstream test suite, CUDA simulator/learner, long convergence or
policy-quality evaluation, W&B model playback, Isaac Sim 6.2 full regression suite,
real robot execution. A trained K1 policy is not delivered. MaxCoord K1 and
sim-to-real controller compatibility remain outside the validated implementation.
The existing G1 tasks remain in unmodified upstream source; only GenCoord K1 is
registered by the Cyclo launcher. Full `uv sync` of the optional CUDA runtime is
not claimed; CPU verification used isolated pip editable installs of both pinned
sources. No global environment changes were made.
