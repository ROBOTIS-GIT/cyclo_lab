# Isaac Sim 6.1 environment migration

This branch prepares the runtime environment. Cyclo task APIs, quaternion conventions,
recorded datasets, and policy compatibility are a separate migration step.

## Pinned stack

- Isaac Sim container: `nvcr.io/nvidia/isaac-sim:6.1.0`
- Isaac Lab: `release/3.0.0`, commit `341109ef41c4c21cb3afaa14337e93cd9a682038`
- Bundled Python: 3.12
- Lab Docker PyTorch stack: torch 2.11.0 / torchvision 0.26.0
- Follow the upstream bundled-Python installer; do not separately install or require
  torchaudio. Its constraint only pins the version if another dependency requests it.
- Lab installer selects CUDA wheels (cu128 on Linux x86_64).
- Use upstream `isaaclab.sh --install` (default: all), including Rerun and Viser.
  Do not constrain psutil or IPython: follow the upstream sequential installation.
  Rerun can upgrade psutil to 7.x after RL-Games installs 5.x, leaving a declared
  RL-Games requirement mismatch. This is recorded, not treated as proof that
  RL-Games training fails; actual training still requires validation.
- LeRobot 0.3.3 stays in its separate OS-Python venv; Sim constraints do not apply there.

The Lab release branch is under active development. Use the recorded gitlink, not a
floating branch checkout. `isaaclab.sh` is deprecated upstream but remains supported
for this downloaded-container installation; do not mix a uv/conda environment into
the bundled Sim interpreter.

## Build and inspect

```bash
git submodule update --init --recursive
./docker/container.sh build
./docker/container.sh start
./docker/container.sh enter
```

Inside the container:

```bash
/isaac-sim/python.sh docker/check_environment.py
/isaac-sim/python.sh -m pip check
/isaac-sim/python.sh docker/check_environment.py --gpu
```

The first check verifies package installation, Python and torch pins without importing
Cyclo tasks. `--gpu` additionally checks CUDA and launches a bare headless Sim via
AppLauncher, then resets and steps an empty GPU PhysX scene 10 times. It does
not validate task migration or run DDS commands. Review `pip check`
output separately. The official 6.1.0 base image already reports missing
`usd-validation-nvidia` (two SimReady packages), `pyjwt` (msal), and
`aiobotocore` (aioboto3). These inherited metadata warnings are not a passing
`pip check`; unrelated cloud/validation features need their own validation.

## Isolation and rebuilds

For side-by-side local validation, set a distinct `DOCKER_NAME_SUFFIX` and
`COMPOSE_PROJECT_NAME` in your local `.env.base`. The suffix separates image and
container names; the project name separates Compose-managed volumes. These local
values are intentionally not committed. The committed defaults retain the existing
names, so set local overrides before building if you need to preserve the old image.

The whole source checkout is bind-mounted. An old container mounting this same checkout
will see its updated code and Lab submodule. Use a separate checkout/worktree for
concurrent 5.1 work. After changes to the image, stop/recreate this branch's container
using Compose so it uses the rebuilt image; `container.sh start` resumes an existing
stopped container without recreating it.

Constraints in `constraints.isaacsim.txt` track the pinned Lab checkout and apply to
Lab, DDS, and Cyclo installation. Reconcile them with Lab's root `pyproject.toml` when
changing the gitlink. Do not restore the old torch 2.7 / torchvision 0.22 constraints.

## Next stage

Keep the first task validation on PhysX (`physics=isaacsim_physx` for upstream unified
commands). Newton being installed as a Lab dependency does not migrate Cyclo tasks to
Newton. Validate a bare Sim launch, then migrate one OMY reach task before broadening
to cameras, Mimic, motion data, and SH5/DDS.
