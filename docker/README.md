# Docker environment

## Pinned stack

- Isaac Sim container: `nvcr.io/nvidia/isaac-sim:6.2.0`
- Isaac Lab: 3.0.0, commit `e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4`
- Python: 3.12, installed environment at `/opt/isaaclab-venv`
- PyTorch: torch 2.11.0 / torchvision 0.26.0 (CUDA 13.0)
- Isaac Lab and Arena dependencies are installed from their pinned `uv.lock` files.
- LeRobot 0.3.3 uses a separate environment at `~/lerobot_env`.

## Build and inspect

```bash
git submodule update --init
./docker/container.sh build
./docker/container.sh start
./docker/container.sh enter
```

Inside the container, from `/workspace/cyclo_lab`:

```bash
python docker/check_environment.py
uv pip check --python /opt/isaaclab-venv/bin/python
python docker/check_environment.py --gpu
```

The container shell's `python` and `python3` aliases select the installed Lab
Python environment through `isaaclab.sh -p`. For non-interactive commands, use
`./third_party/IsaacLab/isaaclab.sh -p` explicitly.

The first check verifies package installation and Python/PyTorch versions.
`--gpu` also checks CUDA and steps an empty headless PhysX scene 10 times.
These checks do not validate individual tasks. Review dependency conflicts
reported by `uv pip check` separately.

## Isolation and rebuilds

For side-by-side local validation, export a distinct `DOCKER_NAME_SUFFIX` and
`COMPOSE_PROJECT_NAME` before running `container.sh`. The suffix separates image
and container names; the project name separates Compose-managed volumes.
The defaults are `robotis/cyclo-lab:latest` and container/project `cyclo_lab`.

The whole source checkout is bind-mounted. Use a separate checkout for concurrent
work on different versions. After rebuilding an image, recreate the container:

```bash
./docker/container.sh build
./docker/container.sh recreate
./docker/container.sh enter
```

`start` or `docker restart` reuses the existing container's image. Recreating
preserves bind mounts and named volumes, but replaces the container's writable layer.
