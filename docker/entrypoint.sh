#!/bin/bash

# Copyright (c) 2024, Cyclo Lab Project Developers.
# All rights reserved.

# This script is the entrypoint for the Docker container
# It sets up symbolic links after volumes are mounted

set -e

# Create symbolic link from Isaac Sim to Isaac Lab (third_party submodule)
if [ ! -L "${ISAACLAB_PATH}/_isaac_sim" ]; then
    echo "[INFO] Creating symbolic link: ${ISAACLAB_PATH}/_isaac_sim -> ${ISAACSIM_ROOT_PATH}"
    ln -sf ${ISAACSIM_ROOT_PATH} ${ISAACLAB_PATH}/_isaac_sim
fi

# Execute the command passed to docker run
exec "$@"
