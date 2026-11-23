#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/.env.base"
container="cyclo_lab${DOCKER_NAME_SUFFIX}"
docker exec -i --user 0 "$container" bash -seu <<'REPAIR'
owner="$(id -u cyclolab):$(id -g cyclolab)"

# New runs only need writable directories within existing output trees.
for path in "$CYCLOLAB_PATH"/{logs,outputs,wandb} "$ISAACLAB_PATH/logs" \
    "$ISAACSIM_PATH/kit/logs/Kit/Isaac-Sim" "$DOCKER_USER_HOME/.nvidia-omniverse/logs"; do
    if [[ -d "$path" && "$path" == "$(realpath "$path")" ]]; then
        find "$path" -xdev -type d -uid 0 -exec chown -h --from=0 "$owner" {} +
    fi
done

# Generated caches can also contain existing files that need updating.
for path in "$ISAACSIM_PATH/kit/cache" "$DOCKER_USER_HOME"/.cache/{ov,pip,nvidia/GLCache,warp,wandb} \
    "$DOCKER_USER_HOME"/{.nv/ComputeCache,.triton,.local/share/wandb}; do
    if [[ -d "$path" && "$path" == "$(realpath "$path")" ]]; then
        find "$path" -xdev -uid 0 \( -type d -o \( -type f -links 1 \) \) \
            -exec chown -h --from=0 "$owner" {} +
    fi
done
REPAIR
echo "Permission repair complete for ${container}."
