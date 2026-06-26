#!/usr/bin/env bash

# Copyright (c) 2024, Cyclo Lab Project Developers.
# All rights reserved.
#
# Based on Isaac Lab container management script

#==
# Configurations
#==

# Exits if error occurs
set -e

# Set tab-spaces
tabs 4

# get source directory
export CYCLOLAB_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." &> /dev/null && pwd )"
export DOCKER_DIR="${CYCLOLAB_PATH}/docker"

#==
# Helper functions
#==

# print the usage description
print_help() {
    echo -e "\nusage: $(basename "$0") [-h] <command> [<args>]"
    echo -e "\nCyclo Lab Docker Container Management Script"
    echo -e "\noptional arguments:"
    echo -e "  -h, --help           Display this help message."
    echo ""
    echo -e "commands:"
    echo -e "  build                Build the docker image for Cyclo Lab"
    echo -e "  start                Start the docker container"
    echo -e "  enter                Enter the running docker container"
    echo -e "  stop                 Stop the docker container"
    echo -e "  clean                Remove the docker container and image"
    echo -e "  logs                 Show logs from the container"
    echo ""
}

# Load environment variables
load_env() {
    if [ -f "${DOCKER_DIR}/.env.base" ]; then
        set -a
        source "${DOCKER_DIR}/.env.base"
        set +a
        echo "[INFO] Loaded environment from .env.base"
    else
        echo "[ERROR] .env.base file not found in ${DOCKER_DIR}"
        exit 1
    fi
}

# Configure X11 forwarding
setup_x11() {
    # Check if xauth is installed
    if ! command -v xauth &> /dev/null; then
        echo "[WARN] xauth is not installed. X11 forwarding will not work."
        echo "[WARN] Install with: sudo apt install xauth"
        return 1
    fi

    # Check if DISPLAY is set
    if [ -z "$DISPLAY" ]; then
        echo "[WARN] DISPLAY variable is not set. X11 forwarding will not work."
        return 1
    fi

    # Create temporary directory for xauth
    export __CYCLOLAB_TMP_DIR=$(mktemp -d)
    export __CYCLOLAB_TMP_XAUTH="${__CYCLOLAB_TMP_DIR}/.xauth"

    # Create xauth file
    touch "${__CYCLOLAB_TMP_XAUTH}"
    xauth nlist "$DISPLAY" | sed -e 's/^..../ffff/' | xauth -f "${__CYCLOLAB_TMP_XAUTH}" nmerge -
    
    echo "[INFO] X11 forwarding configured"
    echo "[INFO] XAUTH file: ${__CYCLOLAB_TMP_XAUTH}"

    return 0
}

# Check if X11 is available
check_x11() {
    if [ -n "$DISPLAY" ] && command -v xauth &> /dev/null; then
        return 0
    else
        return 1
    fi
}

# Build docker image
build_image() {
    echo "[INFO] Building Cyclo Lab docker image..."
    cd "${DOCKER_DIR}"
    docker compose build cyclo_lab
    echo "[INFO] Build complete!"
}

# Start docker container
start_container() {
    echo "[INFO] Starting Cyclo Lab docker container..."

    # Check and initialize git submodules
    echo "[INFO] Checking git submodules..."
    cd "${CYCLOLAB_PATH}"
    if [ -d ".git" ]; then
        if git submodule status | grep -q '^-'; then
            echo "[INFO] Initializing git submodules..."
            git submodule update --init --recursive
            echo "[INFO] Git submodules initialized"
        else
            echo "[INFO] Git submodules already initialized"
        fi
    else
        echo "[WARN] Not a git repository, skipping submodule initialization"
    fi

    cd "${DOCKER_DIR}"

    # Setup X11 forwarding
    X11_COMPOSE_FILE=""
    if check_x11; then
        if setup_x11; then
            X11_COMPOSE_FILE="-f x11.yaml"
            echo "[INFO] X11 forwarding enabled"
        fi
    else
        echo "[INFO] X11 forwarding not available (no DISPLAY or xauth)"
    fi

    # Check if container is already running
    if [ -n "$(docker ps -q --filter "name=^cyclo_lab${DOCKER_NAME_SUFFIX}$")" ]; then
        echo "[INFO] Container is already running"
        return 0
    fi

    # Check if container exists but is stopped
    if [ -n "$(docker ps -aq --filter "name=^cyclo_lab${DOCKER_NAME_SUFFIX}$")" ]; then
        echo "[INFO] Starting existing container..."
        docker start cyclo_lab${DOCKER_NAME_SUFFIX}
    else
        echo "[INFO] Creating and starting new container..."
        docker compose -f docker-compose.yaml ${X11_COMPOSE_FILE} up -d cyclo_lab
    fi

    echo "[INFO] Container started successfully!"
    echo "[INFO] Use './docker/container.sh enter' to access the container"
}

# Enter running container
enter_container() {
    echo "[INFO] Entering Cyclo Lab docker container..."

    # Check if container is running
    if [ -z "$(docker ps -q --filter "name=^cyclo_lab${DOCKER_NAME_SUFFIX}$")" ]; then
        echo "[ERROR] Container is not running. Start it first with './docker/container.sh start'"
        exit 1
    fi

    # Pass DISPLAY environment variable to the container
    docker exec -it -e DISPLAY="${DISPLAY}" cyclo_lab${DOCKER_NAME_SUFFIX} /bin/bash
}

# Stop container
stop_container() {
    echo "[INFO] Stopping Cyclo Lab docker container..."
    cd "${DOCKER_DIR}"
    docker compose stop cyclo_lab
    echo "[INFO] Container stopped"
}

# Clean up container and image
clean_docker() {
    echo "[INFO] Cleaning up Cyclo Lab docker resources..."
    cd "${DOCKER_DIR}"

    read -p "This will remove the container and image. Continue? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker compose down cyclo_lab
        docker rmi cyclolab/cyclo-lab${DOCKER_NAME_SUFFIX}:latest || true
        echo "[INFO] Cleanup complete"
    else
        echo "[INFO] Cleanup cancelled"
    fi
}

# Show container logs
show_logs() {
    echo "[INFO] Showing Cyclo Lab container logs..."
    cd "${DOCKER_DIR}"
    docker compose logs -f cyclo_lab
}

#==
# Main
#==

# check argument provided
if [ -z "$*" ]; then
    echo "[ERROR] No arguments provided." >&2
    print_help
    exit 1
fi

# Load environment variables
load_env

# pass the arguments
case "$1" in
    build)
        build_image
        ;;
    start)
        start_container
        ;;
    enter)
        enter_container
        ;;
    stop)
        stop_container
        ;;
    clean)
        clean_docker
        ;;
    logs)
        show_logs
        ;;
    -h|--help)
        print_help
        exit 0
        ;;
    *)
        echo "[ERROR] Invalid command: $1"
        print_help
        exit 1
        ;;
esac

echo ""
echo "[INFO] Command completed successfully!"
