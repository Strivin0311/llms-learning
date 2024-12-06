#!/bin/bash

# basic info
docker_prefix=docker.io/library
docker_name="a_env_light"
docker_version=0

workspace_dir="workspace"
repo_name="a_repo"

# (optional) set nvidia-docker gpu options
gpu_enable=false
if [[ $gpu_enable = true ]]; then
    docker_gpu_options="
        --gpus all \
        --shm-size=32g \
        --ulimit memlock=-1 \
        --ulimit stack=67108864 \
    "
else docker_gpu_options=""; fi


# load docker image tar (default "true" for the first time, then you can toggle it to "false")
load_enable=true
if [[ $load_enable = true ]]; then
    docker load -i "${docker_name}_v${docker_version}".tar
fi


# run docker image and mount current dir to workspace
docker run -it \
        --ipc=host \
        $docker_gpu_options \
        -v "$(pwd)":/$workspace_dir/$repo_name \
        "${docker_prefix}/${docker_name}:v${docker_version}"