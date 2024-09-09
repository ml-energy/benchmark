#!/usr/bin/env bash

docker run -dit --gpus all --cap-add SYS_ADMIN --name nvml nvidia/cuda:12.3.1-base-ubuntu22.04 bash
