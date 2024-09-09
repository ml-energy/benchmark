FROM nvidia/cuda:12.1.0-base-ubuntu22.04

# Basic installs
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ='America/Detroit'
RUN apt-get update -qq \
    && apt-get -y --no-install-recommends install python3-pip \
    && apt-get clean all \
    && rm -r /var/lib/apt/lists/*

# HuggingFace cache dir
ENV HF_HOME=/root/.cache/huggingface

# Copy over benchmark suite and install dependencies
ADD . /workspace/text-to-image
WORKDIR /workspace/text-to-image
RUN pip install -r requirements.txt

# Benchmark script to run
ENTRYPOINT ["python3", "scripts/benchmark_one_datapoint.py"]
