FROM pytorch/pytorch:2.7.0-cuda12.6-cudnn9-runtime AS base

ARG DEBIAN_FRONTEND=noninteractive
ARG VLLM_VERSION=v0.10.0rc1

RUN apt-get update -y \
    && apt-get install -y --no-install-recommends \
        git \
        curl \
        wget \
        build-essential \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

ENV PATH="/root/.local/bin:$PATH"
ENV VIRTUAL_ENV="/opt/venv"
RUN uv venv --python 3.11 --seed ${VIRTUAL_ENV}
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

WORKDIR /workspace
ENV VLLM_USE_PRECOMPILED=1

RUN git clone -b ${VLLM_VERSION} https://github.com/vllm-project/vllm s-vllm && \
    cd s-vllm && \
    uv pip install -e . .[audio]

ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server"]

