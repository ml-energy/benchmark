FROM nvidia/cuda:11.8.0-base-ubuntu22.04

# Basic installs
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ='America/Detroit'
RUN apt-get update -qq \
    && apt-get -y --no-install-recommends install \
       build-essential software-properties-common wget git tar rsync ninja-build \
    && apt-get clean all \
    && rm -r /var/lib/apt/lists/*

# Install Miniconda3 23.3.1
ENV PATH="/root/.local/miniconda3/bin:$PATH"
RUN mkdir -p /root/.local \
    && wget https://repo.anaconda.com/miniconda/Miniconda3-py39_23.3.1-0-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-py39_23.3.1-0-Linux-x86_64.sh -b -p /root/.local/miniconda3 \
    && rm -f Miniconda3-py39_23.3.1-0-Linux-x86_64.sh \
    && ln -sf /root/.local/miniconda3/etc/profile.d/conda.sh /etc/profile.d/conda.sh

# Install spitfight
ADD . /workspace/leaderboard
RUN cd /workspace/leaderboard && pip install -e .[benchmark]

# Clone lm-evaluation-harness and install
RUN cd /workspace \
      && git clone https://github.com/EleutherAI/lm-evaluation-harness.git \
      && cd lm-evaluation-harness \
      && git checkout d1537059b515511801ae9b742f8e949f1bfcd010 \
      && rm -r .git \
      && pip install -e .

# Apply patches
# Salesforce xgen inference fix (https://github.com/lm-sys/FastChat/pull/2350)
RUN cd /root/.local/miniconda3/lib/python3.9/site-packages \
      && patch -p1 < /workspace/leaderboard/deployment/fastchat_xgen_fix.patch

# Where all the weights downloaded from Hugging Face Hub will go to
ENV TRANSFORMERS_CACHE=/data/leaderboard/hfcache
ENV HF_HOME=/data/leaderboard/hfcache

# So that docker exec container python scripts/benchmark.py will work
WORKDIR /workspace/leaderboard
