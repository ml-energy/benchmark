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
      && git checkout e9f1af36d2f6f8449e3cd132e6885d3b010ec838 \
      && rm -r .git \
      && pip install -e .

# Where all the weights downloaded from Hugging Face Hub will go to
ENV TRANSFORMERS_CACHE=/data/leaderboard/hfcache

# So that docker exec container python scripts/benchmark.py will work
WORKDIR /workspace/leaderboard
