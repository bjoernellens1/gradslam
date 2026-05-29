ARG BASE_IMAGE=docker.io/rocm/pytorch:rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.7.1
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV HIP_VISIBLE_DEVICES=0

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    cmake \
    ninja-build \
    pkg-config \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/gradslam

COPY pyproject.toml README.md ./
COPY gradslam ./gradslam
COPY tests ./tests
COPY examples ./examples
COPY scripts ./scripts
COPY docs ./docs

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install -e ".[optim,mesh,vis,realsense,docs,dev]"

CMD ["bash"]
