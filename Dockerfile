# CUDA dev image. The repo is bind-mounted at /workspace, so code changes need no rebuild.
#
#   docker build -t qwen-rlcd .
#   docker run --rm --gpus '"device=0"' --user $(id -u):$(id -g) \
#     -v $PWD:/workspace -v /big/mazhar/qwen-rlcd:/cache qwen-rlcd pytest -q
#
# /cache holds the HF hub cache and the Triton kernel + autotune cache (the first fla run otherwise
# spends ~2 min compiling and autotuning).
#
# torch comes from the cu126 index: the dev box driver (550) supports CUDA <= 12.4 plus minor-version
# compatibility, and the default PyPI wheels need a CUDA 13 driver.
FROM python:3.12-slim

# Triton (used by flash-linear-attention) compiles a small C launcher at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
ENV UV_SYSTEM_PYTHON=1 UV_NO_CACHE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/cache/hf TRITON_CACHE_DIR=/cache/triton HOME=/tmp PYTHONPATH=/workspace

RUN uv pip install --index-url https://download.pytorch.org/whl/cu126 "torch==2.14.*"

WORKDIR /workspace
COPY pyproject.toml .
RUN mkdir system_one && touch system_one/__init__.py \
    && uv pip install ".[dev,eval,cuda]" scikit-learn matplotlib \
    && uv pip uninstall system-one && rm -rf system_one build *.egg-info
