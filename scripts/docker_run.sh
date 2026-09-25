#!/usr/bin/env bash
# Run a command in the qwen-rlcd image with the repo mounted at /workspace (see Dockerfile).
#
#   scripts/docker_run.sh pytest -q
#   GPU=1 scripts/docker_run.sh python scripts/zero_shot_eval.py --limit 1000
#
# GPU selects the card (default 0; "none" for CPU only). CACHE (HF hub + Triton caches) is mounted at /cache.
set -euo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"
gpu="${GPU:-0}"
gpu_args=()
[[ "$gpu" != none ]] && gpu_args=(--gpus "\"device=$gpu\"")
exec docker run --rm "${gpu_args[@]}" --user "$(id -u):$(id -g)" --shm-size 8g \
  -v "$repo:/workspace" -v "${CACHE:-/big/mazhar/qwen-rlcd}:/cache" \
  "${IMAGE:-qwen-rlcd}" "$@"
