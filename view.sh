#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source cuda-env.sh
export OMP_NUM_THREADS=4
exec .venv/bin/python -u vendor/gsplat/examples/simple_viewer.py \
  --ckpt results/default/ckpts/ckpt_29999_rank0.pt \
  --output_dir results/default --port 8080
