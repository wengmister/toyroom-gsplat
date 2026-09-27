#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source cuda-env.sh
export OMP_NUM_THREADS=4
exec .venv/bin/python -u vendor/gsplat/examples/simple_trainer.py default \
  --data-dir data/aligned --data-factor 1 --result-dir results/default \
  --max-steps 30000 --save-ply --disable-video --port 8080
