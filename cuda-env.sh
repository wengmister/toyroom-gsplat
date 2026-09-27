export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=12.0
export MAX_JOBS=2
export CPATH="$(.venv/bin/python -c 'import pathlib; print(":".join(str(p.resolve()) for p in pathlib.Path(".venv/lib/python3.12/site-packages/nvidia").glob("*/include")))')${CPATH:+:$CPATH}"
