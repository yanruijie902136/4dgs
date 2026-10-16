#!/usr/bin/env bash
# Build the pinned native submodules against the active PyTorch/CUDA environment.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -z "${CONDA_PREFIX:-}" ]]; then
    echo "Activate the 4dgs conda environment first." >&2
    exit 1
fi
export CUDA_HOME="${CUDA_HOME:-$CONDA_PREFIX}"
export CC="${CC:-$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc}"
export CXX="${CXX:-$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++}"
export MAX_JOBS="${MAX_JOBS:-4}"
if [[ -d "$CONDA_PREFIX/targets/x86_64-linux/include" ]]; then
    export CPATH="$CONDA_PREFIX/targets/x86_64-linux/include${CPATH:+:$CPATH}"
    export LIBRARY_PATH="$CONDA_PREFIX/targets/x86_64-linux/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
fi
# PyTorch detects the active GPU unless an architecture list is supplied.
python "$repo_root/scripts/build_extensions.py"
python -c 'import torch; from simple_knn._C import distCUDA2; from diff_gaussian_rasterization import GaussianRasterizer; print("CUDA extensions imported successfully")'
