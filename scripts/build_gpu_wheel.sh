#!/usr/bin/env bash
# Build a CUDA wheel. Requires nvcc (CUDA toolkit 11.8+). Does not need a GPU.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
if ! command -v nvcc >/dev/null 2>&1 && [[ ! -x "${CUDA_HOME:-/usr/local/cuda}/bin/nvcc" ]]; then
  echo "nvcc not found. Install CUDA toolkit and export CUDA_HOME." >&2
  exit 1
fi
export TIDEGBM_CUDA=1
python -m pip install -U pip build pybind11 wheel
python -m pip wheel . -w dist --no-deps
echo "wheels in $ROOT/dist"
ls -1 dist/tidegbm*.whl 2>/dev/null || ls -1 dist
