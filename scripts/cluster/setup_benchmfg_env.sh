#!/usr/bin/env bash
# Install / repair Bench-MFG conda env on scratch (never home).
#
# Usage (from repo root, preferably inside singularity):
#   bash scripts/cluster/setup_benchmfg_env.sh

set -euo pipefail

ENV_PATH="/scratch/js12556/conda/envs/bench-mfg"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/scratch/js12556/.cache/pip}"
export TMPDIR="${TMPDIR:-/scratch/js12556/tmp}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/scratch/js12556/.cache}"
mkdir -p "${PIP_CACHE_DIR}" "${TMPDIR}" "${XDG_CACHE_HOME}"

# Avoid accidental writes to home caches
export HOME_PIP_CACHE_BLOCK="${HOME}/.cache/pip"
if [[ -d "${HOME_PIP_CACHE_BLOCK}" ]]; then
  echo "WARNING: ${HOME_PIP_CACHE_BLOCK} exists; prefer PIP_CACHE_DIR=${PIP_CACHE_DIR}"
fi

CONDA_EXE="${CONDA_EXE:-/share/apps/anaconda3/2025.06/bin/conda}"
# shellcheck disable=SC1091
source "$("${CONDA_EXE}" info --base)/etc/profile.d/conda.sh"

if [[ ! -d "${ENV_PATH}" ]]; then
  echo "Creating conda env at ${ENV_PATH}"
  conda create -y -p "${ENV_PATH}" python=3.11 pip --override-channels -c conda-forge
fi

conda activate "${ENV_PATH}"
cd "${REPO_ROOT}"
pip install -e . --cache-dir "${PIP_CACHE_DIR}"
python -c "import jax; import benchmfg; print('jax', jax.__version__); print('benchmfg', benchmfg.__file__)"
echo "Done. Env: ${ENV_PATH}"
