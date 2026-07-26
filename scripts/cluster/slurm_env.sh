#!/usr/bin/env bash
# Shared Slurm environment inside NYU Torch Singularity container.
# Source after enter_singularity.sh from repo root:
#   source scripts/cluster/slurm_env.sh

set -euo pipefail

# All caches/installs must stay on scratch (never home).
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/scratch/js12556/.cache/pip}"
export TMPDIR="${TMPDIR:-/scratch/js12556/tmp}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/scratch/js12556/.cache}"
export HF_HOME="${HF_HOME:-/scratch/js12556/.cache/huggingface}"
mkdir -p "${PIP_CACHE_DIR}" "${TMPDIR}" "${XDG_CACHE_HOME}"

CONDA_EXE="${CONDA_EXE:-/share/apps/anaconda3/2025.06/bin/conda}"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-/scratch/js12556/conda/envs/bench-mfg}"

set +u
eval "$("${CONDA_EXE}" shell.bash hook)"
conda activate "${CONDA_ENV_PATH}"
set -u

_SLURM_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${_SLURM_ENV_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
