#!/usr/bin/env bash
# Shared Slurm bootstrap (Torch copies job scripts to /opt/slurm/... spool).
#
# Usage (after #SBATCH lines, from repo root via sbatch):
#   source scripts/cluster/slurm_init.sh scripts/cluster/submit_mfpso_particle_ablation_gpu.slurm.sh
#
# Optional env:
#   SINGULARITY_USE_NV=0   # CPU-only jobs

REPO_ROOT="${SLURM_SUBMIT_DIR:-/scratch/js12556/Bench-MFG}"
if [[ $# -lt 1 ]]; then
  echo "Usage: source slurm_init.sh <path/to/job.slurm relative to repo root>" >&2
  return 1 2>/dev/null || exit 1
fi

export SLURM_ENTRY_SCRIPT="${REPO_ROOT}/$1"
source "${REPO_ROOT}/scripts/cluster/enter_singularity.sh"
cd "${REPO_ROOT}"

set -eo pipefail
source "${REPO_ROOT}/scripts/cluster/slurm_env.sh"
set -u
