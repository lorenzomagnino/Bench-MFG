#!/usr/bin/env bash
# Re-exec the Slurm entry script inside NYU Torch Singularity.
# Prefer: source scripts/cluster/slurm_init.sh scripts/cluster/your_job.slurm.sh

if [[ -z "${BENCHMFG_IN_SINGULARITY:-}" ]]; then
  if [[ -z "${SLURM_ENTRY_SCRIPT:-}" ]]; then
    echo "ERROR: SLURM_ENTRY_SCRIPT is not set. Use slurm_init.sh or export it before sourcing." >&2
    exit 1
  fi

  if [[ ! -f "${SLURM_ENTRY_SCRIPT}" ]]; then
    echo "ERROR: Slurm entry script not found: ${SLURM_ENTRY_SCRIPT}" >&2
    exit 1
  fi

  _CALLER_SCRIPT="$(readlink -f "${SLURM_ENTRY_SCRIPT}" 2>/dev/null || realpath "${SLURM_ENTRY_SCRIPT}")"
  _SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${_CALLER_SCRIPT}")/../.." && pwd)}"

  SINGULARITY_IMAGE="${SINGULARITY_IMAGE:-/scratch/js12556/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif}"
  SINGULARITY_OVERLAY="${SINGULARITY_OVERLAY:-/scratch/js12556/overlay-10GB-400K.ext3}"

  _SING_ARGS=(exec)
  if [[ "${SINGULARITY_USE_NV:-1}" != "0" ]]; then
    _SING_ARGS+=(--nv)
  fi
  _SING_ARGS+=(--overlay "${SINGULARITY_OVERLAY}:ro" "${SINGULARITY_IMAGE}")

  exec singularity "${_SING_ARGS[@]}" \
    /bin/bash -lc "export BENCHMFG_IN_SINGULARITY=1 SLURM_SUBMIT_DIR=$(printf '%q' "${_SUBMIT_DIR}") SLURM_ENTRY_SCRIPT=$(printf '%q' "${_CALLER_SCRIPT}"); cd $(printf '%q' "${_SUBMIT_DIR}"); bash $(printf '%q' "${_CALLER_SCRIPT}")"
fi
