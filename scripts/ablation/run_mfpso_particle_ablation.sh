#!/usr/bin/env bash
# MF-PSO particle-number ablation launcher (Four Rooms).
#
# Usage:
#   bash scripts/ablation/run_mfpso_particle_ablation.sh --device cuda
#   bash scripts/ablation/run_mfpso_particle_ablation.sh --device cpu --particles 50 --seeds 42
#   bash scripts/ablation/run_mfpso_particle_ablation.sh --device cuda --array-task-id 0
#
# With --array-task-id N, runs the N-th cell in the Cartesian product
# particles × seeds (0-indexed). Used by SLURM array jobs.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

# Scratch-only caches
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/scratch/js12556/.cache/pip}"
export TMPDIR="${TMPDIR:-/scratch/js12556/tmp}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/scratch/js12556/.cache}"
mkdir -p "${PIP_CACHE_DIR}" "${TMPDIR}" "${XDG_CACHE_HOME}"

DEVICE="cuda"
DEVICE_DIR_OVERRIDE=""
PARTICLES_CSV="50,100,200,400,800"
SEEDS_CSV="42,10,111,1032"
ARRAY_TASK_ID=""
FORCE_RERUN=0
ABLATION_ROOT="${ABLATION_ROOT:-${REPO_ROOT}/outputs/mfpso_particle_ablation_four_rooms}"

NUM_ITERATIONS=150
TEMP=0.7
W=0.3
C1=0.3
C2=1.2
EXP_NAME="pso_particle_ablation"

usage() {
  cat <<'EOF'
Usage: run_mfpso_particle_ablation.sh --device {cpu|cuda} [options]

Options:
  --device {cpu|cuda}       Required compute backend (cuda must be exact string)
  --device-dir NAME         Output subdir under ablation root (default: cpu or gpu_a100)
  --particles CSV           Particle counts (default: 50,100,200,400,800)
  --seeds CSV               Random seeds (default: 42,10,111,1032)
  --array-task-id N         Run only the N-th (P,seed) cell
  --ablation-root DIR       Output root (default: outputs/mfpso_particle_ablation_four_rooms)
  --force                   Re-run even if metrics.npz already exists
  -h, --help                Show help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --device) DEVICE="$2"; shift 2 ;;
    --device-dir) DEVICE_DIR_OVERRIDE="$2"; shift 2 ;;
    --particles) PARTICLES_CSV="$2"; shift 2 ;;
    --seeds) SEEDS_CSV="$2"; shift 2 ;;
    --array-task-id) ARRAY_TASK_ID="$2"; shift 2 ;;
    --ablation-root) ABLATION_ROOT="$2"; shift 2 ;;
    --force) FORCE_RERUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ "${DEVICE}" != "cpu" && "${DEVICE}" != "cuda" ]]; then
  echo "ERROR: --device must be exactly 'cpu' or 'cuda' (got '${DEVICE}')" >&2
  exit 1
fi

IFS=',' read -r -a PARTICLES <<< "${PARTICLES_CSV}"
IFS=',' read -r -a SEEDS <<< "${SEEDS_CSV}"

CELLS=()
for P in "${PARTICLES[@]}"; do
  for S in "${SEEDS[@]}"; do
    CELLS+=("${P}:${S}")
  done
done

if [[ -n "${ARRAY_TASK_ID}" ]]; then
  if [[ "${ARRAY_TASK_ID}" -lt 0 || "${ARRAY_TASK_ID}" -ge "${#CELLS[@]}" ]]; then
    echo "ERROR: --array-task-id ${ARRAY_TASK_ID} out of range [0, $((${#CELLS[@]} - 1))]" >&2
    exit 1
  fi
  CELLS=("${CELLS[${ARRAY_TASK_ID}]}")
fi

if [[ "${DEVICE}" == "cuda" ]]; then
  DEVICE_DIR="${DEVICE_DIR_OVERRIDE:-gpu_a100}"
  HYDRA_DEVICE="cuda"
else
  DEVICE_DIR="${DEVICE_DIR_OVERRIDE:-cpu}"
  HYDRA_DEVICE="cpu"
fi

OUT_ROOT="${ABLATION_ROOT}/${DEVICE_DIR}"
MANIFEST="${ABLATION_ROOT}/manifest.jsonl"
mkdir -p "${OUT_ROOT}" "${ABLATION_ROOT}"
export BENCH_MFG_OUTPUT_ROOT="${OUT_ROOT}"
export BENCHMFG_OUTPUT_ROOT="${OUT_ROOT}"

HOSTNAME_STR="$(hostname)"
GPU_NAME=""
if [[ "${DEVICE}" == "cuda" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | sed 's/^ *//;s/ *$//')"
    nvidia-smi -L || true
  fi
fi

echo "=== MF-PSO particle ablation ==="
echo "repo=${REPO_ROOT}"
echo "device=${HYDRA_DEVICE} device_dir=${DEVICE_DIR}"
echo "out_root=${OUT_ROOT}"
echo "manifest=${MANIFEST}"
echo "hostname=${HOSTNAME_STR} gpu=${GPU_NAME:-n/a}"
echo "cells=${#CELLS[@]}  K=${NUM_ITERATIONS} HPs=temp${TEMP},w${W},c1${C1},c2${C2}"

if ! command -v benchmfg >/dev/null 2>&1; then
  export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
  BENCHMFG_CMD=(python -m benchmfg.cli)
else
  BENCHMFG_CMD=(benchmfg)
fi

append_manifest() {
  # Append one JSON line atomically via temp+flock-friendly redirect.
  local line="$1"
  (
    flock -w 30 9 || true
    printf '%s\n' "${line}" >> "${MANIFEST}"
  ) 9>"${MANIFEST}.lock"
}

is_cell_done() {
  local P="$1" SEED="$2"
  local pattern="${OUT_ROOT}/FourRoomsAversion2D/PSO/seed_${SEED}/pso_particle_ablation_"*"_particles${P}"
  local latest
  # shellcheck disable=SC2086
  latest="$(ls -1d ${pattern}/*/metrics.npz 2>/dev/null | sort | tail -1 || true)"
  if [[ -n "${latest}" && -f "${latest}" ]]; then
    echo "${latest%/*}"
    return 0
  fi
  return 1
}

find_latest_run_dir() {
  local P="$1" SEED="$2"
  local pattern="${OUT_ROOT}/FourRoomsAversion2D/PSO/seed_${SEED}/pso_particle_ablation_"*"_particles${P}"
  # shellcheck disable=SC2086
  ls -1d ${pattern}/*/ 2>/dev/null | sort | tail -1 || true
}

for cell in "${CELLS[@]}"; do
  P="${cell%%:*}"
  SEED="${cell##*:}"
  echo ""
  echo "--- cell P=${P} seed=${SEED} device=${HYDRA_DEVICE} ---"

  if [[ "${FORCE_RERUN}" -eq 0 ]]; then
    if done_dir="$(is_cell_done "${P}" "${SEED}")"; then
      echo "SKIP (metrics.npz exists): ${done_dir}"
      # Ensure manifest has a completed entry (idempotent-ish; aggregator dedups by path)
      runtime_s="$(python - <<PY
import numpy as np
d=np.load("${done_dir}/metrics.npz")
print(float(d["runtime_s"]))
PY
)"
      final_e="$(python - <<PY
import numpy as np
d=np.load("${done_dir}/exploitabilities.npz")
e=d["exploitabilities"]
print(float(e[-1]))
PY
)"
      append_manifest "$(python - <<PY
import json
print(json.dumps({
  "status": "skipped_existing",
  "num_particles": int("${P}"),
  "seed": int("${SEED}"),
  "device": "${HYDRA_DEVICE}",
  "device_dir": "${DEVICE_DIR}",
  "hostname": "${HOSTNAME_STR}",
  "gpu_name": "${GPU_NAME}",
  "runtime_s": float("${runtime_s}"),
  "final_exploitability": float("${final_e}"),
  "path": "${done_dir}",
  "num_iterations": int("${NUM_ITERATIONS}"),
}))
PY
)"
      continue
    fi
  fi

  set +e
  "${BENCHMFG_CMD[@]}" train \
    algorithm=pso \
    environment=four_rooms_obstacles \
    device="${HYDRA_DEVICE}" \
    experiment.name="${EXP_NAME}" \
    experiment.random_seed="${SEED}" \
    experiment.is_saved=true \
    algorithm.pso.num_particles="${P}" \
    algorithm.pso.num_iterations="${NUM_ITERATIONS}" \
    algorithm.pso.temperature="${TEMP}" \
    algorithm.pso.w="${W}" \
    algorithm.pso.c1="${C1}" \
    algorithm.pso.c2="${C2}"
  train_rc=$?
  set -e

  run_dir="$(find_latest_run_dir "${P}" "${SEED}")"
  # Treat metrics.npz as success even if post-train plotting exits non-zero.
  if [[ -z "${run_dir}" || ! -f "${run_dir}/metrics.npz" || ! -f "${run_dir}/exploitabilities.npz" ]]; then
    echo "FAIL: train_rc=${train_rc} run_dir=${run_dir}"
    append_manifest "$(python - <<PY
import json
print(json.dumps({
  "status": "failed",
  "num_particles": int("${P}"),
  "seed": int("${SEED}"),
  "device": "${HYDRA_DEVICE}",
  "device_dir": "${DEVICE_DIR}",
  "hostname": "${HOSTNAME_STR}",
  "gpu_name": "${GPU_NAME}",
  "runtime_s": None,
  "final_exploitability": None,
  "path": "${run_dir}",
  "exit_code": int("${train_rc}"),
  "num_iterations": int("${NUM_ITERATIONS}"),
}))
PY
)"
    exit 1
  fi
  if [[ "${train_rc}" -ne 0 ]]; then
    echo "WARN: train exited ${train_rc} but metrics.npz exists; treating as success"
  fi

  runtime_s="$(python - <<PY
import numpy as np
d=np.load("${run_dir}/metrics.npz")
print(float(d["runtime_s"]))
PY
)"
  final_e="$(python - <<PY
import numpy as np
d=np.load("${run_dir}/exploitabilities.npz")
print(float(d["exploitabilities"][-1]))
PY
)"

  run_dir_clean="${run_dir%/}"
  append_manifest "$(python - <<PY
import json
print(json.dumps({
  "status": "completed",
  "num_particles": int("${P}"),
  "seed": int("${SEED}"),
  "device": "${HYDRA_DEVICE}",
  "device_dir": "${DEVICE_DIR}",
  "hostname": "${HOSTNAME_STR}",
  "gpu_name": "${GPU_NAME}",
  "runtime_s": float("${runtime_s}"),
  "final_exploitability": float("${final_e}"),
  "path": "${run_dir_clean}",
  "num_iterations": int("${NUM_ITERATIONS}"),
}))
PY
)"
  echo "OK P=${P} seed=${SEED} runtime_s=${runtime_s} final_E=${final_e}"
  echo "saved=${run_dir_clean}"
done

echo ""
echo "=== ablation launcher finished ==="
