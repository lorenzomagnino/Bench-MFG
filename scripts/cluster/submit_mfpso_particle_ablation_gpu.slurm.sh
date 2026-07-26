#!/usr/bin/env bash
#SBATCH --job-name=mfpso-pabl-gpu
#SBATCH --output=outputs/mfpso_particle_ablation_four_rooms/slurm_logs/gpu-%A_%a.out
#SBATCH --error=outputs/mfpso_particle_ablation_four_rooms/slurm_logs/gpu-%A_%a.err
#SBATCH --account=torch_pr_643_cds
#SBATCH --partition=a100_cds
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-19
#
# NOTE: partition=rtx6000 is not valid for torch_pr_* accounts on this cluster
# ("GPU job setup is not valid"). a100_cds is the working GPU partition for
# torch_pr_643_cds. Actual GPU name is recorded in manifest.jsonl / nvidia-smi.

# 5 particles × 4 seeds = 20 cells
source scripts/cluster/slurm_init.sh scripts/cluster/submit_mfpso_particle_ablation_gpu.slurm.sh

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
mkdir -p outputs/mfpso_particle_ablation_four_rooms/slurm_logs

echo "=== MF-PSO particle ablation GPU (a100_cds; RTX6000 unavailable for account) ==="
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-} SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-}"
echo "host=$(hostname)"
nvidia-smi -L || true
python - <<'PY'
import jax
print("python ok")
print("jax", jax.__version__)
print("devices", jax.devices())
print("backend", jax.default_backend())
PY

bash scripts/ablation/run_mfpso_particle_ablation.sh \
  --device cuda \
  --device-dir gpu_a100 \
  --array-task-id "${SLURM_ARRAY_TASK_ID}"
