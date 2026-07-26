#!/usr/bin/env bash
#SBATCH --job-name=mfpso-pabl-cpu
#SBATCH --output=outputs/mfpso_particle_ablation_four_rooms/slurm_logs/cpu-%A_%a.out
#SBATCH --error=outputs/mfpso_particle_ablation_four_rooms/slurm_logs/cpu-%A_%a.err
#SBATCH --account=torch_pr_643_cds
#SBATCH --partition=cs
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --array=0-19

# 5 particles × 4 seeds = 20 cells (CPU; no GPU)
# SINGULARITY_USE_NV=0 avoids requiring nvidia inside CPU nodes.
# partition=cs matches prior successful CPU jobs for this account
# (cpu_short rejected this scripted job with "CPU job setup is not valid").
export SINGULARITY_USE_NV=0
source scripts/cluster/slurm_init.sh scripts/cluster/submit_mfpso_particle_ablation_cpu.slurm.sh

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
mkdir -p outputs/mfpso_particle_ablation_four_rooms/slurm_logs

echo "=== MF-PSO particle ablation CPU ==="
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-} SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-}"
echo "host=$(hostname)"
python - <<'PY'
import jax
print("python ok")
print("jax", jax.__version__)
print("devices", jax.devices())
print("backend", jax.default_backend())
PY

bash scripts/ablation/run_mfpso_particle_ablation.sh \
  --device cpu \
  --array-task-id "${SLURM_ARRAY_TASK_ID}"
