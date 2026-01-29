#!/bin/bash
# MF-Garnet DampedFP experiments

set -e

# Activate virtual environment if it exists
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
if [ -d "$PROJECT_ROOT/.venv" ]; then
    source "$PROJECT_ROOT/.venv/bin/activate"
    echo "Activated virtual environment: $PROJECT_ROOT/.venv"
fi

# Configuration
NUM_INSTANCES=10  # number of different MFG instances (and runs)
ALGORITHM_SEEDS=(42 10 111 1032 999 1234 5678 9012 3456 7890)  # one seed per instance
NUM_STATES=25
NUM_ACTIONS=10
BRANCHING_FACTOR=10  # number of nonzero next-states in P0 for each (s,a)
DYNAMICS_STRUCTURE="multiplicative"      # "additive" | "multiplicative"
REWARD_STRUCTURE="additive"  # "additive" | "multiplicative"

# DampedFP hyperparameters
DAMPEDFP_LAMBDA=("damped" "pure" "fictitious_play")
DAMPEDFP_CONSTANT=0.2

echo "=========================================="
echo "MF-Garnet DampedFP Experiments"
echo "=========================================="
echo "  - Instances/Runs (X): ${NUM_INSTANCES}"
echo "  - States: ${NUM_STATES}, Actions: ${NUM_ACTIONS}"
echo "  - Dynamics: ${DYNAMICS_STRUCTURE}, Reward: ${REWARD_STRUCTURE}"
echo "  - Each run pairs (garnet_seed, algo_seed)"
echo "=========================================="
echo ""

# Check that we have enough algorithm seeds
if [ ${#ALGORITHM_SEEDS[@]} -lt ${NUM_INSTANCES} ]; then
    echo "Error: Not enough algorithm seeds. Need ${NUM_INSTANCES}, got ${#ALGORITHM_SEEDS[@]}"
    exit 1
fi
echo ""

cd "$PROJECT_ROOT"

for i in $(seq 0 $((NUM_INSTANCES - 1))); do
    GARNET_SEED=$i
    ALGO_SEED=${ALGORITHM_SEEDS[$i]}
    for lambda in "${DAMPEDFP_LAMBDA[@]}"; do
        echo "Run $((i+1))/${NUM_INSTANCES}: garnet_seed=${GARNET_SEED}, algo_seed=${ALGO_SEED}, lambda=${lambda}"
        python main.py \
            environment.num_states=${NUM_STATES} \
            environment.num_actions=${NUM_ACTIONS} \
            environment.reward.mfgarnet.branching_factor=${BRANCHING_FACTOR} \
            environment.reward.mfgarnet.dynamics_structure=${DYNAMICS_STRUCTURE} \
            environment.reward.mfgarnet.reward_structure=${REWARD_STRUCTURE} \
            environment.reward.mfgarnet.seed=${GARNET_SEED} \
            experiment.random_seed=${ALGO_SEED} \
            algorithm.dampedfp.lambda_schedule=${lambda} \
            algorithm.dampedfp.damped_constant=${DAMPEDFP_CONSTANT} \
            experiment.name="garnet_dampedfp_${DYNAMICS_STRUCTURE}_${REWARD_STRUCTURE}_${lambda}"
    done
done

echo "=========================================="
echo "DampedFP experiments completed."
echo "=========================================="

