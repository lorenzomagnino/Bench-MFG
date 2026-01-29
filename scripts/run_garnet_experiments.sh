#!/bin/bash
# MF-Garnet experiments following the specified protocol:
# 1. Fix class (dynamics_structure, reward_structure)
# 2. Fix model parameters (num_states, num_actions) but NOT coefficients (random via seed)
# 3. Fix algorithm hyperparameters
# 4. Repeat X times: each run pairs (garnet_seed_i, algo_seed_i)
# 5. Aggregate: mean and std of final exploitability over X runs for each algorithm

set -e

# Activate virtual environment if it exists
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
if [ -d "$PROJECT_ROOT/.venv" ]; then
    source "$PROJECT_ROOT/.venv/bin/activate"
    echo "Activated virtual environment: $PROJECT_ROOT/.venv"
fi

# Configuration
NUM_INSTANCES=10  # X: number of different MFG instances (and runs)
ALGORITHM_SEEDS=(42 10 111 1032 999 1234 5678 9012 3456 7890)  # Algorithm seeds (one per instance)
NUM_STATES=5
NUM_ACTIONS=5
BRANCHING_FACTOR=5  # number of nonzero next-states in P0 for each (s,a)
DYNAMICS_STRUCTURE="additive"  # "additive" | "multiplicative"
REWARD_STRUCTURE="multiplicative"  # "additive" | "multiplicative"

# Algorithm hyperparameters (fix these - can be adjusted)
PSO_TEMP=0.2
PSO_W=0.4
PSO_C1=0.5
PSO_C2=1.5

DAMPEDFP_LAMBDA="damped"
DAMPEDFP_CONSTANT=0.2

PI_VARIANT="policy_iteration"
PI_DAMPED=0.4

OMD_LR=0.005
OMD_TEMP=0.2

echo "=========================================="
echo "MF-Garnet Experiments"
echo "=========================================="
echo "Configuration:"
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

# Function to run experiments for one algorithm
run_algorithm() {
    local algo_name=$1
    local experiment_suffix=$2

    echo "=========================================="
    echo "Running ${algo_name} experiments..."
    echo "=========================================="

    cd "$PROJECT_ROOT"

    # Run X instances, each with a paired (garnet_seed, algo_seed)
    for i in $(seq 0 $((NUM_INSTANCES - 1))); do
        GARNET_SEED=$i
        ALGO_SEED=${ALGORITHM_SEEDS[$i]}

        echo "  Run $((i+1))/${NUM_INSTANCES}: garnet_seed=${GARNET_SEED}, algo_seed=${ALGO_SEED}"

        case $algo_name in
            "pso")
                python main.py \
                    environment.num_states=${NUM_STATES} \
                    environment.num_actions=${NUM_ACTIONS} \
                    environment.reward.mfgarnet.branching_factor=${BRANCHING_FACTOR} \
                    environment.reward.mfgarnet.dynamics_structure=${DYNAMICS_STRUCTURE} \
                    environment.reward.mfgarnet.reward_structure=${REWARD_STRUCTURE} \
                    environment.reward.mfgarnet.seed=${GARNET_SEED} \
                    experiment.random_seed=${ALGO_SEED} \
                    algorithm.pso.temperature=${PSO_TEMP} \
                    algorithm.pso.w=${PSO_W} \
                    algorithm.pso.c1=${PSO_C1} \
                    algorithm.pso.c2=${PSO_C2} \
                    experiment.name="garnet_${experiment_suffix}"
                ;;
            "dampedfp")
                python main.py \
                    environment.num_states=${NUM_STATES} \
                    environment.num_actions=${NUM_ACTIONS} \
                    environment.reward.mfgarnet.branching_factor=${BRANCHING_FACTOR} \
                    environment.reward.mfgarnet.dynamics_structure=${DYNAMICS_STRUCTURE} \
                    environment.reward.mfgarnet.reward_structure=${REWARD_STRUCTURE} \
                    environment.reward.mfgarnet.seed=${GARNET_SEED} \
                    experiment.random_seed=${ALGO_SEED} \
                    algorithm.dampedfp.lambda_schedule=${DAMPEDFP_LAMBDA} \
                    algorithm.dampedfp.damped_constant=${DAMPEDFP_CONSTANT} \
                    experiment.name="garnet_${experiment_suffix}"
                ;;
            "pi")
                python main.py \
                    environment.num_states=${NUM_STATES} \
                    environment.num_actions=${NUM_ACTIONS} \
                    environment.reward.mfgarnet.branching_factor=${BRANCHING_FACTOR} \
                    environment.reward.mfgarnet.dynamics_structure=${DYNAMICS_STRUCTURE} \
                    environment.reward.mfgarnet.reward_structure=${REWARD_STRUCTURE} \
                    environment.reward.mfgarnet.seed=${GARNET_SEED} \
                    experiment.random_seed=${ALGO_SEED} \
                    algorithm.pi.variant=${PI_VARIANT} \
                    algorithm.pi.damped_constant=${PI_DAMPED} \
                    experiment.name="garnet_${experiment_suffix}"
                ;;
            "omd")
                python main.py \
                    environment.num_states=${NUM_STATES} \
                    environment.num_actions=${NUM_ACTIONS} \
                    environment.reward.mfgarnet.branching_factor=${BRANCHING_FACTOR} \
                    environment.reward.mfgarnet.dynamics_structure=${DYNAMICS_STRUCTURE} \
                    environment.reward.mfgarnet.reward_structure=${REWARD_STRUCTURE} \
                    environment.reward.mfgarnet.seed=${GARNET_SEED} \
                    experiment.random_seed=${ALGO_SEED} \
                    algorithm.omd.learning_rate=${OMD_LR} \
                    algorithm.omd.temperature=${OMD_TEMP} \
                    experiment.name="garnet_${experiment_suffix}"
                ;;
            *)
                echo "Unknown algorithm: ${algo_name}"
                exit 1
                ;;
        esac
    done

    echo "✓ ${algo_name} experiments completed!"
    echo ""
}

run_algorithm "pso" "pso_${DYNAMICS_STRUCTURE}_${REWARD_STRUCTURE}"
run_algorithm "dampedfp" "dampedfp_${DYNAMICS_STRUCTURE}_${REWARD_STRUCTURE}"
run_algorithm "pi" "pi_${DYNAMICS_STRUCTURE}_${REWARD_STRUCTURE}"
run_algorithm "omd" "omd_${DYNAMICS_STRUCTURE}_${REWARD_STRUCTURE}"

echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
