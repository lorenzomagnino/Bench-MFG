#!/bin/bash
# Generate mean field and policy plots for every environment × algorithm.
# Grid dimensions are auto-detected by plot_single_run (FourRoomsAversion2D 11×11,
# KineticCongestion 5×5). Prerequisite: run collect_best_artifacts first.

set -e

ENVIRONMENTS=(
    ContractionGame
    FourRoomsAversion2D
    KineticCongestion
    LasryLionsChain
    MultipleEquilibriaGame
    NoInteractionChain
    RockPaperScissors
    SISEpidemic
)

ALGORITHMS=(
    DampedFP_damped
    DampedFP_fictitious_play
    DampedFP_pure
    OMD
    PI_boltzmann_policy_iteration
    PI_policy_iteration
    PI_smooth_policy_iteration
    PSO
)

for ENV in "${ENVIRONMENTS[@]}"; do
    echo "=== $ENV ==="
    for ALGO in "${ALGORITHMS[@]}"; do
        echo "  $ALGO"
        benchmfg plot single-run --env "$ENV" --algo "$ALGO" \
            2>&1 | sed 's/^/    /'
    done
    echo ""
done

echo "All mean field and policy plots generated."
