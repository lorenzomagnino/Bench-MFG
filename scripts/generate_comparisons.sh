#!/bin/bash
# Generate comparison plots (exploitability, exploitability log, runtime) for all environments.
# Prerequisite: run plot_sweep.py for each algorithm + environment first.

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

for ENV in "${ENVIRONMENTS[@]}"; do
    echo "=== Generating comparison plots for: $ENV ==="
    benchmfg plot compare "$ENV"
    echo ""
done

echo "=== Collecting best-model NPZ artifacts into results/ ==="
python -m benchmfg.utility.collect_best_artifacts

echo "All comparison plots generated."
