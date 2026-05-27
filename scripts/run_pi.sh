#!/bin/bash
# Policy Iteration experiments (standard, smooth, boltzmann)

set -e

ENVIRONMENT="LasryLionsChain"
# ENVIRONMENT="NoInteractionChain"
# ENVIRONMENT="FourRoomsAversion2D"
# ENVIRONMENT="RockPaperScissors"
# ENVIRONMENT="SISEpidemic"
# ENVIRONMENT="KineticCongestion"
# ENVIRONMENT="MultipleEquilibriaGame"
# ENVIRONMENT="ContractionGame"

echo "Running Policy Iteration sweep..."
benchmfg sweep algorithm=pi environment=lasry_lions_chain \
  experiment.name="policy_iteration_sweep" \
  experiment.random_seed=42,10,111,1032 \
  algorithm.pi.variant=policy_iteration
echo "Generating Policy Iteration sweep plots..."
benchmfg plot sweep "$ENVIRONMENT" PI_policy_iteration

echo "Running Smooth PI sweep..."
benchmfg sweep algorithm=pi environment=lasry_lions_chain \
  experiment.name="smooth_pi_sweep" \
  experiment.random_seed=42,10,111,1032 \
  algorithm.pi.variant=smooth_policy_iteration \
  algorithm.pi.damped_constant=0.1,0.5,0.8
echo "Generating Smooth PI sweep plots..."
benchmfg plot sweep "$ENVIRONMENT" PI_smooth_policy_iteration

echo "Running Boltzmann PI sweep..."
benchmfg sweep algorithm=pi environment=lasry_lions_chain \
  experiment.name="boltzmann_pi_sweep" \
  experiment.random_seed=42,10,111,1032 \
  algorithm.pi.variant=boltzmann_policy_iteration \
  algorithm.pi.damped_constant=0.1,0.5,0.8 \
  algorithm.pi.temperature=0.2,0.5,0.8
echo "Generating Boltzmann PI sweep plots..."
benchmfg plot sweep "$ENVIRONMENT" PI_boltzmann_policy_iteration

echo "All Policy Iteration experiments completed!"
