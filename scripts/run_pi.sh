#!/bin/bash
# Policy Iteration experiments (standard, smooth, boltzmann)

set -e

echo "Running Policy Iteration sweep..."
python main.py -m \
  experiment.name="policy_iteration_sweep" \
  experiment.random_seed=42,10,111,1032 \
  algorithm.pi.variant=policy_iteration

echo "Running Smooth PI sweep..."
python main.py -m \
  experiment.name="smooth_pi_sweep" \
  experiment.random_seed=42,10,111,1032 \
  algorithm.pi.variant=smooth_policy_iteration \
  algorithm.pi.damped_constant=0.1,0.5,0.8

echo "Running Boltzmann PI sweep..."
python main.py -m \
  experiment.name="boltzmann_pi_sweep" \
  experiment.random_seed=42,10,111,1032 \
  algorithm.pi.variant=boltzmann_policy_iteration \
  algorithm.pi.damped_constant=0.1,0.5,0.8 \
  algorithm.pi.temperature=0.2,0.5,0.8

echo "All Policy Iteration experiments completed!"
