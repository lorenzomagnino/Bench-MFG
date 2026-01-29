#!/bin/bash
# Damped Fixed Point experiments (damped, pure, fictitious play)

set -e

echo "Running Damped FP sweep..."
python main.py -m \
  experiment.name="damped_sweep" \
  experiment.random_seed=42,10,111,1032 \
  algorithm.dampedfp.lambda_schedule=damped \
  algorithm.dampedfp.damped_constant=0.1,0.5,0.8

echo "Running Pure FP sweep..."
python main.py -m \
  experiment.name="pure_fp_sweep" \
  experiment.random_seed=42,10,111,1032 \
  algorithm.dampedfp.lambda_schedule=pure

echo "Running Fictitious Play sweep..."
python main.py -m \
  experiment.name="fplay_sweep" \
  experiment.random_seed=42,10,111,1032 \
  algorithm.dampedfp.lambda_schedule=fictitious_play

echo "All Damped FP experiments completed!"
