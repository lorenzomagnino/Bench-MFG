#!/bin/bash
# Online Mirror Descent experiments

set -e

echo "Running OMD sweep..."
python main.py -m \
  experiment.name="omd_sweep" \
  experiment.random_seed=42,10,111,1032 \
  algorithm.omd.learning_rate=0.5,0.05,0.005 \
  algorithm.omd.temperature=0.2,0.5,0.8

echo "All OMD experiments completed!"
