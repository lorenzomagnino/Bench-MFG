#!/bin/bash
# Online Mirror Descent experiments

set -e

ENVIRONMENT="LasryLionsChain"
# ENVIRONMENT="NoInteractionChain"
# ENVIRONMENT="FourRoomsAversion2D"
# ENVIRONMENT="RockPaperScissors"
# ENVIRONMENT="SISEpidemic"
# ENVIRONMENT="KineticCongestion"
# ENVIRONMENT="MultipleEquilibriaGame"
# ENVIRONMENT="ContractionGame"

echo "Running OMD sweep..."
benchmfg sweep algorithm=omd environment=lasry_lions_chain \
  experiment.name="omd_sweep" \
  experiment.random_seed=42,10,111,1032 \
  algorithm.omd.learning_rate=0.5,0.05,0.005 \
  algorithm.omd.temperature=0.2,0.5,0.8

echo "Generating OMD sweep plots..."
benchmfg plot sweep "$ENVIRONMENT" OMD

echo "All OMD experiments completed!"
