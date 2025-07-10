#!/bin/bash

# This script runs a sweep over the number of samples for the DiBS experiment.

echo "Starting sample size sweep..."

for N_SAMPLES in $(seq 100 100 1000)
do
  echo "-----------------------------------------------------"
  echo "Running experiment with num_samples = $N_SAMPLES"
  echo "-----------------------------------------------------"
  
  conda run -n dibs_env python ../debug/dibs_vectorised.py --num_samples $N_SAMPLES
  
  if [ $? -ne 0 ]; then
    echo "Experiment with num_samples = $N_SAMPLES failed. Aborting sweep."
    exit 1
  fi
done

echo "Sample size sweep finished successfully."
