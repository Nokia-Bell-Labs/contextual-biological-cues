#!/bin/sh

PYTHON_SCRIPT="main.py"
METHOD=cornelius12
TASK=embgen
DATASET=wesad

GPU=5          # Set to your available GPU
NOTE=20250208  # Experiment ID for tracking

run_exp() {
  local modals="$1"
  local devices="$2"
  local ws="$3"
  local test_labels="$4"
  local seed="$5"
  local gpu="$6"
  echo "Running experiment with modals:${modals}, ws:${ws}, test_labels:${test_labels}, seed:${seed} and gpu:${gpu}"
  python $PYTHON_SCRIPT --method $METHOD \
                        --task $TASK \
                        --note $NOTE \
                        --dataset $DATASET \
                        --modals ${modals} \
                        --devices ${devices} \
                        --ws ${ws} \
                        --test_labels ${test_labels} \
                        --gpu ${gpu} \
                        --seed ${seed} \
                        --wandb # Disable if you don't have wandb
}

# For Cornelius12, we consider only acc, as it works with only acc data
for seed in 42 43 44 45 46
do
  run_exp  "acc"        "c w"           8     "all"     $seed    $GPU   &
done