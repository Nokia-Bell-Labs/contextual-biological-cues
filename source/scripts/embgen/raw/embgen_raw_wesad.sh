#!/bin/sh

PYTHON_SCRIPT="main.py"
METHOD=raw
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
  echo "Running experiment with modals:${modals}, ws:${ws}, test_labels:${test_labels} and seed:${seed}"
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


for seed in 42 43 44 45 46
  do
    ## Unimodal
    run_exp "acc"        "c w"           8     "all"      $seed    $GPU  &
    run_exp "eda"        "c w"           20    "all"      $seed    $GPU  &
    run_exp "temp"       "c w"           20    "all"      $seed    $GPU  &
done
wait

for seed in 42 43 44 45 46
do
    ## Multimodal
    run_exp "acc eda"       "c w"        20     "all"     $seed    $GPU  &
    run_exp "acc temp"      "c w"        20     "all"     $seed    $GPU  &
    run_exp "eda temp"      "c w"        20     "all"     $seed    $GPU  &
    run_exp "acc eda temp"  "c w"        20     "all"     $seed    $GPU
done