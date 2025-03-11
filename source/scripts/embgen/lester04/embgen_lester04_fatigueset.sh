#!/bin/sh

PYTHON_SCRIPT="main.py"
METHOD=lester04
TASK=embgen
DATASET=fatigueset

GPU=5          # Set to your available GPU
NOTE=20250208  # Experiment ID for tracking


run_exp() {
  local modals="$1"
  local devices="$2"
  local ws="$3"
  local test_labels="$4"
  local seed="$5"
  local gpu="$6"
  echo "Running experiment with modals: ${modals}, ws: ${ws}, test_labels: ${test_labels}, seed: ${seed} and gpu: ${gpu}"
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

# ================================== #
# ======== Main Experiments  ======= #
# ================================== #
for seed in 42 43 44 45 46
do
  run_exp "acc"          "l r w h"     8    "all"       $seed         $GPU  &
done
wait

# ==================================== #
# ===== Device-specific Analysis ===== #
# ==================================== #
for seed in 42 43 44 45 46
do
  run_exp "acc"          "l r"        8    "all"       $seed     0    &    # (1,2)
  run_exp "acc"          "l w"        8    "all"       $seed     1    &    # (1,3)
  run_exp "acc"          "l h"        8    "all"       $seed     2    &    # (1,4)
  run_exp "acc"          "r w"        8    "all"       $seed     3    &    # (2,3)
  run_exp "acc"          "r h"        8    "all"       $seed     4    &    # (2,4)
  run_exp "acc"          "w h"        8    "all"       $seed     5         # (3,4)
done
wait

for seed in 42 43 44 45 46
do
  run_exp "acc"          "l r w"      8    "all"       $seed     $GPU    &    # (1,2,3)
  run_exp "acc"          "l r h"      8    "all"       $seed     $GPU    &    # (1,2,4)
  run_exp "acc"          "l w h"      8    "all"       $seed     $GPU    &    # (1,3,4)
  run_exp "acc"          "r w h"      8    "all"       $seed     $GPU    &    # (2,3,4)
  # acc (1,2,3,4) already exists in the main experiments
done
wait