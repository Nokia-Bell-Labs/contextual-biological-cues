#!/bin/sh

PYTHON_SCRIPT="main.py"
METHOD=bioq
TASK=embgen
DATASET=fatigueset

# Hyperparameters for FatigueSet
CON_LR=0.001
CON_WD=0.001
EM_OP=adam
EM_EP=400


NOTE=20250208 # EmbGen note

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
                        --em_ep $EM_EP \
                        --em_op $EM_OP \
                        --con_lr $CON_LR \
                        --con_wd $CON_WD \
                        --seed ${seed} \
                        --wandb # Disable if you don't have wandb
}


# ================================== #
# ======== Main Experiments  ======= #
# ================================== #
for seed in 42 43 44 45 46
do
  run_exp "acc"          "l r w h"     8    "all"       $seed     7   &
  run_exp "gyro"          "l r h"      8    "all"       $seed     2   &
  run_exp "ppg"           "l r"       20    "all"       $seed     3   &
done
wait


for seed in 42 43 44 45 46
do
  # Multimodal
  run_exp "acc gyro"      "l r"        8    "all"       $seed     4   &
  run_exp "acc gyro"      "l r h"      8    "all"       $seed     5   &
  run_exp "acc ppg"       "l r"       20    "all"       $seed     6   &
  run_exp "gyro ppg"      "l r"       20    "all"       $seed     7   &
  run_exp "acc gyro ppg"  "l r"       20    "all"       $seed     0   &
done
wait


# ==================================== #
# ===== Device-specific Analysis ===== #
# ==================================== #
for seed in 42 43 44 45 46
do
  run_exp "gyro"        "l r"      8    "all"       $seed     0    &    # (1,2,4)
  run_exp "gyro"        "l h"      8    "all"       $seed     1    &    # (1,3,4)
  run_exp "gyro"        "r h"      8    "all"       $seed     2    &    # (2,3,4)
done
wait

for seed in 42 43 44 45 46
do
  run_exp "acc"          "l r"        8    "all"       $seed     0    &    # (1,2)
  run_exp "acc"          "l w"        8    "all"       $seed     1    &    # (1,3)
  run_exp "acc"          "l h"        8    "all"       $seed     2    &    # (1,4)
  run_exp "acc"          "r w"        8    "all"       $seed     3    &    # (2,3)
  run_exp "acc"          "r h"        8    "all"       $seed     4    &    # (2,4)
  run_exp "acc"          "w h"        8    "all"       $seed     5    &    # (3,4)
done
wait

for seed in 42 43 44 45 46
do
  run_exp "acc"          "l r w"      8    "all"       $seed     0    &    # (1,2,3)
  run_exp "acc"          "l r h"      8    "all"       $seed     1    &    # (1,2,4)
  run_exp "acc"          "l w h"      8    "all"       $seed     2    &    # (1,3,4)
  run_exp "acc"          "r w h"      8    "all"       $seed     3    &    # (2,3,4)
  # acc (1,2,3,4) already exists in the main experiments
done
wait