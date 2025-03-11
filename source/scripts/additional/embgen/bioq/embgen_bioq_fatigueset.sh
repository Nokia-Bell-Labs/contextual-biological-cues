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


NOTE=20250208_label_specific # EmbGen note

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


# =================================================== #
# ======== Additional Label-Specific Analysis ======= #
# =================================================== #

# Labels: baseline, other, activity, fatigue, we consider all labels except "other"

for test_labels in "baseline" "activity" "fatigue"
  do
  for seed in 42 43 44 45 46
    do
      run_exp "acc"          "l r w h"     8    $test_labels       $seed     7   &
      run_exp "gyro"          "l r h"      8    $test_labels       $seed     2   &
      run_exp "ppg"           "l r"       20    $test_labels       $seed     3   &
    done
    wait
  done
  wait


for test_labels in "baseline" "activity" "fatigue"
  do
  for seed in 42 43 44 45 46
    do
      # Multimodal
      run_exp "acc gyro"      "l r"        8    $test_labels       $seed     4   &
      run_exp "acc gyro"      "l r h"      8    $test_labels       $seed     5   &
      run_exp "acc ppg"       "l r"       20    $test_labels       $seed     6   &
      run_exp "gyro ppg"      "l r"       20    $test_labels       $seed     7   &
      run_exp "acc gyro ppg"  "l r"       20    $test_labels       $seed     0   &
    done
    wait
  done
