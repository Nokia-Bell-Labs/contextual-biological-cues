#!/bin/sh

PYTHON_SCRIPT="main.py"
METHOD=bioq
TASK=embgen
DATASET=wesad

# Hyperparameters for WESAD
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
  echo "Running experiment with modals: ${modals}, ws: ${ws}, test_labels: ${test_labels}"
  python $PYTHON_SCRIPT --method $METHOD \
                        --task $TASK \
                        --note $NOTE \
                        --dataset $DATASET \
                        --modals ${modals} \
                        --devices ${devices} \
                        --ws ${ws} \
                        --test_labels ${test_labels} \
                        --gpu ${gpu} \
                        --em_op $EM_OP \
                        --em_ep $EM_EP \
                        --con_lr $CON_LR \
                        --con_wd $CON_WD \
                        --seed ${seed} \
                        --wandb # Disable if you don't have wandb
}

for seed in 42 43 44 45 46
  do
    ## Unimodal
    run_exp "acc"        "c w"           8     "all"      $seed    1  &
    run_exp "eda"        "c w"           20    "all"      $seed    2  &
    run_exp "temp"       "c w"           20    "all"      $seed    3  &

    ## Multimodal
    run_exp "acc eda"       "c w"        20     "all"     $seed    4  &
    run_exp "acc temp"      "c w"        20     "all"     $seed    5  &
    run_exp "eda temp"      "c w"        20     "all"     $seed    6  &
    run_exp "acc eda temp"  "c w"        20     "all"     $seed    0  &
done






