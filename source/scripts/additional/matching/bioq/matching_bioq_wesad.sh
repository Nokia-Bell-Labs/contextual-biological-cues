#!/bin/sh

PYTHON_SCRIPT="main.py"
METHOD=bioq
TASK=matching
DATASET=wesad

NOTE=20250208_label_specific  # EmbGen note
MNOTE=20250208_label_specific # Matching note

# Embgen hparams
CON_LR=0.001
CON_WD=0.001
EM_OP=adam
EM_EP=400

# Matching hparams
M_LR=0.001
M_WD=0.001
M_EP=200
M_OP=adam


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
                        --con_lr $CON_LR \
                        --con_wd $CON_WD \
                        --em_op $EM_OP \
                        --em_ep $EM_EP \
                        --m_lr $M_LR \
                        --m_wd $M_WD \
                        --m_ep $M_EP \
                        --m_op $M_OP \
                        --seed ${seed} \
                        --wandb # Disable if you don't have wandb
}

# =================================================== #
# ======== Additional Label-Specific Analysis ======= #
# =================================================== #

for test_labels in 0 1 2 # 0: baseline, 1: stress (public speaking, mental arithmetic), 2: amusement (watching video clips)
  do
  for seed in 42 43 44 45 46
    do
      ## Unimodal
      run_exp "acc"        "c w"           8     $test_labels      $seed    1  &
      run_exp "eda"        "c w"           20    $test_labels      $seed    2  &
      run_exp "temp"       "c w"           20    $test_labels      $seed    3  &

      ## Multimodal
      run_exp "acc eda"       "c w"        20     $test_labels     $seed    4  &
      run_exp "acc temp"      "c w"        20     $test_labels     $seed    5  &
      run_exp "eda temp"      "c w"        20     $test_labels     $seed    6  &
      run_exp "acc eda temp"  "c w"        20     $test_labels     $seed    0  &
    done
    wait
  done



