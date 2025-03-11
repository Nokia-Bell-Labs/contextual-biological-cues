#!/bin/sh

PYTHON_SCRIPT="02_feature_engin_fatigueset.py"
CONFIG_FILE="../../dataset_paths.json"
ALIGNED_FILE="fatigueset_filtered_100Hz.csv"


# Function to run the parser with specific modals and window size
run_parser() {
    local modals="$1"
    local ws="$2"
    echo "Running parser with modals: ${modals} and window size: ${ws}"
    python $PYTHON_SCRIPT --path_config_file $CONFIG_FILE \
                          --aligned_filename $ALIGNED_FILE \
                          --modals ${modals} --ws ${ws}
}


# Settings
# Single-modal
run_parser "acc" 8                # ACC → ws: 8
run_parser "gyro" 8               # GYRO → ws: 8
run_parser "ppg" 20               # PPG → ws: 20

# Multimodal
run_parser "acc gyro" 8           # ACC, GYRO → ws: 8
run_parser "acc ppg" 20           # ACC, PPG → ws: 20
run_parser "gyro ppg" 20          # GYRO, PPG → ws: 20
run_parser "acc gyro ppg" 20      # ACC, GYRO, PPG → ws: 20