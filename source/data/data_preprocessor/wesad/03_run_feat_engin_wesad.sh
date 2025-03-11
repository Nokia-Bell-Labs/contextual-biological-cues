#!/bin/sh

PYTHON_SCRIPT="02_feature_engin_wesad.py"
CONFIG_FILE="../../dataset_paths.json"
ALIGNED_FILE="wesad_filtered_32Hz.csv"


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
run_parser "eda" 20               # EDA → ws: 20
run_parser "temp" 20              # TEMP → ws: 20

# Multimodal
run_parser "acc eda" 20           # ACC, EDA → ws: 20
run_parser "acc temp" 20          # ACC, TEMP → ws: 20
run_parser "eda temp" 20          # EDA, TEMP → ws: 20
run_parser "acc eda temp" 20      # ACC, EDA, TEMP → ws: 20