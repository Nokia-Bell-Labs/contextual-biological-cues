"""
Create aligned data across all sensing modalities for WESAD dataset
"""
import os
import time
import json
import pickle
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import neurokit2 as nk
from scipy import signal
from scipy.stats import mode


# User list
USER_LIST = ['S10', 'S11', 'S13', 'S14', 'S15', 'S16', 'S17', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9']

# We are only interested in common sensors across devices
USED_DEVICE_SENSOR_SET = {
    'chest': ['ACC', 'EDA', 'Temp'],
    'wrist': ['ACC', 'EDA', 'TEMP']
}

# Sensor data sampling rates
SAMPLING_RATES_PER_MODAL = {
    'chest': {
        'ACC': 700, 'ECG': 700, 'EMG': 700, 'EDA': 700, 'Temp': 700, 'Resp': 700
    },
    'wrist': {
        'ACC': 32,
        'BVP': 64,
        'EDA': 4,
        'TEMP': 4
    },
    'label': 700,
}

# Aligned sampling rate
TARGET_SAMPLING_RATE = 32



def parse_args():
    parser = argparse.ArgumentParser(description='WESAD Preprocessing')
    parser.add_argument('--path_config_file', type=str, default='../../dataset_paths.json', help='Location of data path config file')
    return parser.parse_args()


# Step 1
def resample_and_merge(path_to_raw_data):
    start_step_1 = time.time()
    all_users_result_data = {}
    for user_id in tqdm(USER_LIST):
        curr_user_result_data = {}
        # 1. Load raw data and label file
        curr_user_file = f'{path_to_raw_data}/{user_id}/{user_id}.pkl'
        with open(curr_user_file, 'rb') as file:
            curr_user_data_dict = pickle.load(file, encoding='latin1')
        # 2. Resample sensor data
        for device_location in USED_DEVICE_SENSOR_SET.keys():
            available_sensor_names = USED_DEVICE_SENSOR_SET[device_location]
            for sensor_type in available_sensor_names:
                original_sensor_data = curr_user_data_dict['signal'][device_location][sensor_type]
                original_sampling_rate = SAMPLING_RATES_PER_MODAL[device_location][sensor_type]
                if sensor_type in ['EDA', 'Temp', 'TEMP']:
                    resampled_sensor_data = _resample_sensor_data(original_sampling_rate, original_sensor_data)
                    curr_user_result_data[f'{device_location}_{sensor_type.lower()}'] = resampled_sensor_data
                elif sensor_type in ['ACC']:
                    for axis_name, axis_index in zip(['ax', 'ay', 'az'], [0, 1, 2]):
                        original_sensor_data = curr_user_data_dict['signal'][device_location][sensor_type][:,
                                               axis_index]
                        resampled_sensor_data = _resample_sensor_data(original_sampling_rate, original_sensor_data)
                        curr_user_result_data[
                            f'{device_location}_{sensor_type.lower()}_{axis_name}'] = resampled_sensor_data

        # 3. Resample labels
        original_label_data = curr_user_data_dict['label']
        original_sampling_rate = SAMPLING_RATES_PER_MODAL['label']
        resampled_label_data = _resample_labels(original_sampling_rate, original_label_data)
        curr_user_result_data['label'] = resampled_label_data
        curr_user_result_data['user_id'] = [user_id] * len(resampled_label_data)
        all_users_result_data[user_id] = curr_user_result_data
    print(f'Step 1 (Resampling and merging all data) took {time.time() - start_step_1} seconds')
    return all_users_result_data


# Step 1-1
def _resample_sensor_data(original_sampling_rate, original_signal):
    """
    - Resampling (both upsampling and downsampling) can be done using signal.resample()
    - signal.resample() uses FFT to filter the data and assumes that the signal is periodic.
    - There are other options for downsampling (decimate, resample_poly) that might be suitable
        for non-periodic data. However, they do not result in the exact same number of samples we expect
        as (1) decimate expects the integer decimation, and (2) resample_poly needs to use the integer
        for resampling as well. For instance, 700/32 does not result in integer value, making decimate and
        resample_poly functions not result in the same number of samples we expect.
        Moreover, visualizations after resampling seem not to be very different when we apply decimate(),
        resample_poly(), and resample() showing the possibility of using signal.resample() for non-periodic
        (e.g., EDA) signal as well.
    """
    if original_sampling_rate == TARGET_SAMPLING_RATE:
        return original_signal  # no need to apply any resampling method
    else:
        # signal.resample() can be applied for both up- and down-sampling
        number_of_original_samples = len(original_signal)
        duration_in_seconds = number_of_original_samples / original_sampling_rate
        target_number_of_samples = int(duration_in_seconds * TARGET_SAMPLING_RATE)
        resampled_signal = signal.resample(original_signal, target_number_of_samples)
        return resampled_signal


# Step 1-2
def _resample_labels(original_sampling_rate, labels):
    """
    - Resampling labels (categorical data) is different from resampling sensor data.
    - For every second, we extract the original label data,
        and assign the mode (most frequent label) in the resampled label
    """
    resampled_labels = []
    moving_window_size = original_sampling_rate  # 1 second
    start = 0
    end = start + moving_window_size
    while end <= len(labels):
        current_window = labels[start:end]
        mode_val, counts = mode(current_window)
        resampled_labels.extend([mode_val[0]] * TARGET_SAMPLING_RATE)
        start = end
        end = start + moving_window_size
    return resampled_labels


# Step 2
def clean_merged_data(all_user_resampled_data_dict):
    start_step_2 = time.time()
    filtered_all_users_result_dict = {}
    for user_id in tqdm(all_user_resampled_data_dict.keys()):
        curr_user_data_dict = all_user_resampled_data_dict[user_id]
        filtered_curr_user_data_dict = curr_user_data_dict.copy()
        for column_name in curr_user_data_dict.keys():
            if column_name in ['chest_acc_ax', 'chest_acc_ay', 'chest_acc_az',
                               'wrist_acc_ax', 'wrist_acc_ay', 'wrist_acc_az']:
                acc_data = curr_user_data_dict[column_name]
                # Apply butterworth filter with highcut=15 for each axis of ACC
                filtered_acc_data = nk.signal_filter(acc_data, sampling_rate=TARGET_SAMPLING_RATE,
                                                     method='butterworth', highcut=15)
                filtered_curr_user_data_dict[column_name] = filtered_acc_data

            elif column_name in ['chest_eda', 'wrist_eda']:
                eda_data = curr_user_data_dict[column_name]
                # Apply eda clean using biosppy method from neurokit library for EDA
                filtered_eda_data = nk.eda_clean(eda_data, sampling_rate=TARGET_SAMPLING_RATE,
                                                 method='biosppy')
                filtered_curr_user_data_dict[column_name] = filtered_eda_data

        # No filtering applied for Temp, label, user_id data fields
        filtered_all_users_result_dict[user_id] = filtered_curr_user_data_dict
    print(f'Step 2 (Cleaning merged data) took {time.time() - start_step_2} seconds')
    return filtered_all_users_result_dict


# Step 3
def merge_and_save_data(input_dict, save_dir, filter_identifier):
    start_step_3 = time.time()
    all_user_df = None
    for user_id in tqdm(input_dict.keys()):
        curr_user_data_dict = input_dict[user_id]
        # Compute acc mag
        for device_location in ['chest', 'wrist']:
            acc_ax = curr_user_data_dict[f'{device_location}_acc_ax']
            acc_ay = curr_user_data_dict[f'{device_location}_acc_ay']
            acc_az = curr_user_data_dict[f'{device_location}_acc_az']
            acc_mag = _compute_acc_mag(acc_ax, acc_ay, acc_az)
            curr_user_data_dict[f'{device_location}_acc_mag'] = acc_mag

        for column_name, column_values in curr_user_data_dict.items():
            if isinstance(column_values, np.ndarray) and column_values.ndim > 1:
                curr_user_data_dict[column_name] = column_values.flatten()

        # Now merge data
        curr_user_df = pd.DataFrame(curr_user_data_dict)

        # ==========================
        # Add artificial timestamps
        # ==========================
        curr_user_df = curr_user_df.reset_index(drop=True) # start from 0
        curr_user_df['exper_time_second'] = curr_user_df.index // TARGET_SAMPLING_RATE

        if all_user_df is None:
            all_user_df = curr_user_df.copy()
        else:
            all_user_df = pd.concat([all_user_df, curr_user_df], ignore_index=True)


    # Includes all columns (acc_ax, acc_ay, acc_az, acc_mag) for acc data
    all_user_df = all_user_df[[
        # Chestband
        'chest_acc_ax', 'chest_acc_ay', 'chest_acc_az', 'chest_acc_mag',
        'chest_eda', 'chest_temp',

        # Wristband
        'wrist_acc_ax', 'wrist_acc_ay', 'wrist_acc_az', 'wrist_acc_mag',
        'wrist_eda', 'wrist_temp',

        # Meta
        'user_id', 'label', 'exper_time_second'
    ]]

    # Leave only labels (1, 2, 3)
    all_user_df = all_user_df[all_user_df['label'].isin([1, 2, 3])]

    # Make it start from 0
    all_user_df['label'] = all_user_df['label'] - 1

    # Reset index
    all_user_df = all_user_df.reset_index(drop=True)

    # Dropna
    print(f'Before dropping NA: {all_user_df.shape}')
    all_user_df = all_user_df.dropna()
    print(f'After dropping NA: {all_user_df.shape}')

    # Save data now
    save_path = f'{save_dir}/wesad_{filter_identifier}_{TARGET_SAMPLING_RATE}Hz.csv'
    all_user_df.to_csv(save_path, index=False)

    et = time.time() - start_step_3
    print(f'Step 3 (Saving data for {filter_identifier} data) took {et:.2f} seconds')
    return all_user_df


# Step 3-1
def _compute_acc_mag(acc_ax, acc_ay, acc_az):
    """Compute magnitude of acceleration"""
    acc_mag = np.sqrt(acc_ax ** 2 + acc_ay ** 2 + acc_az ** 2)
    return acc_mag



def main():
    st = time.time()
    args = parse_args()

    # Load config file
    with open(args.path_config_file) as f:
        data_paths = json.load(f)

    raw_data_dir = os.path.join(data_paths['wesad']['raw'], 'WESAD')
    processed_data_dir = data_paths['wesad']['processed']

    if not os.path.exists(raw_data_dir):
        raise FileNotFoundError(f"Raw data directory {raw_data_dir} does not exist.")

    # Create processed data directory if not exists
    os.makedirs(processed_data_dir, exist_ok=True)


    # Step 1: Resample all user data
    df_dict = resample_and_merge(raw_data_dir)

    # Step 2: Clean data by applying filters
    filtered_df_dict = clean_merged_data(df_dict.copy())

    # Step 3: Save the processed files (both unfiltered and filtered versions)
    merge_and_save_data(df_dict, processed_data_dir, 'unfiltered')
    merge_and_save_data(filtered_df_dict, processed_data_dir, 'filtered')

    et = time.time() - st
    print(f'Preprocessing WESAD data took {et:.2f} seconds')


if __name__ == '__main__':
    main()