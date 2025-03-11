"""
Create aligned data across all sensing modalities for FatigueSet dataset
"""

import os
import time
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import neurokit2 as nk
from scipy import signal

# User and session information
USER_LIST = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
SESSION_LIST = ['01', '02', '03']

# Valid suffixes to access corresponding sensor data
VALID_SUFFIX_LIST = {
    "ear_acc_left", "ear_acc_right",
    "ear_gyro_left", "ear_gyro_right",
    "ear_ppg_left", "ear_ppg_right",    # Earbuds
    "forehead_acc", "forehead_gyro",    # Headband
    "wrist_acc"                         # Wristband
}

# Column names for sensor data
SENSOR_COLUMN_LIST = [
    # Earbuds
    'ear_acc_left_ax', 'ear_acc_left_ay', 'ear_acc_left_az',
    'ear_acc_right_ax', 'ear_acc_right_ay', 'ear_acc_right_az',
    'ear_gyro_left_gx', 'ear_gyro_left_gy', 'ear_gyro_left_gz',
    'ear_gyro_right_gx', 'ear_gyro_right_gy', 'ear_gyro_right_gz',
    'ear_ppg_left_green', 'ear_ppg_left_ir', 'ear_ppg_left_red',
    'ear_ppg_right_green', 'ear_ppg_right_ir', 'ear_ppg_right_red',

    # Headband
    'forehead_acc_ax', 'forehead_acc_ay', 'forehead_acc_az',
    'forehead_gyro_gx', 'forehead_gyro_gy', 'forehead_gyro_gz',

    # Wristband
    'wrist_acc_ax', 'wrist_acc_ay', 'wrist_acc_az'
]

# Column names and metadata for the aligned dataframe
REORDERED_COL_NAMES = [
    # Sensor data (Earbuds, Headband, Wristband)
    'ear_acc_left_ax', 'ear_acc_left_ay', 'ear_acc_left_az',
    'ear_acc_right_ax', 'ear_acc_right_ay', 'ear_acc_right_az',
    'ear_gyro_left_gx', 'ear_gyro_left_gy', 'ear_gyro_left_gz',
    'ear_gyro_right_gx', 'ear_gyro_right_gy', 'ear_gyro_right_gz',
    'ear_ppg_left_green', 'ear_ppg_left_ir', 'ear_ppg_left_red',
    'ear_ppg_right_green', 'ear_ppg_right_ir', 'ear_ppg_right_red',
    'forehead_acc_ax', 'forehead_acc_ay', 'forehead_acc_az',
    'forehead_gyro_gx', 'forehead_gyro_gy', 'forehead_gyro_gz',
    'wrist_acc_ax', 'wrist_acc_ay', 'wrist_acc_az',

    # Metadata
    'timestamp_sec', 'timestamp', 'user_id', 'session_id'
]

# Sensor data sampling rates
SENSOR_FILE_SAMPLING_RATE_DICT = {
    # Earbuds (100 Hz)
    'ear_acc_left': 100,
    'ear_acc_right': 100,
    'ear_gyro_left': 100,
    'ear_gyro_right': 100,
    'ear_ppg_left': 100,
    'ear_ppg_right': 100,

    # Headband (52 Hz)
    'forehead_acc': 52,
    'forehead_gyro': 52,

    # Wristband (32 Hz)
    'wrist_acc': 32
}

# Aligned sampling rate
TARGET_SAMPLING_RATE = 100  # Unify all at 100 Hz


def parse_args():
    parser = argparse.ArgumentParser(description='FatigueSet Preprocessing')
    parser.add_argument('--path_config_file', type=str, default='../../dataset_paths.json', help='Location of data path config file')
    return parser.parse_args()


# Step 1
def resample_and_merge(raw_data_dir):
    start_step_1 = time.time()
    dfs = []
    for user_id in USER_LIST:
        user_df = _resample_merge_single_user(user_id, raw_data_dir)
        if user_df.empty:
            print(f'!! Resampled user {user_id} dataframe is empty. Skipping... !!')
            continue
        dfs.append(user_df.copy())
    all_user_df = pd.concat(dfs, axis=0, ignore_index=True)
    et = time.time() - start_step_1
    print(f'Step 1 (Resampling and merging all data) took {et:.2f} seconds')
    return all_user_df


# Step 1-1
def _resample_merge_single_user(user_id, raw_data_dir):
    """
    Resample and merge all the sensor data for a single user
    Notes:
        - Only the users who have all the files (earbuds, headband, wristband) for a given session are considered as valid users.
        - Columns are reordered
    Return: re-sampled and merged data (for all available sessions) for a single user
    """
    dfs =[]
    for sid in SESSION_LIST:
        resampled_df_dict = {}
        for file_id in VALID_SUFFIX_LIST:
            file_path = f'{raw_data_dir}/{user_id}/{sid}/{file_id}.csv'
            if not os.path.exists(file_path):
                print(f'File not found: {file_path}')
                continue # Skip if the file is not found

            sensor_df = pd.read_csv(file_path)
            if sensor_df.empty:
                print(f'Empty file: {file_path}')
                continue # Skip if the file is empty
            resampled_sensor_df = _resample_to_exactly_100Hz(sensor_df, file_id)
            resampled_df_dict[file_id] = resampled_sensor_df.copy()

        # They should have all the available files
        if len(resampled_df_dict.keys()) < len(VALID_SUFFIX_LIST):
            print(f'!! Skipping user:{user_id}, session:{sid} due to missing sensor files. !!')
            continue

        merged_df = _merge_resampled_sensor_dfs(resampled_df_dict)
        merged_df['user_id'] = user_id
        merged_df['session_id'] = sid
        dfs.append(merged_df)

    if dfs:
        all_session_df = pd.concat(dfs, axis=0, ignore_index=True) # Merge all sessions
        all_session_df = all_session_df[REORDERED_COL_NAMES] # Reorder columns
    else:
        print(f'No valid data for user: {user_id}')

    return all_session_df if dfs else pd.DataFrame() # Return empty dataframe if no valid data


# Step 1-1-1
def _resample_to_exactly_100Hz(sensor_df, file_id):
    sensor_df.sort_values(by='timestamp', inplace=True)  # 1. Sort the values by timestamp
    org_sr = SENSOR_FILE_SAMPLING_RATE_DICT[file_id]     # 2. Extract original sr

    # 3. Create timestamp_seconds
    sensor_df['timestamp_sec'] = (sensor_df['timestamp'] // 1000).astype(int)

    # 4. Create empty dataframe
    resampled_sensor_df = pd.DataFrame(columns=sensor_df.columns.tolist())

    # 5. Get unique timestamp seconds in the dataframe
    unique_seconds_list = sorted(list(sensor_df['timestamp_sec'].unique()))

    # 6. Everything expect the columns ['timestamp', 'timestamp_sec'] are feature columns
    feature_columns = [col for col in sensor_df.columns.tolist() if col not in ['timestamp', 'timestamp_sec']]

    # 7. Start traversing for each second
    skip_due_to_not_enough_samples = 0
    for unq_sec in tqdm(unique_seconds_list):

        # 7.1 Get the current second dataframe
        original_second_segment_df = sensor_df[sensor_df['timestamp_sec'] == unq_sec].copy()

        # 7.2 At least 90% of the original sr should exist
        if len(original_second_segment_df) < (0.9 * org_sr):
            skip_due_to_not_enough_samples += 1
            continue

        # 7.3 Otherwise, we can do the resampling
        # ==> We do it for each column
        resampled_second_segment_df = pd.DataFrame(columns=original_second_segment_df.columns.tolist())

        try:
            for feat_col in feature_columns:
                resampled_second_segment_df[feat_col] = signal.resample(original_second_segment_df[feat_col],
                                                                        TARGET_SAMPLING_RATE)
        except Exception as e:
            print(f'Error in resampling {feat_col} for {file_id}. e: {e} \n'
                  f'(Probably the data which has the format like this: 3.476715 08789')
            continue
        # 7.4 Once everything is resampled, we add timestamps back (with exactly correct ones)
        resampled_second_segment_df['timestamp_sec'] = unq_sec
        resampled_second_segment_df['timestamp'] = (resampled_second_segment_df['timestamp_sec'] * 1000) + (
                resampled_second_segment_df.index * 10) # 1000//100 = 10

        # 7.5 Add to larger df
        resampled_sensor_df = pd.concat([resampled_sensor_df, resampled_second_segment_df],
                                        axis=0, ignore_index=True)
    print(f'Skip due to not enough samples: {skip_due_to_not_enough_samples}')

    # 8. Return the resampled_data
    return resampled_sensor_df


# Step 1-1-2
def _merge_resampled_sensor_dfs(sensor_df_dict):
    """key: file_id, value: resampled_df"""
    dfs_with_updated_columns = []
    for file_id, resampled_df in sensor_df_dict.items():
        col_names = resampled_df.columns.tolist()
        updated_col_name_dict = {}
        for col_name in col_names:  # only for feature columns
            updated_col_name_dict[col_name] = f'{file_id}_{col_name}' if col_name not in [
                'timestamp', 'timestamp_sec'] else col_name
        # Update column names
        resampled_df.rename(columns=updated_col_name_dict, inplace=True)
        dfs_with_updated_columns.append(resampled_df.copy())

    # Start merging with the first dataframe
    merged_df = dfs_with_updated_columns[0]
    # Merge the rest of the dataframes based on 'timestamp_sec' and 'timestamp'
    for sensor_df in dfs_with_updated_columns[1:]:
        # Use inner join to make sure that we have the same timestamps
        merged_df = merged_df.merge(sensor_df, on=['timestamp_sec', 'timestamp'], how='inner')
    return merged_df


# Step 2
def add_session_intensity_levels(passed_df, path_to_raw_data):
    start_step_2 = time.time()
    session_intensity_dict = _get_session_intensity_dict(path_to_raw_data)
    passed_df['session_intensity'] = passed_df.apply(
        lambda row: session_intensity_dict.get(int(row['user_id']), {}).get(int(row['session_id'])), axis=1)
    ett = time.time() - start_step_2
    print(f'Step 2 (Adding session intensity levels) took {ett:.2f} seconds')
    return passed_df


# Step 2-1
def _get_session_intensity_dict(path_to_raw_data):
    organized_session_intensity_dict = {}
    session_intensity_df = pd.read_csv(f'{path_to_raw_data}/metadata.csv')
    list_of_session_intensity_dict = session_intensity_df.to_dict('records')
    for record in list_of_session_intensity_dict:
        participant_id = record['participant_id']
        del record['participant_id'] # Leave only information on session_id and intensity
        curr_user_dict = {session_id: intensity for intensity, session_id in record.items()}
        # sort based on the session id
        curr_user_dict = {session_id: curr_user_dict[session_id] for session_id in sorted(curr_user_dict.keys())}
        organized_session_intensity_dict[participant_id] = curr_user_dict

    # Organized session intensity dict will look like this:
    # {1: {1: 'low_session', 2: 'medium_session', 3: 'high_session'},
    #  2: {1: 'low_session', 2: 'high_session', 3: 'medium_session'},
    # .. }
    return organized_session_intensity_dict


# Step 3
def add_event_labels(passed_df, path_to_raw_data):
    start_step_3 = time.time()
    df_with_labels = pd.DataFrame()
    user_id_list = passed_df['user_id'].unique().tolist()
    for user_id in tqdm(user_id_list):
        curr_user_df = passed_df[passed_df['user_id'] == user_id].copy()
        session_id_list = curr_user_df['session_id'].unique().tolist()
        for session_id in session_id_list:
            curr_session_df = curr_user_df[curr_user_df['session_id'] == session_id].copy()
            curr_session_df = _assign_label_for_user_and_session(curr_session_df, path_to_raw_data, user_id, session_id)
            curr_session_df = curr_session_df.reset_index(drop=True)  # Reset index
            start_timestamp_sec = curr_session_df['timestamp_sec'].iloc[0]
            curr_session_df['exper_time_sec'] = curr_session_df['timestamp_sec'] - start_timestamp_sec # add exper_time_sec
            df_with_labels = pd.concat([df_with_labels, curr_session_df], axis=0)
    ett = time.time() - start_step_3
    print(f'Step 3 (Adding event labels) took {ett:.2f} seconds')
    return df_with_labels


# Step 3-1
def _assign_label_for_user_and_session(passed_df, path_to_raw_data, user_id, session_id):
    experiment_marker_df = pd.read_csv(f'{path_to_raw_data}/{user_id}/{session_id}/exp_markers.csv')
    experiment_marker_df.dropna(inplace=True)
    experiment_marker_df['timestamp_sec'] = (experiment_marker_df['utcTime'] // 1000).astype(int)

    event_start_end_timestamp_sec_dict = {}
    important_events = ['baseline', 'fatigue', 'activity']
    for event in important_events:
        start_time = experiment_marker_df[experiment_marker_df['eventMarker'] == f'start_{event}']['timestamp_sec'].iloc[0]
        end_time = experiment_marker_df[experiment_marker_df['eventMarker'] == f'end_{event}']['timestamp_sec'].iloc[0]
        assert start_time < end_time
        event_start_end_timestamp_sec_dict[event] = {
            'start_time': start_time, 'end_time': end_time}

    # even_start_end_timestamp_sec_dict will look like this:
    # {'baseline': {'start_time': 1629370423, 'end_time': 1629370603},
    #  'fatigue': {'start_time': 1629371361, 'end_time': 1629371848},
    #  'activity': {'start_time': 1629370864, 'end_time': 1629371048}}

    passed_df['label'] = passed_df['timestamp_sec'].apply(
        lambda x: _assign_labels_for_timestamp(x, event_start_end_timestamp_sec_dict))
    return passed_df


# Step 3-1-1
def _assign_labels_for_timestamp(curr_timestamp_sec, start_end_timestamp_dict):
    for event in ['baseline', 'fatigue', 'activity']:
        if start_end_timestamp_dict[event]['start_time'] <= curr_timestamp_sec < start_end_timestamp_dict[event]['end_time']:
            return event
    return 'other'  # intermediate activity


# Step 4
def clean_merged_data(merged_df):
    start_step_4 = time.time()
    clean_merged_df = pd.DataFrame(columns=merged_df.columns.tolist())
    unique_user_list = list(merged_df['user_id'].unique())
    for en_id, user_id in enumerate(unique_user_list):
        curr_user_df = merged_df[merged_df['user_id'] == user_id].copy()
        cleaned_user_df = curr_user_df.copy()
        for sensor_column_name in SENSOR_COLUMN_LIST:
            clean_sensor_data = _remove_noise_from_signal(curr_user_df[sensor_column_name].values, sensor_column_name)
            cleaned_user_df[sensor_column_name] = clean_sensor_data
        clean_merged_df = pd.concat([clean_merged_df, cleaned_user_df], axis=0, ignore_index=True)
    ett = time.time() - start_step_4
    print(f'Step 4 (Cleaning merged data) took {ett:.2f} seconds')
    return clean_merged_df


# Step 4-1
def _remove_noise_from_signal(org_signal, column_name):
    """
    PPG
        nk.ppg_clean(given_signal, sampling_rate=sampling_rate, method='elgendi')
        --> method=elegandi applies Bandpass filter with lowcut of 0.5Hz and highcut of 8Hz and 3rd order Butterworth filter.

    ACC (for each axis separately)
        nk.signal_filter(given_signal, sampling_rate=sampling_rate, method='butterworth', highcut=15)
        --> Apply Low-pass filter with 15Hz cutoff frequency and 3rd order Butterworth filter.
        99% of human body motion is contained below 15Hz [1]
        [1] D.M. Karantonis, M.R. Narayanan, M. Mathie, N.H. Lovell, and B.G. Celler. Implementation of a real-time human movement classifier using a triaxial accelerometer for ambulatory monitoring. IEEE Transactions on Information Technology in Biomedicine, 10(1):156–167, 2006.
    """

    # 1. PPG
    if column_name in ['ear_ppg_left_green', 'ear_ppg_left_ir', 'ear_ppg_left_red',
                       'ear_ppg_right_green', 'ear_ppg_right_ir', 'ear_ppg_right_red']:
        clean_signal = nk.ppg_clean(org_signal, sampling_rate=TARGET_SAMPLING_RATE, method='elgendi')

    # 2. ACC (for each axis)
    elif column_name in ['ear_acc_left_ax', 'ear_acc_left_ay', 'ear_acc_left_az',
                         'ear_acc_right_ax', 'ear_acc_right_ay', 'ear_acc_right_az',
                         'forehead_acc_ax', 'forehead_acc_ay', 'forehead_acc_az',
                         'wrist_acc_ax', 'wrist_acc_ay', 'wrist_acc_az']:
        clean_signal = nk.signal_filter(org_signal, sampling_rate=TARGET_SAMPLING_RATE,
                                        method='butterworth', highcut=15)
    # 3. GYRO
    elif column_name in ['ear_gyro_left_gx', 'ear_gyro_left_gy', 'ear_gyro_left_gz',
                         'ear_gyro_right_gx', 'ear_gyro_right_gy', 'ear_gyro_right_gz',
                         'forehead_gyro_gx', 'forehead_gyro_gy', 'forehead_gyro_gz']:  # gyro
        clean_signal = nk.signal_filter(org_signal, sampling_rate=TARGET_SAMPLING_RATE,
                                        method='butterworth', highcut=20)
    else:
        raise ValueError(f'Not a valid sensor modality {column_name} passed for FatigueSet dataset')
    return clean_signal



# Step 5
def merge_and_save_data(passed_df, save_dir, filter_identifier):
    start_step_5 = time.time()
    # 1. Add acc magnitude
    passed_df = _compute_acc_mag_for_all_devices(passed_df.copy())

    # 2. Reset index
    passed_df.reset_index(inplace=True)
    passed_df.drop(columns=['index'], inplace=True)

    # 4. Reorder columns: keep all acc axis
    passed_df = passed_df[[
        # Earbuds (left)
        'ear_acc_left_ax', 'ear_acc_left_ay', 'ear_acc_left_az', 'ear_acc_left_mag',
        'ear_gyro_left_gx', 'ear_gyro_left_gy', 'ear_gyro_left_gz',
        'ear_ppg_left_ir', 'ear_ppg_left_red',

        # Earbuds (right)
        'ear_acc_right_ax', 'ear_acc_right_ay', 'ear_acc_right_az', 'ear_acc_right_mag',
        'ear_gyro_right_gx', 'ear_gyro_right_gy', 'ear_gyro_right_gz',
        'ear_ppg_right_ir', 'ear_ppg_right_red',

        # Headband
        'forehead_acc_ax', 'forehead_acc_ay', 'forehead_acc_az', 'forehead_acc_mag',
        'forehead_gyro_gx', 'forehead_gyro_gy', 'forehead_gyro_gz',

        # Wristband
        'wrist_acc_ax', 'wrist_acc_ay', 'wrist_acc_az', 'wrist_acc_mag',

        # Meta
        'timestamp_sec', 'timestamp', 'user_id', 'session_id', 'label', 'session_intensity', 'exper_time_sec'
    ]]

    # 5. Drop nan if any
    passed_df = passed_df.dropna()

    # 6. Save the processed data
    path_to_save = f'{save_dir}/fatigueset_{filter_identifier}_{TARGET_SAMPLING_RATE}Hz.csv'
    passed_df.to_csv(path_to_save, index=False)
    ett = time.time() - start_step_5
    print(f'Step 5 (Saving data for {filter_identifier} data) took {ett:.2f} seconds')
    return passed_df


# Step 5-1
def _compute_acc_mag_for_all_devices(passed_df):
    # Compute magnitude of acc all devices
    passed_df['ear_acc_left_mag'] = _compute_acc_mag(passed_df['ear_acc_left_ax'], passed_df['ear_acc_left_ay'], passed_df['ear_acc_left_az'])
    passed_df['ear_acc_right_mag'] = _compute_acc_mag(passed_df['ear_acc_right_ax'], passed_df['ear_acc_right_ay'], passed_df['ear_acc_right_az'])
    passed_df['forehead_acc_mag'] = _compute_acc_mag(passed_df['forehead_acc_ax'], passed_df['forehead_acc_ay'], passed_df['forehead_acc_az'])
    passed_df['wrist_acc_mag'] = _compute_acc_mag(passed_df['wrist_acc_ax'], passed_df['wrist_acc_ay'], passed_df['wrist_acc_az'])
    return passed_df


# Step 5-1-1
def _compute_acc_mag(acc_ax, acc_ay, acc_az):
    """Compute magnitude of acceleration"""
    acc_mag = np.sqrt(acc_ax ** 2 + acc_ay ** 2 + acc_az ** 2)
    return acc_mag




def main():
    st = time.time()
    args = parse_args()

    # Load config file paths
    with open(args.path_config_file, 'r') as f:
        data_paths = json.load(f)

    raw_data_dir = os.path.join(data_paths['fatigueset']['raw'], 'fatigueset') # 'fatigueset' inside raw is generated as a result of unzipping the original zip file
    processed_data_dir = data_paths['fatigueset']['processed']

    # Check if raw data directory exists:
    if not os.path.exists(raw_data_dir):
        raise FileNotFoundError(f"Raw data directory {raw_data_dir} does not exist.")

    # Create processed data directory if it does not exist
    os.makedirs(processed_data_dir, exist_ok=True)


    # Step 1. Resample and merge all data
    df = resample_and_merge(raw_data_dir)

    # Step 2. Add session intensity (low, medium, high) based on the session_id
    df = add_session_intensity_levels(df, raw_data_dir)

    # Step 3. Assign labels (baseline, fatigue, activity) for each timestamp
    df = add_event_labels(df, raw_data_dir)

    # Step 4. Clean data by applying filters
    filtered_df = clean_merged_data(df.copy())

    # Step 5. Save the processed files (both unfiltered and filtered versions)
    merge_and_save_data(df, processed_data_dir, 'unfiltered')
    merge_and_save_data(filtered_df, processed_data_dir, 'filtered')

    et = time.time() - st
    print(f'Preprocessing FatigueSet took {et:.2f} seconds')


if __name__ == '__main__':
    main()
