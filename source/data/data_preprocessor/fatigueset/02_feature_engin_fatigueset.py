"""
This script processes the aligned and noise-filtered FatigueSet data to:
1. Extract features based on the specified modalities and window size.
2. Save the extracted feature data along with the corresponding raw data.
3. Extract metadata about generated data and save to a .csv file.
   - This is necessary to select valid users with balanced data for training and testing.
   - Metadata will be based on the feature-engineered data.

Note:
    - In some cases (particularly with PPG data), feature extraction for a given window may fail.
      When this happens, both the feature data and the corresponding raw data for that window will be skipped.
"""


import os
import json
import time
import argparse
import warnings
import numpy as np
import pandas as pd
import heartpy as hp
from collections import Counter
warnings.filterwarnings('ignore')

ALLOWED_SECOND_DIFF = 2
OVERLAP_SECONDS = 0 # so that the data does not leak to test

# Annotated global variables
N_ROWS_PER_WIN: int = None      # Number of samples per window
STEP_SIZE: int = None           # Step size for windowing based on sample rate and overlap


FATIGUESET_MODAL_TO_DATA_COLS = {
    'acc': [
        'ear_acc_left_ax', 'ear_acc_left_ay', 'ear_acc_left_az', 'ear_acc_left_mag',
        'ear_acc_right_ax', 'ear_acc_right_ay', 'ear_acc_right_az', 'ear_acc_right_mag',
        'forehead_acc_ax', 'forehead_acc_ay', 'forehead_acc_az', 'forehead_acc_mag',
        'wrist_acc_ax', 'wrist_acc_ay', 'wrist_acc_az', 'wrist_acc_mag'
    ],
    'gyro': [
        'ear_gyro_left_gx', 'ear_gyro_left_gy', 'ear_gyro_left_gz',
        'ear_gyro_right_gx', 'ear_gyro_right_gy', 'ear_gyro_right_gz',
        'forehead_gyro_gx', 'forehead_gyro_gy', 'forehead_gyro_gz'
    ],
    'ppg': [
        'ear_ppg_left_ir', 'ear_ppg_right_ir' # Consider only IR signal
    ]
}

FATIGUESET_DATA_COL_TO_MODAL = {
    # Acc (left ear)
    'ear_acc_left_ax': 'acc',
    'ear_acc_left_ay': 'acc',
    'ear_acc_left_az': 'acc',
    'ear_acc_left_mag': 'acc',

    # Acc (right ear)
    'ear_acc_right_ax': 'acc',
    'ear_acc_right_ay': 'acc',
    'ear_acc_right_az': 'acc',
    'ear_acc_right_mag': 'acc',

    # Acc (forehead)
    'forehead_acc_ax': 'acc',
    'forehead_acc_ay': 'acc',
    'forehead_acc_az': 'acc',
    'forehead_acc_mag': 'acc',

    # Acc (wrist)
    'wrist_acc_ax': 'acc',
    'wrist_acc_ay': 'acc',
    'wrist_acc_az': 'acc',
    'wrist_acc_mag': 'acc',

    # Gyro (left ear)
    'ear_gyro_left_gx': 'gyro',
    'ear_gyro_left_gy': 'gyro',
    'ear_gyro_left_gz': 'gyro',

    # Gyro (right ear)
    'ear_gyro_right_gx': 'gyro',
    'ear_gyro_right_gy': 'gyro',
    'ear_gyro_right_gz': 'gyro',

    # Gyro (forehead)
    'forehead_gyro_gx': 'gyro',
    'forehead_gyro_gy': 'gyro',
    'forehead_gyro_gz': 'gyro',

    # PPG (left ear)
    'ear_ppg_left_ir': 'ppg',
    'ear_ppg_right_ir': 'ppg',
}



def parse_args():
    parser = argparse.ArgumentParser(description='Feature Engineering for FatigueSet dataset')
    parser.add_argument('--path_config_file', type=str, default='../../dataset_paths.json',
                        help='location of data path config file')
    parser.add_argument('--aligned_filename', type=str, default='fatigueset_filtered_100Hz.csv',
                        help='name of the aligned data file')
    parser.add_argument('--modals', nargs='+', default=['acc', 'gyro', 'ppg'], help='list of modalities to use')
    parser.add_argument('--ws', type=int, default=20, choices=[8, 10, 20, 30], help='window size in seconds')
    return parser.parse_args()



# Step 0: Validate window size
def validate_win_size(modalities, win_size_sec):
    if 'ppg' in modalities and win_size_sec < 10:
        raise ValueError('Minimum window size for PPG is 10 seconds')


# Step 1: Load raw aligned data
def load_raw_filtered_data(path_config_file, aligned_filename):
    print("Step 1: Loading raw aligned clean data...")
    st = time.time()
    with open(path_config_file, 'r') as f:
        data_paths = json.load(f)
    processed_data_dir = data_paths['fatigueset']['processed']
    path_to_raw_aligned_data = os.path.join(processed_data_dir, aligned_filename)
    try:
        df = pd.read_csv(path_to_raw_aligned_data)
    except FileNotFoundError:
        raise FileNotFoundError(f'File not found at {path_to_raw_aligned_data}')
    sampling_rate = int(path_to_raw_aligned_data.split('Hz.csv')[0].split('_')[-1])
    print(f'Loaded raw aligned data. // Shape: {df.shape}. Sampling rate: {sampling_rate} Hz'
          f'// Time taken: {time.time() - st:.2f} seconds')
    return df, sampling_rate, processed_data_dir


# Step 2: Get user and session ids
def get_user_session_dfs(df, user_list, session_list):
    print("Step 2: Getting user and session dataframes...")
    user_session_dfs = {}
    for user in user_list:
        curr_user_dict = {}
        for session_id in session_list:
            curr_df = df[(df['user_id'] == user) & (df['session_id'] == session_id)]
            curr_user_dict[session_id] = curr_df.copy()
        user_session_dfs[user] = curr_user_dict
    return user_session_dfs


# Step 3: Feature Engineering
def process_feat_engin(user_session_dfs, user_list, session_list, sampling_rate, modals, ws):
    print("Step 3: Feature Engineering...")
    # Extract sensor data cols for passed modals
    sensor_data_cols = []
    for modal in modals:
        sensor_data_cols += FATIGUESET_MODAL_TO_DATA_COLS[modal]
    print(f'Extracting sensor data columns for modals: {modals} // {sensor_data_cols}')

    all_feature_df = None
    all_success_raw_df = None
    for user_id in user_list:
        for session_id in session_list:
            print(f"Processing user {user_id} and session {session_id}...")
            try:
                raw_df = user_session_dfs[user_id][session_id]
            except Exception as e:
                print(f'Cannot extract raw data for user {user_id} and session: {session_id}. Error: {e}')
                continue

            feature_df, success_raw_df = _feat_engin_single_user_session(raw_df, user_id, session_id,
                                                                         sampling_rate, ws, sensor_data_cols)

            if feature_df.empty:
                print(f'No features extracted for user {user_id} and session {session_id}')
                continue
            if all_feature_df is None:
                if len(raw_df['session_intensity'].unique()) != 1:
                    raise ValueError(f'Multiple session intensities found for user {user_id} and session {session_id}')
                feature_df['session_intensity'] = raw_df['session_intensity'].values[0]
                success_raw_df['session_intensity'] = raw_df['session_intensity'].values[0]

                all_feature_df = feature_df.copy()
                all_success_raw_df = success_raw_df.copy()
            else:
                all_feature_df = pd.concat([all_feature_df, feature_df], axis=0, ignore_index=True)
                all_success_raw_df = pd.concat([all_success_raw_df, success_raw_df], axis=0, ignore_index=True)
        print(f'Unique users so far: {all_feature_df["user_id"].unique()}')

    return all_feature_df, all_success_raw_df



# Step 3-1: Feature Engineering for a single user session
def _feat_engin_single_user_session(raw_df, user_id, session_id, sampling_rate, ws, sensor_data_cols):
    result_feat_df = None
    for sensor_data_col in sensor_data_cols:
        sensor_type = FATIGUESET_DATA_COL_TO_MODAL[sensor_data_col]
        sensor_feat_df = _create_feature_df(raw_df[sensor_data_col], raw_df['timestamp_sec'],
                                               sensor_data_col, sensor_type, sampling_rate, ws)

        # If the feature df is empty, halt this user-session combo
        if sensor_feat_df.empty:
            print(f'Sensor data col {sensor_data_col} for user {user_id} and session {session_id} is empty. '
                  f'Cannot create feature dataframe, skipping this user and session.')
            return pd.DataFrame(), pd.DataFrame()

        if result_feat_df is None:
            result_feat_df = sensor_feat_df.copy()
        else:
            result_feat_df = pd.merge(result_feat_df, sensor_feat_df.copy(), on='win_enum_id', how='inner')


    # Get aligned timestamps and labels
    ts_df = _get_feat_aligned_timestamps(raw_df['exper_time_sec'], raw_df['timestamp_sec'], ws)
    label_df = _get_feat_aligned_labels(raw_df['label'], raw_df['timestamp_sec'], ws)

    # Merge with timestamps and labels
    ts_label_df = pd.merge(ts_df, label_df, on='win_enum_id', how='inner')
    result_feat_df = pd.merge(result_feat_df, ts_label_df, on='win_enum_id', how='inner')


    if result_feat_df.empty:
        print(f'No successful windows found for user {user_id} and session {session_id}. Skipping this user and session.')
        return pd.DataFrame(), pd.DataFrame()

    # Push the 'win_enum_id' column to the end
    ending_columns = ['win_start_second', 'win_enum_id', 'label']
    columns = [col for col in result_feat_df.columns if col not in ending_columns] + ending_columns
    result_feat_df = result_feat_df[columns]

    result_feat_df['user_id'] = int(user_id)
    result_feat_df['session_id'] = int(session_id)
    session_intensity = raw_df['session_intensity'].values[0]
    assert len(raw_df['session_intensity'].unique()) == 1, f'Multiple session intensities ' \
                                                           f'found for user {user_id} and session {session_id}'
    result_feat_df['session_intensity'] = session_intensity

    print(f"Current user {user_id} and session {session_id} processed successfully has length: {len(result_feat_df)}.")
    # print(f"[DEBUG]: Result feature df created successfully. Now, will create success raw df.")
    win_info = result_feat_df[['win_enum_id', 'win_start_second', 'label']]
    result_success_raw_df = _get_success_raw_df(raw_df.copy(), raw_df['timestamp_sec'], sensor_data_cols, win_info, ws)
    result_success_raw_df['user_id'] = int(user_id)
    result_success_raw_df['session_id'] = int(session_id)
    result_success_raw_df['session_intensity'] = session_intensity

    return result_feat_df, result_success_raw_df


# Step 3-2: Get raw data for successful windows
def _get_success_raw_df(raw_df, timestamps, sensor_data_cols, win_info, ws):
    enum_win_id = 0
    valid_enum_win_ids = win_info['win_enum_id'].unique().tolist()
    # print("Valid enum win ids: ", valid_enum_win_ids)
    assert len(valid_enum_win_ids)  == len(win_info), "Duplicate enum win ids found in win_info"

    raw_df.reset_index(inplace=True, drop=True)

    filtered_raw_df = None
    for start in range(0, len(raw_df) - N_ROWS_PER_WIN + 1, STEP_SIZE):
        end = start + N_ROWS_PER_WIN
        # print(f"start: {start}, end: {end}, len(raw_df): {len(raw_df)}")
        timestamp_df = timestamps[start:end]
        computed_win_size = int(timestamp_df.iloc[-1] - timestamp_df.iloc[0])
        diff_in_sec = abs(int(computed_win_size - ws))
        if diff_in_sec > ALLOWED_SECOND_DIFF: # ALLOWED_SECOND_DIFF inside the window
            # print(f"Diff in seconds: {diff_in_sec} is greater than allowed second diff: {ALLOWED_SECOND_DIFF}. Skipping this window.")
            continue
        else:
            if enum_win_id in valid_enum_win_ids:
                curr_raw_window_df = raw_df.loc[start:end - 1, sensor_data_cols]
                if len(curr_raw_window_df) != N_ROWS_PER_WIN:
                    raise ValueError(
                        f"Length of current raw window df is not {N_ROWS_PER_WIN}. it is: {len(curr_raw_window_df)}.")
                curr_raw_window_df['win_enum_id'] = enum_win_id
                curr_raw_window_df['win_start_second'] = \
                win_info[win_info['win_enum_id'] == enum_win_id]['win_start_second'].values[0]
                curr_raw_window_df['label'] = win_info[win_info['win_enum_id'] == enum_win_id]['label'].values[0]
                if filtered_raw_df is None:
                    filtered_raw_df = curr_raw_window_df.copy()
                else:
                    filtered_raw_df = pd.concat([filtered_raw_df, curr_raw_window_df.copy()], axis=0, ignore_index=True)
                enum_win_id += 1

            else:
                enum_win_id+=1 # Skip this window, but still increment the enum_win_id
                continue

    # print(f"Len of filtered raw df: {len(filtered_raw_df)}.")
    __check_for_nans(filtered_raw_df, 'get_success_raw_df')
    return filtered_raw_df




# Step 3-1-1: Create feature dataframe for a single sensor data column
def _create_feature_df(input_data, timestamps, sensor_data_col, sensor_type, sampling_rate, ws):
    if sensor_type == 'ppg':
        modal_feat_df = _ppg_create_feature_df(input_data, timestamps, sensor_data_col, sampling_rate, ws)
    elif sensor_type in ['acc', 'gyro']:
        modal_feat_df = _imu_create_feature_df(input_data, timestamps, sensor_data_col, ws)
    else:
        raise ValueError(f'Modal {sensor_type} is not supported for FatigueSet feature engineering.')
    return modal_feat_df


# Step 3-1-1-A: Create feature dataframe for PPG
def _ppg_create_feature_df(ppg_data, timestamps, sensor_data_col, sampling_rate, ws):
    feature_df_list = []
    for enum_win, window in enumerate(_window_data(ppg_data, timestamps, ws)):
        # print("enum id: ", enum_win)
        try:
            stat_features = _compute_statistical_features(window.copy(), sensor_type='ppg')
            ppg_features = _compute_ppg_specific_features(window.copy(), sampling_rate)

            # Merge statistical and PPG-specific features
            feat_dict = {**stat_features, **ppg_features}

            # Rename keys with sensor column name
            feat_dict = {f'{sensor_data_col}:{feat_name}': value for feat_name, value in feat_dict.items()}
            # Append to feature dataframe
            df_to_append = pd.DataFrame(feat_dict, index=[0])
            if not df_to_append.empty and df_to_append.notna().values.all(): # should not contain any NaN values
                df_to_append['win_enum_id'] = enum_win
                feature_df_list.append(df_to_append)
        except Exception as e:
            pass # Will pass this window as it could not extract PPG-features
            # print(f'Encountered exception while extracting features from PPG. Error: {e}')

    result_df = pd.concat(feature_df_list, ignore_index=True) if feature_df_list else pd.DataFrame()

    __check_for_nans(result_df, '_ppg_create_feature_df')

    return result_df

def __check_for_nans(df, fun_name):
    if df.isnull().values.any():
        print(f"[ERROR]: {fun_name} - NaN values found in the dataframe. Dropping them.")
        df = df.dropna()


# Step 3-1-1-A-i: Window data
def _window_data(data, timestamps, ws):
    for start in range(0, len(data) - N_ROWS_PER_WIN + 1, STEP_SIZE):
        end = start + N_ROWS_PER_WIN
        timestamp_df = timestamps[start:end]
        computed_win_size = int(timestamp_df.iloc[-1] - timestamp_df.iloc[0])
        diff_in_sec = abs(int(computed_win_size - ws))
        if diff_in_sec > ALLOWED_SECOND_DIFF: # ALLOWED_SECOND_DIFF inside the window
            # print(f"Skipping this window.")
            continue
        else:
            # print(f"Yielding data from {start} to {end}.")
            yield data[start:end].values


# Step 3-1-1-A-ii Compute statistical features
def _compute_statistical_features(input_array, sensor_type):
    min_ = np.min(input_array)
    max_ = np.max(input_array)
    mean_ = np.mean(input_array)
    std_ = np.std(input_array)

    if sensor_type in ['ppg']:
        range_ = max_ - min_
        return {'min': min_, 'max': max_, 'mean': mean_, 'std': std_, 'range': range_}

    elif sensor_type in ['acc', 'gyro', 'imu']:
        median_ = np.median(input_array)
        return {'min': min_, 'max': max_, 'mean': mean_, 'std': std_, 'median': median_}


# Step 3-1-1-A-iii Compute ppg-specific features
def _compute_ppg_specific_features(ppg_data, sampling_rate):
    working_data, result_features = hp.process(ppg_data, sample_rate=sampling_rate)
    return result_features

# Step 3-1-1-B: Create feature dataframe for IMU
def _imu_create_feature_df(imu_data, timestamps, sensor_data_col, ws):
    feature_df_list = []
    for enum_win, window in enumerate(_window_data(imu_data, timestamps, ws)):
        try:
            feat_dict = _compute_statistical_features(window.copy(), sensor_type='imu')

            # Rename keys with sensor column name
            feat_dict = {f'{sensor_data_col}:{feat_name}': value for feat_name, value in feat_dict.items()}
            # Append to feature dataframe
            df_to_append = pd.DataFrame(feat_dict, index=[0])
            df_to_append['win_enum_id'] = enum_win
            if not df_to_append.empty:
                feature_df_list.append(df_to_append)
        except Exception as e:
            print(f'Encountered exception while extracting features from IMU. Error: {e}')
    result_df = pd.concat(feature_df_list, ignore_index=True) if feature_df_list else pd.DataFrame()
    __check_for_nans(result_df, '_imu_create_feature_df')
    return result_df



# Step 3-1-2: Get aligned timestamps for feature dataframe
def _get_feat_aligned_timestamps(exp_ts, actual_ts, ws):
    ts_df_list = []
    for enum_win, window in enumerate(_window_data(exp_ts, actual_ts, ws)):
        curr_datapoint = {'win_start_second': window[0], 'win_enum_id': enum_win}
        df_to_append = pd.DataFrame(curr_datapoint, index=[0])
        if not df_to_append.empty:
            ts_df_list.append(df_to_append)
    result_df = pd.concat(ts_df_list, ignore_index=True) if ts_df_list else pd.DataFrame()
    return result_df


# Step 3-1-3: Get aligned labels for feature dataframe
def _get_feat_aligned_labels(labels, actual_ts, ws):
    label_df_list = []
    for enum_win, window in enumerate(_window_data(labels, actual_ts, ws)):
        # Get the most common label value
        common_label = Counter(window).most_common(1)[0][0]
        curr_datapoint = {'label': common_label, 'win_enum_id': enum_win}
        df_to_append = pd.DataFrame(curr_datapoint, index=[0])
        if not df_to_append.empty:
            label_df_list.append(df_to_append)
    result_df = pd.concat(label_df_list, ignore_index=True) if label_df_list else pd.DataFrame()
    return result_df


# Step 4: Save data
def save_data(feature_df, success_raw_df, processed_dir, modals, ws):
    print("Step 4: Saving data...")
    # Drop NaN values
    len_feat_df, len_success_raw_df = len(feature_df), len(success_raw_df)
    feature_df.dropna(inplace=True)
    success_raw_df.dropna(inplace=True)
    print(f'Number of NaN values in feat_df: {len_feat_df - len(feature_df)}, '
          f'\nNumber of NaN values in success_raw_df: {len_success_raw_df - len(success_raw_df)}')

    # Save data
    modals_str = '_'.join(sorted(modals))
    save_filename_feat = f'fatigueset_{modals_str}_ws_{ws}_feats.csv'
    save_filename_raw = f'fatigueset_{modals_str}_ws_{ws}_success_raw.csv'

    save_dir = os.path.join(processed_dir, f'windowed/{modals_str}')
    os.makedirs(save_dir, exist_ok=True)
    feature_df.to_csv(os.path.join(save_dir, save_filename_feat), index=False)
    print(f'Saved feature dataframe to {os.path.join(save_dir, save_filename_feat)}')

    success_raw_df.to_csv(os.path.join(save_dir, save_filename_raw), index=False)
    print(f'Saved success raw dataframe to {os.path.join(save_dir, save_filename_raw)}')

    metadata_df = _extract_metadata(feature_df)
    metadata_df.to_csv(os.path.join(save_dir, f'fatigueset_{modals_str}_ws_{ws}_metadata.csv'), index=False)


def _extract_metadata(df):
    print(f"Computing metadata for generated feature data...")

    user_list = df['user_id'].unique().tolist()
    session_list = df['session_id'].unique().tolist()
    label_list = df['label'].unique().tolist()

    meta_df = pd.DataFrame(columns=['user_id', 'session_id', 'session_intensity', 'label', 'cnt'])
    for uid in user_list:
        print(f'==== User : {uid} ====')
        for sid in session_list:
            for l in label_list:
                curr_df = df[(df['user_id'] == uid) & (df['session_id'] == sid) & (df['label'] == l)]
                len_curr_df = 0 if curr_df is None else len(curr_df)
                session_intensity = 'N/A' if len_curr_df == 0 else curr_df['session_intensity'].unique()
                print(f"\tsid={sid}, label={l}: {len_curr_df}")
                meta_df = pd.concat([meta_df,
                                     pd.DataFrame([{'user_id': uid, 'session_id': sid,
                                                 'session_intensity': session_intensity,
                                                 'label': l, 'cnt': len_curr_df}])], axis=0, ignore_index=True)
    return meta_df



def main():
    stt = time.time()
    args = parse_args()
    # Step 0: Validate window size
    validate_win_size(args.modals, args.ws) # For PPG, minimum window size is 10 seconds

    # Step 1: Load raw filtered aligned data
    df, sampling_rate, processed_dir = load_raw_filtered_data(args.path_config_file, args.aligned_filename)

    # Step 2: Get user and session ids
    user_list = df['user_id'].unique().tolist()
    session_list = df['session_id'].unique().tolist()
    print(f'user_list: {user_list}')
    print(f'session_list: {session_list}')
    user_session_dfs = get_user_session_dfs(df, user_list, session_list)

    global N_ROWS_PER_WIN, STEP_SIZE
    N_ROWS_PER_WIN = int(args.ws * sampling_rate)
    n_samples_overlap = int(OVERLAP_SECONDS * sampling_rate)
    STEP_SIZE = N_ROWS_PER_WIN - n_samples_overlap
    print(f'N_ROWS_PER_WIN: {N_ROWS_PER_WIN}, STEP_SIZE: {STEP_SIZE}')

    # Step 3: Feature Engineering
    feature_df, success_raw_df = process_feat_engin(user_session_dfs, user_list, session_list,
                                                    sampling_rate, args.modals, args.ws)

    # Step 4: Save data
    save_data(feature_df, success_raw_df, processed_dir, args.modals, args.ws)
    ett = time.time() - stt
    print(f'Feature engineering completed successfully! Took: {ett:.2f} seconds in total.')



if __name__ == '__main__':
    main()

