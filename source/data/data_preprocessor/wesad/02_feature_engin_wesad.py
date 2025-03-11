"""
This script processes the aligned and noise-filtered WESAD data to:
1. Extract features based on the specified modalities and window size.
2. Save the extracted feature data along with the corresponding raw data.
3. Extract metadata about generated data and save to a .csv file.
   - This is necessary to select valid users with balanced data for training and testing.
   - Metadata will be based on the feature-engineered data.
"""


import os
import json
import time
import argparse
import warnings
import numpy as np
import pandas as pd
import neurokit2 as nk
warnings.filterwarnings('ignore')

OVERLAP_SECONDS = 0 # No overlap so that the data does not leak to test

# Annotated global variables
N_ROWS_PER_WIN: int = None      # Number of samples per window
STEP_SIZE: int = None           # Step size for windowing based on sample rate and overlap


WESAD_MODAL_TO_DATA_COLS = {
    'acc':  ['chest_acc_ax', 'chest_acc_ay', 'chest_acc_az', 'chest_acc_mag',
             'wrist_acc_ax', 'wrist_acc_ay', 'wrist_acc_az', 'wrist_acc_mag'],
    'eda':  ['chest_eda', 'wrist_eda'],
    'temp': ['chest_temp', 'wrist_temp'],
}

WESAD_DATA_COL_TO_MODAL = {
    # Acc (chestband)
    'chest_acc_ax': 'acc',
    'chest_acc_ay': 'acc',
    'chest_acc_az': 'acc',
    'chest_acc_mag': 'acc',

    # Acc (wristband)
    'wrist_acc_ax': 'acc',
    'wrist_acc_ay': 'acc',
    'wrist_acc_az': 'acc',
    'wrist_acc_mag': 'acc',

    # EDA (chestband)
    'chest_eda': 'eda',

    # EDA (wristband)
    'wrist_eda': 'eda',

    # Temp (chestband)
    'chest_temp': 'temp',

    # Temp (wristband)
    'wrist_temp': 'temp',
}



def parse_args():
    parser = argparse.ArgumentParser(description='Feature Engineering for WESAD dataset')
    parser.add_argument('--path_config_file', type=str, default='../../dataset_paths.json',
                        help='location of data path config file')
    parser.add_argument('--aligned_filename', type=str, default='wesad_filtered_32Hz.csv',
                        help='name of the aligned data file')
    parser.add_argument('--modals', nargs='+', default=['acc', 'eda', 'temp'], help='list of modalities to use')
    parser.add_argument('--ws', type=int, default=20, choices=[8, 10, 20, 30], help='window size in seconds')
    return parser.parse_args()


# Step 0: Validate window size
def validate_win_size(modalities, win_size_sec):
    if len(modalities) != 1 and win_size_sec < 10: # If more than one modality is selected, min win size is 10
        raise ValueError(f'Window size of {win_size_sec} seconds is only valid for acc')
    if len(modalities) == 1 and modalities[0] != 'acc' and win_size_sec < 10: # If one modality is selected and it is not acc, min win size is 10
        raise ValueError(f'Window size of {win_size_sec} seconds is only valid for acc')


# Step 1: Load raw aligned data
def load_raw_filtered_data(path_config_file, aligned_filename):
    print(f"Step 1: Loading raw aligned clean data...")
    st = time.time()
    with open(path_config_file, 'r') as f:
        data_paths = json.load(f)
    processed_data_dir = data_paths['wesad']['processed']
    path_to_raw_aligned_data = os.path.join(processed_data_dir, aligned_filename)
    try:
        df = pd.read_csv(path_to_raw_aligned_data)
    except FileNotFoundError:
        raise FileNotFoundError(f'File not found at {path_to_raw_aligned_data}')
    sampling_rate = int(path_to_raw_aligned_data.split('Hz.csv')[0].split('_')[-1])
    print(f'Loaded raw aligned data. // Shape: {df.shape}. Sampling rate: {sampling_rate} Hz'
          f'// Time taken: {time.time() - st:.2f} seconds')
    return df, sampling_rate, processed_data_dir


# Step 2: Get user data and label dataframes
def get_user_specific_dfs(user_list, label_list, df):
    user_label_df_dict = {}
    for user_id in user_list:
        curr_user_dict = {}
        for label_id in label_list:
            curr_df = df[(df['user_id'] == user_id) & (df['label'] == label_id)]
            curr_user_dict[label_id] = curr_df.copy()
        user_label_df_dict[user_id] = curr_user_dict
    return user_label_df_dict


# Step 3: Feature Engineering
def process_feat_engin(user_dfs, user_list, label_list, sampling_rate, modals, ws):
    print("Step 3: Feature Engineering...")

    # Extract sensor data cols for passed modals
    sensor_data_cols = []
    for modal in modals:
        sensor_data_cols += WESAD_MODAL_TO_DATA_COLS[modal]
    print(f'Extracting sensor data columns for modals: {modals} // {sensor_data_cols}')

    all_feature_df = None
    all_success_raw_df = None
    for user_id in user_list:
        for label in label_list:
            raw_df = user_dfs[user_id][label]
            feature_df, success_raw_df = _feat_engin_single_user_label(raw_df, user_id, label,
                                                                       sampling_rate, ws, sensor_data_cols)
            if feature_df.empty:
                print(f'No features extracted for user {user_id} and label {label}')
                continue
            if all_feature_df is None:
                all_feature_df = feature_df.copy()
                all_success_raw_df = success_raw_df.copy()
            else:
                all_feature_df = pd.concat([all_feature_df, feature_df], axis=0, ignore_index=True)
                all_success_raw_df = pd.concat([all_success_raw_df, success_raw_df], axis=0, ignore_index=True)
        print(f"Unique users so far: {all_feature_df['user_id'].unique()}")

    return all_feature_df, all_success_raw_df


# Step 3-1: Feature Engineering for a single user and label
def _feat_engin_single_user_label(raw_df, user_id, label, sampling_rate, ws, sensor_data_cols):
    result_feat_df = None
    for sensor_data_col in sensor_data_cols:
        sensor_type = WESAD_DATA_COL_TO_MODAL[sensor_data_col]
        sensor_feat_df = _create_feature_df(raw_df[sensor_data_col], sensor_data_col, sensor_type, sampling_rate)

        if sensor_feat_df.empty:
            print(f'No features extracted for user {user_id}, label {label}, and sensor {sensor_data_col}')
            return pd.DataFrame(), pd.DataFrame()

        if result_feat_df is None:
            result_feat_df = sensor_feat_df.copy()
        else:
            result_feat_df = pd.merge(result_feat_df, sensor_feat_df, on='win_enum_id', how='inner')

    ts_df = _get_feat_aligned_timestamps(raw_df['exper_time_second'].values)
    result_feat_df = pd.merge(result_feat_df, ts_df, on='win_enum_id', how='inner')
    if result_feat_df.empty:
        print(f'No successful windows found for user={user_id},label={label}. Skipping this user and label.')
        return pd.DataFrame(), pd.DataFrame()

    # Push the 'win_enum_id' to back
    columns = [col for col in result_feat_df.columns if col != 'win_enum_id'] + ['win_enum_id']
    result_feat_df = result_feat_df[columns]

    result_feat_df['user_id'] = user_id
    result_feat_df['label'] = label

    win_info = result_feat_df[['win_enum_id', 'win_start_second']]
    result_success_raw_df = _get_success_raw_df(raw_df.copy(), sensor_data_cols, win_info)
    result_success_raw_df['user_id'] = user_id
    result_success_raw_df['label'] = label

    return result_feat_df, result_success_raw_df


# Step 3-2: Get aligned timestamps and window enum ids
def _get_feat_aligned_timestamps(exp_ts):
    ts_df_list = []
    for enum_win, window in enumerate(_window_data(exp_ts)):
        start_second = window[0]
        curr_datapoint = {'win_start_second': start_second, 'win_enum_id': enum_win}
        df_to_append = pd.DataFrame(curr_datapoint, index=[0])
        if not df_to_append.empty:
            ts_df_list.append(df_to_append)

    result_df = pd.concat(ts_df_list, ignore_index=True) if ts_df_list else pd.DataFrame()
    return result_df


# Step 3-3: Get raw data for successful windows
def _get_success_raw_df(raw_df, sensor_data_cols, win_info):
    enum_win_id = 0
    valid_enum_win_ids = win_info['win_enum_id'].unique().tolist()

    assert len(valid_enum_win_ids) == len(win_info), 'Duplicate win_enum_id found in win_info'

    raw_df.reset_index(inplace=True)

    filtered_raw_df = None
    for start in range(0, len(raw_df) - N_ROWS_PER_WIN + 1, STEP_SIZE):
        end = start + N_ROWS_PER_WIN
        if enum_win_id in valid_enum_win_ids:
            curr_raw_window_df = raw_df.loc[start:end - 1, sensor_data_cols]
            if len(curr_raw_window_df) != N_ROWS_PER_WIN:
                raise ValueError(
                    f"Length of current raw window df is not {N_ROWS_PER_WIN}. it is: {len(curr_raw_window_df)}.")
            curr_raw_window_df['win_enum_id'] = enum_win_id
            curr_raw_window_df['win_start_second'] = \
                                win_info[win_info['win_enum_id']==enum_win_id]['win_start_second'].values[0]

            if filtered_raw_df is None:
                filtered_raw_df = curr_raw_window_df.copy()
            else:
                filtered_raw_df = pd.concat([filtered_raw_df, curr_raw_window_df.copy()], axis=0, ignore_index=True)
            enum_win_id += 1
        else:
            enum_win_id += 1  # Skip this window, but still increment the enum_win_id
            continue

    return filtered_raw_df


# Step 3-1-1: Create feature df
def _create_feature_df(input_data, sensor_data_col, sensor_type, sampling_rate):
    if sensor_type == 'eda':
        result_df = eda_create_feature_df(input_data, sensor_data_col, sampling_rate)
    elif sensor_type == 'temp':
        result_df = temperature_create_feature_df(input_data, sensor_data_col)
    elif sensor_type == 'acc':
        result_df = acc_create_feature_df(input_data, sensor_data_col)
    else:
        raise ValueError(f'Unexpected sensor type: {sensor_type}')
    return result_df


# Step 3-1-1-Helper
def temperature_create_feature_df(temperature_data, sensor_data_col):
    feature_df_list = []
    for enum_win, window in enumerate(_window_data(temperature_data)):
        feat_dict = _compute_statistical_features(window.copy(), sensor_type='temp')
        feat_dict = {f'{sensor_data_col}:{feat_name}': value for feat_name, value in feat_dict.items()}
        df_to_append = pd.DataFrame(feat_dict, index=[0])

        # There should not be any NaN values in the extracted features
        if not df_to_append.empty and df_to_append.notna().values.all():
            df_to_append['win_enum_id'] = enum_win
            feature_df_list.append(df_to_append)
    result_df = pd.concat(feature_df_list, ignore_index=True) if feature_df_list else pd.DataFrame()
    return result_df


# Step 3-1-1-Helper
def acc_create_feature_df(acc_data, sensor_data_col):
    feat_df_list = []
    for enum_win, window in enumerate(_window_data(acc_data)):
        feat_dict = _compute_statistical_features(window.copy(), sensor_type='acc')
        feat_dict = {f'{sensor_data_col}:{feat_name}': value for feat_name, value in feat_dict.items()}
        df_to_append = pd.DataFrame(feat_dict, index=[0])

        # There should not be any NaN values in the extracted features
        if not df_to_append.empty and df_to_append.notna().values.all():
            df_to_append['win_enum_id'] = enum_win
            feat_df_list.append(df_to_append)

    result_df = pd.concat(feat_df_list, ignore_index=True) if feat_df_list else pd.DataFrame()
    return result_df


# Step 3-1-1-Helper
def eda_create_feature_df(eda_data, sensor_data_col, sampling_rate):
    feature_df_list = []
    for enum_win, window in enumerate(_window_data(eda_data)):
        stat_features = _compute_statistical_features(window.copy(), sensor_type='eda')
        phasic_tonic_features = _eda_compute_phasic_tonic_features(window.copy(), sampling_rate)

        # Merge all features and create dataframe with them
        feat_dict = {**stat_features, **phasic_tonic_features}
        feat_dict = {f'{sensor_data_col}:{feat_name}': value for feat_name, value in feat_dict.items()}
        df_to_append = pd.DataFrame(feat_dict, index=[0])

        if not df_to_append.empty and df_to_append.notna().values.all():
            # There should not be any NaN values in the extracted features
            df_to_append['win_enum_id'] = enum_win
            feature_df_list.append(df_to_append)
    result_df = pd.concat(feature_df_list, ignore_index=True) if feature_df_list else pd.DataFrame()
    return result_df


# Step 3-1-1-Helper
def _eda_compute_phasic_tonic_features(eda_window, sampling_rate):
    # Tonic: SCL (Skin Conductance Level)
    # Phasic SCR (Skin Conductance Response)
    result_features = {}
    eda_tonic_phasic = nk.eda_phasic(eda_window, sampling_rate=sampling_rate)
    result_features['tonic_mean'] = eda_tonic_phasic['EDA_Tonic'].mean()
    result_features['tonic_std'] = eda_tonic_phasic['EDA_Tonic'].std()
    result_features['phasic_mean'] = eda_tonic_phasic['EDA_Phasic'].mean()
    result_features['phasic_std'] = eda_tonic_phasic['EDA_Phasic'].std()

    # Peak related
    peak_info = nk.eda_findpeaks(eda_tonic_phasic['EDA_Phasic'].values, sampling_rate=sampling_rate, method='neurokit')
    result_features['SCR_num_peaks'] = len(peak_info['SCR_Peaks'])
    result_features['SCR_mean_peak_amplitude'] = np.mean(peak_info['SCR_Height'])

    return result_features


# Step 3-1-1-Helper
def _compute_statistical_features(input_array, sensor_type):
    # Common features
    min_ = np.min(input_array)
    max_ = np.max(input_array)
    mean_ = np.mean(input_array)
    std_ = np.std(input_array)

    if sensor_type in ['eda', 'temp']:
        range_ = max_ - min_
        # first-degree polynomial (linear) fit to compute slope
        coefficients = np.polyfit(input_array, range(len(input_array)), 1)
        slope_ = coefficients[0]  # slope
        return {'min': min_, 'max': max_, 'mean': mean_, 'std': std_, 'range': range_, 'slope': slope_}

    elif sensor_type in ['acc']:
        median_ = np.median(input_array)
        return {'min': min_, 'max': max_, 'mean': mean_, 'std': std_, 'median': median_}

    else:
        raise ValueError(f'Unexpected sensor type: {sensor_type}')


# Windower
def _window_data(data):
    for start in range(0, len(data) - N_ROWS_PER_WIN + 1, STEP_SIZE):
        end = start + N_ROWS_PER_WIN
        yield data[start:end]



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
    save_filename_feat = f'wesad_{modals_str}_ws_{ws}_feats.csv'
    save_filename_raw = f'wesad_{modals_str}_ws_{ws}_success_raw.csv'

    save_dir = os.path.join(processed_dir, f'windowed/{modals_str}')
    os.makedirs(save_dir, exist_ok=True)
    feature_df.to_csv(os.path.join(save_dir, save_filename_feat), index=False)
    print(f'Saved feature dataframe to {os.path.join(save_dir, save_filename_feat)}')

    success_raw_df.to_csv(os.path.join(save_dir, save_filename_raw), index=False)
    print(f'Saved success raw dataframe to {os.path.join(save_dir, save_filename_raw)}')

    metadata_df = _extract_metadata(feature_df)
    metadata_df.to_csv(os.path.join(save_dir, f'wesad_{modals_str}_ws_{ws}_metadata.csv'), index=False)


# Step 4-Helper
def _extract_metadata(df):
    print(f"Computing metadata for generated feature data...")

    user_list = df['user_id'].unique().tolist()
    label_list = df['label'].unique().tolist()

    meta_df = pd.DataFrame(columns=['user_id', 'label', 'cnt'])
    for user_id in user_list:
        for label in label_list:
            curr_df = df[(df['user_id'] == user_id) & (df['label'] == label)]
            len_curr_df = 0 if curr_df is None else len(curr_df)
            print(f"User={user_id}, label={label}, cnt={len_curr_df}")
            meta_df = pd.concat([meta_df,
                             pd.DataFrame([{'user_id': user_id, 'label': label, 'cnt': len_curr_df}])],
                             axis=0, ignore_index=True)
    return meta_df



def main():
    st = time.time()
    args = parse_args()

    # Step 0: Validate window size
    validate_win_size(args.modals, args.ws)

    # Step 1: Load raw filtered aligned data
    df, sampling_rate, processed_dir = load_raw_filtered_data(args.path_config_file, args.aligned_filename)

    # Step 2: Get user-specific dataframes
    user_list = df['user_id'].unique().tolist()
    label_list = df['label'].unique().tolist()
    print(f"User list: {user_list}\nLabel list: {label_list}")
    user_dfs = get_user_specific_dfs(user_list, label_list, df.copy())

    global N_ROWS_PER_WIN, STEP_SIZE
    N_ROWS_PER_WIN = int(args.ws * sampling_rate)
    n_samples_overlap = int(OVERLAP_SECONDS * sampling_rate)
    STEP_SIZE = N_ROWS_PER_WIN - n_samples_overlap
    print(f"N_ROWS_PER_WIN: {N_ROWS_PER_WIN}, STEP_SIZE: {STEP_SIZE}")

    # Step 3: Feature Engineering
    feature_df, success_raw_df = process_feat_engin(user_dfs, user_list, label_list, sampling_rate, args.modals, args.ws)


    # Step 4: Save data
    save_data(feature_df, success_raw_df, processed_dir, args.modals, args.ws)
    et = time.time() - st
    print(f'Feature engineering completed successfully! Took: {et:.2f} seconds in total.')



if __name__ == '__main__':
    main()