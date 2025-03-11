
import pandas as pd
from typing import List, Dict, Union, Optional

# FatigueSet
FATIGUESET_SAMPLING_RATE = 100
FATIGUESET_NUM_CLASSES = 4 # FatigueSet (baseline, other, activity, fatigue)
FATIGUESET_FEATURE_COLUMNS = {
    'ppg': {
        'l': [  # left ear PPG
            'ear_ppg_left_ir:min',
            'ear_ppg_left_ir:max',
            'ear_ppg_left_ir:mean',
            'ear_ppg_left_ir:std',
            'ear_ppg_left_ir:range',
            'ear_ppg_left_ir:bpm',
            'ear_ppg_left_ir:ibi',
            'ear_ppg_left_ir:sdnn',
            'ear_ppg_left_ir:sdsd',
            'ear_ppg_left_ir:rmssd',
            'ear_ppg_left_ir:pnn20',
            'ear_ppg_left_ir:pnn50',
            'ear_ppg_left_ir:hr_mad',
            'ear_ppg_left_ir:sd1',
            'ear_ppg_left_ir:sd2',
            'ear_ppg_left_ir:s',
            'ear_ppg_left_ir:sd1/sd2',
            'ear_ppg_left_ir:breathingrate'
        ],
        'r': [  # right ear PPG
            'ear_ppg_right_ir:min',
            'ear_ppg_right_ir:max',
            'ear_ppg_right_ir:mean',
            'ear_ppg_right_ir:std',
            'ear_ppg_right_ir:range',
            'ear_ppg_right_ir:bpm',
            'ear_ppg_right_ir:ibi',
            'ear_ppg_right_ir:sdnn',
            'ear_ppg_right_ir:sdsd',
            'ear_ppg_right_ir:rmssd',
            'ear_ppg_right_ir:pnn20',
            'ear_ppg_right_ir:pnn50',
            'ear_ppg_right_ir:hr_mad',
            'ear_ppg_right_ir:sd1',
            'ear_ppg_right_ir:sd2',
            'ear_ppg_right_ir:s',
            'ear_ppg_right_ir:sd1/sd2',
            'ear_ppg_right_ir:breathingrate'
        ],

    },
    'acc': {
        'l': [
            # Left Earbud ACC
            'ear_acc_left_mag:min',
            'ear_acc_left_mag:max',
            'ear_acc_left_mag:mean',
            'ear_acc_left_mag:std',
            'ear_acc_left_mag:median'
        ],
        'r': [
            # Right Earbud ACC
            'ear_acc_right_mag:min',
            'ear_acc_right_mag:max',
            'ear_acc_right_mag:mean',
            'ear_acc_right_mag:std',
            'ear_acc_right_mag:median'
        ],
        'h': [
            # Headband ACC
            'forehead_acc_mag:min',
            'forehead_acc_mag:max',
            'forehead_acc_mag:mean',
            'forehead_acc_mag:std',
            'forehead_acc_mag:median'
        ],
        'w': [
            # Wristband ACC
            'wrist_acc_mag:min',
            'wrist_acc_mag:max',
            'wrist_acc_mag:mean',
            'wrist_acc_mag:std',
            'wrist_acc_mag:median']
    },
    'gyro': {
        'l': [
            # Left Earbud Gyro
            'ear_gyro_left_gx:min',
            'ear_gyro_left_gx:max',
            'ear_gyro_left_gx:mean',
            'ear_gyro_left_gx:std',
            'ear_gyro_left_gx:median',
            'ear_gyro_left_gy:min',
            'ear_gyro_left_gy:max',
            'ear_gyro_left_gy:mean',
            'ear_gyro_left_gy:std',
            'ear_gyro_left_gy:median',
            'ear_gyro_left_gz:min',
            'ear_gyro_left_gz:max',
            'ear_gyro_left_gz:mean',
            'ear_gyro_left_gz:std',
            'ear_gyro_left_gz:median'
        ],
        'r': [
            # Right Earbud Gyro
            'ear_gyro_right_gx:min',
            'ear_gyro_right_gx:max',
            'ear_gyro_right_gx:mean',
            'ear_gyro_right_gx:std',
            'ear_gyro_right_gx:median',
            'ear_gyro_right_gy:min',
            'ear_gyro_right_gy:max',
            'ear_gyro_right_gy:mean',
            'ear_gyro_right_gy:std',
            'ear_gyro_right_gy:median',
            'ear_gyro_right_gz:min',
            'ear_gyro_right_gz:max',
            'ear_gyro_right_gz:mean',
            'ear_gyro_right_gz:std',
            'ear_gyro_right_gz:median'
        ],
        'h': [
            # Headband Gyro
            'forehead_gyro_gx:min',
            'forehead_gyro_gx:max',
            'forehead_gyro_gx:mean',
            'forehead_gyro_gx:std',
            'forehead_gyro_gx:median',
            'forehead_gyro_gy:min',
            'forehead_gyro_gy:max',
            'forehead_gyro_gy:mean',
            'forehead_gyro_gy:std',
            'forehead_gyro_gy:median',
            'forehead_gyro_gz:min',
            'forehead_gyro_gz:max',
            'forehead_gyro_gz:mean',
            'forehead_gyro_gz:std',
            'forehead_gyro_gz:median'
        ]
    }
}
FATIGUESET_RAW_COLUMNS = {
    'ppg': {
        'l': [  # left ear PPG
            'ear_ppg_left_ir'
        ],
        'r': [  # right ear PPG
            'ear_ppg_right_ir'
        ],

    },
    'acc': {
        'l': [
            # Left Earbud ACC
            'ear_acc_left_mag', 'ear_acc_left_ax', 'ear_acc_left_ay', 'ear_acc_left_az'
        ],
        'r': [
            # Right Earbud ACC
            'ear_acc_right_mag', 'ear_acc_right_ax', 'ear_acc_right_ay', 'ear_acc_right_az'
        ],
        'h': [
            # Headband ACC
            'forehead_acc_mag', 'forehead_acc_ax', 'forehead_acc_ay', 'forehead_acc_az'
        ],
        'w': [
            # Wristband ACC
            'wrist_acc_mag', 'wrist_acc_ax', 'wrist_acc_ay', 'wrist_acc_az'
        ]
    },
    'gyro': {
        'l': [
            # Left Earbud Gyro
            'ear_gyro_left_gx',
            'ear_gyro_left_gy',
            'ear_gyro_left_gz'
        ],
        'r': [
            # Right Earbud Gyro
            'ear_gyro_right_gx',
            'ear_gyro_right_gy',
            'ear_gyro_right_gz'
        ],
        'h': [
            # Headband Gyro
            'forehead_gyro_gx',
            'forehead_gyro_gy',
            'forehead_gyro_gz'
        ]
    }
}


# WESAD
WESAD_SAMPLING_RATE = 32 # Aligned sampling rate
WESAD_NUM_CLASSES = 3
WESAD_FEATURE_COLUMNS = {
    'eda': {
        'c': [
            # chest EDA
            'chest_eda:min',
            'chest_eda:max',
            'chest_eda:mean',
            'chest_eda:std',
            'chest_eda:range',
            'chest_eda:slope',
            'chest_eda:tonic_mean',
            'chest_eda:tonic_std',
            'chest_eda:phasic_mean',
            'chest_eda:phasic_std',
            'chest_eda:SCR_num_peaks',
            'chest_eda:SCR_mean_peak_amplitude'],
        'w': [
        # wrist EDA
            'wrist_eda:min',
            'wrist_eda:max',
            'wrist_eda:mean',
            'wrist_eda:std',
            'wrist_eda:range',
            'wrist_eda:slope',
            'wrist_eda:tonic_mean',
            'wrist_eda:tonic_std',
            'wrist_eda:phasic_mean',
            'wrist_eda:phasic_std',
            'wrist_eda:SCR_num_peaks',
            'wrist_eda:SCR_mean_peak_amplitude']
    },
    'temp': {
        'c': [
            # chest temp
            'chest_temp:min',
            'chest_temp:max',
            'chest_temp:mean',
            'chest_temp:std',
            'chest_temp:range',
            'chest_temp:slope' ],
        'w': [
            # wrist temp
            'wrist_temp:min',
            'wrist_temp:max',
            'wrist_temp:mean',
            'wrist_temp:std',
            'wrist_temp:range',
            'wrist_temp:slope' ],
    },
    'acc': {
        'c': [
            # chest acc
            'chest_acc_mag:min',
            'chest_acc_mag:max',
            'chest_acc_mag:mean',
            'chest_acc_mag:std',
            'chest_acc_mag:median'],
        'w': [
            # wrist acc
            'wrist_acc_mag:min',
            'wrist_acc_mag:max',
            'wrist_acc_mag:mean',
            'wrist_acc_mag:std',
            'wrist_acc_mag:median']
    }
}
WESAD_RAW_COLUMNS = {
    'eda': {
        'c': ['chest_eda'],
        'w': ['wrist_eda']
    },
    'temp': {
        'c': ['chest_temp'],
        'w': ['wrist_temp']
    },
    'acc': {
        'c': ['chest_acc_mag', 'chest_acc_ax', 'chest_acc_ay', 'chest_acc_az'],
        'w': ['wrist_acc_mag', 'wrist_acc_ax', 'wrist_acc_ay', 'wrist_acc_az']
    }
}


def get_dataset_sampling_rate(dataset_name: str) -> int:
    if dataset_name == 'fatigueset':
        return FATIGUESET_SAMPLING_RATE
    elif dataset_name == 'wesad':
        return WESAD_SAMPLING_RATE
    else:
        raise ValueError(f"Dataset name {dataset_name} is not supported.")


def get_feature_column_dict(dataset_name: str) -> Dict[str, Dict[str, List[str]]]:
    if dataset_name == 'fatigueset':
        feat_col_dict = FATIGUESET_FEATURE_COLUMNS
    elif dataset_name == 'wesad':
        feat_col_dict = WESAD_FEATURE_COLUMNS
    else:
        raise ValueError(f"Dataset name {dataset_name} is not supported.")
    return feat_col_dict


def get_meta_columns(dataset_name: str) -> List[str]:
    if dataset_name == 'fatigueset':
        non_feat_col_dict = ['session_id', 'user_id', 'win_start_second', 'label', 'session_intensity']
    elif dataset_name == 'wesad':
        non_feat_col_dict = ['user_id', 'label', 'win_start_second']
    else:
        raise ValueError(f"Dataset name {dataset_name} is not supported.")
    return non_feat_col_dict


def get_valid_users(metadata_df) -> Union[List[int], List[str]]:
    # Get valid users based on metadata
    user_list = metadata_df['user_id'].unique().tolist()
    label_list = metadata_df['label'].unique().tolist()
    MIN_SAMPLES_PER_LABEL = 15
    valid_user_list = []
    label_count_per_user = metadata_df.groupby(['user_id', 'label'])['cnt'].sum()
    # print(f"[debug] label_count_per_user: {label_count_per_user}")

    for user_id in user_list:
        user_data = label_count_per_user.loc[user_id]
        if all(user_data.get(label, 0) > MIN_SAMPLES_PER_LABEL for label in label_list):
            valid_user_list.append(user_id)
    print(f"Valid users ({len(valid_user_list)}): {valid_user_list}")
    return valid_user_list


def get_raw_data_column_dict(dataset_name: str) -> Dict[str, Dict[str, List[str]]]:
    if dataset_name == 'fatigueset':
        raw_col_dict = FATIGUESET_RAW_COLUMNS
    elif dataset_name == 'wesad':
        raw_col_dict = WESAD_RAW_COLUMNS
    else:
        raise ValueError(f"Dataset name {dataset_name} is not supported.")
    return raw_col_dict


def select_raw_acc_cols(col_names, acc_load) -> List[str]:
    if acc_load == 'all':
        selected_cols = col_names
    elif acc_load == 'mag_only': # only select magnitude
        selected_cols = [col for col in col_names if 'mag' in col]
    elif acc_load == 'axis_only': # only select axis (ax, ay, az) columns
        selected_cols = [col for col in col_names if 'mag' not in col]
    else:
        raise ValueError(f'Invalid acc_load value: {acc_load}')
    return selected_cols


def window_raw_data(dataset_name, df, ws):
    overlap_seconds = 0
    if dataset_name == 'fatigueset':
        sampling_rate = FATIGUESET_SAMPLING_RATE
    elif dataset_name == 'wesad':
        sampling_rate = WESAD_SAMPLING_RATE
    else:
        raise ValueError(f"Dataset name {dataset_name} is not supported.")

    n_rows_per_win = int(ws * sampling_rate)
    n_rows_overlap = int(overlap_seconds * sampling_rate)
    step_size = n_rows_per_win - n_rows_overlap

    for start in range(0, len(df) - n_rows_per_win + 1, step_size):
        end = start + n_rows_per_win
        assert df[start:end]['label'].nunique() == 1, 'Window contains multiple labels'
        yield df[start:end]


def get_device_ID_dict(dataset_name: str) -> Dict[str, int]:
    if dataset_name == 'fatigueset':
        device_ID_dict = {'l': 0, 'r': 1, 'h': 2, 'w': 3}
    elif dataset_name == 'wesad':
        device_ID_dict = {'c': 0, 'w': 1}
    else:
        raise ValueError(f"Dataset name {dataset_name} is not supported.")
    return device_ID_dict



def select_sub_df(df: pd.DataFrame, user_id: Union[int, str], label: Union[int, str],
                 session_id: Optional[Union[str, int]]=None) -> pd.DataFrame:
    if session_id is None:
        return df[(df['user_id'] == user_id) & (df['label'] == label)]
    else:
        return df[(df['user_id'] == user_id) & (df['label'] == label) & (df['session_id'] == session_id)]


def get_num_classes(dataset_name: str) -> int:
    if dataset_name == 'fatigueset':
        return FATIGUESET_NUM_CLASSES
    elif dataset_name == 'wesad':
        return WESAD_NUM_CLASSES
    else:
        raise ValueError(f"Dataset name {dataset_name} is not supported.")
