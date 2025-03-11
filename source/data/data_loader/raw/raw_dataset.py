
import os
import time
import json
import random
import logging
import argparse
import pandas as pd
from typing import List, Union, Tuple, Optional
from sklearn.preprocessing import StandardScaler
from source.data.data_loader.loader_utils import (get_valid_users, select_raw_acc_cols,
     get_raw_data_column_dict, get_meta_columns, window_raw_data, select_sub_df)


class Raw_Dataset:
    def __init__(self, dataset_name: str, data_config_path: str,
                 args: argparse.Namespace, logger: logging.Logger):
        self.dataset_name = dataset_name
        self.data_config_path = data_config_path
        self.args = args
        self.logger = logger

        self.modals = args.modals
        self.devices = args.devices
        self.ws = args.ws
        self.train_ratio = args.train_ratio
        self.test_labels = args.test_labels
        self.method = args.method

        self.acc_load = 'axis_only' if self.method in ['lester04', 'cornelius12'] else 'all'
        self.feature_columns = None


    def load_train_test_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        st = time.time()
        df, meta_df = self.load_raw_dataset()
        self.logger.info(f"Loaded raw dataset in {time.time()-st:.2f}s. Shape: {df.shape}")

        # Select valid users and relevant data
        valid_user_list = get_valid_users(meta_df)
        df = df[df['user_id'].isin(valid_user_list)]

        # Select devices and modals
        df = self.select_devices_and_modals(df)
        # print(f"[DEBUG] Raw Dataset // After select_devices_and_modals: {df.shape}")

        # Split data into train and test
        train_df, test_df = self.split_within_user(df)
        # print(f"[DEBUG] Raw Dataset // After split_within_user: {train_df.shape}, {test_df.shape}")

        # Scale data (for each user)
        train_df, test_df = self.scale_data(train_df, test_df)
        # print(f"[DEBUG] Raw Dataset // After scale_data: {train_df.shape}, {test_df.shape}")

        # Filter out labels not in self.test_labels
        if self.test_labels != 'all':
            test_df = self._filter_labels(test_df, self.test_labels, self.args.dataset)
            self.logger.info(f'Unique test labels: {test_df["label"].unique()}')
        return train_df, test_df


    def _filter_labels(self, df: pd.DataFrame, target_labels: Union[str, List[str]], dataset_name):
        if dataset_name == 'fatigueset':
            target_labels = [target_labels] if isinstance(target_labels, str) else target_labels
        elif dataset_name == 'wesad':
            target_labels = [int(label) for label in target_labels]
        return df[df['label'].isin(target_labels)]




    def load_raw_dataset(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        with open(self.data_config_path, 'r') as f:
            data_paths = json.load(f)

        modals_str = '_'.join(sorted(self.modals))
        modal_specific_dir = f'windowed/{modals_str}'
        dir_to_processed_files = os.path.join(data_paths[self.dataset_name]['processed'], modal_specific_dir)

        raw_filename = f'{self.dataset_name}_{modals_str}_ws_{self.ws}_success_raw.csv'
        meta_filename = raw_filename.replace('success_raw.csv', 'metadata.csv') # metadata extracted from feature_df
        try:
            df = pd.read_csv(os.path.join(dir_to_processed_files, raw_filename))
            meta_df = pd.read_csv(os.path.join(dir_to_processed_files, meta_filename))
        except FileNotFoundError:
            raise FileNotFoundError(f'Either {raw_filename} or {meta_filename} not available in {dir_to_processed_files}')

        return df, meta_df


    def select_devices_and_modals(self, df: pd.DataFrame) -> pd.DataFrame:
        target_feat_cols = []
        data_col_dict = get_raw_data_column_dict(self.dataset_name)
        non_data_cols = get_meta_columns(self.dataset_name)
        for modal in self.modals:
            for device in self.devices:
                modal_device_cols = data_col_dict[modal][device]
                if modal == 'acc':
                    modal_device_cols = select_raw_acc_cols(modal_device_cols, self.acc_load)
                target_feat_cols += modal_device_cols
        assert len(target_feat_cols) > 0, 'No target feature columns selected. Check the modals and devices'

        self.feature_columns = target_feat_cols
        return df[target_feat_cols + non_data_cols]


    def split_within_user(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        user_list = df['user_id'].unique().tolist()
        label_list = df['label'].unique().tolist()

        # Session id is present only in fatigueset
        session_list = df['session_id'].unique().tolist() if 'session_id' in df.columns else [None]
        windowed_data = self._get_windowed_data(df, user_list, label_list, session_list)

        train_df_list, test_df_list = [], []
        for user_id in user_list:
            for session_id in session_list:
                for label in label_list:
                    if self.dataset_name == 'fatigueset':
                        curr_data_list = windowed_data.get(user_id, {}).get(session_id, {}).get(label, {})
                    elif self.dataset_name == 'wesad':
                        curr_data_list = windowed_data.get(user_id, {}).get(label, {})
                    else:
                        raise ValueError(f"Unknown dataset {self.dataset_name}")
                    available_win_len = len(curr_data_list)
                    # print(f"available_win_len for {user_id}, {session_id}, {label}: {available_win_len}")
                    if available_win_len == 0:
                        continue

                    n_train_wins = int(available_win_len * self.train_ratio)
                    train_indices = random.sample(range(available_win_len), n_train_wins)
                    test_indices = [i for i in range(available_win_len) if i not in train_indices]
                    # print(f'[DEBUG] <User: {user_id}, Session: {session_id}, Label: {label}>')
                    # print(f'\t Train indices: {train_indices}, test indices: {test_indices}')
                    if len(train_indices) == 0 or len(test_indices) == 0:
                        print(f'[DEBUG] Skipping <User: {user_id}, Session: {session_id}, Label: {label}> due to less number of data')
                        continue
                    curr_train_df = pd.concat([curr_data_list[i] for i in train_indices], axis=0, ignore_index=True)
                    curr_test_df = pd.concat([curr_data_list[i] for i in test_indices], axis=0, ignore_index=True)

                    # Uncomment to check consistency with feat version
                    # print(f'[RAW-DEBUG] <User: {user_id}, Session: {session_id}>'
                    #       f'\n\t Train win_start_second: {curr_train_df["win_start_second"].unique()}'
                    #       f'\n\t Test win_start_second: {curr_test_df["win_start_second"].unique()}')

                    train_df_list.append(curr_train_df)
                    test_df_list.append(curr_test_df)

        train_df = pd.concat(train_df_list, axis=0, ignore_index=True)
        test_df = pd.concat(test_df_list, axis=0, ignore_index=True)

        # Drop index column from both train and test dfs
        train_df = train_df.drop(columns=['index'])
        test_df = test_df.drop(columns=['index'])

        return train_df, test_df



    def _get_windowed_data(self, df: pd.DataFrame, user_list: Union[List[str], List[int]],
                           label_list: Union[List[str], List[int]],
                           session_list: Optional[Union[List[str], List[int]]]=[None]):
        win_data_dict = {}

        for user_id in user_list:
            for session_id in session_list:
                for label in label_list:
                    curr_df = select_sub_df(df, user_id, label, session_id)
                    if curr_df.empty:
                        continue
                    curr_df = curr_df.reset_index()
                    for enum_win, window in enumerate(window_raw_data(self.dataset_name, curr_df, self.ws)):
                        win_data_dict = self._process_window(win_data_dict, user_id, session_id, label, window)

        return win_data_dict



    def _process_window(self, win_data_dict: dict, user_id: Union[str, int],
                        session_id: Union[str, int], label: Union[str, int], window: pd.DataFrame):
        if self.dataset_name == 'fatigueset':
            if user_id not in win_data_dict:
                win_data_dict[user_id] = {}
            if session_id not in win_data_dict[user_id]:
                win_data_dict[user_id][session_id] = {}
            if label not in win_data_dict[user_id][session_id]:
                win_data_dict[user_id][session_id][label] = []
            win_data_dict[user_id][session_id][label].append(window)
        elif self.dataset_name == 'wesad':
            # session_id is not present in wesad
            if user_id not in win_data_dict:
                win_data_dict[user_id] = {}
            if label not in win_data_dict[user_id]:
                win_data_dict[user_id][label] = []
            win_data_dict[user_id][label].append(window)
        else:
            raise ValueError(f"Unknown dataset name {self.dataset_name}")

        return win_data_dict


    def scale_data(self, train_df, test_df) -> Tuple[pd.DataFrame, pd.DataFrame]:
        scaled_train_df = pd.DataFrame()
        scaled_test_df = pd.DataFrame()

        for user_id in train_df['user_id'].unique():
            curr_train_df = train_df[train_df['user_id'] == user_id].copy()
            curr_test_df = test_df[test_df['user_id'] == user_id].copy()

            scaler = StandardScaler()
            curr_train_df.loc[:, self.feature_columns] = scaler.fit_transform(curr_train_df[self.feature_columns])
            curr_test_df.loc[:, self.feature_columns] = scaler.transform(curr_test_df[self.feature_columns])

            scaled_train_df = pd.concat([scaled_train_df, curr_train_df], ignore_index=True)
            scaled_test_df = pd.concat([scaled_test_df, curr_test_df], ignore_index=True)

        return scaled_train_df, scaled_test_df
