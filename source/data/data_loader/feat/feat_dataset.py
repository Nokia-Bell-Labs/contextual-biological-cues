
import os
import json
import random
import logging
import argparse
import pandas as pd
from typing import List, Union, Tuple, Optional
from sklearn.preprocessing import StandardScaler
from source.data.data_loader.loader_utils import (get_feature_column_dict,
                                                  get_meta_columns, get_valid_users)


class Feat_Dataset:
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

        self.feature_columns = None # Will be set in select_devices_and_modals


    def load_train_test_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df, meta_df = self.load_feature_dataset()

        # Select valid users and relevant data
        valid_user_list = get_valid_users(meta_df)
        df = df[df['user_id'].isin(valid_user_list)]

        # Select devices and modals
        df = self.select_devices_and_modals(df)

        # Split data into train and test
        train_df, test_df = self.split_within_user(df) # within evaluation split

        # Scale data (for each user)
        train_df, test_df = self.scale_data(train_df, test_df)

        # Filter out labels not in self.test_labels
        if self.test_labels != 'all':
            test_df = self._filter_labels(test_df, self.test_labels, self.dataset_name)
            self.logger.info(f'Unique test labels: {test_df["label"].unique()}')
        return train_df, test_df


    def _filter_labels(self, df: pd.DataFrame, target_labels: Union[str, List[str]], dataset_name):
        if dataset_name == 'fatigueset':
            target_labels = [target_labels] if isinstance(target_labels, str) else target_labels
        elif dataset_name == 'wesad':
            target_labels = [int(label) for label in target_labels]
        return df[df['label'].isin(target_labels)]


    def load_feature_dataset(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        with open(self.data_config_path, 'r') as f:
            data_paths = json.load(f)

        modals_str = '_'.join(sorted(self.modals))
        modal_specific_dir = f'windowed/{modals_str}'
        dir_to_processed_files = os.path.join(data_paths[self.dataset_name]['processed'], modal_specific_dir)

        feat_filename = f'{self.dataset_name}_{modals_str}_ws_{self.ws}_feats.csv'
        meta_filename = feat_filename.replace('feats.csv', 'metadata.csv')
        try:
            df = pd.read_csv(os.path.join(dir_to_processed_files, feat_filename))
            meta_df = pd.read_csv(os.path.join(dir_to_processed_files, meta_filename))
        except FileNotFoundError:
            raise FileNotFoundError(f'Either {feat_filename} or {meta_filename} not available in {dir_to_processed_files}')

        return df, meta_df


    def select_devices_and_modals(self, df: pd.DataFrame) -> pd.DataFrame:
        target_feat_cols = []
        feat_col_dict = get_feature_column_dict(self.dataset_name)
        non_feat_cols = get_meta_columns(self.dataset_name)
        for modal in self.modals:
            for device in self.devices:
                target_feat_cols += feat_col_dict[modal][device]
        assert len(target_feat_cols) > 0, 'No target feature columns selected. Check the modals and devices'

        self.feature_columns = target_feat_cols
        return df[target_feat_cols + non_feat_cols]


    def split_within_user(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        user_list = df['user_id'].unique().tolist()
        label_list = df['label'].unique().tolist()

        # Session id is present only in fatigueset
        session_list = df['session_id'].unique().tolist() if 'session_id' in df.columns else [None]

        train_dfs, test_dfs = [], []
        for user_id in user_list:
            for session_id in session_list:
                for label in label_list:
                    curr_df = self._filter_data(df, user_id, label, session_id)
                    if curr_df.empty:
                        continue
                    curr_df = curr_df.reset_index(drop=True)

                    # Random sample train and test data (for each user, session, and label)
                    num_train_data = int(len(curr_df) * self.train_ratio)
                    train_indices = random.sample(range(len(curr_df)), num_train_data)
                    test_indices = [index for index in range(len(curr_df)) if index not in train_indices]
                    if len(train_indices) == 0 or len(test_indices) == 0:
                        print(f'[DEBUG] Skipping <User: {user_id}, Session: {session_id}, Label: {label}> due to less number of data')
                        continue
                    curr_train_df = curr_df.iloc[train_indices]
                    curr_test_df = curr_df.iloc[test_indices]

                    # Uncomment to check consistency with feat version
                    # print(f'[FEAT-DEBUG] <User: {user_id}, Session: {session_id}>'
                    #       f'\n\t Train win_start_second: {curr_train_df["win_start_second"].unique()}'
                    #       f'\n\t Test win_start_second: {curr_test_df["win_start_second"].unique()}')

                    train_dfs.append(curr_train_df)
                    test_dfs.append(curr_test_df)

        train_df = pd.concat(train_dfs, ignore_index=True)
        test_df = pd.concat(test_dfs, ignore_index=True)

        return train_df, test_df


    def _filter_data(self, df: pd.DataFrame, user_id: Union[int, str], label: Union[int, str],
                     session_id: Optional[Union[str, int]]=None) -> pd.DataFrame:
        if session_id is None:
            return df[(df['user_id'] == user_id) & (df['label'] == label)]
        else:
            return df[(df['user_id'] == user_id) & (df['label'] == label) & (df['session_id'] == session_id)]


    def scale_data(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Scale columns before returning
        scaled_train_df = pd.DataFrame()
        scaled_test_df = pd.DataFrame()

        for user_id in train_df['user_id'].unique().tolist():
            # self.logger.info(f"Scaling data for user {user_id}")

            curr_train_df = train_df[train_df['user_id'] == user_id].copy()
            curr_test_df = test_df[test_df['user_id'] == user_id].copy()

            scaler = StandardScaler()
            curr_train_df.loc[:, self.feature_columns] = scaler.fit_transform(curr_train_df.loc[:, self.feature_columns])
            curr_test_df.loc[:, self.feature_columns] = scaler.transform(curr_test_df.loc[:, self.feature_columns])

            scaled_train_df = pd.concat([scaled_train_df, curr_train_df], ignore_index=True)
            scaled_test_df = pd.concat([scaled_test_df, curr_test_df], ignore_index=True)
        return scaled_train_df, scaled_test_df