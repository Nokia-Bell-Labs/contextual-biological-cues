
import time
import torch
import random
import logging
import argparse
import itertools
import numpy as np
import pandas as pd
from typing import Union, List, Dict, Tuple
from source.data.data_loader.loader_utils import (get_raw_data_column_dict,
     select_raw_acc_cols, get_device_ID_dict, window_raw_data, select_sub_df)


class Raw_Loader:
    def __init__(self, dataset_name: str, df: pd.DataFrame, is_train: bool,
                 args:argparse.Namespace, logger: logging.Logger):
        self.dataset_name = dataset_name
        self.df = df
        self.is_train = is_train
        self.args = args
        self.logger = logger

        self.method = args.method
        self.modals = sorted(args.modals) if (len(args.modals) > 1 and isinstance(args.modals, list)) else args.modals
        self.devices = sorted(args.devices) # always more than a single device
        self.ws = args.ws

        # Load necessary data columns
        self.acc_load = 'axis_only' if self.method in ['lester04', 'cornelius12'] else 'all'
        self.device_col_dict = self.get_device_columns()
        self.device_ID_dict = get_device_ID_dict(self.dataset_name)

        # Indices
        self.DATA_IDX = 0
        self.DEVICE_IDX = 1
        self.WIN_IDX = 2
        self.USER_IDX = 3
        self.AUX_LABEL_IDX = 4    # auxiliary task label (e.g. stress, fatigue)
        self.SESSION_IDX = 5  # only for fatigueset

        self.all_positive_pairs = []
        self.create_all_positive_pairs()


    def create_all_positive_pairs(self) -> None:
        st = time.time()
        initial_positive_pair_list = []
        windowed_data_list = self.create_windowed_data()

        list_of_unique_device_pairs = list(itertools.combinations(self.devices, 2))
        for idx in range(len(windowed_data_list)):
            curr_win_df = windowed_data_list[idx]
            self._validate_unique_items(curr_win_df)

            # Extract data columns specific to each device
            device_data_dict = {}
            for device_name in self.devices:
                curr_device_data = torch.tensor(
                        np.array(curr_win_df[self.device_col_dict[device_name]].astype(float)), dtype=torch.float)
                device_data_dict[device_name] = curr_device_data

            # Create all possible pairs of devices
            for device_pair_names in list_of_unique_device_pairs:
                curr_datapoint = self._make_positive_pair(device_pair_names, device_data_dict, curr_win_df)
                initial_positive_pair_list.append(curr_datapoint)

        self.logger.info(f"Positive pair creation took {time.time() - st:.2f} seconds")
        self.all_positive_pairs = initial_positive_pair_list


    def _make_positive_pair(self, dev_names: Tuple[str, str], dev_data_dict: Dict[str, torch.Tensor],
                            window_df: pd.DataFrame) -> Dict[str, Tuple]:
        user_id = window_df['user_id'].iloc[0]
        label = self.label_to_int(window_df['label'].iloc[0])
        win_id = window_df['win_start_second'].iloc[0]
        session_id = window_df['session_id'].iloc[0] if self.dataset_name == 'fatigueset' else None

        dev1_name, dev2_name = dev_names
        dev1_ID, dev2_ID = self.device_ID_dict[dev1_name], self.device_ID_dict[dev2_name]
        dev1_data, dev2_data = dev_data_dict[dev1_name], dev_data_dict[dev2_name]

        if self.dataset_name == 'fatigueset':
            return {
                'anchor': (dev1_data, dev1_ID, win_id, user_id, label, session_id),
                'positive': (dev2_data, dev2_ID, win_id, user_id, label, session_id)}
        elif self.dataset_name == 'wesad':
            return {
                'anchor': (dev1_data, dev1_ID, win_id, user_id, label),
                'positive': (dev2_data, dev2_ID, win_id, user_id, label)}


    def __len__(self) -> int:
        return len(self.all_positive_pairs)


    def __getitem__(self, idx: int) -> Dict[str, Tuple]:
        if self.is_train:
            data = self.getitem_general_train(idx)
        else:
            data = self.getitem_general_test(idx)
        return data


    def getitem_general_test(self, idx: int) -> Dict[str, Tuple]:
        anchor_and_pos = self.all_positive_pairs[idx]
        neg_sameuser = self.get_neg_sameuser(anchor_and_pos['anchor'], 'random')
        neg_diffuser = self.get_neg_diffuser(anchor_and_pos['anchor'], 'random')
        data = {
            'anchor': anchor_and_pos['anchor'],
            'positive': anchor_and_pos['positive'],
            'neg_sameuser': neg_sameuser,
            'neg_diffuser': neg_diffuser
        }
        return data


    def getitem_general_train(self, idx: int) -> Dict[str, Tuple]: # for training matching network
        anchor_and_pos_pair = self.all_positive_pairs[idx]
        prob_of_diff = 0.5
        random_pick_result = 1 if random.random() < prob_of_diff else 0
        if random_pick_result == 1:
            negative = self.get_neg_diffuser(anchor_and_pos_pair['anchor'], 'random')
        elif random_pick_result == 0:
            negative = self.get_neg_sameuser(anchor_and_pos_pair['anchor'], 'random')
        else:
            raise ValueError('Invalid random pick result. Select from [0, 1]')
        data = {
            'anchor': anchor_and_pos_pair['anchor'],
            'positive': anchor_and_pos_pair['positive'],
            'negative': negative
        }
        return data


    def get_neg_sameuser(self, anchor: Tuple, sampling_method: str) -> Tuple:
        anc_data, anc_devID, anc_winID, anc_userID, anc_label, *rest = anchor

        mask = [(tup['anchor'][self.USER_IDX] == anc_userID) &
                              (tup['anchor'][self.WIN_IDX] != anc_winID) # same user diff window
                              for tup in self.all_positive_pairs]
        valid_neg_pairs = list(itertools.compress(self.all_positive_pairs, mask))

        # We will consider 'positive' as negative candidate, and it should be different from the anchor device
        valid_neg_pairs = [tup for tup in valid_neg_pairs if tup['positive'][self.DEVICE_IDX] != anc_devID]
        if len(valid_neg_pairs) == 0:
            raise ValueError('No negative pair found for the same user, different window')

        if sampling_method == 'random':
            random_idx = random.randint(0, len(valid_neg_pairs) - 1)
            assert anc_devID != valid_neg_pairs[random_idx]['positive'][self.DEVICE_IDX], \
                '<><>======Same device selected as negative======<><>'
            negative_sample = valid_neg_pairs[random_idx]['positive']
        else:
            raise ValueError(f'Invalid negative sampling method {sampling_method}')
        return negative_sample


    def get_neg_diffuser(self, anchor: Tuple, sampling_method: str) -> Tuple:
        anc_data, anc_devID, anc_winID, anc_userID, anc_label, *rest = anchor

        mask = [tup['anchor'][self.USER_IDX] != anc_userID for tup in self.all_positive_pairs]
        valid_neg_pairs = list(itertools.compress(self.all_positive_pairs, mask))

        # We will consider 'positive' as negative candidate, and it should be different from the anchor device
        valid_neg_pairs = [tup for tup in valid_neg_pairs if tup['positive'][self.DEVICE_IDX] != anc_devID]
        if len(valid_neg_pairs) == 0:
            raise ValueError('No negative pair found for different user')

        if sampling_method == 'random':
            random_idx = random.randint(0, len(valid_neg_pairs) - 1)
            assert anc_devID != valid_neg_pairs[random_idx]['positive'][
                self.DEVICE_IDX], '<><>======Same device selected as negative======<><>'
            negative_sample = valid_neg_pairs[random_idx]['positive']
        else:
            raise ValueError(f'Invalid negative sampling method {sampling_method}')
        return negative_sample


    def create_windowed_data(self) -> List[pd.DataFrame]:
        user_list = self.df['user_id'].unique().tolist()
        label_list = self.df['label'].unique().tolist()
        session_list = self.df['session_id'].unique().tolist() if self.dataset_name == 'fatigueset' else [None]

        windowed_list = []
        for user_id in user_list:
            for session_id in session_list:
                for label in label_list:
                    curr_df = select_sub_df(self.df, user_id, label, session_id)
                    if curr_df.empty:
                        continue
                    curr_df = curr_df.reset_index()
                    for enum_win, window in enumerate(window_raw_data(self.dataset_name, curr_df, self.ws)):
                        windowed_list.append(window)
        return windowed_list


    def label_to_int(self, label: Union[int, str]) -> int:
        """Convert label name to integer. Valid for only Fatigueset (WESAD already has integer labels)"""
        if self.dataset_name == 'fatigueset':
            label_dict = {'baseline': 0, 'activity': 1, 'fatigue': 2, 'other': 3}
            return label_dict[label]
        elif self.dataset_name == 'wesad':
            return label


    def _validate_unique_items(self, window_df: pd.DataFrame) -> None:
        item_list = ['user_id', 'label', 'win_start_second']
        if self.dataset_name == 'fatigueset':
            item_list.append('session_id')
        for item in item_list:
            assert window_df[item].nunique() == 1, f'Error: {item} is not unique inside window'


    def get_device_columns(self) -> Dict[str, List[str]]:
        device_data_cols = {}
        raw_col_dict = get_raw_data_column_dict(self.dataset_name)
        for device in self.devices:
            curr_device_cols = []
            for modal in self.modals:
                col_names = raw_col_dict[modal][device]
                if modal == 'acc':
                    col_names = select_raw_acc_cols(col_names, self.acc_load)
                curr_device_cols.extend(col_names)
            device_data_cols[device] = curr_device_cols

        # self.logger.info(f"Device data columns:\n {device_data_cols}")
        return device_data_cols