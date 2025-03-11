
import torch
import random
import logging
import argparse
import itertools
import numpy as np
import pandas as pd
from typing import Union, List, Dict, Tuple
from source.data.data_loader.loader_utils import (get_feature_column_dict,
                                        get_device_ID_dict, get_num_classes)

FILTER_PERCENTAGE = 0.1
PROB_DIFF_USER_SELECTION = 0.8 # To encourage the selection of more diverse users


class Feat_Loader:
    def __init__(self, dataset_name: str, df: pd.DataFrame, is_train: bool,
                 args: argparse.Namespace, logger: logging.Logger):
        self.dataset_name = dataset_name
        self.df = df
        self.is_train = is_train
        self.args = args
        self.logger = logger

        self.method = args.method
        self.modals = sorted(args.modals) if (len(args.modals) > 1 and isinstance(args.modals, list)) else args.modals
        self.devices = sorted(args.devices) # always more than a single device
        self.device_col_dict = self.get_device_columns() # {'h': ['ear_acc_left_mag:min', ..],.. }
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
        initial_positive_pair_list = []

        list_of_unique_device_pairs = list(itertools.combinations(self.devices, 2))
        for _, row in self.df.iterrows():
            device_data_dict = {}  # h: torch_tensor_data, w: torch_tensor_data, etc.
            for device_name in self.devices:
                curr_device_data = torch.tensor(
                    np.array(row[self.device_col_dict[device_name]].astype(float)), dtype=torch.float)
                device_data_dict[device_name] = curr_device_data

            # Create all possible pairs of devices
            for device_pair_names in list_of_unique_device_pairs:
                curr_datapoint = self._make_positive_pair(device_pair_names, device_data_dict, row)
                initial_positive_pair_list.append(curr_datapoint)

        self.all_positive_pairs = initial_positive_pair_list


    def _make_positive_pair(self, dev_names: Tuple[str, str], dev_data_dict: Dict[str, torch.Tensor],
                            row: pd.Series) -> Dict[str, Tuple]:
        dev1_name, dev2_name = dev_names
        dev1_ID, dev2_ID = self.device_ID_dict[dev1_name], self.device_ID_dict[dev2_name]
        dev1_data, dev2_data = dev_data_dict[dev1_name], dev_data_dict[dev2_name]
        label = self.label_to_int(row['label'])

        if self.dataset_name == 'fatigueset':
            return {
                'anchor': (dev1_data, dev1_ID, row['win_start_second'], row['user_id'], label, row['session_id']),
                'positive': (dev2_data, dev2_ID, row['win_start_second'], row['user_id'], label, row['session_id'])}
        elif self.dataset_name == 'wesad':
            return {
                'anchor': (dev1_data, dev1_ID, row['win_start_second'], row['user_id'], label),
                'positive': (dev2_data, dev2_ID, row['win_start_second'], row['user_id'], label)}
        else:
            raise ValueError(f"Invalid dataset name: {self.dataset_name}")


    def __len__(self) -> int:
        return len(self.all_positive_pairs)


    def __getitem__(self, idx) -> Dict[str, Tuple]:
        if self.is_train:
            if self.method == 'bioq':
                data = self.getitem_bioq_train(idx)  # Same for both 'matching' and 'embgen'
            elif self.args.task == 'matching':  # NOT BioQ method, and mode is matching
                data = self.getitem_matching_general_train(idx)
            elif self.args.task == 'embgen':  # NOT BioQ method, and mode is embgen
                raise ValueError(f'Invalid method {self.method} for embgen mode')
            else:
                raise ValueError(f'Invalid task {self.args.task}')
        else:  # Test mode
            data = self.getitem_general_test(idx)
        return data



    def getitem_bioq_train(self, idx) -> Dict[str, Tuple]:
        anchor_and_pos = self.all_positive_pairs[idx]
        random_pick_result = 1 if random.random() < PROB_DIFF_USER_SELECTION else 0
        if random_pick_result == 1:
            negative = self.get_neg_diffuser(anchor_and_pos['anchor'], sampling_method=self.args.neg_sampling)
        elif random_pick_result == 0:
            try:
                negative = self.get_neg_sameuser(anchor_and_pos['anchor'], sampling_method=self.args.neg_sampling)
            except ValueError as e:
                print(e)
                negative = self.get_neg_diffuser(anchor_and_pos['anchor'], sampling_method=self.args.neg_sampling)
        else:
            raise ValueError('Invalid random pick result. Select from [0, 1]')

        data = {
            'anchor': anchor_and_pos['anchor'],
            'positive': anchor_and_pos['positive'],
            'negative': negative
        }
        return data


    def getitem_matching_general_train(self, idx) -> Dict[str, Tuple]:
        anchor_and_pos_pair = self.all_positive_pairs[idx]
        prob_of_diff = 0.5
        random_pick_result = 1 if random.random() < prob_of_diff else 0
        if random_pick_result == 1:
            negative = self.get_neg_diffuser(anchor_and_pos_pair['anchor'], sampling_method='random')
        elif random_pick_result == 0:
            negative = self.get_neg_sameuser(anchor_and_pos_pair['anchor'], sampling_method='random')
        else:
            raise ValueError('Invalid random pick result. Select from [0, 1]')
        data = {
            'anchor': anchor_and_pos_pair['anchor'],
            'positive': anchor_and_pos_pair['positive'],
            'negative': negative
        }
        return data


    def getitem_general_test(self, idx) -> Dict[str, Tuple]:
        anchor_and_pos = self.all_positive_pairs[idx]
        neg_sameuser = self.get_neg_sameuser(anchor_and_pos['anchor'], 'random')
        neg_diffuser = self.get_neg_diffuser(anchor_and_pos['anchor'], 'random')
        data = {
            'anchor': anchor_and_pos['anchor'],
            'positive': anchor_and_pos['positive'],
            'neg_sameuser': neg_sameuser,
            'neg_diffuser': neg_diffuser}
        return data


    def get_neg_sameuser(self, anchor: Tuple, sampling_method: str) -> Tuple:
        anc_data, anc_devID, anc_winID, anc_userID, anc_label, *rest = anchor

        # Find all negative pairs with the same user, but different window
        mask = [(tup['anchor'][self.USER_IDX] == anc_userID) &
                            (tup['anchor'][self.WIN_IDX] != anc_winID)
                            for tup in self.all_positive_pairs]
        valid_neg_pairs = list(itertools.compress(self.all_positive_pairs, mask))

        # We will consider 'positive' as negative candidate, and it should be different from the anchor device
        valid_neg_pairs = [tup for tup in valid_neg_pairs if tup['positive'][self.DEVICE_IDX] != anc_devID]
        if len(valid_neg_pairs) == 0:
            raise ValueError('No negative pair found for the same user, different window')

        # Sampling strategy for negative pair: [random, cut_top, cut_bottom, middle]
        if sampling_method == 'random':
            random_idx = random.randint(0, len(valid_neg_pairs)-1)
            assert anc_devID != valid_neg_pairs[random_idx]['positive'][self.DEVICE_IDX], \
                '<><>======Same device selected as negative======<><>'
            negative_sample = valid_neg_pairs[random_idx]['positive']
        elif sampling_method in ['cut_bottom', 'middle', 'cut_top']:
            min_samples = self.args.min_neg
            randomly_selected_candidates = random.sample(valid_neg_pairs, min(min_samples, len(valid_neg_pairs)))
            negative_sample = self.distance_based_negative_sampling(sampling_method, randomly_selected_candidates, anc_data)
        else:
            raise ValueError(f'Invalid negative sampling method {sampling_method}')
        return negative_sample


    def get_neg_diffuser(self, anchor: Tuple, sampling_method: str) -> Tuple:
        anc_data, anc_devID, anc_winID, anc_userID, anc_label, *rest = anchor

        # Find all negative pairs with different user
        mask = [tup['anchor'][self.USER_IDX] != anc_userID for tup in self.all_positive_pairs]
        valid_neg_pairs = list(itertools.compress(self.all_positive_pairs, mask))

        # We will consider 'positive' as negative candidate, and it should be different from the anchor device
        valid_neg_pairs = [tup for tup in valid_neg_pairs if tup['positive'][self.DEVICE_IDX] != anc_devID]
        if len(valid_neg_pairs) == 0:
            raise ValueError('No negative pair found for different user')

        # Sampling strategy for negative pair: [random, cut_top, cut_bottom, middle]
        if sampling_method == 'random':
            random_idx = random.randint(0, len(valid_neg_pairs)-1)
            assert anc_devID != valid_neg_pairs[random_idx]['positive'][
            self.DEVICE_IDX], '<><>======Same device selected as negative======<><>'
            negative_sample = valid_neg_pairs[random_idx]['positive']
        elif sampling_method in ['cut_bottom', 'middle', 'cut_top']:
            min_samples = self.args.min_neg
            randomly_selected_candidates = random.sample(valid_neg_pairs, min(min_samples, len(valid_neg_pairs)))
            negative_sample = self.distance_based_negative_sampling(sampling_method, randomly_selected_candidates, anc_data)
        else:
            raise ValueError(f'Invalid negative sampling method {sampling_method}')
        return negative_sample


    def distance_based_negative_sampling(self, sampling_method: str, candidate_list: List[Dict[str, Tuple]],
                                         anchor_data: torch.Tensor) -> Tuple:
        l1_dist_list = []
        for neg_sample in candidate_list:
            data, *rest = neg_sample['positive']
            l1_dist = torch.sum(torch.abs(anchor_data - data))
            l1_dist_list.append(l1_dist)

        num_pairs_to_exclude = int(FILTER_PERCENTAGE * len(l1_dist_list))
        sorted_l1_dist_list = sorted(l1_dist_list)
        threshold_bottom = float('-inf')
        threshold_top = float('inf')
        if sampling_method in ['cut_bottom', 'middle']:
            threshold_bottom = sorted_l1_dist_list[num_pairs_to_exclude]
        if sampling_method in ['cut_top', 'middle']:
            threshold_top = sorted_l1_dist_list[-num_pairs_to_exclude]

        filtered_negative_samples = [tup for tup, dist in zip(candidate_list, l1_dist_list) if
                                     (dist >= threshold_bottom) & (dist < threshold_top)]
        if len(filtered_negative_samples) == 0:
            raise ValueError('No negative samples with different user id and different activity label')
        random_idx = random.randint(0, len(filtered_negative_samples) - 1)
        negative_sample = filtered_negative_samples[random_idx]['positive']
        return negative_sample



    def label_to_int(self, label: Union[int, str]) -> int:
        """Convert label name to integer. Valid for only Fatigueset (WESAD already has integer labels)"""
        if self.dataset_name == 'fatigueset':
            label_dict = {'baseline': 0, 'activity': 1, 'fatigue': 2, 'other': 3}
            return label_dict[label]
        elif self.dataset_name == 'wesad':
            return label


    def get_device_columns(self) -> dict:
        """Get the columns of the dataframe that correspond to the target devices"""
        feat_col_dict = get_feature_column_dict(self.dataset_name)
        device_columns = {}
        for device in self.devices:
            device_columns[device] = []
            for modal in self.modals:
                device_columns[device] += feat_col_dict[modal][device]
        return device_columns



    def get_aux_task_num_classes(self) -> int:
        """Number of classes in an auxiliary task"""
        num_classes = get_num_classes(self.dataset_name)
        return num_classes


    def get_single_device_input_dim(self) -> int:
        return len(self.device_col_dict[self.devices[0]])
