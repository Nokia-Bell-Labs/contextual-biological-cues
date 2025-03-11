"""
Raw_EmbGen.py
"""

import logging
import argparse
import numpy as np
import pandas as pd
import source.utils as utils
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
from source.data.data_loader import load_windowed_dataset
from source.data.data_loader.loader_utils import get_meta_columns
from source.methods import (init_wandb, create_exp_dir, load_batch_data,
                            keep_metadata, extract_and_log_test_results)


EMB_DIST_METRIC = 'l1'

class Raw_Embgen:
    def __init__(self, train_df: pd.DataFrame, test_df: pd.DataFrame, device: str,
                 args: argparse.Namespace, logger: logging.Logger):
        self.train_df = train_df
        self.test_df = test_df
        self.device = device
        self.args = args
        self.logger = logger

        self.dataset_name = args.dataset
        meta_columns = get_meta_columns(self.dataset_name)
        self.sensor_columns = [col for col in self.test_df.columns if col not in meta_columns]

        self.task = self.args.task
        self.method = self.args.method
        self.exp_dir = create_exp_dir(self.task, self.method, args, logger)
        init_wandb(self.task, self.method, args, logger) if self.args.wandb else None


    def run(self):
        """Raw_Embgen does not train any model."""
        # For Raw_Embgen, we need to scale the features to [0, 1] before evaluating
        # This is because we will be computing average across all raw sensor types and they
        # might be in different scales, we need to normalize them to [0, 1] before computing the average
        self.min_max_scale_test_data()
        self.eval_embgen()

    def min_max_scale_test_data(self):
        # Scale each sensor column to [0, 1] range
        scaler = MinMaxScaler()
        self.logger.info(f"MinMax scaling the feature columns to [0, 1] range..."
                         f"Columns: {self.sensor_columns}")
        self.test_df.loc[:, self.sensor_columns] = scaler.fit_transform(self.test_df.loc[:, self.sensor_columns])


    def eval_embgen(self):
        self.logger.info(f"Evaluating embedding generation algorithm {self.args.method}...")

        test_set = load_windowed_dataset(self.dataset_name, self.test_df, is_train=False,
                                         args=self.args, logger=self.logger)
        test_loader = DataLoader(test_set, batch_size=self.args.em_bs, shuffle=False)

        dc = utils.DataContainer()
        for batch in test_loader:
            data, metadata = load_batch_data(batch, test_set, self.dataset_name)
            keep_metadata(data, metadata, dc, self.dataset_name)
            self._compute_and_keep_raw_distances(data, dc)

        # Extract FDR results, log to wand and save to disk (as json)
        extract_and_log_test_results(self.task, self.method, dc, self.exp_dir, self.args, self.logger)


    def _compute_and_keep_raw_distances(self, data, dc):
        anchor = data['anchor'].clone().cpu().numpy()
        positive = data['positive'].clone().cpu().numpy()
        neg_sameuser = data['neg_sameuser'].clone().cpu().numpy()
        neg_diffuser = data['neg_diffuser'].clone().cpu().numpy()

        pos_dist_list = []
        neg_sameuser_dist_list = []
        neg_diffuser_dist_list = []
        # print(f"anchor_.shape: {anchor.shape}")

        num_sensor_cols = anchor.shape[2]

        for col_idx in range(num_sensor_cols):
            anchor_col = anchor[:, :, col_idx] # (bs, seq_len) # seq_len: ws * sampling_rate
            positive_col = positive[:, :, col_idx]
            neg_sameuser_col = neg_sameuser[:, :, col_idx]
            neg_diffuser_col = neg_diffuser[:, :, col_idx]
            # print(f"anchor_col_shape: {anchor_col.shape}") # debug

            # Compute distance between anchor and positive
            dist_pos = utils.compute_embedding_distance(anchor_col, positive_col, EMB_DIST_METRIC)
            dist_neg_sameuser = utils.compute_embedding_distance(anchor_col, neg_sameuser_col, EMB_DIST_METRIC)
            dist_neg_diffuser = utils.compute_embedding_distance(anchor_col, neg_diffuser_col, EMB_DIST_METRIC)

            pos_dist_list.append(dist_pos)
            neg_sameuser_dist_list.append(dist_neg_sameuser)
            neg_diffuser_dist_list.append(dist_neg_diffuser)

        # Average distance across all sensor types
        pos_dist = np.nanmean(pos_dist_list, axis=0)
        neg_sameuser_dist = np.nanmean(neg_sameuser_dist_list, axis=0)
        neg_diffuser_dist = np.nanmean(neg_diffuser_dist_list, axis=0)

        # self.logger.info(f"Shape of pos_dist: {pos_dist.shape}")
        # self.logger.info(f"Shape of neg_sameuser_dist: {neg_sameuser_dist.shape}")
        # self.logger.info(f"Shape of neg_diffuser_dist: {neg_diffuser_dist.shape}")

        dc.emb_dist['pos_dist'].extend(pos_dist)
        dc.emb_dist['neg_sameuser_dist'].extend(neg_sameuser_dist)
        dc.emb_dist['neg_diffuser_dist'].extend(neg_diffuser_dist)


