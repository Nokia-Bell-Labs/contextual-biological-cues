"""
Raw_Matching.py
"""

import logging
import argparse
import numpy as np
import pandas as pd
from sklearn.svm import SVC
import source.utils as utils
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
from source.data.data_loader import load_windowed_dataset
from source.data.data_loader.loader_utils import get_meta_columns
from sklearn.metrics import f1_score, precision_score, recall_score
from source.methods import init_wandb, create_exp_dir, load_batch_data, extract_and_log_test_results



EMB_DIST_METRIC = 'l1'

class Raw_Matching:
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
        # scale the data (as the emb-dist needs to be scaled)
        self.min_max_scale_data() # both train and test

        trained_model = self.train_matching()
        self.eval_matching(trained_model)


    def min_max_scale_data(self):
        for df in [self.train_df, self.test_df]:
            scaler = MinMaxScaler()
            df.loc[:, self.sensor_columns] = scaler.fit_transform(df.loc[:, self.sensor_columns])

        # Check if the scaling is correct
        tolerance = 1e-9
        for df in [self.train_df, self.test_df]:
            max_vals = df[self.sensor_columns].max()
            min_vals = df[self.sensor_columns].min()
            # Assert max is close to 1, allowing for small floating-point errors
            assert np.all(np.isclose(max_vals, 1, atol=tolerance) | (max_vals <= 1)), "Max value exceeds 1"
            assert np.all(np.isclose(min_vals, 0, atol=tolerance) | (min_vals >= 0)), "Min value is below 0"



    def train_matching(self):
        train_set = load_windowed_dataset(self.dataset_name, self.train_df, is_train=True,
                                          args=self.args, logger=self.logger)
        train_loader = DataLoader(train_set, batch_size=self.args.m_bs, shuffle=True)

        pos_dist_list = []
        neg_dist_list = []  # includes both neg_sameuser and neg_diffuser

        for batch in train_loader:
            data, _, = load_batch_data(batch, train_set, self.dataset_name, train_matching=True)
            pos_dist, neg_dist = self.compute_raw_distance_train(data)
            pos_dist_list.extend(pos_dist)
            neg_dist_list.extend(neg_dist)

        # Train SVM classifier
        svm_model = SVC(kernel='rbf', class_weight='balanced') # adjust class weight
        X_train = np.array(pos_dist_list + neg_dist_list)
        y_train = np.array([1] * len(pos_dist_list) + [0] * len(neg_dist_list))

        self.logger.info(f"[TRAIN] Sample size before removing NaNs: {X_train.shape}")
        nan_row_indices = np.any(np.isnan(X_train), axis=1)
        X_train = X_train[~nan_row_indices]
        y_train = y_train[~nan_row_indices]
        self.logger.info(f"[TRAIN] Sample size after removing NaNs: {X_train.shape}")

        svm_model.fit(X_train, y_train)
        return svm_model


    def compute_raw_distance_train(self, data):
        anchor = data['anchor'].clone().cpu().numpy()
        positive = data['positive'].clone().cpu().numpy()
        negative = data['negative'].clone().cpu().numpy()

        res_pos_dist, res_neg_dists = self.compute_distances(anchor, positive, [negative])
        return res_pos_dist, res_neg_dists[0] # Only one negative set in train


    def compute_raw_distance_test(self, data):
        anchor = data['anchor'].clone().cpu().numpy()
        positive = data['positive'].clone().cpu().numpy()
        neg_sameuser = data['neg_sameuser'].clone().cpu().numpy()
        neg_diffuser = data['neg_diffuser'].clone().cpu().numpy()

        res_pos_dist, res_neg_dists = self.compute_distances(anchor, positive, [neg_sameuser, neg_diffuser])
        res_neg_dist = np.concatenate(res_neg_dists, axis=0) # (2*bs, num_sensor_cols)
        return res_pos_dist, res_neg_dist


    def compute_distances(self, anchor, positive, negatives_list):
        """Helper function to compute distances between anchor, positive, and multiple negative sets."""
        pos_dist_list = []
        neg_dist_lists = [[] for _ in range(len(negatives_list))]

        num_sensor_cols = anchor.shape[2]
        for col_idx in range(num_sensor_cols): # traverse over sensor columns
            anchor_col = anchor[:, :, col_idx]
            positive_col = positive[:, :, col_idx]

            # Compute positive distance
            # distance is computed for (batch_size, num_pairs_pos)
            dist_pos = utils.compute_embedding_distance(anchor_col, positive_col, EMB_DIST_METRIC)
            pos_dist_list.append(dist_pos)

            # Compute negative distances for each negative set
            for i, negative in enumerate(negatives_list):
                neg_col = negative[:, :, col_idx]
                # distance is computed for (batch_size, num_pairs_neg)
                dist_neg = utils.compute_embedding_distance(anchor_col, neg_col, EMB_DIST_METRIC)
                neg_dist_lists[i].append(dist_neg)

        res_pos_dist = np.stack(pos_dist_list, axis=1)  # (bs, num_feat_cols)
        res_neg_dists = [np.stack(neg_dist_list, axis=1) for neg_dist_list in neg_dist_lists]  # List of (bs, num_feat_cols)

        return res_pos_dist, res_neg_dists



    def eval_matching(self, trained_model):
        test_set = load_windowed_dataset(self.dataset_name, self.test_df, is_train=False,
                                         args=self.args, logger=self.logger)
        test_loader = DataLoader(test_set, batch_size=self.args.m_bs, shuffle=False)

        pos_dist_list = []
        neg_dist_list = []  # includes both neg_sameuser and neg_diffuser
        for batch in test_loader:
            data, _ = load_batch_data(batch, test_set, self.dataset_name)
            pos_dist, neg_dist = self.compute_raw_distance_test(data)
            pos_dist_list.extend(pos_dist)
            neg_dist_list.extend(neg_dist)
        X_test = np.array(pos_dist_list + neg_dist_list)
        y_test = np.array([1] * len(pos_dist_list) + [0] * len(neg_dist_list))

        self.logger.info(f"[TEST] Sample size before removing NaNs: {X_test.shape}")
        nan_row_indices = np.any(np.isnan(X_test), axis=1)
        X_test = X_test[~nan_row_indices]
        y_test = y_test[~nan_row_indices]
        self.logger.info(f"[TEST] Sample size after removing NaNs: {X_test.shape}")

        y_pred = trained_model.predict(X_test)
        test_eer, _ = utils.calculate_eer(y_test, y_pred)
        test_f1_macro = f1_score(y_test, y_pred, average='macro')
        test_precision = precision_score(y_test, y_pred, average='macro')
        test_recall = recall_score(y_test, y_pred, average='macro')

        result_dict = {
            'test_eer': test_eer,
            'test_f1_macro': test_f1_macro,
            'test_precision': test_precision,
            'test_recall': test_recall
        }
        self.logger.info(f"[{self.task}-{self.method}] Test EER: {test_eer:.4f} | Test F1-macro: {test_f1_macro:.4f}")
        extract_and_log_test_results(self.task, self.method, result_dict, self.exp_dir, self.args, self.logger)





















