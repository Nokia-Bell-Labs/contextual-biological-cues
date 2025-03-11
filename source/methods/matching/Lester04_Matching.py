"""
Lester04_Matching.py
"""

import logging
import argparse
import numpy as np
import pandas as pd
from sklearn.svm import SVC
import source.utils as utils
from torch.utils.data import DataLoader
from source.data.data_loader import load_windowed_dataset
from source.utils.coherence_utils import compute_coherence_raw
from sklearn.metrics import f1_score, precision_score, recall_score
from source.data.data_loader.loader_utils import get_dataset_sampling_rate
from source.methods import init_wandb, create_exp_dir, load_batch_data, extract_and_log_test_results


class Lester04_Matching:
    def __init__(self, train_df: pd.DataFrame, test_df: pd.DataFrame, device: str,
                 args: argparse.Namespace, logger: logging.Logger):
        self.train_df = train_df
        self.test_df = test_df
        self.device = device
        self.args = args
        self.logger = logger

        self.dataset_name = args.dataset
        self.fs = get_dataset_sampling_rate(self.args.dataset)

        self.task = self.args.task
        self.method = self.args.method
        self.exp_dir = create_exp_dir(self.task, self.method, args, logger)
        init_wandb(self.task, self.method, args, logger) if self.args.wandb else None


    def run(self):
        trained_svm = self.train_matching()
        self.eval_matching(trained_svm)

    def train_matching(self):
        train_set = load_windowed_dataset(self.args.dataset, self.train_df, is_train=True,
                                          args=self.args, logger=self.logger)
        train_loader = DataLoader(train_set, batch_size=self.args.m_bs, shuffle=True)

        pos_sim_list = []
        neg_sim_list = []  # includes both neg_sameuser and neg_diffuser

        for batch in train_loader:
            data, _ = load_batch_data(batch, train_set, self.dataset_name, train_matching=True)
            pos_sim, neg_sim = self.compute_coh_similarity_train(data)
            pos_sim_list.extend(pos_sim)
            neg_sim_list.extend(neg_sim)

        # Train SVM classifier
        svm_model = SVC(kernel='rbf', class_weight='balanced') # adjust class weight
        X_train = np.array(pos_sim_list + neg_sim_list).reshape(-1, 1) # reshape to 2D
        y_train = np.array([1] * len(pos_sim_list) + [0] * len(neg_sim_list))

        self.logger.info(f"[TRAIN] Sample size before removing NaNs: {X_train.shape}")
        nan_row_indices = np.any(np.isnan(X_train), axis=1)
        X_train = X_train[~nan_row_indices]
        y_train = y_train[~nan_row_indices]
        self.logger.info(f"[TRAIN] Sample size after removing NaNs: {X_train.shape}")

        svm_model.fit(X_train, y_train)

        return svm_model

    def compute_coh_similarity_train(self, data):
        anchor = data['anchor'].clone().cpu().numpy()
        positive = data['positive'].clone().cpu().numpy()
        negative = data['negative'].clone().cpu().numpy()

        res_pos_sim, res_neg_sim = self._compute_coh_similarity(anchor, positive, [negative])
        print(f"[train] shape of res_pos_sim: {res_pos_sim.shape}, shape of res_neg_sim: {res_neg_sim[0].shape}")
        return res_pos_sim, res_neg_sim[0] # Only one negative set in train


    def compute_coh_similarity_test(self, data):
        anchor = data['anchor'].clone().cpu().numpy()
        positive = data['positive'].clone().cpu().numpy()
        neg_sameuser = data['neg_sameuser'].clone().cpu().numpy()
        neg_diffuser = data['neg_diffuser'].clone().cpu().numpy()

        res_pos_dist, res_neg_dists = self._compute_coh_similarity(anchor, positive, [neg_sameuser, neg_diffuser])
        res_neg_dist = np.concatenate(res_neg_dists, axis=0) # (2*bs, num_sensor_cols)
        print(f"[test] shape of res_pos_dist: {res_pos_dist.shape}, shape of res_neg_dist: {res_neg_dist.shape}")
        return res_pos_dist, res_neg_dist


    def _compute_coh_similarity(self, anchor, positive, negatives_list):
        pos_sim_list = []
        neg_sim_list = [[] for _ in range(len(negatives_list))]
        for i in range(anchor.shape[0]): # traversing across the batch size (accessing each row)
            # Compute pos similarities, distance computed for a pair
            pos_sim = compute_coherence_raw(anchor[i], positive[i], self.fs)
            pos_sim_list.append(pos_sim)

            # Compute neg similarities
            for j, negative in enumerate(negatives_list):
                neg_sim = compute_coherence_raw(anchor[i], negative[i], self.fs)
                neg_sim_list[j].append(neg_sim)

        res_pos_sim = np.stack(pos_sim_list, axis=0) # (bs, num_sensor_cols)
        res_neg_sim = [np.stack(neg_sim_list_i, axis=0) for neg_sim_list_i in neg_sim_list] # (bs, num_sensor_cols)
        return res_pos_sim, res_neg_sim



    def eval_matching(self, trained_svm):
        test_set = load_windowed_dataset(self.args.dataset, self.test_df, is_train=False,
                                         args=self.args, logger=self.logger)
        test_loader = DataLoader(test_set, batch_size=self.args.m_bs, shuffle=False)

        pos_sim_list = []
        neg_sim_list = []
        for batch in test_loader:
            data, _ = load_batch_data(batch, test_set, self.dataset_name)
            pos_sim, neg_sim = self.compute_coh_similarity_test(data)
            pos_sim_list.extend(pos_sim)
            neg_sim_list.extend(neg_sim)

        X_test = np.array(pos_sim_list + neg_sim_list).reshape(-1, 1) # reshape to 2D
        y_test = np.array([1] * len(pos_sim_list) + [0] * len(neg_sim_list))

        self.logger.info(f"[TEST] Sample size before removing NaNs: {X_test.shape}")
        nan_row_indices = np.any(np.isnan(X_test), axis=1)
        X_test = X_test[~nan_row_indices]
        y_test = y_test[~nan_row_indices]
        self.logger.info(f"[TEST] Sample size after removing NaNs: {X_test.shape}")

        y_pred = trained_svm.predict(X_test)
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


