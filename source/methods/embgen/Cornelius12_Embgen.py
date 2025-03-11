"""
Embedding generation method using Cornelius12
- Cornelius, C. T., & Kotz, D. F. (2012). Recognizing whether sensors are on the same body. Pervasive and Mobile Computing..
"""

import logging
import argparse
import pandas as pd
import source.utils as utils
from torch.utils.data import DataLoader
from source.data.data_loader import load_windowed_dataset
from source.utils.coherence_utils import compute_coherence_features
from source.data.data_loader.loader_utils import get_dataset_sampling_rate
from source.methods import (init_wandb, create_exp_dir, load_batch_data,
                            keep_metadata, extract_and_log_test_results)


class Cornelius12_Embgen:
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
        """For embedding generation, we do not train, as we are only interested in distances.
           Cornelius12 trains model for matching and it is implemented in the matching method."""
        self.eval_embgen()


    def eval_embgen(self):
        self.logger.info(f"Evaluating embedding generation algorithm {self.args.method}...")
        test_set = load_windowed_dataset(self.args.dataset, self.test_df, is_train=False,
                                         args=self.args, logger=self.logger)
        test_loader = DataLoader(test_set, batch_size=self.args.em_bs, shuffle=False)

        dc = utils.DataContainer()
        for batch in test_loader:
            data, metadata = load_batch_data(batch, test_set, self.dataset_name)
            keep_metadata(data, metadata, dc, self.dataset_name)
            self._compute_and_keep_coherence_features(data, dc)

        # Extract FDR results, log to wandb and save to disk (as json)
        extract_and_log_test_results(self.task, self.method, dc, self.exp_dir, self.args, self.logger)



    def _compute_and_keep_coherence_features(self, data, dc):
        anchor = data['anchor'].clone().cpu().numpy()
        positive = data['positive'].clone().cpu().numpy()
        neg_sameuser = data['neg_sameuser'].clone().cpu().numpy()
        neg_diffuser = data['neg_diffuser'].clone().cpu().numpy()

        for i in range(anchor.shape[0]):
            # for each data in the batch
            pos_sim = compute_coherence_features(anchor[i], positive[i], self.fs, self.task)
            neg_sameuser_sim = compute_coherence_features(anchor[i], neg_sameuser[i], self.fs, self.task)
            neg_diffuser_sim = compute_coherence_features(anchor[i], neg_diffuser[i], self.fs, self.task)
            dc.coherence_sim['pos_sim'].append(pos_sim)
            dc.coherence_sim['neg_sameuser_sim'].append(neg_sameuser_sim)
            dc.coherence_sim['neg_diffuser_sim'].append(neg_diffuser_sim)

