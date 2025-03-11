"""
Bioq_EmbGen.py
"""

import os
import torch
import wandb
import logging
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import source.utils as utils
import torch.nn.functional as F
from torch.utils.data import DataLoader
from source.models import get_embgen_model
from source.data.data_loader import load_windowed_dataset
from source.methods import (init_wandb, create_exp_dir,
                            load_batch_data, keep_metadata,
                            extract_and_log_test_results)


EMB_DIST_METRIC = 'l1'

class Bioq_Embgen:
    def __init__(self, train_df: pd.DataFrame, test_df: pd.DataFrame, device: str,
                 args: argparse.Namespace, logger: logging.Logger):
        self.train_df = train_df
        self.test_df = test_df
        self.device = device
        self.args = args
        self.logger = logger
        self.dataset_name = args.dataset

        self.task = self.args.task
        self.method = self.args.method
        self.exp_dir = create_exp_dir(self.task, self.method, args, logger)
        init_wandb(self.task, self.method, args, logger) if self.args.wandb else None


    def run(self):
        trained_model = self.load_trained_model()
        if trained_model is None:
            trained_model = self.train_embgen()
        self.eval_embgen(trained_model)


    # Part 1: Embedding Generator
    def train_embgen(self):
        self.logger.info('Training embedding generator...')
        train_set = load_windowed_dataset(self.args.dataset, self.train_df, is_train=True, args=self.args, logger=self.logger)
        self.logger.info(f'Loaded windowed train set with {len(train_set)} samples.')

        train_loader = DataLoader(train_set, batch_size=self.args.em_bs, shuffle=True)
        input_dim = train_set.get_single_device_input_dim()

        embed_loss_fn = utils.get_embedding_loss(EMB_DIST_METRIC, self.args.margin, reduction='sum', swap=True).to(self.device)

        # Initialize models
        embgen_model = get_embgen_model(self.args.dataset, input_dim, self.args.em_dim)
        embgen_model = embgen_model.to(self.device)
        embgen_model.train()
        optimizer = self.get_optimizer_embgen(embgen_model)

        print_every = 2
        for epoch in tqdm(range(self.args.em_ep)):
            con_loss_list, total_loss_list = [], []
            for batch_idx, data in enumerate(train_loader):
                optimizer.zero_grad()
                data_dict, label_dict = {}, {}
                for data_type in ['anchor', 'positive', 'negative']:
                    data_dict[data_type] = data[data_type][train_set.DATA_IDX].to(self.device)
                    label_dict[data_type] = data[data_type][train_set.AUX_LABEL_IDX].to(self.device)

                # Get embeddings
                anchor_embeds = F.normalize(embgen_model(data_dict['anchor']), dim=1)
                positive_embeds = F.normalize(embgen_model(data_dict['positive']), dim=1)
                negative_embeds = F.normalize(embgen_model(data_dict['negative']), dim=1)

                # Loss_contrastive
                loss = embed_loss_fn(anchor_embeds, positive_embeds, negative_embeds)
                loss.backward()
                optimizer.step()

                con_loss_list.append(loss.item())

            avg_cntr_loss = np.mean(con_loss_list)
            if epoch % print_every == 0:
                print(f'[{epoch}] Contrastive Loss: {avg_cntr_loss:.3f}')
                self.log_training_wandb(epoch, avg_cntr_loss)

        self.save_model_cpt(embgen_model, input_dim)
        return embgen_model



    def eval_embgen(self, embedder):
        self.logger.info("Evaluating embedding generator model...")
        test_set = load_windowed_dataset(self.args.dataset, self.test_df, is_train=False, args=self.args, logger=self.logger)
        test_loader = DataLoader(test_set, batch_size=self.args.em_bs, shuffle=False)

        dc = utils.DataContainer() # data container to store results
        embedder = embedder.to(self.device)
        embedder.eval()
        with torch.no_grad():
            for batch in test_loader:
                data, metadata = load_batch_data(batch, test_set, self.dataset_name)
                embeddings = self._extract_and_keep_embeds(embedder, data, dc)
                keep_metadata(data, metadata, dc, self.dataset_name)
                self._compute_and_keep_embdists(embeddings, dc)

        # Extract FDR results, log to wandb, and save to disk (as json)
        extract_and_log_test_results(self.task, self.method, dc, self.exp_dir, self.args, self.logger)



    def _extract_and_keep_embeds(self, embedder, data, dc):
        embeddings = {}
        for data_type, d in data.items():
            d = d.to(self.device)
            embeddings[data_type] = F.normalize(embedder(d), dim=1).detach().cpu().numpy()
        for data_type, embedding in embeddings.items():
            dc.embedding[data_type].extend(embedding.tolist())
        return embeddings


    def _compute_and_keep_embdists(self, embeds, dc):
        pos_dist = utils.compute_embedding_distance(embeds['anchor'], embeds['positive'], EMB_DIST_METRIC)
        neg_sameuser_dist = utils.compute_embedding_distance(embeds['anchor'], embeds['neg_sameuser'], EMB_DIST_METRIC)
        neg_diffuser_dist = utils.compute_embedding_distance(embeds['anchor'], embeds['neg_diffuser'], EMB_DIST_METRIC)
        dc.emb_dist['pos_dist'].extend(pos_dist)
        dc.emb_dist['neg_sameuser_dist'].extend(neg_sameuser_dist)
        dc.emb_dist['neg_diffuser_dist'].extend(neg_diffuser_dist)


    def load_trained_model(self):
        full_cpt_path = os.path.join(self.exp_dir, 'cpt.pt')
        if not os.path.exists(full_cpt_path):
            return None

        cpt = torch.load(full_cpt_path)
        model = get_embgen_model(self.args.dataset, cpt['input_dim'], self.args.em_dim)
        model.load_state_dict(cpt['state_dict'])
        return model



    def save_model_cpt(self, model, input_dim):
        cpt = {'input_dim': input_dim,
               'state_dict': model.state_dict()}
        torch.save(cpt, os.path.join(self.exp_dir, 'cpt.pt'))


    def get_optimizer_embgen(self, embgen_model):
        if self.args.em_op == 'adam':
            optimizer = torch.optim.Adam(embgen_model.parameters(), lr=self.args.con_lr, weight_decay=self.args.con_wd)
        elif self.args.em_op == 'sgd':
            optimizer = torch.optim.SGD(embgen_model.parameters(), lr=self.args.con_lr, weight_decay=self.args.con_wd)
        else:
            raise ValueError(f'Invalid optimizer {self.args.optim}')
        return optimizer


    def log_training_wandb(self, epoch, con_loss):
        try:
            wandb.log({'epoch': epoch, 'contrast_loss': con_loss}) if self.args.wandb else None
        except Exception as e:
            pass



