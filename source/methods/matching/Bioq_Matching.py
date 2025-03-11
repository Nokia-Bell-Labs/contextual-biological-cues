"""
Bioq_Matching.py
"""

import os
import wandb
import torch
import logging
import argparse
import pandas as pd
import torch.nn as nn
import source.utils as utils
import torch.nn.functional as F
from torch.utils.data import DataLoader
from source.data.data_loader import load_windowed_dataset
from source.models import get_embgen_model, get_matching_model
from sklearn.metrics import f1_score, precision_score, recall_score
from source.methods import init_wandb, create_exp_dir, extract_and_log_test_results


EMB_DIST_METRIC = 'l1'

class Bioq_Matching:
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
        self.matching_exp_dir, self.embgen_exp_dir = create_exp_dir(self.task, self.method, args, logger)
        init_wandb(self.task, self.method, args, logger) if self.args.wandb else None



    def run(self):
        """Matching requires trained embedding generator model."""
        trained_embgen_model = self.load_trained_model('embgen')
        if trained_embgen_model is None:
            raise ValueError('Embedding generator must be trained before matching model.')

        trained_matching_model = self.load_trained_model('matching')
        if trained_matching_model is None:
            trained_matching_model = self.train_matching(trained_embgen_model)

        self.eval_matching(trained_embgen_model, trained_matching_model)


    def load_trained_model(self, task):
        if task == 'embgen':
            full_cpt_path = os.path.join(self.embgen_exp_dir, 'cpt.pt')
            if not os.path.exists(full_cpt_path):
                return None
            cpt = torch.load(full_cpt_path)
            model = get_embgen_model(self.args.dataset, cpt['input_dim'], self.args.em_dim)
            model.load_state_dict(cpt['state_dict'])
            return model

        elif task == 'matching':
            full_cpt_path = os.path.join(self.matching_exp_dir, 'matching_cpt.pt')
            if not os.path.exists(full_cpt_path):
                return None
            cpt = torch.load(full_cpt_path)
            model = get_matching_model(self.args.dataset, self.args.em_dim)
            model.load_state_dict(cpt['state_dict'])
            return model


    def train_matching(self, embgen_model):
        train_set = load_windowed_dataset(self.args.dataset, self.train_df, is_train=True, args=self.args, logger=self.logger)
        train_loader = DataLoader(train_set, batch_size=self.args.m_bs, shuffle=True)
        matching_model = get_matching_model(self.args.dataset, self.args.em_dim)

        # Move embgen model to device and set evaluation mode
        embgen_model = embgen_model.to(self.device)
        embgen_model.eval() # embgen is in eval mode
        matching_model = matching_model.to(self.device)
        matching_model.train()

        # Initialize optimizer and loss function
        optimizer = self.get_optimizer_matching(matching_model)
        matching_loss_fn = nn.BCEWithLogitsLoss()

        print_every = 2
        for epoch in range(self.args.m_ep):
            batch_loss = 0
            y_pred_list, y_true_list = [], []
            for batch_idx, data in enumerate(train_loader):
                optimizer.zero_grad()
                data_dict = {dtype: data[dtype][train_set.DATA_IDX].to(self.device) for dtype in ['anchor', 'positive', 'negative']}

                # Add .detach() to prevent backpropagation to embgen model
                anchor_emb = F.normalize(embgen_model(data_dict['anchor']).detach(), dim=1)
                positive_emb = F.normalize(embgen_model(data_dict['positive'].detach()), dim=1)
                negative_emb = F.normalize(embgen_model(data_dict['negative'].detach()), dim=1)

                # Get matching predictions
                emb_matching_true = torch.cat([anchor_emb, positive_emb], dim=1)
                emb_matching_false = torch.cat([anchor_emb, negative_emb], dim=1)
                emb_all = torch.cat([emb_matching_true, emb_matching_false], dim=0)

                labels_1 = torch.ones(emb_matching_true.shape[0]).to(self.device)
                labels_0 = torch.zeros(emb_matching_false.shape[0]).to(self.device)
                labels_all = torch.cat([labels_1, labels_0], dim=0)

                y_preds = matching_model(emb_all).squeeze()
                loss = matching_loss_fn(y_preds, labels_all.float())

                y_preds = torch.round(torch.sigmoid(y_preds))
                y_pred_list.extend(y_preds.cpu().detach().numpy())
                y_true_list.extend(labels_all.cpu().detach().numpy())

                loss.backward()
                optimizer.step()
                batch_loss += loss.item()

            epoch_loss = batch_loss / len(train_loader)
            train_eer, _ = utils.calculate_eer(y_true_list, y_pred_list)
            train_f1_macro = f1_score(y_true_list, y_pred_list, average='macro')
            if epoch % print_every == 0:
                print(f'[{epoch}] Train Loss: {epoch_loss:.4f}, Train EER: {train_eer:.4f}, Train F1: {train_f1_macro:.4f}')
                self.log_matching_training_wandb(epoch, epoch_loss, train_eer, train_f1_macro)

        self.save_matching_model_cpt(matching_model)  # input_dim not needed for matching model
        return matching_model


    def save_matching_model_cpt(self, model):
        cpt = {'state_dict': model.state_dict()}
        torch.save(cpt, os.path.join(self.matching_exp_dir, 'matching_cpt.pt'))


    def get_optimizer_matching(self, matching_model):
        if self.args.m_op == 'adam':
            optimizer = torch.optim.Adam(matching_model.parameters(), lr=self.args.m_lr, weight_decay=self.args.m_wd)
        elif self.args.m_op == 'sgd':
            optimizer = torch.optim.SGD(matching_model.parameters(), lr=self.args.m_lr, weight_decay=self.args.m_wd)
        else:
            raise ValueError('Invalid optimizer for matching model.')
        return optimizer


    def log_matching_training_wandb(self, epoch, epoch_loss, train_eer, train_f1_macro):
        try:
            wandb.log({'epoch': epoch,
                       'epoch_loss': epoch_loss,
                       'train_eer': train_eer,
                       'train_f1_macro': train_f1_macro}) if self.args.wandb else None
        except Exception as e:
            pass


    def eval_matching(self, embgen_model, matching_model):
        test_set = load_windowed_dataset(self.args.dataset, self.test_df, is_train=False, args=self.args, logger=self.logger)
        test_loader = DataLoader(test_set, batch_size=self.args.m_bs, shuffle=False)

        embgen_model = embgen_model.to(self.device)
        embgen_model.eval()
        matching_model = matching_model.to(self.device)
        matching_model.eval()

        y_pred_list, y_true_list = [], []
        with torch.no_grad():
            for batch_idx, data in enumerate(test_loader):
                data_dict = {dtype: data[dtype][test_set.DATA_IDX].to(self.device)
                             for dtype in ['anchor', 'positive', 'neg_sameuser', 'neg_diffuser']}
                anchor_emb = F.normalize(embgen_model(data_dict['anchor']), dim=1)
                positive_emb = F.normalize(embgen_model(data_dict['positive']), dim=1)
                neg_sameuser_emb = F.normalize(embgen_model(data_dict['neg_sameuser']), dim=1)
                neg_diffuser_emb = F.normalize(embgen_model(data_dict['neg_diffuser']), dim=1)

                emb_matching_true = torch.cat([anchor_emb, positive_emb], dim=1)
                emb_matching_false = torch.cat([torch.cat([anchor_emb, neg_sameuser_emb], dim=1),
                                                torch.cat([anchor_emb, neg_diffuser_emb], dim=1)], dim=0)
                emb_all = torch.cat([emb_matching_true, emb_matching_false], dim=0)

                labels_1 = torch.ones(emb_matching_true.shape[0]).to(self.device)
                labels_0 = torch.zeros(emb_matching_false.shape[0]).to(self.device)
                labels_all = torch.cat([labels_1, labels_0], dim=0)

                y_preds = matching_model(emb_all).squeeze()
                y_preds = torch.round(torch.sigmoid(y_preds))
                y_pred_list.extend(y_preds.cpu().detach().numpy())
                y_true_list.extend(labels_all.cpu().detach().numpy())

        test_eer, _ = utils.calculate_eer(y_true_list, y_pred_list)
        test_f1_macro = f1_score(y_true_list, y_pred_list, average='macro')
        test_precision = precision_score(y_true_list, y_pred_list, average='macro')
        test_recall = recall_score(y_true_list, y_pred_list, average='macro')

        result_dict = {
            'test_eer': test_eer,
            'test_f1_macro': test_f1_macro,
            'test_precision': test_precision,
            'test_recall': test_recall
        }
        self.logger.info(f"[{self.task}-{self.method}] Test EER: {test_eer:.4f} | Test F1-macro: {test_f1_macro:.4f}")
        extract_and_log_test_results(self.task, self.method, result_dict, self.matching_exp_dir, self.args, self.logger)

