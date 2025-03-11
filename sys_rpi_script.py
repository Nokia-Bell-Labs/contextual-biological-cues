"""
Script to run on the Raspberry Pi to measure system cost.
"""

import os
import time
import torch
import argparse
import numpy as np
import heartpy as hp
import torch.nn as nn
import neurokit2 as nk
import torch.nn.functional as F



def parse_args():
    parser = argparse.ArgumentParser(description='Raspberry Pi script to measure system cost')
    parser.add_argument('--ws', type=int, default=20, help='Window size in seconds')
    parser.add_argument('--sr', type=int, default=100, help='Sampling rate in Hz')
    parser.add_argument('--modals', nargs='+', default=['ppg'], help='List of modals to use')
    parser.add_argument('--em_dim', type=int, default=16, help='Embedding dimension')
    parser.add_argument('--cpt_dir', type=str, default='./model_cpts', help='Directory to save model checkpoints')
    parser.add_argument('--verbose', action='store_true', help='Print verbose logs')
    return parser.parse_args()


def get_simulated_data(sr, ws, modality_list, frequency):
    num_cols = 0
    num_channel_modality_dict = {'ppg': 1, 'acc': 3, 'gyro': 3}
    simulated_data = []
    for modality in modality_list:
        if modality not in num_channel_modality_dict.keys():
            raise ValueError(f'Modality {modality} is not supported')
        num_cols += num_channel_modality_dict[modality]
        if modality == 'ppg':
            curr_data = nk.signal_simulate(duration=ws, frequency=frequency, sampling_rate=sr)
            curr_data = np.expand_dims(curr_data, axis=1)
        elif modality in ['acc', 'gyro']:
            curr_data = np.random.rand(sr * ws, num_channel_modality_dict[modality])
        else:
            raise ValueError(f'Modality {modality} is not supported')
        simulated_data.append(curr_data)

    simulated_data = np.concatenate(simulated_data, axis=1)

    # For splitting the data into channels
    if len(modality_list) == 1 and modality_list[0] not in ['acc', 'gyro']:
        simulated_data = np.expand_dims(simulated_data, axis=1)
    return simulated_data


def print_log(message, space_before=0, space_after=1):
    len_message = len(message)
    for _ in range(space_before):
        print()
    print(f'{"=" * (len_message + 4)}')
    print(f'| {message} |')
    print(f'{"=" * (len_message + 4)}')

    for _ in range(space_after):
        print()


def compute_acc_mag(acc_ax, acc_ay, acc_az):
    """Compute magnitude of acceleration"""
    acc_mag = np.sqrt(acc_ax ** 2 + acc_ay ** 2 + acc_az ** 2)
    return acc_mag


def filter_modality_data(data, sampling_rate, modality):
    if modality == 'ppg':
        filtered_data = nk.ppg_clean(data, sampling_rate=sampling_rate, method='elgendi')
        filtered_data = np.expand_dims(filtered_data, axis=1)
    elif modality =='acc':
        filtered_data = []
        for i in range(3):
            filtered_axis = nk.signal_filter(data[:, i], sampling_rate=sampling_rate, method='butterworth', highcut=15)
            filtered_data.append(filtered_axis)
        filtered_data = np.stack(filtered_data, axis=1)
        filtered_data = compute_acc_mag(filtered_data[:, 0], filtered_data[:, 1], filtered_data[:, 2])
        filtered_data = np.expand_dims(filtered_data, axis=1)
    elif modality == 'gyro':
        filtered_data = []
        for i in range(3):
            filtered_axis = nk.signal_filter(data[:, i], sampling_rate=sampling_rate, method='butterworth', highcut=20)
            filtered_data.append(filtered_axis)
        filtered_data = np.stack(filtered_data, axis=1)
    else:
        raise ValueError(f'Modality {modality} is not supported')

    # print(f' ( {modality} ) shape of filtered_data: {filtered_data.shape}')
    return filtered_data


def filter_data(data, sampling_rate, modality_list):
    filtered_data = []
    num_channel_modality_dict = {'ppg': 1, 'acc': 3, 'gyro': 3}
    curr_pointer = 0
    for idx, modality in enumerate(modality_list):
        if modality == 'ppg':
            curr_modality_data = data[:, curr_pointer]
            curr_pointer += 1
        elif modality in ['acc', 'gyro']:
            curr_modality_data = data[:, curr_pointer:curr_pointer + 3]
            curr_pointer += 3
        else:
            raise ValueError(f'Modality {modality} is not supported')
        curr_modality_filtered_data = filter_modality_data(curr_modality_data, sampling_rate, modality)
        filtered_data.append(curr_modality_filtered_data)

    return np.concatenate(filtered_data, axis=1)


def compute_statistical_features(input_array, sensor_type):
    min_ = np.min(input_array)
    max_ = np.max(input_array)
    mean_ = np.mean(input_array)
    std_ = np.std(input_array)

    if sensor_type in ['ppg']:
        range_ = max_ - min_
        return {'min': min_, 'max': max_, 'mean': mean_, 'std': std_, 'range': range_}

    elif sensor_type in ['acc', 'gyro', 'imu']:
        median_ = np.median(input_array)
        return {'min': min_, 'max': max_, 'mean': mean_, 'std': std_, 'median': median_}


def compute_ppg_specific_features(ppg_data, aligned_sampling_rate):
    working_data, result_features = hp.process(ppg_data, sample_rate=aligned_sampling_rate)
    #     print(f'<<<< extracted ppg features >>>:\n{result_features}')
    return result_features


def ppg_create_feature_df(ppg_data, sampling_rate):
    stat_features = compute_statistical_features(ppg_data.copy(), sensor_type='ppg')
    ppg_features = compute_ppg_specific_features(ppg_data.copy(), sampling_rate)
    feature_dict = {**stat_features, **ppg_features}
    return feature_dict


def compute_modality_features(data, sampling_rate, modality):
    if modality == 'ppg':
        features = ppg_create_feature_df(data, sampling_rate)
    elif modality in ['acc', 'gyro']:
        features = compute_statistical_features(data, modality)
    else:
        raise ValueError(f'Modality {modality} is not supported')
    return features.values()


def compute_features(data, sampling_rate, modality_list):
    all_features = []
    curr_pointer = 0
    for idx, modality in enumerate(modality_list):
        if modality in ['ppg', 'acc']:
            curr_modality_data = data[:, curr_pointer]
            curr_pointer += 1
        elif modality =='gyro':
            curr_modality_data = data[:, curr_pointer:curr_pointer + 3]
            curr_pointer += 3
        else:
            raise ValueError(f'Modality {modality} is not supported')
        curr_modality_features = compute_modality_features(curr_modality_data, sampling_rate, modality)
        all_features.extend(curr_modality_features)
    # print(f'Shape of all features: {len(all_features)}')
    all_features = np.array(all_features)
    return all_features


class Embedding_Model(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(Embedding_Model, self).__init__()
        self.emb_dim = emb_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )

        self.projection_head = nn.Sequential(
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, self.emb_dim)
        )

    def forward(self, x):
       x = self.encoder(x)
       x = self.projection_head(x)
       return x


class Matching_Model(nn.Module):
    def __init__(self, emb_dim):
        super(Matching_Model, self).__init__()
        self.concat_embedding_dim = emb_dim * 2 # A pair of embeddings from two devices
        self.user_matching_model = nn.Sequential(
            nn.Linear(self.concat_embedding_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1) # Output is a single value (Binary classification)
        )

    def forward(self, x):
        x = self.user_matching_model(x)
        return x


def load_trained_model(cpt_dir, model_type, modals, model):
    is_loaded = False
    modals_id = '_'.join(sorted(modals) if isinstance(modals, list) else [modals])
    assert model_type in ['embgen', 'matching']
    full_cpt_path = os.path.join(cpt_dir, f'{model_type}_m:{modals_id}.pt')

    if not os.path.exists(full_cpt_path):
        return None, is_loaded

    print(f"Loading model from {full_cpt_path}...")
    cpt = torch.load(full_cpt_path)
    model.load_state_dict(cpt['state_dict'])
    is_loaded = True
    return model, is_loaded


def save_model(cpt_dir, model_type, modals, model):
    modals_id = '_'.join(sorted(modals) if isinstance(modals, list) else [modals])
    assert model_type in ['embgen', 'matching']
    full_cpt_path = os.path.join(cpt_dir, f'{model_type}_m:{modals_id}.pt')

    torch.save({'state_dict': model.state_dict()}, full_cpt_path)
    print(f"Saved model to {full_cpt_path}...")


def main():
    print_log('Start of Raspberry Pi script to measure system cost')
    args = parse_args()
    # 1. Simulate Data
    d1_data = get_simulated_data(args.sr, args.ws, args.modals, frequency=1)
    d2_data = get_simulated_data(args.sr, args.ws, args.modals, frequency=2)

    start_filtering = time.time()
    # 2. Filter data
    d1_data = filter_data(d1_data, args.sr, args.modals)
    d2_data = filter_data(d2_data, args.sr, args.modals)
    end_filtering = time.time()

    start_feat_extraction = time.time()
    # 2. Extract features based on the modality type
    d1_features = compute_features(d1_data, args.sr, args.modals)
    d2_features = compute_features(d2_data, args.sr, args.modals)
    # data with one batch
    d1_features = np.expand_dims(d1_features, axis=0)
    d2_features = np.expand_dims(d2_features, axis=0)
    # print(f'Shape of d1_features: {d1_features.shape}\nd1_features: {d1_features}')
    end_feat_extraction = time.time()


    # 3. Load embedder and generate embeddings
    start_embgen_model_inference = time.time()
    embedder = Embedding_Model(d1_features.shape[1], args.em_dim)
    matcher = Matching_Model(args.em_dim)

    if not os.path.exists(args.cpt_dir):
        os.makedirs(args.cpt_dir)

    _, is_loaded_embgen = load_trained_model(args.cpt_dir, 'embgen', args.modals, embedder)
    _, is_loaded_matching = load_trained_model(args.cpt_dir, 'matching', args.modals, matcher)

    if not is_loaded_embgen or not is_loaded_matching:
        print(f'No model found. Generating a new model and saving it.')
        save_model(args.cpt_dir, 'embgen', args.modals, embedder)
        save_model(args.cpt_dir, 'matching', args.modals, matcher)
        print(f"Please, run the script again to load the model checkpoints.")
        return

    with torch.no_grad():
        d1_embeddings = F.normalize(embedder(torch.tensor(d1_features).float()), dim=1)
        d2_embeddings = F.normalize(embedder(torch.tensor(d2_features).float()), dim=1)
    end_embgen_model_inference = time.time()

    # 4. Load matcher and generate matching results
    start_matching_inference = time.time()
    with torch.no_grad():
        concat_embeddings = torch.cat((d1_embeddings, d2_embeddings), dim=1)
        matching_result = matcher(concat_embeddings)[0].item()
        matching_result = torch.round(torch.sigmoid(torch.tensor(matching_result))).item()
        # print(f'matching_result: {matching_result}')
    end_matching_inference = time.time()

    # 5. Print time taken for each step
    print(f'\nTime taken for each step:')
    print(f'[1] Filtering: {end_filtering - start_filtering:.5f} s')
    print(f'[2] Feature extraction: {end_feat_extraction - start_feat_extraction:.5f} s')
    print(f'[3] Embedding generation model inference: {end_embgen_model_inference - start_embgen_model_inference:.5f} s')
    print(f'[4] Matching model inference: {end_matching_inference - start_matching_inference:.5f} s')
    print(f'[5] Total time: {end_matching_inference - start_filtering:.5f} s')


if __name__ == '__main__':
    main()

