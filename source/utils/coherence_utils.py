"""
coherence_utils.py
Helper functions used in Lester04 and Cornelius12 to compute coherence features
"""
import numpy as np
from scipy.stats import iqr
from scipy.signal import coherence
from source.data.data_loader.loader_utils import (FATIGUESET_SAMPLING_RATE,
                                                  WESAD_SAMPLING_RATE)


def compute_coherence_raw(acc_1, acc_2, fs):
    acc_1 = _remove_gravity_offset(acc_1)
    acc_2 = _remove_gravity_offset(acc_2)

    acc_1_mag = _compute_magnitude(acc_1)
    acc_2_mag = _compute_magnitude(acc_2)

    # lef of acc_1_mag (ws:8 sec, sr=100Hz) = 800,
    # Here, we are using the raw data, and the nfft should be 256 or 512 to capture the motion data
    # At the same time, nfft should be same or higher than nperseg (which is in this case, 400 (800//2))
    # So, we set nfft=512
    f, Cxy = coherence(acc_1_mag, acc_2_mag,
                       fs=fs, window='hann',
                       nperseg=len(acc_1_mag) // 2, noverlap=0, # two windows with no overlap
                       nfft=512 # 256 or 512 recommended for motion data
                       )
    similarity = _integrate_coherence(f, Cxy) # to normalize the range
    return similarity


def compute_coherence_features(acc_1, acc_2, fs, task='embgen'):
    if fs == FATIGUESET_SAMPLING_RATE:
        num_feats_per_second = 10 # 10 features per second, SR = 100 Hz
    elif fs == WESAD_SAMPLING_RATE:
        num_feats_per_second = 4 # 4 features per second, SR = 32 Hz
    else:
        raise ValueError(f"Sampling rate {fs} not supported")

    acc_1 = _remove_gravity_offset(acc_1)
    acc_2 = _remove_gravity_offset(acc_2)

    acc_1_mag = _compute_magnitude(acc_1)
    acc_2_mag = _compute_magnitude(acc_2)

    f, Cxy_dict = _compute_feat_coherence_dict(acc_1_mag, acc_2_mag, fs, num_feats_per_second)

    # Integrate the coherence values for each feature
    similarities = []
    for feat_name in Cxy_dict.keys():
        Cxy_curr_feat = Cxy_dict[feat_name]
        similarity_curr_feat = _integrate_coherence(f, Cxy_curr_feat)
        similarities.append(similarity_curr_feat)

    avg_similarity = np.mean(similarities)
    if task == 'embgen':
        return avg_similarity
    elif task == 'matching':
        return similarities
    else:
        raise ValueError(f"Task {task} not supported")



def _remove_gravity_offset(acc_data):
    # Subtract the mean from each axis to remove the DC component (gravity)
    return acc_data - np.mean(acc_data, axis=0)


def _compute_magnitude(acc_data):
    return np.linalg.norm(acc_data, axis=1)


def _integrate_coherence(frequencies, coherence_values, freq_range=(0, 10)):
    # Select the frequency range 0-10 Hz (human motion data)
    mask = (frequencies >= freq_range[0]) & (frequencies <= freq_range[1])

    # Compute the integral of the coherence values in the specified frequency range
    integral = np.trapz(coherence_values[mask], frequencies[mask])

    # Normalize by multiplying by 0.1 as per the equation in the paper
    normalized_integral = 0.1 * integral  # According to the paper
    return normalized_integral



# Cornelius '12 helper functions
def _compute_sub_window_features(sub_window):
    feature_dict = {
        'mean': np.mean(sub_window),
        'std': np.std(sub_window),
        'var': np.var(sub_window),
        'mad': np.mean(np.abs(sub_window - np.mean(sub_window))), # mean abs. deviation
        'iqr': iqr(sub_window), # interquartile range
        'power': np.sum(np.square(sub_window)),
        'energy': np.sum(np.square(sub_window)) / len(sub_window)
    }
    return feature_dict

def _compute_feature_windows(coh_window, fs, n_feats_per_sec):
    feat_win_rows = int(fs / n_feats_per_sec)
    n_sub_windows = len(coh_window) // feat_win_rows

    feature_dict = {}
    for enum, idx in enumerate(range(n_sub_windows)):
        start_idx = idx * feat_win_rows
        end_idx = (idx + 1) * feat_win_rows
        sub_window = coh_window[start_idx:end_idx]
        sub_window_features = _compute_sub_window_features(sub_window)
        for feat_name in sub_window_features.keys():
            if feat_name not in feature_dict.keys():
                feature_dict[feat_name] = []
            feature_dict[feat_name].append(sub_window_features[feat_name])
    return feature_dict


def _compute_feat_coherence_dict(acc_1_mag, acc_2_mag, fs, n_feats_per_sec):
    feat_wins_1 = _compute_feature_windows(acc_1_mag, fs, n_feats_per_sec)
    feat_wins_2 = _compute_feature_windows(acc_2_mag, fs, n_feats_per_sec)

    feat_names = feat_wins_1.keys()
    f = None
    feat_coh_dict = {}
    for feat_name in feat_names:
        curr_feat_win_1 = feat_wins_1[feat_name]
        cuff_feat_win_2 = feat_wins_2[feat_name]

        # Len of curr_feat_win_1: 80, nperseg: 40
        # As coherence is computed for features (not from raw data), and nperseg is 40,
        # we let nfft be default (nfft=nperseg)
        f_temp, Cxy = coherence(curr_feat_win_1, cuff_feat_win_2,
                               fs=fs, window='hann',
                               nperseg=len(curr_feat_win_1) // 2, noverlap=0, # two windows with no overlap
                               )
        f = f_temp if f is None else f
        if feat_name not in feat_coh_dict.keys():
            feat_coh_dict[feat_name] = []
        feat_coh_dict[feat_name] = Cxy
    return f, feat_coh_dict



# For testing
def generate_accelerometer_data(duration=8, fs=100, correlated=True, noise=3):
    t = np.arange(0, duration, 1 / fs)
    # Walking pattern simulation for each axis (sinusoidal patterns)
    # X, Y, Z axis for device 1 (sinusoidal variations)
    data1_x = 0.5 * np.sin(2 * np.pi * 1.5 * t)  # X-axis walking movement
    data1_y = 0.7 * np.sin(2 * np.pi * 1.2 * t + np.pi / 4)  # Y-axis walking movement
    data1_z = 9.81 + 0.6 * np.sin(2 * np.pi * 1.0 * t)  # Z-axis, constant gravity + motion

    if correlated:
        # Correlated data: Device 2 has similar frequency and phase to Device 1
        data2_x = 0.45 * np.sin(2 * np.pi * 1.6 * t + np.pi / 6)  # X-axis walking movement
        data2_y = 0.65 * np.sin(2 * np.pi * 1.1 * t + np.pi / 3)  # Y-axis walking movement
        data2_z = 9.81 + 0.55 * np.sin(2 * np.pi * 0.9 * t)  # Z-axis, constant gravity + motion
    else:
        # Non-correlated data: Different frequencies, phases, and added noise
        data2_x = 0.45 * np.sin(2 * np.pi * 5.0 * t + np.pi / 8) + noise * np.random.randn(len(t))
        data2_y = 0.85 * np.sin(2 * np.pi * 3.0 * t + np.pi / 2) + noise * np.random.randn(len(t))
        data2_z = 9.81 + 0.55 * np.sin(2 * np.pi * 3.5 * t + np.pi / 3) + noise * np.random.randn(len(t))

    # Combine into data arrays
    data1 = np.stack((data1_x, data1_y, data1_z), axis=-1)
    data2 = np.stack((data2_x, data2_y, data2_z), axis=-1)

    return data1, data2, t



