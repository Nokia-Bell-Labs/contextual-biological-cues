
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity


######################################
# ========= Embedding Loss ========= #
######################################
def get_embedding_loss(distance_fun, margin_value, reduction, swap):
    distance_func_dict = {'l1': l1_distance,
                         'linf': l_infinity,
                         'cosine': cosine_distance}
    if distance_fun not in distance_func_dict.keys():
        raise ValueError(f'Error: Distance function {distance_fun} not implemented')
    loss_function = nn.TripletMarginWithDistanceLoss(margin=margin_value,
                                                     distance_function=distance_func_dict[distance_fun],
                                                     reduction=reduction, swap=swap)
    return loss_function


def l_infinity(x1, x2):
    return torch.max(torch.abs(x1 - x2), dim=1).values

def l1_distance(x1, x2):
    return torch.sum(torch.abs(x1 - x2), dim=1)

def cosine_distance(x1, x2):
    return 1.0 - F.cosine_similarity(x1, x2)



##########################################
# ========= Embedding Distance ========= #
##########################################
def compute_embedding_distance(embedding1, embedding2, metric_type='l1'):
    if metric_type == 'spearmancorr': # Closer to 1 means more similar
        spearman_corr_list = []
        for i in range(len(embedding1)):
            spearman_corr, _ = spearmanr(embedding1[i], embedding2[i])
            spearman_corr_list.append(spearman_corr)
        return spearman_corr_list

    elif metric_type == 'cosine_sim': # Closer to 1 means more similar
        cosine_sim_list = []
        for i in range(len(embedding1)):
            cosine_sim = cosine_similarity(embedding1[i].reshape(1, -1), embedding2[i].reshape(1, -1))
            cosine_sim_list.append(cosine_sim)
        return cosine_sim_list

    elif metric_type == 'euclidean': # Closer to 0 means more similar
        euclidean_list = []
        for i in range(len(embedding1)):
            euclidean = np.linalg.norm(embedding1[i] - embedding2[i])
            euclidean_list.append(euclidean)
        return euclidean_list

    elif metric_type == 'l1': # Closer to 0 means more similar
        # print(f'Computing L1 distance!!!!')
        l1_list = []
        for i in range(len(embedding1)):
            l1 = np.sum(np.abs(embedding1[i] - embedding2[i]))
            l1_list.append(l1)
        return l1_list

    else:
        raise NotImplementedError(f'Invalid metric type: {metric_type}')

#######################################################
# ========= Fisher Discriminant Ratio (FDR) ========= #
#######################################################
def fisher_discriminant_ratio(dist1, dist2):
    mu1, mu2 = np.mean(dist1), np.mean(dist2)
    var1, var2 = np.var(dist1), np.var(dist2)
    fdr = (mu1 - mu2) ** 2 / (var1 + var2)
    return fdr