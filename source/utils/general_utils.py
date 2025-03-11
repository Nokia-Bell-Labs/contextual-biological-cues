
import os
import torch
import random
import numpy as np
from sklearn.metrics import roc_curve

###############################################
# ========= Seed for Reproducibility ======== #
###############################################
def set_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(random_seed)


##########################################
# ========= Performance Metrics ======== #
##########################################
def calculate_eer(y_true, y_pred):
    # Credits: https://github.com/YuanGongND/python-compute-eer
    # FRR: False Rejection rate, when authorized user is falsely rejected
    # FAR: False Acceptance rate, when unauthorized user is falsely accepted
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1) # 1 is auth=true (same aligned)
    frr = 1 - tpr # FRR = 1 - True Positive Rate (TPR)
    far = fpr # FAR = False Positive Rate (FPR)

    # Find the point where FAR and FRR are closest
    eer_threshold_index = np.nanargmin(np.abs(far - frr))
    eer = (far[eer_threshold_index] + frr[eer_threshold_index]) / 2
    return eer, thresholds[eer_threshold_index]


def calculate_tpr_fpr_fnr(y_true, y_pred):
    # Convert lists to numpy arrays if they aren't already
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # True positives (TP): Predicted = 1, Actual = 1
    TP = np.sum((y_pred == 1) & (y_true == 1))

    # False positives (FP): Predicted = 1, Actual = 0
    FP = np.sum((y_pred == 1) & (y_true == 0))

    # False negatives (FN): Predicted = 0, Actual = 1
    FN = np.sum((y_pred == 0) & (y_true == 1))

    # True negatives (TN): Predicted = 0, Actual = 0
    TN = np.sum((y_pred == 0) & (y_true == 0))

    # True Positive Rate (TPR) or Sensitivity = TP / (TP + FN)
    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0

    # False Positive Rate (FPR) = FP / (FP + TN)
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0

    # False Negative Rate (FNR) = FN / (TP + FN)
    FNR = FN / (TP + FN) if (TP + FN) > 0 else 0

    result_dict = {
        'TPR': round(TPR, 2),
        'FPR': round(FPR, 2),
        'FNR': round(FNR, 2)
    }
    return result_dict


#################################################
# ========= Conversion to Python Types ======== #
#################################################
def convert_to_python_types(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert numpy arrays to lists
    elif isinstance(obj, np.generic):
        return obj.item()  # Convert numpy types (like float32, int64) to their Python equivalents
    elif isinstance(obj, dict):
        return {k: convert_to_python_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_python_types(i) for i in obj]
    else:
        return obj


