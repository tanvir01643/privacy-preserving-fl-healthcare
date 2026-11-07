import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score


# Compute classification performance metrics (accuracy, precision, recall, roc_auc)
# for model evaluation, safely handling single-class edge cases for ROC AUC.
def evaluate_model(y_true, y_pred, y_score):
    accuracy = float(accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))

    y_true_arr = np.asarray(y_true)
    if len(np.unique(y_true_arr)) < 2:
        roc_auc = float("nan")
    else:
        try:
            roc_auc = float(roc_auc_score(y_true, y_score))
        except ValueError:
            roc_auc = float("nan")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "roc_auc": roc_auc,
    }
