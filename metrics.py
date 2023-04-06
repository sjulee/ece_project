import numpy as np
from sklearn.metrics import *

METRICS = ['AUC', 'Accuracy', 'F-1 Score', 'Kappa Statistic', 'Precision',
           'Recall', 'MCC', 'Balanced Accuracy']

def get_metrics(y_true, y_pred):
    score = np.nan()
    y_pred_class = np.round(y_pred)
    out_metrics = np.zeros(len(METRICS))

    for metric_index in range(len(METRICS)):
        metric = metrics[metric_index]
        if metric == 'AUC':
            score = roc_auc_score(y_true, y_pred)
        elif metric == 'Accuracy':
            score = accuracy_score(y_true, y_pred_class)
        elif metric == 'F-1 Score':
            score = f1_score(y_true, y_pred_class)
        elif metric == 'Balanced Accuracy':
            score = balanced_accuracy_score(y_true, y_pred_class)
        elif metric == 'Kappa Statistic':
            score = cohen_kappa_score(y_true, y_pred_class)
        elif metric == 'Precision':
            score = precision_score(y_true, y_pred_class)
        elif metric == 'Recall':
            score = recall_score(y_true, y_pred_class)
        elif metric == 'MCC':
            score = matthews_corrcoef(y_true, y_pred_class)
        else:
            print("Metric aint available!!!")

        out_metrics[metric_index] = score

    return out_metrics
