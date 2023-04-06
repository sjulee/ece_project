from sklearn.metrics import roc_auc_score
import numpy as np

metrics = ['AUC']

def get_metrics(y_true, y_pred):
    out_metrics = np.zeros(len(metrics))
    for metric_index in range(len(metrics)):
        metric = metrics[metric_index]
        if metric == 'AUC':
            metric_value = roc_auc_score(y_true, y_pred)

        out_metrics[metric_index] = metric_value
    return out_metrics