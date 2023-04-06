from sklearn.metrics import *
import numpy as np

metrics = ['AUC', 'Accuracy','F-1 Score', 'Average Accuracy', 'Kappa Statistic', 'Precision',
           'Recall','MCC', 'Balanced Accuracy']
def get_metrics(y_true, y_pred):
    y_pred_class = np.round(y_pred)
    out_metrics = np.zeros(len(metrics))
    for metric_index in range(len(metrics)):
        metric = metrics[metric_index]
        if metric == 'AUC':
            metric_value = auc_m(y_true, y_pred)
        elif metric == 'Accuracy':



        out_metrics[metric_index] = metric_value
    return out_metrics

def auc_m(y_true, y_pred):
    score = roc_auc_score(y_true, y_pred)
    return score

def accuracy_m(y_true, y_pred):
    score = accuracy_score()
    return score

def f_1_score_m(y_true, y_pred):
    return score
