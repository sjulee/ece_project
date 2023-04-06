import dataset
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# Datasets to work on
datasets = ['spambase']
ratios = [100, 50, 10, 5, 1]
methods = ['SMOTEBagging']
#classifiers = ['DecisionTree']
metrics = ['AUC']
B = 1 # Number of bootstraps to do
folds = 5

# For each dataset: Conduct bootstrap stratified sampling with following ratios: 0.01, 0.05, 0.1, 0.2, 0.5
# Each of these will serve as our dataset

for data in datasets:
    x_orig, y_orig = dataset.loadData(data)

    for ratio in ratios:
        for b in range(B):
            x, y = dataset.bootstrap_data(b, ratio, x_orig, y_orig)

            skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = b)
            metric_values = np.zeros((len(methods), len(metrics), folds))
            for i, (train_index, test_index) in enumerate(skf.split(x, y)):
                x_train = x[train_index, :]
                x_test = x[test_index, :]
                y_train = y[train_index]
                y_test = y[test_index]

                for method_index in range(len(methods)):
                    method = methods[method_index]
                    y_pred = get_prediction(x_train, y_train, x_test, method)
                    metric_values[method_index, :, i] = get_metrics(y_test, y_pred)
