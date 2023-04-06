import dataset
import numpy as np
from sklearn.model_selection import StratifiedKFold
import metrics as met
import prediction

# Datasets to work on
datasets = ['adult']
ratios = [100, 50, 10, 5, 1]
methods = ['SMOTEBagging', 'RUSBoost', 'SMOTEBoost', 'UnderBagging', 'RandomForest', 'AdaCost']
METRICS = ['AUC', 'Accuracy', 'F-1 Score', 'Kappa Statistic', 'Precision',
           'Recall', 'MCC', 'Balanced Accuracy']
B = 100  # Number of bootstraps to do
folds = 5

for data in datasets:
    x_orig, y_orig = dataset.loadData(data)

    for ratio in ratios:
        metric_values = np.zeros((len(methods), len(METRICS), B, folds))
        for b in range(B):
            x, y = dataset.bootstrap_data(b, ratio, x_orig, y_orig)

            skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=b)
            for i, (train_index, test_index) in enumerate(skf.split(x, y)):
                x_train = x[train_index, :]
                x_test = x[test_index, :]
                y_train = y[train_index]
                y_test = y[test_index]

                for method_index in range(len(methods)):
                    method = methods[method_index]
                    y_pred = prediction.get_prediction(x_train, y_train, x_test, method, b)
                    metric_values[method_index, :, b, i] = met.get_metrics(y_test, y_pred, METRICS)

    np.save("test_file.npy", metric_values)