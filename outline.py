import dataset
from imblearn.over_sampling import SMOTE
from imblearn.ensemble import BalancedBaggingClassifier

# Datasets to work on
datasets = ['breast-cancer', 'spambase']
ratios = [100, 50, 10, 5, 1]
B = 1 # Number of bootstraps to do

# For each dataset: Conduct bootstrap stratified sampling with following ratios: 0.01, 0.05, 0.1, 0.2, 0.5
# Each of these will serve as our dataset
for data in datasets:
    x, y = dataset.loadData(data)

    for ratio in ratios:
        for b in range(B):
            x_new, y_new = dataset.bootstrap_data(b, ratio, x, y)


            smote_bagging = BalancedBaggingClassifier(sampler=SMOTE())
cv_results = cross_validate(smote_bagging, X, y, scoring="balanced_accuracy")

print(f"{cv_results['test_score'].mean():.3f} +/- {cv_results['test_score'].std():.3f}")

# Then try the following:
# Upsampling of the minority class
# Downsampling of the majority class
# AdaCost
# SMOTEBagging
# RUSBoost
# UnderBagging?

#Try decision tree, SVM, neural networks?