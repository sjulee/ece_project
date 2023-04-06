import dataset

# Datasets to work on
datasets = ['breast-cancer', 'spambase']

# For each dataset: Conduct bootstrap stratified sampling with following ratios: 0.01, 0.05, 0.1, 0.2, 0.5
# Each of these will serve as our dataset
for data in datasets:
    x, y = dataset.loadData(data)

# Then try the following:
# Upsampling of the minority class
# Downsampling of the majority class
# AdaCost
# SMOTEBagging
# RUSBoost
# UnderBagging?

#Try decision tree, SVM, neural networks?