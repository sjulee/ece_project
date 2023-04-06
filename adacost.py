import numpy as np
from sklearn.tree import DecisionTreeClassifier

class AdaCost:
    def __init__(self, base_estimator=DecisionTreeClassifier(max_depth=1), beta=1):
        # Constructor for the AdaCost class
        # Initializes the base estimator, beta parameter, sample_weight and list of estimators
        self.base_estimator = base_estimator
        self.beta = beta
        self.sample_weight = None
        self.classes_ = None
        self.estimators_ = []

    def fit(self, X, y, sample_weight=None):
        # Fits the Adacost classifier to the training data
        # Inputs:
        # X: array-like of shape (n_samples, n_features)
        # y: array-like of shape (n_samples,) containing the labels
        # sample_weight: array-like of shape (n_samples,) containing the weights for each sample (default: None)
        # Outputs: None
        self.sample_weight = np.ones(X.shape[0]) if sample_weight is None else sample_weight
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        self.estimators_ = []

        # For each pair of consecutive classes in the sorted list of unique class labels
        for i in range(n_classes - 1):
            pos_class = self.classes_[i]
            neg_class = self.classes_[i + 1]

            # Convert the labels to binary (1 for positive class, -1 for negative class)
            y_bin = np.where(y == pos_class, 1, -1)

            # Fit a base estimator to the training data with sample weights
            estimator = self.base_estimator
            estimator.fit(X, y_bin, sample_weight=self.sample_weight)

            # Compute the predicted labels and error rate
            y_pred = estimator.predict(X)
            error = np.sum(self.sample_weight[y != pos_class] * (y_pred[y != pos_class] != 1))

            # Update the beta and alpha parameters
            beta = error / (1 - error)
            alpha = np.log(beta)
            self.estimators_.append((estimator, alpha))

            # Update the sample weights for the positive and negative classes
            self.sample_weight[y == pos_class] *= beta ** (y_pred[y == pos_class] != 1)
            self.sample_weight[y != pos_class] *= (1 - beta) ** (y_pred[y != pos_class] == 1)

    def predict(self, X):
        # Predict the labels for the test data using the Adacost classifier
        # Inputs:
        # X: array-like of shape (n_samples, n_features) containing the test data
        # Outputs: array-like of shape (n_samples,) containing the predicted labels
        n_samples = X.shape[0]
        y_pred = np.zeros((n_samples, len(self.classes_)))
        for estimator, alpha in self.estimators_:
            # Compute the predicted labels for each binary classifier and weight them using the alpha parameter
            y_pred[:, 0] -= alpha * estimator.predict(X)
            y_pred[:, 1] += alpha * estimator.predict(X)

        # Return the predicted labels for the class with the highest weighted sum of predicted labels
        return self.classes_.take(np.argmax(y_pred, axis=1), axis=0)
