import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier


class AdaCost(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator=None, beta=1):
        self.base_estimator = base_estimator or DecisionTreeClassifier(max_depth=1)
        self.beta = beta
        self.estimators_ = []
        self.weights_ = []

    def fit(self, X, y, sample_weight=None):
        sample_weight = sample_weight or np.ones(len(y))
        self.estimators_ = []
        self.weights_ = []
        n_samples = len(y)
        weights_pos = np.where(y == 1, sample_weight, 0)
        weights_neg = np.where(y == -1, sample_weight, 0)
        beta_pos = 1.0 / (1 + self.beta)
        beta_neg = self.beta / (1 + self.beta)
        for i in range(self.n_estimators):
            # Train base estimator on weighted samples
            estimator = self.base_estimator.fit(X, y, sample_weight=sample_weight)

            # Compute misclassification costs
            y_pred = estimator.predict(X)
            errors_pos = np.sum(weights_pos * (y != y_pred))
            errors_neg = np.sum(weights_neg * (y != y_pred))

            # Update sample weights
            alpha_pos = beta_pos * np.log((1 - errors_pos) / errors_pos)
            alpha_neg = beta_neg * np.log((1 - errors_neg) / errors_neg)
            sample_weight *= np.exp(alpha_pos * (y == 1) + alpha_neg * (y == -1))

            # Save estimator and weight
            self.estimators_.append(estimator)
            self.weights_.append(alpha_pos * (y == 1) + alpha_neg * (y == -1))

        return self

    def predict(self, X):
        pred_sum = np.zeros(len(X))
        for estimator, weight in zip(self.estimators_, self.weights_):
            pred_sum += weight * estimator.predict(X)
        return np.sign(pred_sum)

    def predict_proba(self, X):
        proba_sum = np.zeros(len(X))
        for estimator, weight in zip(self.estimators_, self.weights_):
            proba = estimator.predict_proba(X)
            proba_sum += weight * proba[:, 1]  # use the probability of the positive class
        proba_avg = proba_sum / np.sum(self.weights_)
        proba_other = 1 - proba_avg
        return np.column_stack((proba_other, proba_avg))  # stack the probabilities for both classes
