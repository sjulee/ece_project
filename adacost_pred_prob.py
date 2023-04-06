from sklearn.tree import DecisionTreeClassifier
import numpy as np

class AdaCostClassifier(DecisionTreeClassifier):
    def __init__(self, cost_multiplier=1.0, **kwargs):
        super().__init__(**kwargs)
        self.cost_multiplier = cost_multiplier

    def fit(self, X, y):
        # Compute class weights based on AdaCost
        n_pos = np.sum(y == 1)
        n_neg = np.sum(y == 0)
        total = n_pos + n_neg
        beta = n_neg / total
        alpha = n_pos / total
        cost_ratio = (1 - beta) / beta
        cost_array = np.ones(y.shape[0])
        cost_array[y == 1] = cost_ratio * self.cost_multiplier

        # Fit the decision tree using weighted samples
        super().fit(X, y, sample_weight=cost_array)

    def predict_proba(self, X):
        probas = super().predict_proba(X)
        # Map probability of class 1 to weighted probability
        probas[:, 1] *= self.cost_multiplier
        probas /= np.sum(probas, axis=1)[:, np.newaxis]
        return probas
