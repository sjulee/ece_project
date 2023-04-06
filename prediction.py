from imblearn.ensemble import BalancedBaggingClassifier
from imblearn.over_sampling import SMOTE

def get_prediction(x_train, y_train, x_test, method):
    if method == 'SMOTEBagging':
        smotebag = BalancedBaggingClassifier(sampler=SMOTE(), random_state=b)
        smotebag.fit(x_train, y_train)
        y_pred = smotebag.predict_proba(x_test)[:,1]
    return y_pred