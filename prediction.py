from imblearn.ensemble import BalancedBaggingClassifier, RUSBoostClassifier
from imblearn.over_sampling import SMOTE
from SMOTEBoost import SMOTEBoost
from adacost_pred_prob import AdaCostClassifier
from sklearn.ensemble import RandomForestClassifier

def get_prediction(x_train, y_train, x_test, method, b):
    if method == 'SMOTEBagging':
        smotebag = BalancedBaggingClassifier(sampler=SMOTE(), random_state=b)
        smotebag.fit(x_train, y_train)
        y_pred = smotebag.predict_proba(x_test)[:, 1]
    if method == 'RUSBoost':
        rusboost = RUSBoostClassifier(random_state=b)
        rusboost.fit(x_train, y_train)
        y_pred = rusboost.predict_proba(x_test)[:, 1]
    if method == 'SMOTEBoost':
        smoteboost = SMOTEBoost(random_state=b)
        smoteboost.fit(x_train, y_train)
        y_pred = smoteboost.predict_proba(x_test)[:, 1]
    if method == 'UnderBagging':
        underbag = BalancedBaggingClassifier(random_state=b)
        underbag.fit(x_train, y_train)
        y_pred = underbag.predict_proba(x_test)[:, 1]
    if method == 'RandomForest':
        randomforest = RandomForestClassifier(random_state=b)
        randomforest.fit(x_train, y_train)
        y_pred = randomforest.predict_proba(x_test)[:,1]
    if method == 'AdaCost':
        adacost = AdaCostClassifier(cost_multiplier=5.0)
        adacost.fit(x_train, y_train)
        y_pred = adacost.predict_proba(x_test)[:,1]

    return y_pred
