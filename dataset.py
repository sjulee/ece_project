import numpy as np
import pandas as pd
import math

def loadData(dataname):
    if dataname == 'breast-cancer':
        x, y = loadBreastCancer()
        return x, y
    if dataname == 'spambase':
        x, y = loadSpam()
        return x, y

def loadBreastCancer():
    """
    load breast-cancer dataset
    """
    df = pd.read_csv('data/breast-cancer.data', header=None, delimiter=',')
    for i in range(9):
        df = df[df[i] != '?']
    df = df.apply(lambda x: pd.factorize(x)[0])
    x, y = df[[1,2,3,4,5,6,7,8,9]], df[0]
    return np.array(x), np.array(y)

def loadSpam():
    """
    load spam dataset
    """
    df = pd.read_csv('data/spambase.data', header=None, delimiter=',')
    x, y = df.iloc[:,:-1], df.iloc[:,-1]
    return np.array(x), np.array(y)

def loadAdult():
    """
    :return:
    """
    df = pd.read_csv('data/adult.csv')

    df["Y"] = df["income"].replace("<=50`K", 0, regex=True)
    df["Y"] = df["Y"].replace(">50K", 1, regex=True)

    x, y = df.iloc[:, :-2], df.iloc[:, -1]
    return np.array(x), np.array(y)

def bootstrap_data(seed, ratio, x, y):
    n = len(y)
    n_pos = math.ceil(n / (ratio + 1))
    n_neg = n - n_pos

    rng = np.random.default_rng(seed=seed)
    ind_pos = rng.choice(np.where(y == 1)[0], n_pos)
    ind_neg = rng.choice(np.where(y == 0)[0], n_neg)

    x_new = np.vstack((x[ind_pos, :], x[ind_neg, :]))
    y_new = np.concatenate((np.repeat(1, n_pos), np.repeat(0, n_neg)))

    return x_new, y_new