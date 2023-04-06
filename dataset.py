import numpy as np
import pandas as pd

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