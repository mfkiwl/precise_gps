import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd


def normalize(X):
    X_mean = np.average(X, 0)[None, :]
    X_std = 1e-6 + np.std(X, 0)[None, :]
    return (X - X_mean) / X_std, X_mean, X_std

class Dataset(object):

    def __init__(self, path, split = 0.2):
        self.path = path 
        X, y, cols = self.read_data()
        Xnp, ynp = self.preprocess(X,y)
        train_Xnp, test_Xnp, train_ynp, test_ynp = train_test_split(Xnp, ynp, test_size=split, random_state=42)

        self.train_X = train_Xnp
        self.train_y = train_ynp
        self.test_X = test_Xnp
        self.test_y = test_ynp
        self.cols = cols 
    
    def preprocess(self, X, y):
        X, self.X_mean, self.X_std = normalize(X)
        y, self.y_mean, self.y_std = normalize(y)
        return X, y 
    
    def read_data(self):
        raise NotImplementedError
    
    def get_cols(path):
        return [d.strip("'") for d in list(pd.read_csv(path + "/features.csv", delimiter=','))]


class Redwine(Dataset):

    def __init__(self, split):
        super(Redwine, self).__init__(path = "data/redwine", split=split)
    
    def read_data(self):
        data = np.genfromtxt(self.path + "/data.csv", delimiter=';')
        X = data[1:,0:-1]
        y = data[1:,-1]
        cols = self.get_cols(self.path)
        return X, y, cols 

class Whitewine(Dataset):

    def __init__(self, split):
        super(Redwine, self).__init__(path = "data/whitewine", split=split)
    
    def read_data(self):
        data = np.genfromtxt(self.path + "/data.csv", delimiter=';')
        X = data[1:,0:-1]
        y = data[1:,-1]
        cols = self.get_cols(self.path)
        return X, y, cols 

class Naval(Dataset):

    def __init__(self, split):
        super(Redwine, self).__init__(path = "data/naval", split=split)
    
    def read_data(self):
        data = np.genfromtxt(self.path + "/data.txt", delimiter='  ')
        X = data[:,0:-2]
        X = np.delete(X, [8, 11], axis=1) # these have 0 variance
        y = data[:,-2]
        cols = self.get_cols(self.path)
        return X, y, cols 

class Boston(Dataset):

    def __init__(self, split):
        super().__init__(path = "data/boston", split=split)

    def read_data(self):
        data = np.genfromtxt(self.path + "/data.txt", delimiter='  ')
        data = pd.read_fwf(self.path + "/housing.data", header=None).values
        X = data[:, :-1]
        y = data[:, -1]
        cols = self.get_cols(self.path)
        return X, y, cols 