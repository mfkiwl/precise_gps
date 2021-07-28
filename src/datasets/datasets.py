import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd


# This file provides dataset object that automatically preprocess the raw data,
# and creates train and test sets.

def normalize(X):
    """
    Normalizes tensor X

    Args:
        X : tensor
    
    Returns:
        Normalized tensor
    """
    X_mean = np.average(X, 0)[None, :]
    X_std = 1e-6 + np.std(X, 0)[None, :]
    return (X - X_mean) / X_std, X_mean, X_std

class Dataset(object):
    """
    Automatically preprocesses data and creates train and test set for the specified dataset.

    Args:
        path (sitring to directory) : contains data and the input column names
        split (float)               : test/train split specifies testset size (between 0-1) 
    """

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
    
    def get_cols(self, path):
        return [d.strip("'") for d in list(pd.read_csv(path + "/features.csv", delimiter=','))]


class Redwine(Dataset):

    def __init__(self, split):
        super(Redwine, self).__init__(path = "data/redwine", split=split)
    
    def read_data(self):
        data = np.genfromtxt(self.path + "/data.csv", delimiter=';')
        X = data[1:,0:-1]
        y = data[1:,-1].reshape(-1,1)
        cols = self.get_cols(self.path)
        return X, y, cols 

class Whitewine(Dataset):

    def __init__(self, split):
        super(Whitewine, self).__init__(path = "data/whitewine", split=split)
    
    def read_data(self):
        data = np.genfromtxt(self.path + "/data.csv", delimiter=';')
        X = data[1:,0:-1]
        y = data[1:,-1].reshape(-1,1)
        cols = self.get_cols(self.path)
        return X, y, cols 

class Naval(Dataset):

    def __init__(self, split):
        super(Naval, self).__init__(path = "data/naval", split=split)
    
    def read_data(self):
        data = np.genfromtxt(self.path + "/data.txt", delimiter='  ')
        X = data[:,0:-2]
        X = np.delete(X, [8, 11], axis=1) # these have 0 variance
        y = data[:,-2].reshape(-1,1)
        cols = self.get_cols(self.path)
        return X, y, cols 

class Boston(Dataset):

    def __init__(self, split):
        super(Boston, self).__init__(path = "data/boston", split=split)

    def read_data(self):
        data = pd.read_fwf(self.path + "/housing.data", header=None).values
        X = data[:, :-1]
        y = data[:, -1].reshape(-1,1)
        cols = self.get_cols(self.path)
        return X, y, cols 

class Concrete(Dataset):

    def __init__(self, split):
        super(Concrete, self).__init__(path = "data/concrete", split=split)

    def read_data(self):
        data = pd.read_excel(self.path + "/Concrete_Data.xls").values
        return data[:, :-1], data[:, -1].reshape(-1,1), self.get_cols(self.path)

class Energy(Dataset):

    def __init__(self, split):
        super(Energy, self).__init__(path = "data/energy", split=split)
    
    def read_data(self):
        # NB this is the first output (aka Energy1, as opposed to Energy2)
        data = pd.read_excel(self.path + "/ENB2012_data.xlsx").values[:, :-1]
        return data[:, :-1], data[:, -1].reshape(-1,1), self.get_cols(self.path)

class Power(Dataset):

    def __init__(self, split):
         super(Power, self).__init__(path = "data/power", split=split)
    
    def read_data(self):
        data = pd.read_excel(self.path + "/Folds5x2_pp.xlsx").values
        return data[:, :-1], data[:, -1].reshape(-1,1), self.get_cols(self.path)

class Protein(Dataset):

    def __init__(self, split):
        super(Protein, self).__init__(path = "data/protein", split=split)
    
    def read_data(self):
        data = pd.read_csv(self.path + "/CASP.csv").values
        return data[:, 1:], data[:, 0].reshape(-1,1), self.get_cols(self.path)

class Yacht(Dataset):

    def __init__(self, split):
        super(Yacht, self).__init__(path = "data/yacht", split=split)
    
    def read_data(self):
        data = pd.read_fwf(self.path + "/yacht_hydrodynamics.data", header=None).values[:-1, :]
        return data[:, :-1], data[:, -1].reshape(-1,1), self.get_cols(self.path)

class Kin8nm(Dataset):
    def __init__(self, split):
        super(Kin8nm, self).__init__(path = "data/kin8nm", split=split)
        
    def read_data(self):
        data = pd.read_csv(self.path + "/kin8nm.csv", header=None).values
        return data[:, :-1], data[:, -1:].reshape(-1,1), self.get_cols(self.path)