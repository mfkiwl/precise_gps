import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd


# This file provides dataset object that automatically preprocess the 
# raw data, and creates train and test sets.

URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases'

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
    Automatically preprocesses data and creates train and test set for 
    the specified dataset.

    Args:
        path (str to directory) : contains data and the input 
        column names
        split (float) : test/train split specifies testset size 
        (between 0-1) 
    """

    def __init__(self, path, split = 0.2):
        self.path = path 
        X, y, cols = self.read_data()
        Xnp, ynp = self.preprocess(X,y)
        train_Xnp, test_Xnp, train_ynp, test_ynp = \
            train_test_split(Xnp, ynp, test_size=split, random_state=42)

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
        return [f'${chr(92)}mathrm{{{d}}}$'.replace("'", '') for d in list(
            pd.read_csv(path + "/features.csv", delimiter=','))]

class Redwine(Dataset):

    def __init__(self, split):
        super(Redwine, self).__init__(path = "data/redwine", split=split)
        self.task = 'Regression'
    
    def read_data(self):
        data = np.genfromtxt(self.path + "/data.csv", delimiter=';')
        X = data[1:,0:-1]
        y = data[1:,-1].reshape(-1,1)
        cols = self.get_cols(self.path)
        return X, y, cols 

class Whitewine(Dataset):

    def __init__(self, split):
        super(Whitewine, self).__init__(path = "data/whitewine", split=split)
        self.task = 'Regression'
    
    def read_data(self):
        data = np.genfromtxt(self.path + "/data.csv", delimiter=';')
        X = data[1:,0:-1]
        y = data[1:,-1].reshape(-1,1)
        cols = self.get_cols(self.path)
        return X, y, cols 

class Naval(Dataset):

    def __init__(self, split):
        super(Naval, self).__init__(path = "data/naval", split=split)
        self.task = 'Regression'
    
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
        self.task = 'Regression'

    def read_data(self):
        data = pd.read_fwf(self.path + "/housing.data", header=None).values
        X = data[:, :-1]
        y = data[:, -1].reshape(-1,1)
        cols = self.get_cols(self.path)
        return X, y, cols 

class Concrete(Dataset):

    def __init__(self, split):
        super(Concrete, self).__init__(path = "data/concrete", split=split)
        self.task = 'Regression'

    def read_data(self):
        data = pd.read_excel(self.path + "/Concrete_Data.xls").values
        return data[:, :-1], data[:, -1].reshape(-1,1), self.get_cols(self.path)

class Energy(Dataset):

    def __init__(self, split):
        super(Energy, self).__init__(path = "data/energy", split=split)
        self.task = 'Regression'
    
    def read_data(self):
        # NB this is the first output (aka Energy1, as opposed to Energy2)
        data = pd.read_excel(self.path + "/ENB2012_data.xlsx").values[:, :-1]
        return data[:, :-1], data[:, -1].reshape(-1,1), self.get_cols(self.path)

class Power(Dataset):

    def __init__(self, split):
         super(Power, self).__init__(path = "data/power", split=split)
         self.task = 'Regression'
    
    def read_data(self):
        data = pd.read_excel(self.path + "/Folds5x2_pp.xlsx").values
        return data[:, :-1], data[:, -1].reshape(-1,1), self.get_cols(self.path)

class Protein(Dataset):

    def __init__(self, split):
        super(Protein, self).__init__(path = "data/protein", split=split)
        self.task = 'Regression'
    
    def read_data(self):
        data = pd.read_csv(self.path + "/CASP.csv").values
        return data[:, 1:], data[:, 0].reshape(-1,1), self.get_cols(self.path)

class Yacht(Dataset):

    def __init__(self, split):
        super(Yacht, self).__init__(path = "data/yacht", split=split)
        self.task = 'Regression'
    
    def read_data(self):
        data = pd.read_fwf(
            self.path + "/yacht_hydrodynamics.data", header=None).values[:-1, :]
        return data[:, :-1], data[:, -1].reshape(-1,1), self.get_cols(self.path)

class Kin8nm(Dataset):
    def __init__(self, split):
        super(Kin8nm, self).__init__(path = "data/kin8nm", split=split)
        self.task = 'Regression'
        
    def read_data(self):
        data = pd.read_csv(self.path + "/kin8nm.csv", header=None).values
        return data[:,:-1], data[:,-1:].reshape(-1,1), self.get_cols(self.path)

class Year(Dataset):
    def __init__(self, split):
        self.path = "data/year"
        X, y, cols = self.read_data()
        Xnp, ynp = self.preprocess(X,y)

        TRAIN_SIZE = 463_715
        self.train_X = Xnp[:TRAIN_SIZE]
        self.train_y = ynp[:TRAIN_SIZE]
        self.test_X = Xnp[TRAIN_SIZE:]
        self.test_y = ynp[TRAIN_SIZE:]
        self.cols = cols 
        self.task = 'Regression'
        
    
    def get_cols(self, path):
        return np.arange(90)
        
    def read_data(self):
        path = f'{URL}/00203/YearPredictionMSD.txt.zip'
        data = pd.read_csv(path,compression='zip', header = None).values
        return data[:,1:], data[:,0].reshape(-1,1), self.get_cols(self.path)

class Susy(Dataset):
    def __init__(self, split):
        self.path = 'data/susy'
        X, y, cols = self.read_data()
        Xnp, _ = self.preprocess(X,y)

        TRAIN_SIZE = 4_500_000
        self.train_X = Xnp[:TRAIN_SIZE]
        self.train_y = y[:TRAIN_SIZE]
        self.test_X = Xnp[TRAIN_SIZE:]
        self.test_y = y[TRAIN_SIZE:]
        self.cols = cols 
        self.task = 'Classification'
        
    def read_data(self):
        path = f'{URL}/00279/SUSY.csv.gz'
        #data = pd.read_csv(path,compression = 'gzip', header=None).values
        data = pd.read_csv('data/susy/SUSY.csv', header=None).values
        return data[:,1:], data[:,0].reshape(-1,1), self.get_cols(self.path)

class Higgs(Dataset):
    def __init__(self, split):
        self.path = 'data/higgs'
        X, y, cols = self.read_data()
        Xnp, _ = self.preprocess(X,y)

        TRAIN_SIZE = 10_500_000
        self.train_X = Xnp[:TRAIN_SIZE]
        self.train_y = y[:TRAIN_SIZE]
        self.test_X = Xnp[TRAIN_SIZE:]
        self.test_y = y[TRAIN_SIZE:]
        self.cols = cols 
        self.task = 'Classification'
        
    def read_data(self):
        path = f'{URL}/00280/HIGGS.csv.gz'
        #data = pd.read_csv(path,compression = 'gzip', header=None).values
        data = pd.read_csv('data/susy/HIGGS.csv', header=None).values
        return data[:,1:], data[:,0].reshape(-1,1), self.get_cols(self.path)