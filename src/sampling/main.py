import numpy as np
from scipy.stats import norm
import pandas
import tensorflow as tf

from src.sampling.models import RegressionModel
from src.datasets.datasets import Concrete
from src.select import select_dataset
from sklearn.metrics import mean_squared_error

def _calculate_density(self, Xs, Ys, Ystd):
    ms, vs = self._predict(Xs, 100)
    logps = norm.logpdf(np.repeat(Ys[None, :, :], 100, axis=0)*Ystd, ms*Ystd, np.sqrt(vs)*Ystd)
    return np.log(np.sum(np.exp(logps), axis = 0)) - np.log(100)

def run():
    # path = 'src/sampling/data/kin8nm.csv'
    # data = pandas.read_csv(path, header=None).values

    # X_full = data[:, :-1]
    # Y_full = data[:, -1:]


    # N = X_full.shape[0]
    # n = int(N * 0.8)
    # ind = np.arange(N)

    # np.random.shuffle(ind)
    # train_ind = ind[:n]
    # test_ind = ind[n:]

    # X = X_full[train_ind]
    # Xs = X_full[test_ind]
    # Y = Y_full[train_ind]
    # Ys = Y_full[test_ind]

    # X_mean = np.mean(X, 0)
    # X_std = np.std(X, 0)
    # X = (X - X_mean) / X_std
    # Xs = (Xs - X_mean) / X_std
    # Y_mean = np.mean(Y, 0)
    # Y = (Y - Y_mean)
    # Ys = (Ys - Y_mean)
    ds = "Yacht"
    dataset = select_dataset(ds, 0.2)
    Ystd = dataset.y_std[0][0]


    X = dataset.train_X
    Xs = dataset.test_X
    Y = dataset.train_y
    Ys = dataset.test_y
    mses = []
    mlls = []
    coefs = [0]
    for coef in coefs:
        model = RegressionModel(coef = coef, dataset = dataset, mdl = "ARD", ds_name = ds)
        model.fit(X, Y)

        m, v = model.predict(Xs)
        print('MSE', mean_squared_error(Ys, m, squared=False))
        print('MLL', np.mean(model.calculate_density(Xs, Ys)))
        mses.append(mean_squared_error(Ys, m, squared=False))
        mlls.append(np.mean(model.calculate_density(Xs, Ys)))
    
    np.save(f"results/raw/{ds}/mses_ard", mses)
    np.save(f"results/raw/{ds}/mlls_ard", mlls)

    mses = []
    mlls = []
    coefs = np.arange(0.0001,1,0.1)
    for coef in coefs:
        model = RegressionModel(coef = coef, dataset = dataset, mdl = "Test",  ds_name = ds)
        model.fit(X, Y)

        m, v = model.predict(Xs)
        print('MSE', mean_squared_error(Ys, m, squared=False))
        print('MLL', np.mean(model.calculate_density(Xs, Ys)))
        mses.append(mean_squared_error(Ys, m, squared=False))
        mlls.append(np.mean(model.calculate_density(Xs, Ys)))
    
    np.save(f"results/raw/{ds}/mses_full", mses)
    np.save(f"results/raw/{ds}/mlls_full", mlls)