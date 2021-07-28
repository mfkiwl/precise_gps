import numpy as np
from scipy.stats import norm
import pandas
import tensorflow as tf

from src.sampling.models import RegressionModel
from src.datasets.datasets import Concrete
from src.select import select_dataset
from sklearn.metrics import mean_squared_error

def run():
    
    ds = "Yacht"
    dataset = select_dataset(ds, 0.2)

    X = dataset.train_X
    Xs = dataset.test_X
    Y = dataset.train_y
    Ys = dataset.test_y
    coefs = [0]
    num_runs = 10
    for coef in coefs:
        mses = []
        mlls = []
        for num_run in num_runs:
            model = RegressionModel(coef = coef, data = dataset)
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
        model = RegressionModel(coef = coef, dataset = dataset, kernel = "Test",  ds_name = ds)
        model.fit(X, Y)

        m, v = model.predict(Xs)
        print('MSE', mean_squared_error(Ys, m, squared=False))
        print('MLL', np.mean(model.calculate_density(Xs, Ys)))
        mses.append(mean_squared_error(Ys, m, squared=False))
        mlls.append(np.mean(model.calculate_density(Xs, Ys)))
    
    np.save(f"results/raw/{ds}/mses_full", mses)
    np.save(f"results/raw/{ds}/mlls_full", mlls)