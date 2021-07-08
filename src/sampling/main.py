import numpy as np
from scipy.stats import norm

from src.sampling.models import RegressionModel
from src.datasets.datasets import Boston
from src.select import select_dataset

def run():
    dataset = select_dataset("Boston", 0.2)


    X = dataset.train_X
    Xs = dataset.test_X
    Y = dataset.train_y
    Ys = dataset.test_y

    X_mean = np.mean(X, 0)
    X_std = np.std(X, 0)
    X = (X - X_mean) / X_std
    Xs = (Xs - X_mean) / X_std
    Y_mean = np.mean(Y, 0)
    Y = (Y - Y_mean)
    Ys = (Ys - Y_mean)

    model = RegressionModel()
    model.fit(X, Y)

    m, v = model.predict(Xs)
    print('MSE', np.mean(np.square(Ys - m)))
    print('MLL', np.mean(model.calculate_density(Xs, Ys)))