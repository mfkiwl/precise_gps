from src.sampling.sghmc_gp import DGP
from src.models.kernels import *

import numpy as np
from scipy.stats import norm
from src.sampling.likelihoods import Gaussian
from src.select import select_kernel
from src.sampling.kernels import SquaredExponential
import scipy


class RegressionModel(object):
    def __init__(self, coef, dataset,mdl,ds_name):
        class ARGS:
            num_inducing = 100
            iterations = 10000
            minibatch_size = 10000
            window_size = 100
            num_posterior_samples = 100
            posterior_sample_spacing = 50
        self.ARGS = ARGS
        self.model = None
        self.coef = coef
        self.dataset = dataset
        self.ds_name = ds_name.lower()
        self.mdl = mdl

    def fit(self, X, Y):
        lik = Gaussian(np.var(Y, 0))
        return self._fit(X, Y, lik)

    def _fit(self, X, Y, lik, **kwargs):
        if len(Y.shape) == 1:
            Y = Y[:, None]

        kerns = []
        if not self.model:
            for _ in range(1):
                kernel_kwargs = {"dim": X.shape[1], "randomized": True}
                if self.mdl == "ARD":
                    _kernel = select_kernel("Test_ARD", **kernel_kwargs)
                else:
                    _kernel = select_kernel("Test", **kernel_kwargs)
                #_kernel = SquaredExponential(X.shape[1], ARD=True, lengthscales=float(X.shape[1])**0.5)
                kerns.append(_kernel)

            mb_size = self.ARGS.minibatch_size if X.shape[0] > self.ARGS.minibatch_size else X.shape[0]

            self.model = DGP(X, Y, 100, kerns, lik,
                             minibatch_size=mb_size,
                             window_size=self.ARGS.window_size, coef = self.coef, ds_name = self.ds_name,
                             **kwargs)

        self.model.reset(X, Y)

        try:
            for _ in range(self.ARGS.iterations):
                self.model.sghmc_step()
                self.model.train_hypers()
                if _ % 100 == 1:
                    print('Iteration {}'.format(_))
                    self.model.print_sample_performance()
            self.model.collect_samples(self.ARGS.num_posterior_samples, self.ARGS.posterior_sample_spacing)

        except KeyboardInterrupt:  # pragma: no cover
            pass

    def _predict(self, Xs, S):
        ms, vs = [], []
        n = max(len(Xs) / 100, 1)  # predict in small batches
        for xs in np.array_split(Xs, n):
            m, v = self.model.predict_y(xs, S)
            ms.append(m)
            vs.append(v)

        return np.concatenate(ms, 1), np.concatenate(vs, 1)  # num_posterior_samples, N_test, D_y

    def predict(self, Xs):
        ms, vs = self._predict(Xs, self.ARGS.num_posterior_samples)
        m = np.average(ms, 0)
        v = np.average(vs + ms**2, 0) - m**2
        return m, v

    def calculate_density(self, Xs, Ys):
        Y_std = self.dataset.y_std[0][0]
        ms, vs = self._predict(Xs, self.ARGS.num_posterior_samples)
        logps = norm.logpdf(np.repeat(Ys[None, :, :], self.ARGS.num_posterior_samples, axis=0)*Y_std, ms*Y_std, np.sqrt(vs)*Y_std)
        #return np.log(np.sum(np.exp(logps), axis = 0)) - np.log(self.ARGS.num_posterior_samples)
        return scipy.special.logsumexp(logps, axis = 0) - np.log(self.ARGS.num_posterior_samples)


    def sample(self, Xs, S):
        ms, vs = self._predict(Xs, S)
        return ms + vs**0.5 * np.random.randn(*ms.shape)