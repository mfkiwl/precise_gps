# MIT License

# Copyright (c) 2018 Hava842

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from src.sampling_update.sghmc_gp import DGP

import numpy as np
import tensorflow as tf
import scipy
import gpflow

from scipy.stats import norm

from functools import wraps
def print_name(f):
    @wraps(f)
    def wrapper(*args, **kwds):
        #print(f.__name__)
        return f(*args, **kwds)
    return wrapper


class RegressionModel(object):
    def __init__(self, data, kernel, lasso, n, V, penalty):
        class ARGS:
            num_inducing = 100
            iterations = 1_000
            minibatch_size = 10000
            window_size = 100
            num_posterior_samples = 100
            posterior_sample_spacing = 50
        self.ARGS = ARGS
        self.model = None
        self.lasso = lasso
        self.n = n
        self.V = V
        self.data = data
        self.kernel = kernel
        self.penalty = penalty
        self.nlls = []

    @print_name
    def fit(self, X, Y):
        #lik = Gaussian(np.var(Y, 0))
        return self._fit(X, Y)

    @print_name
    def _fit(self, X, Y, **kwargs):
        if len(Y.shape) == 1:
            Y = Y[:, None]

        kerns = []
        if not self.model:
            for _ in range(1):
                kerns.append(self.kernel)

            mb_size = self.ARGS.minibatch_size \
                if X.shape[0] > self.ARGS.minibatch_size else X.shape[0]

            self.model = DGP(X, Y, 100, kerns,
                             minibatch_size=mb_size,
                             window_size=self.ARGS.window_size, 
                             lasso = self.lasso, n = self.n, V = self.V,
                             penalty = self.penalty, **kwargs)

        self.model.reset(X, Y)

        try:
            for _ in range(self.ARGS.iterations):
                self.model.sghmc_step()
                self.model.train_hypers()
                #step_count = self.model.optimizer.minimize(self.model.nll, [var1]).numpy()
                if _ % 100 == 1:
                    print('Iteration {}'.format(_))
                    nll = self.model.print_sample_performance()
                    self.nlls.append(nll)
            self.model.collect_samples(self.ARGS.num_posterior_samples, 
                                       self.ARGS.posterior_sample_spacing)

        except KeyboardInterrupt:  # pragma: no cover
            pass

    @print_name
    def _predict(self, Xs, S):
        ms, vs = [], []
        n = max(len(Xs) / 100, 1)  # predict in small batches
        ps_d = []
        vs_d = []
        for xs in np.array_split(Xs, n):
            m, v, precisions, variances = self.model.predict_y(xs, S)
            ms.append(m)
            vs.append(v)
            ps_d.append(precisions)
            vs_d.append(variances)

        return np.concatenate(ms, 1), np.concatenate(vs, 1), ps_d, vs_d

    @print_name
    def predict(self, Xs):
        ms, vs, ps_d, vs_d = self._predict(Xs, self.ARGS.num_posterior_samples)
        m = np.average(ms, 0)
        v = np.average(vs + ms**2, 0) - m**2
        return m, v, ps_d, vs_d

    @print_name
    def calculate_density(self, Xs, Ys):
        Y_std = self.data.y_std[0][0]
        ms, vs, _, _ = self._predict(Xs, self.ARGS.num_posterior_samples)
        #print('Ys:', Ys.shape, 'Xs:', Xs.shape, ms.shape, vs.shape)
        #print(vs[0])
        #print(vs[1])
        logps = norm.logpdf(
            np.repeat(Ys[None, :, :], 
                      self.ARGS.num_posterior_samples, axis=0)*Y_std, 
            ms*Y_std, np.sqrt(vs)*Y_std)
        return scipy.special.logsumexp(logps, axis = 0) - \
            np.log(self.ARGS.num_posterior_samples)

    @print_name
    def sample(self, Xs, S):
        ms, vs, _, _ = self._predict(Xs, S)
        return ms + vs**0.5 * np.random.randn(*ms.shape)
