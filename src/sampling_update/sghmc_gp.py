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

from src.models.initialization import select_inducing_points
import tensorflow as tf
import numpy as np
import gpflow

from src.sampling_update.sghmc_base import BaseModel
from src.sampling_update.conditionals import conditional as cond
from src.models.prior import Prior
from src.sampling_update.likelihoods import Gaussian

from functools import wraps
def print_name(f):
    @wraps(f)
    def wrapper(*args, **kwds):
        #print(f.__name__)
        return f(*args, **kwds)
    return wrapper

class Layer(object):
    def __init__(self, kern, outputs, n_inducing, fixed_mean, X):
        self.inputs, self.outputs, self.kernel = kern.dim, outputs, kern
        self.M, self.fixed_mean = n_inducing, fixed_mean
        
        self.Z = tf.Variable(select_inducing_points(X, self.M), 
                             dtype=tf.float64, name='Z')
        if self.inputs == outputs:
            self.mean = np.eye(self.inputs)
        elif self.inputs < self.outputs:
            self.mean = np.concatenate(
                [np.eye(self.inputs), 
                 np.zeros((self.inputs, self.outputs - self.inputs))], axis=1)
        else:
            _, _, V = np.linalg.svd(X, full_matrices=False)
            self.mean = V[:self.outputs, :].T

        self.U = tf.Variable(np.random.randn(*(self.M, self.outputs)), 
                             dtype=tf.float64, trainable=False, name='U')
    @print_name
    def conditional(self, X):
        # Caching the covariance matrix from the sghmc steps gives a significant 
        # speedup. This is not being done here.
        mean, var = cond(X, self.Z, self.kernel, self.U, white=True)

        if self.fixed_mean:
            mean += tf.matmul(X, tf.cast(self.mean, tf.float64))
        return mean, var

    @print_name
    def prior(self):
        return -tf.reduce_sum(tf.square(self.U)) / 2.0


class DGP(BaseModel):
    @print_name
    def propagate(self, X):
        Fs = [X, ]
        Fmeans, Fvars = [], []

        for layer in self.layers:
            mean, var = layer.conditional(Fs[-1])
            eps = tf.random.normal(tf.shape(mean), dtype=tf.float64)
            F = mean + eps * tf.sqrt(var)
            Fs.append(F)
            Fmeans.append(mean)
            Fvars.append(var)

        return Fs[1:], Fmeans, Fvars

    def __init__(self, X, Y, n_inducing, kernels, minibatch_size, 
                 window_size, lasso, n, V,penalty, adam_lr=0.01, epsilon=0.01, 
                 mdecay=0.05):
        
        self.exp_variance = tf.Variable(np.log(0.07), 
                        dtype=tf.float64, name='lik_log_variance')
        self.variance = tf.exp(self.exp_variance)
        likelihood = Gaussian(self.variance)
        
        self.n_inducing = n_inducing
        self.kernels = kernels
        self.kernel = kernels[0]
        self.likelihood = likelihood
        self.minibatch_size = minibatch_size
        self.window_size = window_size
        self.lasso = lasso
        self.n = n
        self.V = V
        self.penalty = penalty
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

        n_layers = len(kernels)
        self.N = X.shape[0]

        self.layers = []
        X_running = X.copy()
        for l in range(n_layers):
            outputs = self.kernels[l+1].dim if l+1 < n_layers else Y.shape[1]
            self.layers.append(
                Layer(self.kernels[l], outputs, n_inducing, 
                      fixed_mean=(l+1 < n_layers), X=X_running))
            X_running = np.matmul(X_running, self.layers[-1].mean)
        new_vars = [l.U for l in self.layers] + [self.kernels[0].variance] #+ [self.kernels[0].L]
        
        #new_vars = [self.kernels[0].variance] #+ [self.kernels[0].L] 
        
        
        super().__init__(X, Y, new_vars, minibatch_size, window_size)
        
    @tf.function
    def calculate_nll(self):
        self.f, self.fmeans, self.fvars = self.propagate(self.X_batch)
        self.y_mean, self.y_var = self.likelihood.predict_mean_and_var(
            self.fmeans[-1], self.fvars[-1])                                                       

        self.prior = tf.add_n([l.prior() for l in self.layers])
        self.log_likelihood = self.likelihood.predict_density(
            self.fmeans[-1], self.fvars[-1], self.Y_batch)

        self.nll = - tf.reduce_sum(
             self.log_likelihood) / tf.cast(tf.shape(self.X_batch)[0], 
             tf.float64)*self.N - self.prior #- getattr(Prior(), self.penalty)(self)
        self.nll /= self.N
        
        #self.varexps = self.likelihood.variational_expectations(
        #    self.fmeans[-1], self.fvars[-1], self.Y_placeholder)
        #self.nll = - tf.reduce_sum(self.varexps) / tf.cast(
        #    tf.shape(self.X_placeholder)[0], tf.float64) \
        #          - (self.prior / N) - getattr(Prior(), self.penalty)(self) / N
        return self.nll
        
        #self.generate_update_step(self.nll, epsilon, mdecay)
        #self.adam = tf.compat.v1.train.AdamOptimizer(adam_lr)
        #self.hyper_train_op = self.adam.minimize(self.nll)

        #config = tf.compat.v1.ConfigProto()
        #config.gpu_options.allow_growth = True
        #self.session = tf.compat.v1.Session(config=config)
        #init_op = tf.compat.v1.global_variables_initializer()
        #self.session.run(init_op)

    # @print_name
    # def predict_y(self, X, S):
    #     assert S <= len(self.posterior_samples)
    #     ms, vs = [], []
    #     precisions = []
    #     variances = []
    #     for i in range(S):
    #         f, fmeans, fvars = self.propagate(X)
    #         y_mean, y_var = self.likelihood.predict_mean_and_var(
    #         fmeans[-1], fvars[-1]) 
    #         #feed_dict = {self.X_placeholder: X}
    #         #feed_dict.update(self.posterior_samples[i])
    #         #L = list(self.posterior_samples[i].values())[-2].tolist()
    #         #precisions.append(L)
    #         #variances.append(list(self.posterior_samples[i].values())[-1])
    #         m, v = self.session.run((self.y_mean, self.y_var), 
    #                                 feed_dict=feed_dict)
    #         ms.append(m)
    #         vs.append(v)
    #     return np.stack(ms, 0), np.stack(vs, 0), precisions, variances
    
    def collect_samples(self, num, spacing):
        self.posterior_samples = []
        for i in range(num):
            for j in range(spacing):
                X_batch, Y_batch = self.get_minibatch()
                self.X_batch = X_batch
                self.Y_batch = Y_batch
                nll = self.calculate_nll
                self.generate_update_step(nll, epsilon=0.01, mdecay=0.05, burn_in = False)

            values = self.vars
            sample = []
            for var, value in zip(self.vars, values):
                sample.append(value)
            self.posterior_samples.append(sample)
    
    @print_name
    def sghmc_step(self):
        X_batch, Y_batch = self.get_minibatch()
        self.X_batch = X_batch
        self.Y_batch = Y_batch
        nll = self.calculate_nll
        self.generate_update_step(nll, epsilon=0.01, mdecay=0.05, burn_in = True)
        #print("Var1", self.burn_in_op[8])
        values = self.vars
        sample = []
        for var, value in zip(self.vars, values):
            sample.append(value)
        self.window.append(sample)
        if len(self.window) > self.window_size:
            self.window = self.window[-self.window_size:]
    
    @print_name
    def print_sample_performance(self, posterior=False):
        X_batch, Y_batch = self.get_minibatch()
        self.X_batch = X_batch
        self.Y_batch = Y_batch
        nll = self.calculate_nll()
        print(' Training NLL of a sample: {}'.format(nll))
        return nll

    @print_name
    def train_hypers(self):
        X_batch, Y_batch = self.get_minibatch()
        self.X_batch = X_batch
        self.Y_batch = Y_batch
        i = np.random.randint(len(self.window))
        window_ = self.window[i]
        self.layers[0].U = window_[0]
        #self.kernels[0].variance = window_[1]
        self.optimizer.minimize(self.calculate_nll, [self.kernels[0].L, self.layers[0].Z, self.kernels[0].variance, self.exp_variance])
        #self.session.run(self.hyper_train_op, feed_dict=feed_dict)