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

import numpy as np
import tensorflow as tf

from functools import wraps
def print_name(f):
    @wraps(f)
    def wrapper(*args, **kwds):
        #print(f.__name__)
        return f(*args, **kwds)
    return wrapper


class BaseModel(object):
    def __init__(self, X, Y, vars, minibatch_size, window_size):
        self.X_placeholder = tf.compat.v1.placeholder(tf.float64, 
                                                      shape=[None, X.shape[1]])
        self.Y_placeholder = tf.compat.v1.placeholder(tf.float64, 
                                                      shape=[None, Y.shape[1]])
        self.X = X
        self.Y = Y
        self.N = X.shape[0]
        self.vars = vars
        self.minibatch_size = min(minibatch_size, self.N)
        self.data_iter = 0
        self.window_size = window_size
        self.window = []
        self.posterior_samples = []
        self.sample_op = None
        self.burn_in_op = None

    @print_name
    def generate_update_step(self, nll, epsilon, mdecay):
        self.epsilon = epsilon
        burn_in_updates = []
        sample_updates = []

        grads = tf.gradients(nll, self.vars)

        for theta, grad in zip(self.vars, grads):
            xi = tf.Variable(tf.ones_like(theta), dtype=tf.float64, 
                             trainable=False)
            g = tf.Variable(tf.ones_like(theta), dtype=tf.float64, 
                            trainable=False)
            g2 = tf.Variable(tf.ones_like(theta), dtype=tf.float64, 
                             trainable=False)
            p = tf.Variable(tf.zeros_like(theta), dtype=tf.float64, 
                            trainable=False)

            r_t = 1. / (xi + 1.)
            g_t = (1. - r_t) * g + r_t * grad
            g2_t = (1. - r_t) * g2 + r_t * grad ** 2
            xi_t = 1. + xi * (1. - g * g / (g2 + 1e-16))
            Minv = 1. / (tf.sqrt(g2 + 1e-16) + 1e-16)

            burn_in_updates.append((xi, xi_t))
            burn_in_updates.append((g, g_t))
            burn_in_updates.append((g2, g2_t))

            epsilon_scaled = epsilon / tf.sqrt(tf.cast(self.N, tf.float64))
            noise_scale = 2. * epsilon_scaled ** 2 * mdecay * Minv
            sigma = tf.sqrt(tf.maximum(noise_scale, 1e-16))
            sample_t = tf.random.normal(tf.shape(theta), 
                                        dtype=tf.float64) * sigma
            p_t = p - epsilon ** 2 * Minv * grad - mdecay * p + sample_t
            theta_t = theta + p_t

            sample_updates.append((theta, theta_t))
            sample_updates.append((p, p_t))

        self.sample_op = [
            var.assign(var_t) for var, var_t in sample_updates]
        self.burn_in_op = [
            var.assign(var_t) for var, var_t in burn_in_updates \
                + sample_updates]

    @print_name
    def reset(self, X, Y):
        self.X, self.Y, self.N = X, Y, X.shape[0]
        self.data_iter = 0

    #@print_name
    def get_minibatch(self):
        assert self.N >= self.minibatch_size
        if self.N == self.minibatch_size:
            return self.X, self.Y

        if self.N < self.data_iter + self.minibatch_size:
            shuffle = np.random.permutation(self.N)
            self.X = self.X[shuffle, :]
            self.Y = self.Y[shuffle, :]
            self.data_iter = 0

        X_batch = self.X[self.data_iter:self.data_iter + self.minibatch_size, :]
        Y_batch = self.Y[self.data_iter:self.data_iter + self.minibatch_size, :]
        self.data_iter += self.minibatch_size
        return X_batch, Y_batch

    @print_name
    def collect_samples(self, num, spacing):
        self.posterior_samples = []
        for i in range(num):
            for j in range(spacing):
                X_batch, Y_batch = self.get_minibatch()
                feed_dict = {self.X_placeholder: X_batch, 
                             self.Y_placeholder: Y_batch}
                self.session.run((self.sample_op), feed_dict=feed_dict)

            values = self.session.run((self.vars))
            sample = {}
            for var, value in zip(self.vars, values):
                sample[var] = value
            self.posterior_samples.append(sample)

    @print_name
    def sghmc_step(self):
        X_batch, Y_batch = self.get_minibatch()
        #print(X_batch.shape)
        feed_dict = {self.X_placeholder: X_batch, self.Y_placeholder: Y_batch}
        #print("burn", len(self.burn_in_op))
        #print("feed", len(feed_dict))
        #print("vars", len(self.vars))
        #print(self.session.run((self.vars)))
        #print(self.session.run((self.vars)))
        self.session.run(self.burn_in_op, feed_dict=feed_dict)
        #print("Var1", a[8])
        #a = self.session.run(self.burn_in_op, feed_dict=feed_dict)
        #print("Var2", a[8])
        #print(self.burn_in_op, len(self.burn_in_op))
        #print(self.session.run((self.vars)))
        values = self.session.run((self.vars))
        sample = {}
        for var, value in zip(self.vars, values):
            sample[var] = value
        self.window.append(sample)
        if len(self.window) > self.window_size:
            self.window = self.window[-self.window_size:]

    @print_name
    def train_hypers(self):
        X_batch, Y_batch = self.get_minibatch()
        feed_dict = {self.X_placeholder: X_batch, self.Y_placeholder: Y_batch}
        i = np.random.randint(len(self.window))
        feed_dict.update(self.window[i])
        self.session.run(self.hyper_train_op, feed_dict=feed_dict)

    @print_name
    def print_sample_performance(self, posterior=False):
        X_batch, Y_batch = self.get_minibatch()
        feed_dict = {self.X_placeholder: X_batch, self.Y_placeholder: Y_batch}
        if posterior:
            feed_dict.update(np.random.choice(self.posterior_samples))
        nll = self.session.run((self.nll), feed_dict = feed_dict)
        print(' Training NLL of a sample: {}'.format(nll))
        return nll