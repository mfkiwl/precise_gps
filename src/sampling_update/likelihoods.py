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

import tensorflow as tf
import numpy as np

from functools import wraps
def print_name(f):
    @wraps(f)
    def wrapper(*args, **kwds):
        #print(f.__name__)
        return f(*args, **kwds)
    return wrapper


class Gaussian(object):
    def logdensity(self, x, mu, var):
        return -0.5 * (np.log(2 * np.pi) + tf.math.log(var) + \
            tf.square(mu-x) / var)

    def __init__(self, variance, **kwargs):
        self.variance = variance
    @print_name
    def logp(self, F, Y):
        return self.logdensity(Y, F, self.variance)

    @print_name
    def conditional_mean(self, F):
        return tf.identity(F)

    @print_name
    def conditional_variance(self, F):
        return tf.fill(tf.shape(F), tf.squeeze(self.variance))

    @print_name
    def predict_mean_and_var(self, Fmu, Fvar):
        return tf.identity(Fmu), Fvar + self.variance

    @print_name
    def predict_density(self, Fmu, Fvar, Y):
        return self.logdensity(Y, Fmu, Fvar + self.variance)

    @print_name
    def variational_expectations(self, Fmu, Fvar, Y):
        return -0.5 * np.log(2 * np.pi) - 0.5 * tf.math.log(self.variance) \
               - 0.5 * (tf.square(Y - Fmu) + Fvar) / self.variance