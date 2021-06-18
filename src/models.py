import gpflow
import tensorflow as tf 
import tensorflow_probability as tfp 
from src.kernels import *

class GPRLasso(gpflow.models.GPR):
    """
    Basic Gaussian process regression, but L1 penalty term is added to the loss. This model
    assumes that the underlying kernel is full Gaussian kernel. See: src/kernels/.
    """
    
    def __init__(self, data, kernel, lasso):
        super(GPRLasso, self).__init__(data, kernel)
        assert type(kernel) == FullGaussianKernel # assumes that full Gaussian kernel is used
        self.lasso = lasso # lasso coefficient
    
    def lasso_penalty(self):
        if type(self.kernel) == FullGaussianKernel:
            L = tfp.math.fill_triangular(self.kernel.L)
            return self.lasso*tf.math.reduce_sum(tf.abs(L @ tf.transpose(L)))
        else:
            return self.lasso*tf.math.reduce_sum(tf.abs(tf.linalg.diag(self.kernel.lengthscales**(-2))))

    def maximum_log_likelihood_objective(self):
        """
        Overwrites the gpflow.models.GPR.maximum_likelihood_objective
        See: https://gpflow.readthedocs.io/en/master/_modules/gpflow/models/gpr.html
        """
        return self.log_marginal_likelihood() - self.lasso_penalty()