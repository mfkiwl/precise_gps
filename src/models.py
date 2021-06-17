import gpflow
import tensorflow as tf 
import tensorflow_probability as tfp 
from src.kernels import *

class GPRLassoFull(gpflow.models.GPR):
    """
    Basic Gaussian process regression, but L1 penalty term is added to the loss. This model
    assumes that the underlying kernel is full Gaussian kernel. See: src/kernels/.
    """
    
    def __init__(self, data, kernel, lasso):
        super(GPRLassoFull, self).__init__(data, kernel)
        assert type(kernel) == FullGaussianKernel # assumes that full Gaussian kernel is used
        self.lasso = lasso # lasso coefficient
    
    def lml_lasso(self):
        """
        Returns margial log-likelihood which is maximized during optimization process.
        """
        L = tfp.math.fill_triangular(self.kernel.L)
        return self.log_marginal_likelihood() - self.lasso*tf.math.reduce_sum(tf.abs(L @ tf.transpose(L)))

    def maximum_log_likelihood_objective(self):
        """
        Overwrites the gpflow.models.GPR.maximum_likelihood_objective
        See: https://gpflow.readthedocs.io/en/master/_modules/gpflow/models/gpr.html
        """
        return self.lml_lasso()

class GPRLassoARD(gpflow.models.GPR):
    """
    Basic Gaussian process regression, but L1 penalty term is added to the loss. This model
    assumes that the underlying kernel is the ARD kernel kernel. See: src/kernels/.
    """
    
    def __init__(self, data, kernel, lasso):
        super(GPRLassoARD, self).__init__(data, kernel)
        assert type(kernel) == ARD or gpflow.kernels.SquaredExponential # assumes ARD kernel
        self.lasso = lasso # lasso coefficient
    
    def lml_lasso(self):
        """
        Returns margial log-likelihood which is maximized during optimization process.
        """
        return self.log_marginal_likelihood() - self.lasso*tf.math.reduce_sum(tf.abs(tf.linalg.diag(self.kernel.lengthscales**(-2))))
    
    def maximum_log_likelihood_objective(self):
        """
        Overwrites the gpflow.models.GPR.maximum_likelihood_objective
        See: https://gpflow.readthedocs.io/en/master/_modules/gpflow/models/gpr.html
        """
        return self.lml_lasso()