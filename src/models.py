import gpflow
import tensorflow as tf 
import tensorflow_probability as tfp 
from src.kernels import *

class GPRLasso(gpflow.models.GPR):
    """
    Basic Gaussian process regression, but L1 penalty term is added to the loss. This model
    assumes that the underlying kernel is either full Gaussian kernel or ARD kernel.
    """
    
    def __init__(self, data, kernel, lasso):
        super(GPRLasso, self).__init__(data, kernel)
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

class GPRHorseshoe(gpflow.models.GPR):
    """
    Basic Gaussian process regression, but L1 penalty term is added to the loss. This model
    assumes that the underlying kernel is either full Gaussian kernel or ARD kernel.
    """
    
    def __init__(self, data, kernel, horseshoe):
        super(GPRHorseshoe, self).__init__(data, kernel)
        self.horseshoe = horseshoe # lasso coefficient
    
    def horseshoe_penalty(self):
        if type(self.kernel) == FullGaussianKernel:
            L = tfp.math.fill_triangular(self.kernel.L)
            return self.horseshoe*tf.math.log(tf.math.reduce_sum(tf.math.log(1 + 2*self.kernel.L**(-2))))
        #else:
        #    return self.lasso*tf.math.reduce_sum(tf.abs(tf.linalg.diag(self.kernel.lengthscales**(-2))))

    def maximum_log_likelihood_objective(self):
        """
        Overwrites the gpflow.models.GPR.maximum_likelihood_objective
        See: https://gpflow.readthedocs.io/en/master/_modules/gpflow/models/gpr.html
        """
        return self.log_marginal_likelihood() - self.horseshoe_penalty()

class SVILasso(gpflow.models.SVGP):
    """
    Stochastic variational Gaussian processes, but L1 penalty term is added to the loss. This model
    assumes that the underlying kernel is either full Gaussian kernel or ARD kernel.
    """
    
    def __init__(self, data, kernel, lasso, M):
        
        N = len(data[1])
        indusing_points = np.random.choice(N, M)
        new_X = data[0][indusing_points]
        new_Y = data[1][indusing_points]
        # data = (new_X, new_Y)

        super(SVILasso, self).__init__(kernel, gpflow.likelihoods.Gaussian(), new_X, num_data = N)
        self.lasso = lasso # lasso coefficient
    
    def lasso_penalty(self):
        if type(self.kernel) == FullGaussianKernel:
            L = tfp.math.fill_triangular(self.kernel.L)
            return self.lasso*tf.math.reduce_sum(tf.abs(L @ tf.transpose(L)))
        else:
            return self.lasso*tf.math.reduce_sum(tf.abs(tf.linalg.diag(self.kernel.lengthscales**(-2))))

    def maximum_log_likelihood_objective(self, data):
        """
        Overwrites the gpflow.models.SVGP.maximum_likelihood_objective
        See: https://gpflow.readthedocs.io/en/master/_modules/gpflow/models/gpr.html
        """
        return self.elbo(data) - self.lasso_penalty()