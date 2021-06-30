import gpflow
import tensorflow as tf 
import tensorflow_probability as tfp 
from src.models.kernels import *
from src.models.initialization import select_inducing_points

def _lasso_penalty(model):
    if type(model.kernel) == FullGaussianKernel:
        L = tfp.math.fill_triangular(model.kernel.L)
        return model.lasso*tf.math.reduce_sum(tf.abs(L @ tf.transpose(L)))
    elif type(model.kernel) == LowRankFullGaussianKernel:
        L = fill_lowrank_triangular(model.kernel.L)
        return model.lasso*tf.math.reduce_sum(tf.abs(L @ tf.transpose(L)))
    elif type(model.kernel) == ARD:
        return model.lasso*tf.math.reduce_sum(tf.abs(tf.linalg.diag(model.kernel.lengthscales**(2))))
    else:
        return model.lasso*tf.math.reduce_sum(tf.abs(tf.linalg.diag(model.kernel.lengthscales**(-2))))

class GPRLasso(gpflow.models.GPR):
    """
    Basic Gaussian process regression, but L1 penalty term is added to the loss. This model
    assumes that the underlying kernel is either full Gaussian kernel or ARD kernel.
    """
    
    def __init__(self, **kwargs):
        data = kwargs["data"]
        kernel = kwargs["kernel"]
        lasso = kwargs["lasso"]
        super(GPRLasso, self).__init__(data, kernel)
        self.lasso = lasso # lasso coefficient
    
    def maximum_log_likelihood_objective(self):
        """
        Overwrites the gpflow.models.GPR.maximum_likelihood_objective
        See: https://gpflow.readthedocs.io/en/master/_modules/gpflow/models/gpr.html
        """
        return self.log_marginal_likelihood() - _lasso_penalty(self)

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
            P = L@tf.transpose(L)
            return self.horseshoe*tf.math.log(tf.math.reduce_sum(tf.math.log(1 + 2*P**(-2))))

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
    
    def __init__(self, **kwargs):
        data = kwargs["data"]
        kernel = kwargs["kernel"]
        lasso = kwargs["lasso"]
        M = kwargs["M"]
        
        N = len(data[1])
        new_X = select_inducing_points(data[0], M)

        super(SVILasso, self).__init__(kernel, gpflow.likelihoods.Gaussian(), new_X, num_data = N)
        self.lasso = lasso # lasso coefficient
        self.train_data = data

    def maximum_log_likelihood_objective(self, data):
        """
        Overwrites the gpflow.models.SVGP.maximum_likelihood_objective
        See: https://gpflow.readthedocs.io/en/master/_modules/gpflow/models/gpr.html
        """
        return self.elbo(data) - _lasso_penalty(self)

class Standard_GPR(gpflow.models.GPR):

    def __init__(self, **kwargs):
        data = kwargs["data"]
        kernel = kwargs["kernel"]
        super(Standard_GPR, self).__init__(data, kernel)