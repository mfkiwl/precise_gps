import gpflow
import tensorflow as tf 
import tensorflow_probability as tfp 
from src.models.kernels import *
from src.models.initialization import select_inducing_points
from functools import wraps
from src.models.penalty import Penalty

class GPRPenalty(gpflow.models.GPR):
    """
    Basic Gaussian process regression, but L1 penalty term is added to the loss. This model
    assumes that the underlying kernel is either full Gaussian kernel or ARD kernel.
    """
    
    def __init__(self, **kwargs):

        data = kwargs["data"]
        kernel = kwargs["kernel"]
        super(GPRPenalty, self).__init__(data, kernel)

        self.lasso = 0 if "lasso" not in kwargs else kwargs["lasso"]

        size = self.data[0].shape[1]
        self.p = size if "p" not in kwargs else kwargs["p"]
        self.n = self.p if "n" not in kwargs else kwargs["n"]
        self.V = tf.eye(self.p, dtype = tf.float64) if "V" not in kwargs else kwargs["V"]
        self.penalty = "lasso" if "penalty" not in kwargs else kwargs["penalty"]

    def maximum_log_likelihood_objective(self) -> tf.Tensor:
        """
        Overwrites the gpflow.models.GPR.maximum_likelihood_objective
        See: https://gpflow.readthedocs.io/en/master/_modules/gpflow/models/gpr.html
        """
        return super().log_marginal_likelihood() - getattr(Penalty(), self.penalty)(self)

class SVIPenalty(gpflow.models.SVGP):
    """
    Stochastic variational Gaussian processes, but L1 penalty term is added to the loss. This model
    assumes that the underlying kernel is either full Gaussian kernel or ARD kernel.
    """
    
    def __init__(self, **kwargs):
        data = kwargs["data"]
        kernel = kwargs["kernel"]
        M = kwargs["M"]
        
        N = len(data[1])
        new_X = select_inducing_points(data[0], M)        
        super(SVIPenalty, self).__init__(kernel, gpflow.likelihoods.Gaussian(), new_X, num_data = N)

        self.lasso = 0 if "lasso" not in kwargs else kwargs["lasso"]

        size = data[0].shape[1]
        self.p = size if "p" not in kwargs else kwargs["p"]
        self.n = self.p if "n" not in kwargs else kwargs["n"]
        self.V = tf.eye(self.p, dtype = tf.float64) if "V" not in kwargs else kwargs["V"]
        self.penalty = "lasso" if "penalty" not in kwargs else kwargs["penalty"]
        self.train_data = data 

    def maximum_log_likelihood_objective(self, data) -> tf.Tensor:
        """
        Overwrites the gpflow.models.SVGP.maximum_likelihood_objective
        See: https://gpflow.readthedocs.io/en/master/_modules/gpflow/models/gpr.html
        """
        return super().elbo(data) - getattr(Penalty(), self.penalty)(self)

class Standard_GPR(gpflow.models.GPR):

    def __init__(self, **kwargs):
        data = kwargs["data"]
        kernel = kwargs["kernel"]
        super(Standard_GPR, self).__init__(data, kernel)