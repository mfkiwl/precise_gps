import gpflow
import tensorflow as tf 
import tensorflow_probability as tfp 
from src.models.kernels import *
from src.models.initialization import select_inducing_points
from src.models.prior import Prior
from src.sampling.sghmc_models import RegressionModel
from gpflow import set_trainable

class GPRPenalty(gpflow.models.GPR):
    """
    Basic Gaussian process regression, but L1 penalty term is added to 
    the loss. This model assumes that the underlying kernel is either 
    full Gaussian kernel or ARD kernel.
    """
    
    def __init__(self, **kwargs):

        data = kwargs["data"]
        kernel = kwargs["kernel"]
        super(GPRPenalty, self).__init__((data.train_X, data.train_y), kernel)

        self.lasso = 0 if "lasso" not in kwargs else kwargs["lasso"]

        size = self.data[0].shape[1]
        self.p = size if "p" not in kwargs else kwargs["p"]
        self.n = self.p if "n" not in kwargs else kwargs["n"]
        self.V = tf.eye(self.p, dtype = tf.float64) if "V" not in kwargs else  \
            kwargs["V"]
        self.penalty = "lasso" if "penalty" not in kwargs else kwargs["penalty"]

    def maximum_log_likelihood_objective(self) -> tf.Tensor:
        """
        Overwrites the gpflow.models.GPR.maximum_likelihood_objective
        """
        return super().log_marginal_likelihood() + \
            getattr(Prior(), self.penalty)(self)

class SVIPenalty(gpflow.models.SVGP):
    """
    Stochastic variational Gaussian processes, but L1 penalty term is 
    added to the loss. This model assumes that the underlying kernel is 
    either full Gaussian kernel or ARD kernel.
    """
    
    def __init__(self, **kwargs):
        data = kwargs["data"]
        kernel = kwargs["kernel"]
        M = kwargs["M"]
        
        N = len(data.train_y)
        new_X = select_inducing_points(data.train_X, M)
        if data.task == 'Classification':    
            super(SVIPenalty, self).__init__(kernel, 
                                             gpflow.likelihoods.Bernoulli(), 
                                             new_X, num_data = N)
        else:
            super(SVIPenalty, self).__init__(kernel, 
                                             gpflow.likelihoods.Gaussian(), 
                                             new_X, num_data = N)
            

        self.lasso = 0 if "lasso" not in kwargs else kwargs["lasso"]

        size = data.train_X.shape[1]
        self.p = size if "p" not in kwargs else kwargs["p"]
        self.n = self.p if "n" not in kwargs else kwargs["n"]
        self.V = tf.eye(self.p, dtype = tf.float64) if "V" not in kwargs else \
            kwargs["V"]
        self.penalty = "lasso" if "penalty" not in kwargs else kwargs["penalty"]
        self.train_data = (data.train_X, data.train_y)
        
        set_trainable(self.q_mu, False)
        set_trainable(self.q_sqrt, False)
        
        self.variational_params = [(self.q_mu, self.q_sqrt)]

    def maximum_log_likelihood_objective(self, data) -> tf.Tensor:
        """
        Overwrites the gpflow.models.SVGP.maximum_likelihood_objective
        """
        return super().elbo(data) + getattr(Prior(), self.penalty)(self)

class Standard_GPR(gpflow.models.GPR):

    def __init__(self, **kwargs):
        data = kwargs["data"]
        kernel = kwargs["kernel"]
        super(Standard_GPR, self).__init__(data, kernel)

class SGHMC(RegressionModel, gpflow.models.SVGP):
    def __init__(self, **kwargs):
        data = kwargs["data"]
        kernel = kwargs["kernel"]
        
        lasso = 0 if "lasso" not in kwargs else kwargs["lasso"]
        size = data.train_X.shape[1]
        self.p = size if "p" not in kwargs else kwargs["p"]
        penalty = "lasso" if "penalty" not in kwargs else kwargs["penalty"]
        n = self.p if "n" not in kwargs else kwargs["n"]
        V = tf.eye(self.p, dtype = tf.float64) if "V" not in kwargs else \
            kwargs["V"]
        
        M = kwargs["M"]
        
        N = len(data.train_y)
        new_X = select_inducing_points(data.train_X, M)
        
        q_mu= np.zeros((M,1))
        q_sqrt = [np.eye(M, dtype = np.float64) for _ in range(1)]
        q_sqrt = np.array(q_sqrt)
        
        #gpflow.models.SVGP.__init__(self, kernel, 
        #                                     gpflow.likelihoods.Gaussian(), 
        #                                     new_X, num_data = N, q_mu = q_mu,
        #                                     q_sqrt = q_sqrt)
        
        #set_trainable(self.q_mu, False)
        #set_trainable(self.q_sqrt, False)
        
        #self.variational_params = [(self.q_mu, self.q_sqrt)]
        self.n = n
        self.V = V
        RegressionModel.__init__(self, data, kernel, lasso, n, V, penalty)
        