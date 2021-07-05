import numpy as np
import tensorflow as tf
import gpflow
import tensorflow_probability as tfp 
from src.models.initialization import *
from src.models.base_kernel import BaseKernel
    

class ARD(BaseKernel, gpflow.kernels.Kernel):
    """
    Own implementation of the squared exponential kernel with ard property. Should work
    the same way as gpflow.kernels.SquaredExponential(ARD = True). Lengthscales and variance 
    can be randomized. This should be handled when initializing the kernel.
    See : https://gpflow.readthedocs.io/en/master/gpflow/kernels/index.html#gpflow-kernels-squaredexponential

    Args:
        variance (float)           : signal variance which scales the whole kernel
        lengthscales (numpy array) : list of lengthscales (should match the dimension of the input)
    """
    def __init__(self, **kwargs):        
        super().__init__()
        randomized = kwargs["randomized"]
        dim = kwargs["dim"]
        if not randomized:
            lengthscales = np.ones(dim)
            variance = 1.0
        else:
            lengthscales = np.random.uniform(0.5,3,dim)
            variance = 1.0

        self.variance = gpflow.Parameter(variance, transform = gpflow.utilities.positive())
        self.lengthscales = gpflow.Parameter(lengthscales, transform = gpflow.utilities.positive())
        
    
    def K_diag(self, X) -> tf.Tensor:
        """
        Returns the diagonal vector when X1 == X2 (used in the background of gpflow)
        """
        return self.variance * tf.ones_like(X[:,0])
    
    def K(self, X1, X2=None) -> tf.Tensor:
        """
        Returns the squared exponential ard kernel.

        Args:
            X1 (numpy array) : shaped N x D
            X2 (numpy array) : shaped M x D (D denotes the number of dimensions of the input)
        """
        if X2 is None:
            X2 = X1
            
        # Precision is the inverse squared of the lengthscales
        P = tf.linalg.diag(self.lengthscales**(2))    
        X11 = tf.squeeze(tf.expand_dims(X1,axis = 1) @ P @ tf.expand_dims(X1,axis = -1),-1)  # (N,1)
        X22 = tf.transpose(tf.squeeze(tf.expand_dims(X2,axis = 1) @ P @ tf.expand_dims(X2,axis = -1),-1))  # (1,M)
        X12 = X1 @ P @ tf.transpose(X2) # (N,M)

        # kernel  (N,1) - (N,M) + (1,M)
        K = self.variance * tf.exp(-0.5 * (X11 - 2*X12 + X22))

        return K
    
    def precision(self) -> tf.Tensor:
        return tf.linalg.diag(self.lengthscales**(2))    

class ARD_gpflow(BaseKernel, gpflow.kernels.SquaredExponential):
    def __init__(self, **kwargs):
        randomized = kwargs["randomized"]
        dim = kwargs["dim"]
        if not randomized:
            lengthscales = np.ones(dim)
            variance = 1.0
        else:
            lengthscales = np.random.uniform(0.5,3,dim)
            variance = 1.0       
        super().__init__(variance, lengthscales)
    
    def precision(self) -> tf.Tensor:
        return tf.linalg.diag(self.lengthscales**(-2))   

class FullGaussianKernel(BaseKernel, gpflow.kernels.Kernel):
    """
    Implementation of the full Gaussian kernel which introduces also the off-diagonal
    covariates of the precision matrix. Randomizing the initialization should be handled outside
    of this class.

    Args:
        variance (float) : signal variance which scales the whole kernel
        L (numpy array)  : vector representation of L, where LL^T = P : precision
    """
    
    def __init__(self, **kwargs):
        super().__init__()
        randomized = kwargs["randomized"]
        dim = kwargs["dim"]
        if not randomized:
            L = np.ones((dim*(dim+1))//2)
            variance = 1.0
        else:
            L = init_precision(dim)
            variance = 1.0 

        self.variance = gpflow.Parameter(variance, transform = gpflow.utilities.positive())
        self.L = gpflow.Parameter(L)

    def K_diag(self, X) -> tf.Tensor:
        """
        Returns the diagonal vector when X1 == X2 (used in the background of gpflow)
        """
        return self.variance * tf.ones_like(X[:,0])
    
    def K(self, X1, X2=None) -> tf.Tensor:
        """
        Returns the full Gaussian kernel.

        Args:
            X1 (numpy array) : shaped N x D
            X2 (numpy array) : shaped M x D (D denotes the number of dimensions of the input)
        """
        if X2 is None:
            X2 = X1
        
        L = tfp.math.fill_triangular(self.L) # matrix representation of L

        A = X1 @ L
        B = X2 @ L 

        X11 = tf.squeeze(tf.expand_dims(A, axis = 1) @ tf.expand_dims(A, axis = -1), axis = -1) # (N, 1)
        X22 = tf.transpose(tf.squeeze(tf.expand_dims(B, axis = 1) @ tf.expand_dims(B, axis = -1), axis = -1))  # (1,M)
        X12 = A @ tf.transpose(B) # (N,M)

        # kernel  (N,1) - (N,M) + (1,M)
        K = self.variance*tf.exp(-0.5 * (X11 - 2*X12 + X22))

        return K
    
    def precision(self) -> tf.Tensor:
        L = tfp.math.fill_triangular(self.L)
        return L@tf.transpose(L)

class LowRankFullGaussianKernel(BaseKernel, gpflow.kernels.Kernel):
    """
    Implementation of the full Gaussian kernel which introduces also the off-diagonal
    covariates of the precision matrix. Randomizing the initialization should be handled outside
    of this class.

    Args:
        variance (float) : signal variance which scales the whole kernel
        L (numpy array)  : vector representation of L, where LL^T = P : precision
    """
    
    def __init__(self, **kwargs):
        super().__init__()
        randomized = kwargs["randomized"]
        dim = kwargs["dim"]
        rank = kwargs["rank"]
        if not randomized:
            L = np.ones((dim*(dim+1))//2)
            variance = 1.0
        else:
            L = init_lowrank_precision(dim, rank) 
            variance = 1.0 

        self.variance = gpflow.Parameter(variance, transform = gpflow.utilities.positive())
        self.L = gpflow.Parameter(L)

    def K_diag(self, X) -> tf.Tensor:
        """
        Returns the diagonal vector when X1 == X2 (used in the background of gpflow)
        """
        return self.variance * tf.ones_like(X[:,0])
    
    def K(self, X1, X2=None) -> tf.Tensor:
        """
        Returns the full Gaussian kernel.

        Args:
            X1 (numpy array) : shaped N x D
            X2 (numpy array) : shaped M x D (D denotes the number of dimensions of the input)
        """
        if X2 is None:
            X2 = X1
        
        L = fill_lowrank_triangular(self.L) # matrix representation of L

        A = X1 @ L
        B = X2 @ L 

        X11 = tf.squeeze(tf.expand_dims(A, axis = 1) @ tf.expand_dims(A, axis = -1), axis = -1) # (N, 1)
        X22 = tf.transpose(tf.squeeze(tf.expand_dims(B, axis = 1) @ tf.expand_dims(B, axis = -1), axis = -1))  # (1,M)
        X12 = A @ tf.transpose(B) # (N,M)

        # kernel  (N,1) - (N,M) + (1,M)
        K = self.variance*tf.exp(-0.5 * (X11 - 2*X12 + X22))

        return K
    
    def precision(self) -> tf.Tensor:
        L = fill_lowrank_triangular(self.L)
        return L@tf.transpose(L)