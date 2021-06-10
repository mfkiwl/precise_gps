import numpy as np
import tensorflow as tf
import gpflow
import tensorflow_probability as tfp 

class FullGaussianKernel(gpflow.kernels.Kernel):
    """
    This kernel calculates the 'full' Gaussian kernel defined in ...
    It is used for learning the off-diagonal covariates of the precion matrix.
    """
    
    def __init__(self, covariates, sf = 1, init_factor = 1, randomized = False):
        
        # init as 1D kernel 
        super().__init__()
        self.covariates = covariates
        
        # gpflow parameters
        self.sf = gpflow.Parameter(sf)
        
        # TODO : select some sparsity prior 
        # For now, use np.eye since it seems to work
        if not randomized:
            self.L = gpflow.Parameter(np.eye(self.covariates))
        else:
            self.L = gpflow.Parameter(2*init_factor*np.random.random(self.covariates)-init_factor)
        
        
    def K_diag(self, X):
        return self.sf**2 * tf.ones_like(X[:,0])
    
    def K(self, X1, X2=None):
        if X2 is None:
            X2 = X1
        
        A = X1 @ self.L
        B = X2 @ self.L

    
        # batched products (N_,1,D) (D,D) (N_,D,1)
        X11 = tf.squeeze(tf.expand_dims(A, axis = 1) @ tf.expand_dims(A, axis = -1), axis = -1)
        X22 = tf.transpose(tf.squeeze(tf.expand_dims(B, axis = 1) @ tf.expand_dims(B, axis = -1), axis = -1))  # (1,N2)

        # regular product -2 (N1,D) (D,D) (D,N2)
        X12 = A @ tf.transpose(B) # (N1,N2)

        # kernel  (N1,1) - (N1,N2) + (1,N2)
        K = self.sf**2 * tf.exp(-0.5 * (X11 - 2*X12 + X22))

        return K
    
class FullGaussianKernelCholesky(gpflow.kernels.Kernel):
    """
    This kernel calculates the 'full' Gaussian kernel defined in ...
    It is used for learning the off-diagonal covariates of the precion matrix.
    """
    
    def __init__(self, covariates, sf = 1, init_factor = 1, randomized = False):
        
        # init as 1D kernel 
        super().__init__()
        self.covariates = covariates
        
        # gpflow parameters
        self.sf = gpflow.Parameter(sf)
        
        # TODO : select some sparsity prior 
        # For now, use np.eye since it seems to work
        if not randomized:
          self.L = gpflow.Parameter(init_factor*np.ones(self.covariates*(self.covariates +1) // 2))
        else:
          self.L = gpflow.Parameter(2*init_factor*np.random.random(self.covariates*(self.covariates +1) // 2)-init_factor) 
          #self.L = gpflow.Parameter(np.array([1,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0]))
        
        
    def K_diag(self, X):
        return self.sf**2 * tf.ones_like(X[:,0])
    
    def K(self, X1, X2=None):
        if X2 is None:
            X2 = X1
        
        L_as_matrix = tfp.math.fill_triangular(self.L)
        A = X1 @ L_as_matrix
        B = X2 @ L_as_matrix 

    
        # batched products (N_,1,D) (D,D) (N_,D,1)
        X11 = tf.squeeze(tf.expand_dims(A, axis = 1) @ tf.expand_dims(A, axis = -1), axis = -1)
        X22 = tf.transpose(tf.squeeze(tf.expand_dims(B, axis = 1) @ tf.expand_dims(B, axis = -1), axis = -1))  # (1,N2)

        # regular product -2 (N1,D) (D,D) (D,N2)
        X12 = A @ tf.transpose(B) # (N1,N2)

        # kernel  (N1,1) - (N1,N2) + (1,N2)
        K = self.sf**2*tf.exp(-0.5 * (X11 - 2*X12 + X22))

        return K

class ARD(gpflow.kernels.Kernel):
    """
    ARD kernel
    """
    
    def __init__(self, covariates, sf = 1, init_factor = 1, randomized = False):        
        # init as 1D kernel 
        super().__init__()
        self.covariates = covariates
        
        # gpflow parameters
        self.sf = gpflow.Parameter(sf)
        
        # TODO : select some sparsity prior 
        # For now, use np.eye since it seems to work
        self.ell = gpflow.Parameter(np.ones(self.covariates))
        if not randomized:
            self.ell = gpflow.Parameter(init_factor*np.ones(self.covariates))
        else:
            self.ell = gpflow.Parameter(init_factor*np.random.random(self.covariates))
        
        
    def K_diag(self, X):
        return self.sf**2 * tf.ones_like(X[:,0])
    
    def K(self, X1, X2=None):
        if X2 is None:
            X2 = X1
            
        P = tf.linalg.diag(self.ell**(-2))    
        # batched products (N_,1,D) (D,D) (N_,D,1)
        X11 = tf.squeeze(tf.expand_dims(X1,axis = 1) @ P @ tf.expand_dims(X1,axis = -1),-1)  # (N1,1)
        X22 = tf.transpose(tf.squeeze(tf.expand_dims(X2,axis = 1) @ P @ tf.expand_dims(X2,axis = -1),-1))  # (1,N2)

        # regular product -2 (N1,D) (D,D) (D,N2)
        X12 = X1 @ P @ tf.transpose(X2) # (N1,N2)

        # kernel  (N1,1) - (N1,N2) + (1,N2)
        K = self.sf**2 * tf.exp(-0.5 * (X11 - 2*X12 + X22))

        return K