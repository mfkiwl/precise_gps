import numpy as np
import tensorflow as tf
import gpflow
import tensorflow_probability as tfp 
    
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
        self.sf = gpflow.Parameter(sf, transform = tfp.bijectors.Power(2))
        
        # TODO : select some sparsity prior 
        # For now, use np.eye since it seems to work
        if not randomized:
          self.L = gpflow.Parameter(init_factor*np.ones(self.covariates*(self.covariates +1) // 2))
        else:
          self.L = gpflow.Parameter(2*init_factor*np.random.random(self.covariates*(self.covariates +1) // 2)-init_factor) 
        
        
    def K_diag(self, X):
        return self.sf * tf.ones_like(X[:,0])
    
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
        K = self.sf * tf.exp(-0.5 * (X11 - 2*X12 + X22))

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
        self.sf = gpflow.Parameter(sf, transform = tfp.bijectors.Power(2))
        
        # TODO : select some sparsity prior 
        # For now, use np.eye since it seems to work
        if not randomized:
            self.lengthscales = gpflow.Parameter(init_factor*np.ones(self.covariates), transform = tfp.bijectors.Power(-2))
        else:
            self.lengthscales = gpflow.Parameter(init_factor*np.random.random(self.covariates), transform = tfp.bijectors.Power(-2))
        
        
    def K_diag(self, X):
        return self.sf * tf.ones_like(X[:,0])
    
    def K(self, X1, X2=None):
        if X2 is None:
            X2 = X1
            
        P = tf.linalg.diag(self.lengthscales)    
        # batched products (N_,1,D) (D,D) (N_,D,1)
        X11 = tf.squeeze(tf.expand_dims(X1,axis = 1) @ P @ tf.expand_dims(X1,axis = -1),-1)  # (N1,1)
        X22 = tf.transpose(tf.squeeze(tf.expand_dims(X2,axis = 1) @ P @ tf.expand_dims(X2,axis = -1),-1))  # (1,N2)

        # regular product -2 (N1,D) (D,D) (D,N2)
        X12 = X1 @ P @ tf.transpose(X2) # (N1,N2)

        # kernel  (N1,1) - (N1,N2) + (1,N2)
        K = self.sf * tf.exp(-0.5 * (X11 - 2*X12 + X22))

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
        self.sf = gpflow.Parameter(sf, transform = tfp.bijectors.Power(2))
        
        # TODO : select some sparsity prior 
        if not randomized:
            self.diagonal = gpflow.Parameter(init_factor*np.ones(self.covariates),transform=tfp.bijectors.Power(-2))
            self.off_diagonal = gpflow.Parameter(np.zeros(self.covariates*(self.covariates -1) // 2))
        else:
            self.diagonal = gpflow.Parameter(init_factor*np.random.random(self.covariates),transform=tfp.bijectors.Power(-2))
            self.off_diagonal = gpflow.Parameter(init_factor*np.random.random(self.covariates*(self.covariates -1) // 2)-2*init_factor)

        
        
    def K_diag(self, X):
        return self.sf * tf.ones_like(X[:,0])
    
    def K(self, X1, X2=None):
        if X2 is None:
            X2 = X1
        
        paddings = tf.constant([[1, 0,], [0, 1]])
        L_as_matrix = tfp.math.fill_triangular(self.off_diagonal)
        L_as_matrix = tf.pad(L_as_matrix, paddings, "CONSTANT")
        L_as_matrix = tf.linalg.set_diag(L_as_matrix, self.diagonal)


        #L_as_matrix = tfp.math.fill_triangular(self.L)
        A = X1 @ L_as_matrix
        B = X2 @ L_as_matrix 

    
        # batched products (N_,1,D) (D,D) (N_,D,1)
        X11 = tf.squeeze(tf.expand_dims(A, axis = 1) @ tf.expand_dims(A, axis = -1), axis = -1)
        X22 = tf.transpose(tf.squeeze(tf.expand_dims(B, axis = 1) @ tf.expand_dims(B, axis = -1), axis = -1))  # (1,N2)

        # regular product -2 (N1,D) (D,D) (D,N2)
        X12 = A @ tf.transpose(B) # (N1,N2)

        # kernel  (N1,1) - (N1,N2) + (1,N2)
        K = self.sf*tf.exp(-0.5 * (X11 - 2*X12 + X22))

        return K