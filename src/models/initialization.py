import numpy as np
from scipy.cluster.vq import kmeans2
import tensorflow_probability as tfp 


def init_precision(dim):
    """
    Initializes full gaussian kernel with random precision

    Args:
        dim (int) : dimension of the precision matrix (dim x dim)

    Returns:
        Cholesky decomposition of the precision in vector format
    """
    full_L = np.random.uniform(-1,1,(dim,dim))
    P = full_L@np.transpose(full_L)

    lower_L = np.linalg.cholesky(P)
    return tfp.math.fill_triangular_inverse(lower_L)

def select_inducing_points(X, k):
    """
    Select inducing points for SVI using k-means clustering.

    Args:
        X (tensor) : training set of the inputs
        k (int)    : number of inducing points
    
    Returns:
        Tensor of inducing points
    """
    try:
        _k = kmeans2(X, k, minit='points', missing="raise")[0]
    except:
        return select_inducing_points(X, k)
    
    return _k

def init_lowrank_precision(dim):
    """
    Initializes full gaussian kernel with random precision

    Args:
        dim (int) : dimension of the precision matrix (dim x dim)

    Returns:
        Cholesky decomposition of the precision in vector format
    """
    full_L = np.random.uniform(-1,1,(dim,dim))
    P = full_L@np.transpose(full_L)

    lowrank_L = tfp.math.pivoted_cholesky(P)
    return tfp.math.fill_lowrank_triangular_inverse(lowrank_L)

# TODO:
def fill_lowrank_triangular(n, d):
    pass

def fill_lowrank_triangular_inverse(L):
    pass