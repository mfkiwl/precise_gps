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
    
    _k = kmeans2(X, k, minit='points')[0]
    if _k.shape[0] != k:
        return select_inducing_points(X, k)
    else:
        return _k


