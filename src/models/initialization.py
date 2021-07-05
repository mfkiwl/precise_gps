import numpy as np
from scipy.cluster.vq import kmeans2
import tensorflow_probability as tfp 
import tensorflow as tf


def init_precision(dim) -> tf.Tensor:
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

def select_inducing_points(X, k) -> np.array:
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

def init_lowrank_precision(dim, rank) -> tf.Tensor:
    """
    Initializes full gaussian kernel with random precision

    Args:
        dim (int) : dimension of the precision matrix (dim x dim)

    Returns:
        Cholesky decomposition of the precision in vector format
    """
    full_L = np.random.uniform(-1,1,(dim,dim))
    P = full_L@np.transpose(full_L)

    lowrank_L = tfp.math.pivoted_cholesky(P, rank)
    return fill_lowrank_triangular_inverse(lowrank_L)

def fill_lowrank_triangular(vect, dim) -> tf.Tensor:
    """
    Create lowrank matrix from vector 

    Args:
        vect (tensor) : tensor of float (parameters of a lowrank full Gaussian kernel)
        dim (int)     : dimension of the inputs (higher value)
    
    Returns:
        matrix M that is shaped dim x len(vect) / dim
    """
    length = len(vect)
    if length % dim != 0:
        raise ValueError("Dimension mismatch!")
    
    lowrank_matrix = vect.reshape(dim, int(len(vect) / dim))
    return lowrank_matrix

def fill_lowrank_triangular_inverse(L) -> tf.Tensor:
    """
    Transform lowrank matrix into vector format

    Args:
        M (tensor) : lowrank matrix 
    
    Returns:
        M in vector format 
    """
    return tf.reshape(L, -1)