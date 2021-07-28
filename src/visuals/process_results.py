import numpy as np
import tensorflow as tf 
import tensorflow_probability as tfp
from sklearn.decomposition import PCA
from src.select import select_model, select_kernel
from src.models.initialization import fill_lowrank_triangular

def sub_kernel(kernel, dim1, dim2):
    """
    Constructs a sub-kernel of a kernel.

    Args:
        kernel (tensor) : kernel matrix
        dim1 (tuple)    : start, end
        dim2 (tuple)    : start, end
    """

    sub_kernel = kernel[dim1[0]:dim1[1],dim2[0]:dim2[1]]
    return sub_kernel

def pca_to_params(list_of_params, gradient):
    """
    Create two new features with PCA that are used to visualize the loss-landscape.

    Args:
        list_of_params (list) : list of parameters
        gradient (bool)       : whether PCA is calculted through difference (True) or just through parameters (False) 
    """
    num_of_params = len(list_of_params[0])
    num_of_rows = len(list_of_params)
    
    last_element = list_of_params[-1]

    M = np.zeros((num_of_rows, num_of_params))
    if gradient:
        for idx, param in enumerate(list_of_params[:-1]):
            M[idx] = param - last_element
    else:
        for idx, param in enumerate(list_of_params):
            M[idx] = param
            
    _pca = PCA(n_components = 2) # Choose number of components
    _pca.fit(M)
    return M, _pca.components_, _pca.explained_variance_ratio_, _pca

def transform_M(pca, M):
    """
    Transform values to the new feature space.

    Args:
        pca (sklearn.decomposition.PCA) : fitted PCA object
        M (numpy array)                 : parameters or parameter differences
    
    Returns:
        Transformed values
    """
    return pca.transform(M)

def loss_landscape(model, kernel, lasso, num_Z, data, params, variances, log_variances, directions, alphas, betas):
    kernel_kwargs = {"randomized": True, "dim": data[0].shape[1]}
    _kernel = select_kernel(kernel, **kernel_kwargs)
    
    model_kwargs = {"data": data, "kernel": _kernel, "lasso": lasso, "M": num_Z}
    _model = select_model(model, **model_kwargs)
    
    center_params = params[-1]
    center_var = variances[-1]
    center_logvar = log_variances[-1]
    
    losses = np.zeros((len(alphas), len(betas)))
    
    for idx_alpha, alpha in enumerate(alphas):
        for idx_beta, beta in enumerate(betas):
            if kernel == "ARD" or kernel == "ARD_gpflow":
                _model.kernel.lengthscales = center_params + alpha*directions[0] + beta*directions[1]
            else:
                _model.kernel.L = center_params + alpha*directions[0] + beta*directions[1]
            
            _model.kernel.variance = center_var
            _model.likelihood.variance = center_logvar
            if model == "SVIPenalty":
                loss = -_model.maximum_log_likelihood_objective(data)
            else:
                loss = -_model.maximum_log_likelihood_objective()
            losses[idx_alpha, idx_beta] = loss
            
    return losses

def eigen(M):
    """
    Calculate eigen values of a matrix

    Args:
        M (tensor) : matrix
    
    Returns:
        eigen values and vectors
    """
    values, vectors = np.linalg.eig(M)
    return values, vectors

def average_frobenius(kernels, num_runs): 
    """
    Returns the mean frobenius norm of a dictionary of kernels.

    Args:
        kernels (dict) : A dictionary of parameters
        num_runs (int) : number of random iterations
    
    Returns:
        mean of the norm (float)
    """
    norms = []
    for i in range(num_runs): 
        norms.append(tf.norm(kernels[i], 'euclidean'))
    
    return np.mean(norms)

def params_to_precision(kernel):
    """
    Transform parameters (L or lengthscales) to precision matrix

    Args:
        params (tensor) : either L or lengthscales
        kernl (string)  : name of the kernel
    
    Returns:
        precision matrix
    """

    return kernel.precision()

def params_to_precision_vis(params, kernel, dim, length):
    """
    Transform parameters (L or lengthscales) to precision matrix

    Args:
        params (tensor) : either L or lengthscales
        kernl (string)  : name of the kernel

    Returns:
        precision matrix
    """

    if kernel == "ARD":
        P = tf.linalg.diag(params**(2))
        return P

    if kernel == "ARD_gpflow":
        P = tf.linalg.diag(params**(-2))
        return P

    if kernel == "FullGaussianKernel":
        L = tfp.math.fill_triangular(params)
        P = L@tf.transpose(L)
        return P

    if kernel == "LowRankFullGaussianKernel":
        L = fill_lowrank_triangular(params, dim, length)
        return tf.transpose(L)@L

def eigen_count_mean(values, treshhold = 0.001):
    """
    Calculates number of eigenvalues above a certain threshold

    Args:
        values (list) : list of the eigenvalues
        threshhold ()
    """
    pass

def best_coef(log_liks):
    best_keys = []
    for log_lik in log_liks:
        max_ll = -np.inf
        for key in log_lik.keys():
            current_ll = np.mean(log_lik[key])
            if max_ll < current_ll:
                max_ll = current_ll
                best_key = key
        best_keys.append(best_key)
    return best_keys
    