import numpy as np
from sklearn.decomposition import PCA
from src.select import select_model, select_kernel


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
    return M, _pca.components_, _pca.explained_variance_, _pca

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

def loss_landscape(model, kernel, lasso, data, params, variances, log_variances, directions, alphas, betas):
    kernel_kwargs = {"randomized": True, "dim": data[0].shape[1]}
    _kernel = select_kernel(kernel, **kernel_kwargs)
    
    model_kwargs = {"data": data, "kernel": _kernel, "lasso": lasso}
    model = select_model(model, **model_kwargs)
    
    center_params = params[-1]
    center_var = variances[-1]
    center_logvar = log_variances[-1]
    
    losses = np.zeros((len(alphas), len(betas)))
    
    for idx_alpha, alpha in enumerate(alphas):
        for idx_beta, beta in enumerate(betas):
            model.kernel.L = center_params + alpha*directions[0] + beta*directions[1]
            model.kernel.variance = center_var
            model.likelihood.variance = center_logvar
            
            loss = -model.maximum_log_likelihood_objective()
            losses[idx_alpha, idx_beta] = loss
            
    return losses

def eigen(M):
    values, vectors = np.linalg.eig(M)
    return values, vectors