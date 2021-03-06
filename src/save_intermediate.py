from src.models.models import *
from src.models.kernels import *

_full_kernels = ['FullGaussianKernel', 'LowRankFullGaussianKernel']

def save_results(model, step, params, counter, variances, likelihood_variances, 
                 mlls, coefficient, q_mus, q_sqrts, Zs, train_iter):
    '''
    Save intermediate results during the optimizations process. Used for
    standard Gaussian process regresssion and stochastic variational 
    inference.

    Args:
        model (src.models.models) : instance of a model to be trained
        step (int) : current iteration step
        params (dict) : precision matrix parameters
        coefficient (float) : current lasso coefficient or n (Wishart)
        counter (int) : used for printing intermediate results
        variances (dict) : kernel variances
        likelihood_variances (dict) : model likelihood varinces
        mlls (dict) : maximum log likelihood objectives
        q_mus, q_sqrt (gpflow.Parameter, gpflow.Parameter) : variational
        parameters
        Zs (gpflow.Parameter) : inducing points
    '''
    if type(model).__name__ == 'SVIPenalty':
        value = model.maximum_log_likelihood_objective(train_iter)
        #q_mus[coefficient].append(model.q_mu)
        #q_sqrts[coefficient].append(model.q_sqrt)
        #Zs[coefficient].append(model.inducing_variable.Z)
    else:
        value = model.maximum_log_likelihood_objective()
        
    if type(model.kernel).__name__ in _full_kernels: 
        kernel_params = model.kernel.L.numpy()
        params[coefficient][counter].append(list(kernel_params))
    else:
        lengthscales = model.kernel.lengthscales.numpy()
        params[coefficient][counter].append(list(lengthscales))
    
    var = model.kernel.variance.numpy()
    variances[coefficient][counter].append(var)
    mlls[coefficient][counter].append(value)
    
    if type(model.likelihood).__name__ == 'Gaussian':
        lik_var = model.likelihood.variance.numpy()
        likelihood_variances[coefficient][counter].append(lik_var)

    if step % 100 == 0:
        print('Lasso', coefficient, 'Step:', step, 'MLL:', value)
