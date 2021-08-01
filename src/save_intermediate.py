from src.models.models import *
from src.models.kernels import *

_full_kernels = ['FullGaussianKernel', 'LowRankFullGaussianKernel']

def save_results(model, step, params, counter, variances, likelihood_variances, 
                 mlls, coefficient, q_mus, q_sqrts, Zs):
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
        value = model.maximum_log_likelihood_objective(model.train_data)
        q_mus[coefficient].append(model.q_mu)
        q_sqrts[coefficient].append(model.q_sqrt)
        Zs[coefficient].append(model.inducing_variable.Z)
    else:
        value = model.maximum_log_likelihood_objective()
        
    if type(model.kernel).__name__ in _full_kernels: 
        coefficient = model.kernel.coefficient.numpy()
        params[coefficient][counter].append(list(coefficient))
    else:
        lengthscales = model.kernel.lengthscales.numpy()
        params[coefficient][counter].append(list(lengthscales))
    
    lik_var = model.likelihood.variance.numpy()
    var = model.kernel.variance.numpy()
    variances[coefficient][counter].append(var)
    likelihood_variances[coefficient][counter].append(lik_var)
    mlls[coefficient][counter].append(value)

    if step % 100 == 0:
        print('Lasso', coefficient, 'Step:', step, 'MLL:', value)
