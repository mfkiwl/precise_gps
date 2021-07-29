from src.models.models import *
from src.models.kernels import *

def save_results(model, step, params, counter, variances, likelihood_variances, mlls, l, q_mus, q_sqrts, Zs):
    """
    Save intermediate results of the optimization.

    Args:
        see: src.train.train
    """
    if type(model).__name__ == 'SVIPenalty':
        value = model.maximum_log_likelihood_objective(model.train_data)
        q_mus[l].append(model.q_mu)
        q_sqrts[l].append(model.q_sqrt)
        Zs[l].append(model.inducing_variable.Z)
    else:
        value = model.maximum_log_likelihood_objective()
        
    if type(model.kernel).__name__ in ["FullGaussianKernel", "LowRankFullGaussianKernel"]: 
        L = model.kernel.L.numpy()
        params[l][counter].append(list(L))
    else:
        lengthscales = model.kernel.lengthscales.numpy()
        params[l][counter].append(list(lengthscales))
    
    lik_var = model.likelihood.variance.numpy()
    var = model.kernel.variance.numpy()
    variances[l][counter].append(var)
    likelihood_variances[l][counter].append(lik_var)
    mlls[l][counter].append(value)

    if step % 100 == 0:
        print("Lasso", l, "Step:", step, "MLL:", value)
