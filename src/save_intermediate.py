from src.models.models import *
from src.models.kernels import *
import tensorflow as tf

def save_results(model, step, params, counter, variances, likelihood_variances, mlls, l):
    """
    Save intermediate results of the optimization.

    Args:
        see: src.train.train
    """
    if type(model).__name__ == 'SVILasso':
        value = model.maximum_log_likelihood_objective(model.train_data)
    else:
        value = model.maximum_log_likelihood_objective()
    if step % 10 == 0:
        if type(model.kernel).__name__ == "FullGaussianKernel":
            L = model.kernel.L.numpy()
            params[l][counter].append(list(L))
        else:
            P = tf.linalg.diag(model.kernel.lengthscales.numpy()**(-2))
            params[l][counter].append(list(P))

    if step % 100 == 0:
        print("Lasso", l, "Step:", step, "MLL:", value)

    lik_var = model.likelihood.variance
    var = model.kernel.variance
    variances[l][counter] = var
    likelihood_variances[l][counter] = lik_var
    mlls[l][counter].append(value)