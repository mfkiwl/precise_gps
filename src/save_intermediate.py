from src.models.models import *
from src.models.kernels import *
import tensorflow as tf

def save_results(model, step, params, counter, variances, likelihood_variances, mlls, l):
    print(type(model).__name__, type(model).__name__ == 'SVILasso')
    if type(model).__name__ == 'SVILasso':
        value = model.maximum_log_likelihood_objective(model.train_data)
    else:
        value = model.maximum_log_likelihood_objective()
    value = model.maximum_log_likelihood_objective()
    if step % 100 == 0:
        if type(model.kernel).__name__ == "FullGaussianKernel":
            L = model.kernel.L.numpy()
            params[l][counter].append(list(L))
    else:
        P = tf.linalg.diag(model.kernel.lengthscales.numpy()**(-2))
        params[l][counter].append(list(P))

    print("Step:", step, "MLL:", value)

    lik_var = model.likelihood.variance
    var = model.kernel.variance
    variances[l][counter] = var
    likelihood_variances[l][counter] = lik_var
    mlls[l][counter].append(value)