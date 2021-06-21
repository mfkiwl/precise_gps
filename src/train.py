from src.models import *
from src.kernels import *
from src.utils import * 
from src.visuals import *
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np
import gpflow 
import tensorflow as tf 
import tensorflow_probability as tfp 


possible_models = ["GPR", "GPRLasso", "SVILasso"] # current possible models to train
possible_kernels = ["full", "own_ard", "gpflow_ard"] # current possible kernels to use

def train(model, kernel, data, lassos, max_iter, num_runs, randomized, show):
    """
    
    """

    if model not in possible_models:
        print(f"Model {model} is not part of the models. Changed to gpflow.models.GPR!")
        model = "GPR"
    if kernel not in possible_kernels:
        print(f"Model {kernel} is not part of the kernels. Changed to gpflow.SquaredExponential!")
        kernel = "gpflow_ard"
    
    if model == "GPR":
        lassos = [0]

    train_Xnp = data["train_X"]
    train_ynp = data["train_y"]
    test_Xnp = data["test_X"]
    test_ynp = data["test_y"]
    cols = data["cols"]
    dim = len(cols)
    df = {}

    train_ynp = np.expand_dims(train_ynp,-1)
    test_ynp = np.expand_dims(test_ynp,-1)

    test_errors_full = []
    train_errors_full = []
    mlls = {}
    params = {}
    likelihood_variances = {}
    variances = {}
    log_likelihoods = {}
    for l in lassos:
        mlls[l] = {}
        params[l] = {}
        likelihood_variances[l] = {}
        variances[l] = {}
        log_likelihoods[l] = []
        errors = []
        train_errors = []
        counter = 0
        df[l] = []
        for _ in range(num_runs):
            mlls[l][counter] = []
            params[l][counter] = []
            likelihood_variances[l][counter] = []
            variances[l][counter] = []

            """
            Selecting the correct kernel TODO: remove if-else structure
            """
            if kernel == "full":
                if not randomized:
                    L = np.ones((dim*(dim+1))//2)
                else:
                    L = init_precision(dim)
                _kernel = FullGaussianKernel(variance=1, L=L)
            elif model == "own_ard":
                if not randomized:
                    lengthscales = np.ones(dim)
                else:
                    lengthscales = np.random.uniform(0.5,3,dim)
                _kernel = ARD(variance=1, lengthscales=lengthscales)
            else:
                if not randomized:
                    lengthscales = np.ones(dim)
                else:
                    lengthscales = np.random.uniform(0.5,3,dim)
                _kernel = gpflow.kernels.SquaredExponential(variance=1, lengthscales=lengthscales)
            """
            Selecting the correct model TODO: remove if-else structure
            """
            if model == "GPRLasso":
                gpr_model = GPRLasso((train_Xnp,train_ynp),_kernel,l)
            elif model == "SVILasso":
                gpr_model = SVILasso((train_Xnp, train_ynp), _kernel, l, 100)
            else:
                gpr_model = gpflow.models.GPR((train_Xnp, train_ynp), _kernel)
            
            """
            Optimizer
            """
            def save_results():
                if type(_kernel) == FullGaussianKernel:
                    L = gpr_model.kernel.L
                    params[l][counter].append(list(L))
                else:
                    P = tf.linalg.diag(gpr_model.kernel.lengthscales**(-2))
                    params[l][counter].append(list(P))

                value = gpr_model.maximum_log_likelihood_objective()
                lik_var = gpr_model.likelihood.variance
                var = gpr_model.kernel.variance
                variances[l][counter] = var
                likelihood_variances[l][counter] = lik_var
                mlls[l][counter].append(value)


            optimizer = gpflow.optimizers.Scipy()
            def step_callback(step, variables, values):
                if step % 100 == 0:
                    save_results()
            if model == "SVILasso":
                tensor_data = tuple(map(tf.convert_to_tensor, (train_Xnp, train_ynp)))
                training_loss = gpr_model.training_loss_closure((train_Xnp, train_ynp))
                optimizer.minimize(
                    training_loss, gpr_model.trainable_variables, options={'maxiter': max_iter,'disp': False}, step_callback = step_callback)
            else:
                optimizer.minimize(
                    gpr_model.training_loss, gpr_model.trainable_variables, options={'maxiter': max_iter,'disp': False}, step_callback = step_callback)


            save_results()
            # Calculating error
            pred_mean, _ = gpr_model.predict_f(test_Xnp)
            pred_train,_ = gpr_model.predict_f(train_Xnp)
            rms = mean_squared_error(test_ynp, pred_mean.numpy(), squared=False)
            rms_train = mean_squared_error(train_ynp, pred_train.numpy(), squared=False)
            errors.append(rms)
            train_errors.append(rms_train)
            counter += 1
            log_lik = tf.math.reduce_sum(gpr_model.predict_log_density(data = (test_Xnp, test_ynp)))
            log_likelihoods[l].append(log_lik)

            if show:
                if kernel == "full":
                    L = gpr_model.kernel.L
                    L = tfp.math.fill_triangular(L)
                    show_kernel(L @ tf.transpose(L), "Optimized precision matrix LL^T", cols, "", "center", 1)
                else:
                    P = tf.linalg.diag(gpr_model.kernel.lengthscales**(-2))
                    show_kernel(P, "Optimized precision matrix LL^T", cols, "", "center", 1)

        current_mean = np.mean(errors)
        train_mean = np.mean(train_errors)
        print("Train error:", train_mean, "Test error", current_mean)
        test_errors_full.append(current_mean)
        train_errors_full.append(train_mean)

    df["data"] = data
    df["lassos"] = lassos
    df["max_iter"] = max_iter 
    df["test_errors"] = test_errors_full
    df["train_errors"] = train_errors_full
    df["mll"] = mlls 
    df["params"] = parse_traceL(params, lassos, max_iter = max_iter)
    df["likelihood_variances"] = likelihood_variances
    df["variances"] = variances
    df["log_likelihoods"] = log_likelihoods
    return df 


