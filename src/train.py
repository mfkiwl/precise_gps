from src.models.models import *
from src.models.kernels import *
from src.parse_results import * 
from src.visuals.visuals import *
from src.models.initialization import *
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np
import gpflow 
import tensorflow as tf 
import tensorflow_probability as tfp 
from gpflow.ci_utils import ci_niter
from src.select import select_kernel, select_model
from src.save_intermediate import save_results

def run_adam(model, iterations, train_dataset, minibatch_size, lasso, train_Xnp, train_ynp, params, l, counter, variances, likelihood_variances, mlls, kernel, model_name):
    """
    Utility function running the Adam optimizer

    :param model: GPflow model
    :param interations: number of iterations
    """
    # Create an Adam Optimizer action
    train_iter = iter(train_dataset.batch(minibatch_size))
    training_loss = model.training_loss_closure(train_iter, compile=True)
    optimizer = tf.optimizers.Adam()

    @tf.function
    def optimization_step():
        optimizer.minimize(training_loss, model.trainable_variables)

    for step in range(iterations):
        optimization_step()
        save_results(model, step, params, counter, variances, likelihood_variances, mlls, l)


def train(model, kernel, data, lassos, max_iter, num_runs, randomized, show, num_Z, minibatch_size, batch_iter):
    """
    
    """

    # There is no lasso penalty in standard GPR
    if type(model).__name__ == "Standard_GPR":
        lassos = [0]

    train_Xnp = data.train_X
    train_ynp = data.train_y
    test_Xnp = data.test_X
    test_ynp = data.test_y
    cols = data.cols
    dim = len(cols)
    df = {}

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
        for num_run in range(num_runs):
            print(f"Starting run: {num_run}")
            mlls[l][counter] = []
            params[l][counter] = []
            likelihood_variances[l][counter] = []
            variances[l][counter] = []

            """
            Selecting the correct kernel TODO: remove if-else structure
            """
            kernel_kwargs = {"randomized": randomized, "dim": dim}
            _kernel = select_kernel(kernel, **kernel_kwargs)
            model_kwargs = {"data": (train_Xnp, train_ynp), "kernel": _kernel, "lasso": l, "M": num_Z, "horseshoe": l}
            gpr_model = select_model(model, **model_kwargs)

            """
            Optimizer
            """
            def step_callback(step, variables, values):
                save_results(gpr_model, step, params, counter, variances, likelihood_variances, mlls, l)
            if type(gpr_model) == SVILasso:
                train_dataset = tf.data.Dataset.from_tensor_slices((train_Xnp, train_ynp)).repeat().shuffle(len(train_ynp))
                #minibatch_size = minibatch_size
                run_adam(gpr_model,batch_iter,train_dataset,minibatch_size,l,train_Xnp,train_ynp,params,l,counter,variances,likelihood_variances,mlls,kernel,model)

            else:
                optimizer = gpflow.optimizers.Scipy()
                optimizer.minimize(
                    gpr_model.training_loss, gpr_model.trainable_variables, options={'maxiter': max_iter,'disp': False}, step_callback = step_callback)


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
        print("Lasso:", l, "Train error:", train_mean, "Test error", current_mean)
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


