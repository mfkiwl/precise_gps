from src.models.models import *
from src.models.kernels import *
from src.parse_results import * 
from src.visuals.visuals import *
from src.models.initialization import *
from sklearn.metrics import mean_squared_error
import numpy as np
import gpflow 
import tensorflow as tf 
import tensorflow_probability as tfp 
from src.select import select_kernel, select_model
from src.save_intermediate import save_results

def run_adam(model, iterations, train_dataset, minibatch_size, params, l, counter, variances, likelihood_variances, mlls, N):
    """
    Utility function running the Adam optimizer

    Args:
        see function train() 
    """
    # Create an Adam Optimizer action
    if minibatch_size == -1:
        train_iter = iter(train_dataset.batch(N))
    else:
        train_iter = iter(train_dataset.batch(minibatch_size))

    training_loss = model.training_loss_closure(train_iter, compile=True)
    optimizer = tf.optimizers.Adam()

    @tf.function()
    def optimization_step():
        optimizer.minimize(training_loss, model.trainable_variables)

    for step in range(iterations):
        optimization_step()
        if step % 50 == 0:
            save_results(model, step, params, counter, variances, likelihood_variances, mlls, l)


def train(model, kernel, data, lassos, max_iter, num_runs, randomized, show, num_Z, minibatch_size, batch_iter, rank):
    """
    Training different models and kernels, commands specified by a json-file. 

    Args:
        model (src.models.models)    : instance of a model to be trained
        kernel (src.models.kernels)  : instance of a kernel to be trained
        data (src.datasets.datasets) : instance of a dataset
        lassos (list)                : lost of lasso coefficients
        max_iter (int)               : max number of iterations for Scipy
        num_runs (int)               : number of runs with same initialization
        randomized (bool)            : initialization is randomized if True
        show (bool)                  : plot intermediate results if True
        num_Z (int)                  : number of inducing points
        minibatch_size (int)         : SVI minibatch size
        batch_iter (int)             : number of iterations for Adam
    
    Returns:
        Saves a dictionary into "results/<instance name>.pkl where <instance name> is specified in the
        json-file. The saved dictionary has the following keys
        
        dictionary (each key has structure dict[<lasso>][<num_runs>] if key is dictionary): 
            data_train (tuple)          : X, y
            data_test (tuple)           : X, y
            model (string)              : model that was used
            kernel (string)             : kernel that was used 
            dataset (string)            : dataset that was used
            lassos (list)               : lasso coefficients
            rank (int)                  : lower rank dimensions (if lowrank is used) 
            test_errors (dict)          : test errors
            train_errors (dict)         : train errors
            mll (dict)                  : marginal log likelihoods 
            params (dict)               : kernel params every 10th iteration
            likelihood_variances (dict) : likelihood variances
            variances (dict)            : signal variances
            log_likelihoods (dict)      : log-likelihoods for test set
    """

    # Change default jitter level
    gpflow.config.set_default_jitter(0.001)

    # There is no lasso penalty in standard GPR
    if type(model).__name__ == "Standard_GPR":
        lassos = [0]

    dim = len(data.cols) # number of features for inputs
    
    #Initialize dataframe and dataframe structure
    df = {}
    test_errors, train_errors, mlls, params = {}, {}, {}, {}
    likelihood_variances, variances, log_likelihoods = {}, {}, {}

    for l in lassos:
        test_errors[l], train_errors[l], mlls[l], params[l] = [], [], {}, {}
        likelihood_variances[l], variances[l], log_likelihoods[l] = {}, {}, []

        for num_run in range(num_runs):
            print(f"Starting run: {num_run+1} / {num_runs}")
            mlls[l][num_run] = []
            params[l][num_run] = []
            likelihood_variances[l][num_run] = []
            variances[l][num_run] = []

            # Initializing kernel and model
            kernel_kwargs = {"randomized": randomized, "dim": dim, "rank": rank}
            _kernel = select_kernel(kernel, **kernel_kwargs)
            model_kwargs = {"data": (data.train_X, data.train_y), "kernel": _kernel, "lasso": l, "M": num_Z, "horseshoe": l}
            _model = select_model(model, **model_kwargs)

            # Optimizing either using Scipy or Adam
            def step_callback(step, variables, values):
                if step % 5 == 0:
                    save_results(_model, step, params, num_run, variances, likelihood_variances, mlls, l)
            
            if type(_model) == SVILasso:
                train_dataset = tf.data.Dataset.from_tensor_slices((data.train_X, data.train_y)).repeat().shuffle(len(data.train_y))
                gpflow.set_trainable(_model.inducing_variable, False) # Inducing variables not trainable
                run_adam(_model,batch_iter,train_dataset,minibatch_size,params,l,num_run,variances,likelihood_variances,mlls, len(data.train_y))
            else:
                optimizer = gpflow.optimizers.Scipy()
                optimizer.minimize(
                    _model.training_loss, _model.trainable_variables, options={'maxiter': max_iter,'disp': False}, step_callback = step_callback)

            # Calculating error and log-likelihood
            pred_mean, _ = _model.predict_f(data.test_X)
            pred_train,_ = _model.predict_f(data.train_X)

            rms_test = mean_squared_error(data.test_y, pred_mean.numpy(), squared=False)
            rms_train = mean_squared_error(data.train_y, pred_train.numpy(), squared=False)

            test_errors[l].append(rms_test)
            train_errors[l].append(rms_train)

            log_lik = tf.math.reduce_sum(_model.predict_log_density(data = (data.test_X, data.test_y)))
            log_likelihoods[l].append(log_lik)

            if show:
                if kernel == "FullGaussianKernel":
                    L = _model.kernel.L
                    L = tfp.math.fill_triangular(L)
                    show_kernel(L @ tf.transpose(L), "Optimized precision matrix P", data.cols, "", "center", 1)
                else:
                    P = tf.linalg.diag(_model.kernel.lengthscales**(-2))
                    show_kernel(P, "Optimized precision matrix P", data.cols, "", "center", 1)

        current_mean = np.mean(test_errors[l])
        train_mean = np.mean(train_errors[l])
        print("Lasso:", l, "Train error:", train_mean, "Test error", current_mean)

    # Save results after running the experiments
    df["data_train"] = (data.train_X, data.train_y)
    df["data_test"] = (data.test_X, data.test_y)
    df["model"] = type(_model).__name__
    df["kernel"] = type(_kernel).__name__
    df["dataset"] = type(data).__name__
    df["num_Z"] = num_Z
    df["lassos"] = lassos
    df["test_errors"] = test_errors
    df["train_errors"] = train_errors
    df["mll"] = mlls 
    df["params"] = params 
    df["likelihood_variances"] = likelihood_variances
    df["variances"] = variances
    df["log_likelihoods"] = log_likelihoods
    df["rank"] = rank
    return df 


