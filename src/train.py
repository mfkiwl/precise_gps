from src.models.models import *
from src.models.kernels import *
from src.parse_results import * 
from src.visuals.visuals import *
from src.models.initialization import *
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import numpy as np
import gpflow 
import tensorflow as tf 
import tensorflow_probability as tfp 
from src.select import select_kernel, select_model
from src.save_intermediate import save_results
from scipy.stats import norm 

def run_adam_and_natgrad(model, iterations, train_dataset, minibatch_size, params, l, counter, variances, likelihood_variances, mlls, N, q_mus, q_sqrts, Zs):
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
    optimizer = tf.optimizers.Adam(learning_rate = 0.01) # using default learning rate 
    natgrad_optimizer = gpflow.optimizers.NaturalGradient(gamma=0.1)

    @tf.function()
    def optimization_step():
        optimizer.minimize(training_loss, model.trainable_variables)
        natgrad_optimizer.minimize(training_loss, model.variational_params)
        
    for step in range(iterations):
        optimization_step()
        if step % 50 == 0:
            save_results(model, step, params, counter, variances, likelihood_variances, mlls, l, q_mus, q_sqrts, Zs)


def train(model, kernel, data, lassos, max_iter, num_runs, randomized, num_Z, minibatch_size, batch_iter, rank, penalty, n, V) -> dict:
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
        rank (int)                   : rank of the precision matrix
        penalty (string)             : name of the penalty used (src.models.penalty)
        n (int)                      : wishart degrees of freedom
        V (list)                     : wishart process V matrix
    
    Returns:
        Saves a dictionary into "results/raw/<instance name>.pkl where <instance name> is specified in the
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
    
    if model == "SGHMC":
        tf.compat.v1.disable_eager_execution()

    dim = len(data.cols) # number of features for inputs
    
    #Initialize dataframe and dataframe structure
    df = {}
    test_errors, train_errors, mlls, params, sghmc_params = {}, {}, {}, {}, {}
    likelihood_variances, variances, log_likelihoods = {}, {}, {}
    sghmc_vars = {}
    
    q_mus = {}
    q_sqrts = {}
    Zs = {}

    # Iterating through lassos or n:s
    for l in lassos if penalty == "lasso" else n:
        test_errors[l], train_errors[l], mlls[l], params[l], sghmc_params[l] = [], [], {}, {}, []
        likelihood_variances[l], variances[l], log_likelihoods[l] = {}, {}, []
        sghmc_vars[l] = []
        q_mus[l] = []
        q_sqrts[l] = [] 
        Zs[l] = [] 
        

        #kf = KFold(n_splits=5, shuffle=True)
        for num_run in range(num_runs):
            print(f"Starting run: {num_run+1} / {num_runs}")
            mlls[l][num_run] = []
            params[l][num_run] = []
            likelihood_variances[l][num_run] = []
            variances[l][num_run] = []
            # Cross validation
            #train_index, valid_index = next(kf.split(data.train_y))
            #train_X, train_y = data.train_X[train_index], data.train_y[train_index]
            #valid_X, valid_y = data.train_X[valid_index], data.train_y[valid_index]
            
            # Initializing kernel and model
            kernel_kwargs = {"randomized": randomized, "dim": dim, "rank": rank}
            _kernel = select_kernel(kernel, **kernel_kwargs)
            if model != "SGHMC":
                model_kwargs = {"data": (data.train_X, data.train_y), "kernel": _kernel, "lasso": l, "M": num_Z, "horseshoe": l, "n": l, "V": V}
            else:
                model_kwargs = {"data": data, "kernel": _kernel, "lasso": l, "M": num_Z, "horseshoe": l, "n": l, "V": V}
            if V is not None:
                model_kwargs["V"] = V
            _model = select_model(model, **model_kwargs)
            # Optimizing either using Scipy or Adam
            def step_callback(step, variables, values):
                if step % 5 == 0:
                    save_results(_model, step, params, num_run, variances, likelihood_variances, mlls, l, q_mus, q_sqrts, Zs)
            
            if type(_model) == SVIPenalty:
                train_dataset = tf.data.Dataset.from_tensor_slices((data.train_X, data.train_y)).repeat().shuffle(len(data.train_y))
                run_adam_and_natgrad(_model,batch_iter,train_dataset,minibatch_size,params,l,num_run,variances,likelihood_variances,mlls, len(data.train_y), q_mus, q_sqrts, Zs)
            elif type(_model) == GPRPenalty:
                optimizer = gpflow.optimizers.Scipy()
                optimizer.minimize(
                    _model.training_loss, _model.trainable_variables, options={'maxiter': max_iter,'disp': False}, step_callback = step_callback)
            else:
                _model.fit(data.train_X, data.train_y)

            # Calculating error and log-likelihood
            if type(_model).__name__ == "SGHMC":
                pred_mean, pred_var, opt_params, opt_variances = _model.predict(data.test_X)
                sghmc_params[l].append(opt_params)
                sghmc_vars[l].append(opt_variances)
                pred_train,_,_,_ = _model.predict(data.train_X)
                rms_test = mean_squared_error(data.test_y, pred_mean, squared=False)
                rms_train = mean_squared_error(data.train_y, pred_train, squared=False)
                log_lik = np.average(_model.calculate_density(data.test_X, data.test_y))
            else:
                pred_mean, pred_var = _model.predict_y(data.test_X)
                pred_train,_ = _model.predict_y(data.train_X)
                rms_test = mean_squared_error(data.test_y, pred_mean.numpy(), squared=False)
                rms_train = mean_squared_error(data.train_y, pred_train.numpy(), squared=False)
                log_lik = np.average(norm.logpdf(data.test_y*data.y_std, loc=pred_mean*data.y_std, scale=pred_var**0.5*data.y_std))

            test_errors[l].append(rms_test)
            train_errors[l].append(rms_train)
            log_likelihoods[l].append(log_lik)
            

        current_mean = np.mean(test_errors[l])
        #train_mean = np.mean(train_errors[l])
        current_ll = np.mean(log_likelihoods[l])
        print("Lasso:", l, "LL:", current_ll, "Test error", current_mean)
        
    # Save results after running the experiments
    df["data_train"] = (data.train_X, data.train_y)
    df["data_test"] = (data.test_X, data.test_y)
    df["model"] = type(_model).__name__
    df["kernel"] = type(_kernel).__name__
    df["dataset"] = type(data).__name__
    df["num_Z"] = num_Z
    df["num_runs"] = num_runs
    df["lassos"] = lassos
    df["test_errors"] = test_errors
    df["train_errors"] = train_errors
    df["mll"] = mlls 
    df["params"] = params 
    df["likelihood_variances"] = likelihood_variances
    df["variances"] = variances
    df["log_likelihoods"] = log_likelihoods
    df["rank"] = rank
    df["cols"] = data.cols 
    df["n"] = n
    df["V"] = _model.V 
    df["penalty"] = penalty
    df["sghmc_vars"] = sghmc_vars
    df["sghmc_params"] = sghmc_params
    df["q_mu"] = q_mus
    df["q_sqrt"] = q_sqrts
    df["Z"] = Zs
    return df  

