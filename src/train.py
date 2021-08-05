from src.models.models import *
from src.models.kernels import *
from src.parse_results import * 
from src.visuals.visuals import *
from src.models.initialization import *
from src.select import select_kernel, select_model
from src.save_intermediate import save_results
from src.sampling.mcmc_kernel import sample_posterior_params

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from scipy.stats import norm 
from scipy.stats import multinomial

import numpy as np
import tensorflow as tf 
import tensorflow_probability as tfp 
import gpflow 

# Learning rate for Adam and natural gradient
ADAM_LR, GAMMA = 0.01, 0.1

def run_adam_and_natgrad(model, iterations, train_dataset, minibatch_size,
                         params, coefficient, counter, variances, 
                         likelihood_variances, mlls, N, q_mus, q_sqrts, Zs):
    '''
    Function to run Adam optimizer, and natural gradient for variational
    parameters.

    Args:
        model (src.models.models) : instance of a model to be trained
        iterations (int) : number of iterations for Adam
        train_dataset (tf.data.Dataset) : values and targets
        minibatch_size (int) : SVI minibatch size
        params (dict) : precision matrix parameters
        coefficient (float) : current lasso coefficient or n (Wishart)
        counter (int) : used for printing intermediate results
        variances (dict) : kernel variances
        likelihood_variances (dict) : model likelihood varinces
        mlls (dict) : maximum log likelihood objectives
        N (int) : size of the dataset
        q_mus, q_sqrt (gpflow.Parameter, gpflow.Parameter) : variational
        parameters
        Zs (gpflow.Parameter) : inducing points
    '''
    if minibatch_size == -1:
        train_iter = iter(train_dataset.batch(N))
    else:
        train_iter = iter(train_dataset.batch(minibatch_size))

    training_loss = model.training_loss_closure(train_iter, compile=True)
    optimizer = tf.optimizers.Adam(learning_rate = ADAM_LR) 
    natgrad_optimizer = gpflow.optimizers.NaturalGradient(gamma=GAMMA)

    @tf.function()
    def optimization_step():
        optimizer.minimize(training_loss, model.trainable_variables)
        natgrad_optimizer.minimize(training_loss, model.variational_params)
        
    for step in range(iterations):
        optimization_step()
        if step % 50 == 0:
            save_results(model, step, params, counter, variances, 
                         likelihood_variances, mlls, coefficient, 
                         q_mus, q_sqrts, Zs)


def train(model, kernel, data, lassos, max_iter, num_runs, randomized, 
          num_Z, minibatch_size, batch_iter, rank, penalty, n, V) -> dict:
    '''
    Training different models and kernels, commands specified in a 
    json-file. 

    Args:
        model (src.models.models) : instance of a model to be trained
        kernel (src.models.kernels) : instance of a kernel to be trained
        data (src.datasets.datasets) : instance of a dataset
        lassos (list) : list of lasso coefficients
        max_iter (int) : number of iterations for Scipy
        num_runs (int) : number of initializations
        randomized (bool) : initialization is randomized if True
        show (bool) : plot intermediate results if True
        num_Z (int) : number of inducing points
        minibatch_size (int) : SVI minibatch size
        batch_iter (int) : number of iterations for Adam
        rank (int) : rank of the precision matrix
        penalty (string) : name of the penalty used (src.models.penalty)
        n (int) : wishart degrees of freedom
        V (list) : wishart process V matrix
    
    Returns:
        Saves a dictionary into 'results/raw/<instance name>.pkl where 
        <instance name> is specified in the json-file. 
        The saved dictionary has the following keys
        
        dictionary (each key has structure 
        dict[<coefficient>][<num_runs>] if key is dictionary): 
        
            data_train (tuple) : X, y
            data_test (tuple) : X, y
            model (str) : model that was used
            kernel (str) : kernel that was used 
            dataset (str) : dataset that was used
            lassos (list) : lasso coefficients
            rank (int) : lower rank dimensions (if lowrank is used) 
            test_errors (dict) : test errors
            train_errors (dict) : train errors
            mll (dict) : maximum log likelihood objectives
            params (dict) : precision matrix parameters
            likelihood_variances (dict) : likelihood variances
            variances (dict) : kernel variances
            log_likelihoods (dict) : log-likelihoods for test set
            num_Z (int) : number of inducing points used
            num_runs (int) : number of random initializations used
            rank (int) : rank of the precision matrix used
            cols (list) : list of the data features
            n (int) : degrees of freedom Wishart prior
            V (ndarray) : scale matrix for Wishart prior
            penalty (str) : prior that was used
            sghmc_vars (dict) : kernel variance for SGHMC
            sghmc_params (dict) : precision matrix parameters for SGHMC
            q_mu (dict) : variational parameter
            q_sqrt (dict) : variational parameter
            Z (dict) : optimized inducing points
            nlls (dict) : negative log-likelihood
    '''

    # Change default jitter level. This reduces the chance of Cholesky
    # decomposition errors.
    gpflow.config.set_default_jitter(0.001)
    
    # SGHMC is implemented using tensorflow 1.13.
    if model == 'SGHMC':
        tf.compat.v1.disable_eager_execution()

    dim = len(data.cols) # number of features for inputs
    
    results = {}
    
    test_errors, train_errors, mlls, params, sghmc_params = {}, {}, {}, {}, {}
    likelihood_variances, variances, log_likelihoods = {}, {}, {}
    sghmc_vars, nlls, q_mus, q_sqrts, Zs = {}, {}, {}, {}, {}
    
    sample_test_errors, sample_log_liks = {}, {}
    best_sample_test_errors = {}
    best_sample_log_liks = {}

    # Iterating through coefficients defined by the prior that is used:
    # Lasso, Wishart.
    for coefficient in lassos if penalty == 'lasso' else n:
        test_errors[coefficient], train_errors[coefficient] = [], []
        mlls[coefficient], params[coefficient], nlls[coefficient] = {}, {}, {}
        sghmc_params[coefficient], sghmc_vars[coefficient] = [], []
        likelihood_variances[coefficient], variances[coefficient] = {}, {}
        log_likelihoods[coefficient] = []
        q_mus[coefficient], q_sqrts[coefficient], Zs[coefficient] = [], [], []
        sample_test_errors[coefficient] = []
        sample_log_liks[coefficient] = []
        best_sample_test_errors[coefficient] = []
        best_sample_log_liks[coefficient] = []
        
        #kf = KFold(n_splits=5, shuffle=True)
        for num_run in range(num_runs):
            print(f'Starting run: {num_run+1} / {num_runs}')
            mlls[coefficient][num_run] = []
            params[coefficient][num_run] = []
            likelihood_variances[coefficient][num_run] = []
            variances[coefficient][num_run] = []
            
            # Cross validation
            #train_idx, valid_idx = next(kf.split(data.train_y))
            #train_X, train_y = data.train_X[train_idx], data.train_y[train_idx]
            #valid_X, valid_y = data.train_X[valid_idx], data.train_y[valid_idx]
            
            kernel_kwargs = {'randomized': randomized, 'dim': dim, 'rank': rank}
            _kernel = select_kernel(kernel, **kernel_kwargs)
            
            model_kwargs = {'data': data, 
                            'kernel': _kernel, 'lasso': coefficient, 
                            'M': num_Z, 'horseshoe': coefficient, 
                            'n': coefficient, 'V': V}
                
            _model = select_model(model, **model_kwargs)
            
            def step_callback(step, variables, values):
                if step % 5 == 0:
                    save_results(_model, step, params, num_run, variances, 
                                 likelihood_variances, mlls, coefficient, 
                                 q_mus, q_sqrts, Zs)
            
            if type(_model) == SVIPenalty:
                train_dataset = tf.data.Dataset.from_tensor_slices(
                    (data.train_X, 
                     data.train_y)).repeat().shuffle(len(data.train_y))
                
                run_adam_and_natgrad(_model,batch_iter, train_dataset,
                                     minibatch_size, params, coefficient,
                                     num_run,variances, likelihood_variances,
                                     mlls, len(data.train_y), 
                                     q_mus, q_sqrts, Zs)
                
            elif type(_model) == GPRPenalty:
                optimizer = gpflow.optimizers.Scipy()
                optimizer.minimize(
                    _model.training_loss, _model.trainable_variables, 
                    options={'maxiter': max_iter,'disp': False}, 
                    step_callback = step_callback)
            else:
                _model.fit(data.train_X, data.train_y)

            # Calculating error and log-likelihood
            if type(_model).__name__ == 'SGHMC':
                mean, _, opt_params, opt_variances = _model.predict(data.test_X)
                sghmc_params[coefficient].append(opt_params)
                sghmc_vars[coefficient].append(opt_variances)
                pred_train,_,_,_ = _model.predict(data.train_X)
                rms_test = mean_squared_error(data.test_y, mean, 
                                              squared=False)
                rms_train = mean_squared_error(data.train_y, pred_train, 
                                               squared=False)
                log_lik = np.average(_model.calculate_density(data.test_X, 
                                                              data.test_y))
                nlls[coefficient][num_run] = _model.nlls
            else:
                mean, var = _model.predict_y(data.test_X)
                pred_train,_ = _model.predict_y(data.train_X)
                
                if data.task == 'Regression':
                    rms_test = mean_squared_error(data.test_y, mean.numpy(), 
                                                squared=False)
                    rms_train = mean_squared_error(data.train_y, 
                                                   pred_train.numpy(), 
                                                   squared=False)
                    log_lik = np.average(norm.logpdf(data.test_y*data.y_std, 
                                                    loc=mean*data.y_std, 
                                                    scale=var**0.5*data.y_std))
                else:
                    def onehot(Y, K):
                        return np.eye(K)[Y.flatten().astype(int)].reshape(Y.shape[:-1]+(K,))
                    
                    Y_oh = onehot(data.test_y, 2)[None, :, :]  # 1, N_test, K
                    one_hot_mean =  np.concatenate([1 - mean, mean], 1)

                    # clip very large and small probs
                    eps = 1e-12
                    p = np.clip(one_hot_mean, eps, 1 - eps)
                    p = p / np.expand_dims(np.sum(p, -1), -1)

                    log_lik = multinomial.logpmf(Y_oh, n=1, p=p)
                    pred = np.argmax(p, axis=-1)

                    rms_test = np.average(np.array(pred == data.test_y.flatten()).astype(float))
                    rms_train = 0
                
            print('Intermediate,', 'Coef:', coefficient, 'LL:', log_lik,
              'Test error', rms_test)
            test_errors[coefficient].append(rms_test)
            train_errors[coefficient].append(rms_train)
            log_likelihoods[coefficient].append(log_lik)

            if type(_model).__name__ == 'GPRPenalty' and  \
                type(_model.kernel).__name__ != 'ARD':
                rms, loglik, best_loglik, best_rms = sample_posterior_params(
                    _model, data)
                print('Intermediate_sample,', 'Coef:', coefficient, 'LL:', 
                      loglik, 'Test error', rms, "Best-ll", best_loglik, 
                      "Best-rmse", best_rms)
                
                sample_test_errors[coefficient].append(rms)
                sample_log_liks[coefficient].append(loglik)
                
                best_sample_test_errors[coefficient].append(best_rms)
                best_sample_log_liks[coefficient].append(best_loglik)
                
            

        current_mean = np.mean(test_errors[coefficient])
        current_ll = np.mean(log_likelihoods[coefficient])
        print('Coef:', coefficient, 'LL:', current_ll,
              'Test error', current_mean)
        
    # Save results after running the experiments
    results['data_train'] = (data.train_X, data.train_y)
    results['data_test'] = (data.test_X, data.test_y)
    results['model'] = type(_model).__name__
    results['kernel'] = type(_kernel).__name__
    results['dataset'] = type(data).__name__
    results['num_Z'] = num_Z
    results['num_runs'] = num_runs
    results['lassos'] = lassos
    results['test_errors'] = test_errors
    results['train_errors'] = train_errors
    results['mll'] = mlls 
    results['params'] = params 
    results['likelihood_variances'] = likelihood_variances
    results['variances'] = variances
    results['log_likelihoods'] = log_likelihoods
    results['rank'] = rank
    results['cols'] = data.cols 
    results['n'] = n
    results['V'] = _model.V 
    results['penalty'] = penalty
    results['sghmc_vars'] = sghmc_vars
    results['sghmc_params'] = sghmc_params
    results['q_mu'] = q_mus
    results['q_sqrt'] = q_sqrts
    results['Z'] = Zs
    results['nlls'] = nlls
    results['sample_test_errors'] = sample_test_errors
    results['sample_log_lik'] = sample_log_liks
    results['best_sample_test_errors'] = best_sample_test_errors
    results['best_sample_log_lik'] = best_sample_log_liks
    return results  

