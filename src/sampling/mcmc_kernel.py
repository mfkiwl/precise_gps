import gpflow 
import numpy as np
import tensorflow_probability as tfp
import tensorflow as tf

from tensorflow_probability import distributions as tfd
from gpflow.ci_utils import ci_niter
from gpflow import set_trainable
from sklearn.metrics import mean_squared_error
from scipy.stats import norm 

NUM_BURN_IN = ci_niter(300)
NUM_SAMPLES = ci_niter(50)

@tf.function
def run_chain_fn(hmc_helper, adaptive_hmc):
    return tfp.mcmc.sample_chain(
        num_results=NUM_SAMPLES,
        num_burnin_steps=NUM_BURN_IN,
        current_state=hmc_helper.current_state,
        kernel=adaptive_hmc,
        trace_fn=lambda _, pkr: pkr.inner_results.is_accepted,
    )

def sample_posterior_params(model, dataset):
    if type(model).__name__ == 'SVIPenalty':
        set_trainable(model.inducing_variable, False)
    
    f64 = gpflow.utilities.to_default_float
    model.kernel.L.prior = tfd.Normal(f64(0.0), f64(1.0))
    model.kernel.variance.prior = tfd.Normal(f64(0.0), f64(1.0))
    model.likelihood.variance.prior = tfd.Gamma(f64(1.0), f64(1.0))

    # Note that here we need model.trainable_parameters, not trainable_variables
    # - only parameters can have priors!
    
    hmc_helper = gpflow.optimizers.SamplingHelper(
        model.log_posterior_density, model.trainable_parameters
    )

    hmc = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=hmc_helper.target_log_prob_fn, num_leapfrog_steps=10, 
        step_size=0.01
    )
    adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(
        hmc, num_adaptation_steps=10, target_accept_prob=f64(0.75), 
        adaptation_rate=0.1
    )

    samples, _ = run_chain_fn(hmc_helper, adaptive_hmc)
    parameter_samples = hmc_helper.convert_to_constrained_values(samples)

    means = []
    vars = []
    
    best_loglik, best_rmse = -np.inf, np.inf
    for L, alpha, lik_var  in zip(parameter_samples[0], parameter_samples[1], 
                   parameter_samples[2]):
        new_model = gpflow.utilities.deepcopy(model)
        new_model.kernel.L = L
        new_model.kernel.variance = alpha
        new_model.likelihood.variance = lik_var
        mean, var = new_model.predict_y(dataset.test_X)
        
        rms_test = mean_squared_error(dataset.test_y, np.array(mean), 
                                  squared=False)
        log_lik = np.average(
            norm.logpdf(dataset.test_y*dataset.y_std, 
                        loc=np.array(mean)*dataset.y_std, 
                        scale=np.array(var)**0.5*dataset.y_std))

        if log_lik > best_loglik:
            best_loglik = log_lik
            best_rmse = rms_test
        
        means.append(mean.numpy())
        vars.append(var.numpy())

    bayesian_mean = np.mean(np.array(means), 0)
    bayesian_var = np.mean(np.array(vars), 0)
    
    rms_test = mean_squared_error(dataset.test_y, np.array(bayesian_mean), 
                                  squared=False)
    log_lik = np.average(
        norm.logpdf(dataset.test_y*dataset.y_std, 
                    loc=np.array(bayesian_mean)*dataset.y_std, 
                    scale=np.array(bayesian_var)**0.5*dataset.y_std))
    
    return rms_test, log_lik, best_loglik, best_rmse

