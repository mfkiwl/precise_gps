import cmdstanpy
import pickle, os, time

from src.datasets.datasets import *
from src.select import *

def run_stan():
    datasets_names = ['Boston', 'Concrete', 'Yacht', 'Energy']
    stan_model = 'test2'


    results = {}
    for name in datasets_names:
        
        start = time.time()
        
        results[name] = {}
        dataset = select_dataset(name, 0.2)
        
        data = {'N': len(dataset.train_y), 'D':dataset.train_X.shape[1], 
                'x':dataset.train_X, 'y':dataset.train_y.flatten(), 
                'N_test':len(dataset.test_y), 'x_test':dataset.test_X}

        path = f'stan_models/{stan_model}.stan'
        model = cmdstanpy.CmdStanModel(stan_file = path)
        samples = model.sample(data, chains = 4, iter_warmup = 200, 
                            iter_sampling = 100)
        
        y_pred = samples.stan_variable('y_pred')
        precisions = samples.stan_variable('precision')
        lp = samples.sampler_variables()['lp__']
        r_hat = samples.summary()['R_hat']
        
        results[name]['y_pred'] = y_pred
        results[name]['precisions'] = precisions
        results[name]['lp__'] = lp
        results[name]['R_hat'] = r_hat
        
        end = time.time()
        print(f'Time taken: {end-start}')
        
    save_path = f'results/raw/stan/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with open(f'{save_path}/{stan_model}.pkl', 'wb') as save:
        pickle.dump(results, save)

if __name__ == '__main__':
    run_stan()
        
        
        