import cmdstanpy
import pickle, os, time, argparse

from src.datasets.datasets import *
from src.select import *

ap = argparse.ArgumentParser()
ap.add_argument('-o', '--optimize', required=True, 
                type = int, help='Wheter optimized or sampled.')
ap.add_argument('-m', '--model', required=True)
ap.add_argument('-s', '--samples', type = int, required=True)
ap.add_argument('-w', '--warmup', type = int, required=True)
ap.add_argument('-c', '--chains', type = int, required=True)
ap.add_argument('-d', '--dataset', required=True)
args = vars(ap.parse_args())

optimize = args['optimize']
model = args['model']
num_samples = args['samples']
warmup = args['warmup']
chains = args['chains']
dataset = args['dataset']

def run_stan(optimize, model, num_samples, warmup, chains, dataset):
    datasets_names = [dataset]
    stan_model = model

    results = {}
    for name in datasets_names:
        
        start = time.time()
        
        results[name] = {}
        dataset = select_dataset(name, 0.2)
        
        data = {'N': len(dataset.train_y), 'D':dataset.train_X.shape[1], 
                'x':dataset.train_X, 'y':dataset.train_y.flatten(), 
                'N_test':len(dataset.test_y), 'x_test':dataset.test_X, 
                'D_real_inv': 1/dataset.train_X.shape[1]}

        path = f'stan_models/{stan_model}.stan'
        model = cmdstanpy.CmdStanModel(stan_file = path)
        
        if optimize:
            print('Optimizing...')
            optim = model.optimize(data, algorithm = 'LBFGS', iter = 10_000)
            dictionary = optim.optimized_params_dict
            results[name]['dictionary'] = dictionary
        else:
            print('Sampling...')
            samples = model.sample(data, chains = chains, iter_warmup = warmup,
                                iter_sampling = num_samples)
            alpha = samples.stan_variable('alpha')
            sigma = samples.stan_variable('sigma')
            Ls = samples.stan_variable('transformed_L')
            lp = samples.sampler_variables()['lp__']
            r_hat = samples.summary()['R_hat']
            
            results[name]['sigma'] = sigma 
            results[name]['alpha'] = alpha
            results[name]['L'] = Ls
            results[name]['lp__'] = lp
            results[name]['R_hat'] = r_hat
            results[name]['num_samples'] = num_samples
            results[name]['warmup'] = warmup
        
        end = time.time()
        print(f'Time taken: {end-start}')
    
        if optimize:
            save_path = f'results/raw/stan/opt'
        else:
            save_path = f'results/raw/stan/{name}'
            
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        with open(f'{save_path}/{stan_model}.pkl', 'wb') as save:
            pickle.dump(results, save)
        print(f'Saved {name} dataset!')

if __name__ == '__main__':
    run_stan(optimize, model, num_samples, warmup, chains, dataset)
        
        
        