import os 
import tensorflow as tf
import tensorflow_probability as tfp
import pandas as pd 
from src.parse_results import parse_pickle
from src.visuals.visuals import * 
from src.datasets.datasets import *
from src.visuals.process_results import *
from src.tex.create_tables import * 
from src.select import select_dataset

#plt.rcParams.update({'font.size': 16})

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 22
BIG_SIZE = 14

plt.rc('font', size=BIG_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIG_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIG_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIG_SIZE)      # fontsize of the tick labels
plt.rc('ytick', labelsize=BIG_SIZE)      # fontsize of the tick labels
plt.rc('legend', fontsize=BIG_SIZE)      # legend fontsize
plt.rc('figure', titlesize=BIG_SIZE)  # fontsize of the figure title

def combined_results(datasets):
    '''
    Create visualizations form the raw .pkl files. This function also 
    forms some dataframes. 

    Args:
        dataset (string)   : path to the dataset folder
        directory (string) : dataset directory for analysis
        num_lassos (int)   : number of lasso coefficients (used for some 
                             visualizations)
        step (int)         : step between the lasso coefficients
        show (bool)        : wheter figures are shown during running the 
                             program 
    
    Returns:
        Saves visualizations and dataframes to results/processed
    '''
    combined_df = {}
    for dataset in datasets:
        #if dataset == 'Boston':
        #    data_path = f'results/raw/{dataset.lower()}_plots/{dataset.lower()}/'
        #else:
        data_path = f'results/raw/{dataset.lower()}/{dataset.lower()}/' 
        pkl_files = [file for file in os.listdir(data_path) 
                        if '.pkl' in file]

        df = {}
        for idx, current_file in enumerate(pkl_files):
            data = parse_pickle(data_path + current_file)
            df[idx] = data
        combined_df[dataset] = df
    
    # Each directory should only contain results from one dataset.

    result_path = 'results/processed/'
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    plot_path = result_path + '/plots'
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    results = {}
    for dataset in combined_df.keys():
        df = combined_df[dataset]
        names = []
        log_liks, test_errors = [], []
        num_runs = []
        all_mlls = []
        all_precisions = []
        for key in df.keys():
            data = df[key]
            model = data['model']
            kernel = data['kernel']
            penalty = data['penalty']
            rank = data['rank']
            number_of_initializations = data['num_runs']
            
            num_runs.append(number_of_initializations)

            # Simplify model and kernel names 
            kernel = 'ARD' if 'ARD' in kernel else 'FULL'
            kernel = kernel if 'Low' not in data['kernel'] else f'{kernel} {rank}'
            if 'GPR' in model:
                model = 'GPR'
            elif 'SVI' in model:
                model = 'SVI'
            else:
                model = 'SGHMC'
            penalty = 'W' if penalty == 'wishart' else 'L'

            if kernel == 'ARD':
                names.append(f'{model} {kernel}')
            else:
                names.append(f'{model} {kernel} {penalty}')
                    

            log_liks.append(data['log_likelihoods'])
            test_errors.append(data['test_errors'])
            if model != 'SGHMC':
                all_mlls.append(data['mll'])
            else:
                _nlls = data['nlls']
                new_nlls = {}
                for key1 in _nlls.keys():
                   new_nlls[key1] = {}
                   for key2 in _nlls[key1].keys():
                       new_nlls[key1][key2] = -1*np.array(_nlls[key1][key2])
                all_mlls.append(new_nlls)
            precisions = []
            for l in data['lassos'] if data['penalty'] == 'lasso' else data['n']:
                new_params = {}
                for i in range(data['num_runs']):
                    if model == 'SGHMC':
                        # new_params[i] = params_to_precision_vis(
                        #     np.array(data['sghmc_params'][l][i][-1][90]), 
                        #     data['kernel'], data['rank'], 
                        #     len(data['sghmc_params'][l][i][-1][90]))
                        new_params[i] = []
                    else:  
                        new_params[i] = params_to_precision_vis(
                            np.array(data['params'][l][i][-1]), 
                            data['kernel'], data['rank'], 
                            len(data['params'][l][i][-1]))
                        
                precisions.append(new_params) 
            all_precisions.append(precisions)
         
        data_instance = select_dataset(dataset, 0.2)
        
        _N = len(data_instance.test_y) + len(data_instance.train_y)
        _D = len(data_instance.cols)
        info = f'{dataset}, D={_D}, N={_N}'
        results[dataset] = {}
        results[dataset]['info'] = info
        results[dataset]['test_errors'] = test_errors
        results[dataset]['log_likelihoods'] = log_liks
        results[dataset]['names'] = names
        results[dataset]['mlls'] = all_mlls
        results[dataset]['num_runs'] = num_runs
        results[dataset]['precisions'] = all_precisions
        
        
        
        
    log_likelihood_mean(results, model = 'SVI', savefig=f'{plot_path}/log_liks_svi_yacht.pdf', fro = True)
    #combined_visualize_best(results, savefig=f'{plot_path}/lls.pdf')
    #combined_visualize_best(results, plot_log_lik = False, savefig=f'{plot_path}/mrmse.pdf')
    #log_likelihood_mean(results, model = 'GPR', savefig=f'{plot_path}/log_liks_gpr.pdf', fro = True)
    #log_likelihood_mean(results, model = 'SGHMC', savefig=f'{plot_path}/log_liks_sghmc.pdf', fro = True)
    #log_likelihood_against_mll(results, model='GPR', savefig=f'{plot_path}/ll_vs_mll_gpr.pdf')
    #log_likelihood_against_mll(results, model='SVI', savefig=f'{plot_path}/ll_vs_elbo_svi.pdf')
    #combined_visualize_mll(results, savefig=f'{plot_path}/mlls.pdf')
    #combined_visualize_best_rmse(results, savefig=f'{plot_path}/rmses.pdf')