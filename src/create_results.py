import os 
import tensorflow as tf
import tensorflow_probability as tfp
import pandas as pd 
from src.parse_results import parse_pickle
from src.visuals.visuals import * 
from src.datasets.datasets import *
from src.visuals.process_results import *
from src.tex.create_tables import * 

plt.rcParams.update({'font.size': 14}) # Global fontsize

def create_results(dataset, directory, num_lassos, step = 1, show = 0, 
                   loss_landscape = 0):
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
    if len(directory) > 1:
        pkl_files = []
        for file_path in directory:
            data_path = f'results/raw/{dataset.lower()}/{file_path}'
            if '.pkl' in file_path:
                pkl_files.append(data_path)
        df = {}
        for idx, current_file in enumerate(pkl_files):
            data = parse_pickle(current_file)
            df[idx] = data
        
    else:
        data_path = f'results/raw/{dataset.lower()}/{directory[0]}/'
        pkl_files = [file for file in os.listdir(data_path) 
                     if '.pkl' in file]

        df = {}
        for idx, current_file in enumerate(pkl_files):
            data = parse_pickle(data_path + current_file)
            df[idx] = data
    
    # Each directory should only contain results from one dataset.
    dataset = data['dataset']

    result_path = 'results/processed/' + f'{dataset.lower()}/{directory[0]}'
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    plot_path = result_path + '/plots'
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    
    names, mll_names_gpr, mll_names_svi = [], [], []
    mlls_gpr, mlls_svi = [], []
    log_liks, train_errors, test_errors = [], [], []
    all_lassos, all_precisions = [], []
    num_runs = []

    for key in df.keys():
        precisions = []
        data = df[key]
        model = data['model']
        kernel = data['kernel']
        penalty = data['penalty']
        rank = data['rank']
        number_of_initializations = data["num_runs"]
        
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
        penalty = 'W' if penalty == 'wishart' else 'L1'

        if kernel == 'ARD':
            names.append(f'{model} {kernel}')
        else:
            names.append(f'{model} {kernel} {penalty}')
             
        # Some coefficients for MLL visualization
        lassos = data['lassos'][0::step] if data['penalty'] == 'lasso'  \
            else data['n'][0::step]
        lassos = lassos[0:min(len(lassos), num_lassos)]
        for l in lassos:
            if model == 'SVI':
                mlls_svi.append(data['mll'][l])
                mll_names_svi.append(f'{model} {kernel} {str(np.round(l,2))}')
            if model == 'GPR':
                mlls_gpr.append(data['mll'][l])
                mll_names_gpr.append(f'{model} {kernel} {str(np.round(l,2))}')

        for l in data['lassos'] if data['penalty'] == 'lasso' else data['n']:
            new_params = {}
            for i in range(data['num_runs']):
                if model == 'SGHMC':
                    new_params[i] = params_to_precision_vis(
                        np.array(data['sghmc_params'][l][i][-1][90]), 
                        data['kernel'], data['rank'], 
                        len(data['sghmc_params'][l][i][-1][90]))
                else:  
                    new_params[i] = params_to_precision_vis(
                        np.array(data['params'][l][i][-1]), 
                        data['kernel'], data['rank'], 
                        len(data['params'][l][i][-1]))
                    
            precisions.append(new_params) 
        
        all_precisions.append(precisions)
        log_liks.append(data['log_likelihoods'])
        train_errors.append(data['train_errors'])
        test_errors.append(data['test_errors'])
        all_lassos.append(data['lassos'])
        
    best_coefs = best_coef(log_liks)
    comparison_plot_names = list(
        map(lambda x: f'{x[0]} {str(np.round(x[1],2))}', 
            zip(names,best_coefs)))
    
    visualize_best_lls(log_liks, comparison_plot_names, 
                       savefig=f'{plot_path}/lls.pdf')
    visualize_best_rmse(log_liks, test_errors, comparison_plot_names, 
                        savefig=f'{plot_path}/rmses.pdf')
    visualize_mlls(mlls_svi, mll_names_svi, 
                   f'{plot_path}/mlls_svi.pdf', show)
    if mlls_gpr:
        visualize_mlls(mlls_gpr, mll_names_gpr, 
                       f'{plot_path}/mlls_gpr.pdf', show)
        
    visualize_log_likelihood(log_liks, names, all_precisions, num_runs, True, 
                             f'{plot_path}/log_liks_fro.pdf', show)
    visualize_log_likelihood_mean(log_liks, names, all_precisions, num_runs, 
                                  True, f'{plot_path}/log_liks_fro_mean.pdf', 
                                  show)
    visualize_errors(test_errors,names,'test', all_precisions, num_runs, True, 
                     f'{plot_path}/test_errors_fro.pdf', show)
    visualize_errors_mean(test_errors,names,'test', all_precisions, num_runs, 
                          True, f'{plot_path}/test_errors_fro_mean.pdf', show)

    # Kernels
    for idx, key in enumerate(df.keys()):
        data = df[key]
        for l in data['lassos'] if data['penalty'] == 'lasso' else data['n']:
            precisions = []
            p_names = []
            if data['model'] != 'SGHMC':
                for i in range(9):
                    P = params_to_precision_vis(
                        np.array(data['params'][l][i][-1]), data['kernel'], 
                        data['rank'], len(data['params'][l][i][-1]))
                    precisions.append(P)
                    lll = data['log_likelihoods'][l][i]
                    p_names.append('LL: ' + str(round(lll,3)) +', TE: ' + \
                        str(round(data['test_errors'][l][i],2)) + ', Var: ' + \
                            str(np.round(data['variances'][l][i][-1],2)))
            else:
                indices = np.random.choice(np.arange(100), 9)
                for i in indices:
                    P = params_to_precision_vis(
                        np.array(data['sghmc_params'][l][-1][-1][i]), 
                        data['kernel'], data['rank'], 
                        len(data['sghmc_params'][l][-1][-1][i]))
                    precisions.append(P)
                    lll = data['log_likelihoods'][l][-1]
                    p_names.append('LL: ' + str(round(lll,3)) +', TE: ' + \
                        str(round(data['test_errors'][l][-1],2)) + ', Var: ' + \
                            str(np.round(
                                np.exp(data['sghmc_vars'][l][-1][-1][i]),2)))
            data_instance = globals()[dataset](0.2)
            cols = data_instance.cols
            if not os.path.exists(plot_path + '/kernels'):
                os.makedirs(plot_path + '/kernels')
            show_kernels(
                precisions,p_names,
                cols,'global',-1,
                f'{plot_path}/kernels/{names[idx]}{str(np.round(l,2))}.pdf', 
                show)

    table_path = result_path + '/tables'
    if not os.path.exists(table_path):
        os.makedirs(table_path)

    # save_results_table(new_log_liks, new_train_errors, new_test_errors, 
    # new_names, f'{table_path}/')
    save_overview(
        log_liks, 
        test_errors, 
        list(map(lambda x: f'{x[0]} {str(np.round(x[1],2))}', 
                 zip(names,best_coefs))), f'{table_path}/', best_coefs)

    # Eigenvalues
    if not os.path.exists(table_path + '/eigen'):
        os.makedirs(table_path + '/eigen')

    eigen_value_dict = {}
    eigen_names = []
    for key in df.keys():
        data = df[key]
        if 'SGHMC' not in data['kernel']:
            model_name = 'SVI' if 'SVI' in data['model'] else 'GPR'
            kernel_name = 'ARD' if 'ARD' in data['kernel'] else 'FULL'
            eigen_names.append(f'{model_name} {kernel_name}')
            if 'ARD' in data['kernel']:
                kernel_path = 'ARD'
            else:
                kernel_path = 'FULL'
            if not os.path.exists(table_path + f'/eigen/{kernel_path}'):
                os.makedirs(table_path + f'/eigen/{kernel_path}')

            model_eigen_values = {}
            if 'penalty' not in data:
                data['penalty'] = 'lasso'
            for l in data['lassos'] if data['penalty'] == 'lasso' else data['n']:
                ret = []
                for i in range(data['num_runs']):
                    P = params_to_precision_vis(
                        np.array(data['params'][l][i][-1]), 
                        data['kernel'], data['rank'], 
                        len(data['params'][l][i][-1]))
                    eigen_vals, _ = eigen(P)
                    ret.append(list(eigen_vals))
                model_eigen_values[l] = ret
            
            eigen_value_dict[eigen_names[-1]] = model_eigen_values
    visualize_eigen_threshhold(
        eigen_value_dict,eigen_names,0.001,plot_path + '/eigen_th.pdf',show)

    if loss_landscape:
        # Loss landscape
        loss_param_path = plot_path + '/loss_landscape/param'
        if not os.path.exists(loss_param_path):
            os.makedirs(loss_param_path)
        for key in df.keys():
            data = df[key]
            if 'Z' not in data:
                data['Z'] = []
            if 'q_mu' not in data:
                data['q_mu'] = []
            if 'q_sqrt' not in data:
                data['q_sqrt'] = []
            the_best_coef = best_coef([data['log_likelihoods']])[0]
            if 'GPR' in data['model']:
                visualize_loss_landscape(
                    data, 
                    data['model'], 
                    data['kernel'], 
                    data['data_train'], 
                    the_best_coef, 
                    False,
                    10, 
                    loss_param_path + \
                        '/{}_{}_{}_{}.pdf'.format(data['model'], 
                                                  data['kernel'], 
                                                  data['penalty'], 
                                                  str(round(the_best_coef,1))), 
                        show)
        
        

