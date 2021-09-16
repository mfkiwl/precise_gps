import matplotlib.pyplot as plt 
import matplotlib.colors as colors
import numpy as np
import tensorflow as tf
import math 

from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.lines import Line2D
from matplotlib.ticker import StrMethodFormatter

from src.visuals.process_results import average_frobenius
from src.visuals.process_results import pca_to_params
from src.visuals.process_results import transform_M
from src.visuals.process_results import loss_landscape
from src.visuals.process_results import best_coef
from src.visuals.process_results import best_mll

COLORS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 
          'tab:olive', 'tab:cyan', 'tab:pink', 'tab:brown', 'tab:gray']
NAME_COLORS = {'GPR': 'tab:blue', 'SVI':'tab:orange', 'SGHMC':'tab:red'}
KERNEL_COLORS = {'ARD':'tab:blue', 'FullL1':'tab:purple', 'FullW':'tab:red', 
                 'LowrankL1':'tab:green', 'LowrankW':'tab:orange'}
KERNEL_GRADIENT_COLORS = {'ARD':'Blues_r', 'FullL1':'Purples_r',
                          'FullW':'Reds_r', 'LowrankL1':'Greens_r', 
                          'LowrankW':'Oranges_r'}
CMAP = colors.LinearSegmentedColormap.from_list(name='red_white_blue', 
                                                 colors =['tab:blue', 
                                                          '#FAFAFA', 
                                                          'tab:red'],
                                                 N=200,
                                                 )

class MidpointNormalize(colors.Normalize):
    
    '''
    Normalise the colorbar so that diverging bars work there way either 
    side from a prescribed midpoint value) e.g. 
    im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,
    vmin=-100, vmax=100))
    '''
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, min(vmin,0), max(0,vmax), clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))

def show_kernel(kernel, title, cols, aspect = 'minmax', show_nums = -1, 
                savefig = None, show = 0):
    '''
    Visualizes the kernel with colors.

    Args:
        kernel (numpy array) : D x D shaped numpy array or tensor
        title (str) : text for the title
        cols (list) : D x 1 list for the covariates
        aspect (str) : 'minmax' -> colors are scaled from minimum to 
        maximum (center is not zero) '<anything else>' -> centered to 
        zero
        show_nums (int) : -1 -> dont show numeric values of the kernel
        > -1 -> show values of the kernel
        save_fig (string) : path in which the figure is saved
        show (bool) : wheter figure is shown or just closed
    
    Returns:
        Shows the figure and saves it to the specified path if 
        <savefig not None>.
    '''
    maximum = max(abs(np.min(kernel)), np.max(kernel))
    fig = plt.figure(figsize = (6,6))
    fig.set_facecolor('#FAFAFA')
    ax = plt.gca()
    ax.set_facecolor('#FAFAFA')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.15)
    if aspect != 'minmax':
        im = ax.imshow(kernel, cmap = CMAP, 
                       norm = MidpointNormalize(midpoint = 0, 
                                                vmin = -maximum, 
                                                vmax = maximum))
    else:
        im = ax.imshow(kernel, cmap = CMAP, 
                       norm = MidpointNormalize(midpoint = 0, 
                                                vmin = np.min(kernel), 
                                                vmax = np.max(kernel)))
    
    ax.set_yticks(np.arange(len(cols)))
    ax.set_yticklabels(cols)
    ax.set_title(title)
    
    ax.set_xticks(np.arange(len(cols)))
    
    # Minor ticks
    ax.set_xticks(np.arange(-.5, len(cols), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(cols), 1), minor=True)

    # Gridlines based on minor ticks
    ax.grid(which='minor', color='#FAFAFA', linestyle='-', linewidth=2)
    ax.set_frame_on(False)
    
    num_rows = kernel.shape[0]
    # Loop over data dimensions and create text annotations.
    if show_nums > -1:
        for i in range(num_rows):
            for j in range(num_rows):
                text = ax.text(j, i, np.around(kernel[i, j],show_nums),
                              ha='center', va='center', color='black')

    cb = plt.colorbar(im, cax=cax)
    cb.outline.set_edgecolor('#FAFAFA')
    if savefig:
        plt.savefig(savefig, bbox_inches='tight', facecolor='#FAFAFA')
    if show:
        plt.show()
    else:
        plt.close()

def show_kernels(kernels, titles, cols, aspect = 'own', show_nums = -1, 
                 savefig = None, show = 0):
    '''
    Visualizes a list of kernel with colors.

    Args:
        kernels (list of numpy arrays) : [D x D] list of numpy arrays or 
        tensorflow tensors
        title (list) : list of the title texts
        cols (list) : D x 1 list for the covariates
        aspect (str) : 'own' -> centered to zero with respect to won 
        values '<anything else>' -> centered to zero with respect to 
        global values
        show_nums (int) : -1 -> dont show numeric values of the kernel
        > -1 -> show values of the kernel
        savefig (str) : path in which the figure is saved
        show (bool) : wheter figure is shown or just closed
    
    Returns:
        Shows the figure and saves it to the specified path if 
        <save_fig == True>.
    '''
    num_rows = math.ceil(len(kernels) / 3)
    global_maximum = np.max(
        [max(abs(np.min(k)), np.max(k)) for k in kernels])

    fig, axs = plt.subplots(num_rows,3, figsize = (24,num_rows*6))
    fig.set_facecolor('#FAFAFA')
    axs = axs.ravel()
    for j in range(num_rows):
      for i in range(3):
        kernel = kernels[j*3+i]
        maximum = max(abs(np.min(kernel)), np.max(kernel))
        ax = axs[j*3+i]
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.15)
        if aspect == 'own':
            pcm = ax.imshow(kernel,cmap=CMAP, 
                            norm = MidpointNormalize(midpoint = 0, 
                                                     vmin = -maximum, 
                                                     vmax = maximum))
        else:
            pcm = ax.imshow(kernel,cmap=CMAP, 
                            norm = MidpointNormalize(midpoint = 0, 
                                                     vmin = -global_maximum, 
                                                     vmax = global_maximum))
        ax.set_yticks(np.arange(len(cols)))
        if i == 0:
            ax.set_yticklabels(cols)

        num_rows = kernel.shape[0]
        # Loop over data dimensions and create text annotations.
        if show_nums > -1:
            for i in range(num_rows):
                for j in range(num_rows):
                    text = ax.text(j, i, np.around(kernel[i, j],show_nums),
                                ha='center', va='center', color='black')
        ax.set_title(titles[j*3+i])
        ax.set_facecolor('#FAFAFA')
        
        ax.set_xticks(np.arange(len(cols)))
    
        # Minor ticks
        ax.set_xticks(np.arange(-.5, len(cols), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(cols), 1), minor=True)

        # Gridlines based on minor ticks
        ax.grid(which='minor', color='#FAFAFA', linestyle='-', linewidth=2)
        ax.set_frame_on(False)
        
        cb = fig.colorbar(pcm, cax=cax)
        cb.outline.set_edgecolor('#FAFAFA')
    if savefig:
        plt.savefig(savefig, bbox_inches='tight', facecolor='#FAFAFA')
    if show:
        plt.show()
    else:
        plt.close()

def visualize_mlls(mlls, log_liks, names, savefig = None, show = 0):
    '''
    Visualizes marginal log likelihood through iterations for different 
    models.

    Args:
        mlls (list) : dictionary of different runs for specific model 
        and lasso
        names (list) : list of strings
        savefig (str) : path in which figure is saved, 
        if None -> not saving anywhere
        show (bool) : wheter figure is shown or just closed
    '''

    minimum = np.inf
    maximum = -np.inf 
    
    zipped = sorted(zip(names, mlls))
    zipped2 = zip(*zipped)
    names, mlls = [list(elem) for elem in zipped2]
    best_log_liks = list(map(lambda x : x[0], best_mll(mlls)))


    plt.figure(figsize = (10,6))
    for idx, mll in enumerate(mlls):
        mll_as_list = mll[best_log_liks[idx]]
        counter = 0
        for jdx in range(10):
            current = mll_as_list[jdx][-1]
            if current < minimum:
                minimum = current
            if current > maximum:
                maximum = current 
            
            if 'SVI' in names[idx]:
                x_value = np.arange(len(mll_as_list[jdx]))*50
            else:
                x_value = np.arange(len(mll_as_list[jdx]))*5
            
            if counter == 0:
                counter += 1
                plt.plot(x_value, mll_as_list[jdx], 
                         color = COLORS[idx], label = names[idx], alpha = 0.8)
            else:
                plt.plot(x_value, mll_as_list[jdx], 
                         color = COLORS[idx], alpha = 0.8)
    
    plt.grid(True)
    plt.ylim([maximum - 2*(maximum - minimum), 
              maximum + int(1/4*(maximum - minimum))])
    plt.xlabel('Iteration')
    plt.ylabel('MLL (train)')
    plt.legend(prop = {'size': 14})
    if savefig:
        plt.savefig(savefig, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()    

def visualize_log_likelihood(log_liks, names, kernels, num_runs, fro = False, 
                             savefig = None, show = 0):
    '''
    Visualizes log-likelihoods for different models for all 
    lasso-coefficients used in optimization.

    Args:
        log_liks (list) : dictionary of log_likelihoods 
        (all lasso-coefficients included)
        names (list) : list of strings
        kernels (list) : dictionary of parameters 
        (all lasso-coefficients included)
        num_runs (list) : number of random initializations
        fro (bool) : whether log-likelihood is plotted against Frobenius 
        norm
        savefig (string) : path in which figure is saved, 
        if None -> not saving anywhere
        show (bool) : wheter figure is shown or just closed
    '''

    plt.figure(figsize = (10,6))
    for idx, log_lik in enumerate(log_liks):
        lassos = log_lik.keys()
        counter = 0
        log_liks_as_list = np.array([ll for ll in log_lik.values()])
        max_ll_model = np.unravel_index(np.argmax(log_liks_as_list, axis=None), 
                                        log_liks_as_list.shape)
        for lasso_idx, l in enumerate(lassos):
            max_ll = np.argmax(log_lik[l])
            if fro:
                frobenius = average_frobenius(kernels[idx][lasso_idx], 
                                              num_runs[idx])
            for jdx, ll in enumerate(log_lik[l]):
                if fro:
                    x_value = frobenius
                    if (lasso_idx, jdx) == max_ll_model:
                        plt.plot(x_value, ll, '.', 
                                 color = COLORS[idx % 10], 
                                 markersize = 15, label = names[idx])
                    else:
                        plt.plot(x_value, ll, '.', color = COLORS[idx % 10], 
                                 alpha = 0.25, markersize = 5)
                else:
                    x_value = l
                    if jdx == max_ll:
                        if counter == 0:
                            counter += 1
                            plt.plot(x_value, ll, '.', 
                                     color = COLORS[idx], markersize = 10, 
                                     label = names[idx])
                        else:
                            plt.plot(x_value, ll, '.', 
                                     color = COLORS[idx], markersize = 10)
                    else:
                        plt.plot(x_value, ll, '.', 
                                 color = COLORS[idx], alpha = 0.2)

    plt.grid(True)
    if fro:
        plt.xlabel('Frobenius norm')
        leg = plt.legend(prop = {'size': 14})
        for lh in leg.legendHandles: 
            lh.set_alpha(1)
    else:
        plt.xlabel('Coefficient')
        plt.legend(prop = {'size': 14})
    plt.ylabel('log-likelihood (test)')
    if savefig:
        plt.savefig(savefig, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()

def visualize_log_likelihood_mean(log_liks, names, kernels, num_runs, model, 
                                  fro = False, savefig = None, show = 0):
    '''
    Visualizes log-likelihoods for different models for all 
    lasso-coefficients used in optimization.

    Args:
        log_liks (list) : dictionary of log_likelihoods 
        (all lasso-coefficients included)
        names (list) : list of strings
        kernels (list) : dictionary of parameters 
        (all lasso-coefficients included)
        num_runs (list) : number of random initializations
        fro (bool) : whether log-likelihood is plotted against Frobenius 
        norm
        savefig (string) : path in which figure is saved, 
        if None -> not saving anywhere
        show (bool) : wheter figure is shown or just closed
    '''

    plt.figure(figsize = (10,6))
    for idx, log_lik in enumerate(log_liks):
        if model in names[idx]:
            lassos = log_lik.keys()
            log_liks_as_list = np.array([ll for ll in log_lik.values()])
            mean_ll = np.mean(log_liks_as_list, axis = 1)
            std_ll = np.std(log_liks_as_list, axis = 1)
            max_ll_model = np.argmax(mean_ll)
            for lasso_idx, l in enumerate(lassos):
                frobenius = average_frobenius(kernels[idx][lasso_idx], 
                                            num_runs[idx])
                x_value = frobenius
                if lasso_idx == max_ll_model:
                    plt.plot(x_value, mean_ll[lasso_idx], '.', 
                            color = COLORS[idx % 10], markersize = 15, 
                            label = names[idx], alpha = 0.8)
                    plt.errorbar(x_value, mean_ll[lasso_idx], 
                                yerr=std_ll[lasso_idx], fmt='-', 
                                color = COLORS[idx % 10], alpha = 0.25)
                else:
                    plt.plot(x_value, mean_ll[lasso_idx], '.', 
                            color = COLORS[idx % 10], alpha = 0.8, markersize = 5)


    plt.grid(True)
    if fro:
        plt.xlabel('Frobenius norm')
        leg = plt.legend(prop = {'size': 14})
        for lh in leg.legendHandles: 
            lh.set_alpha(1)
    else:
        plt.xlabel('Coefficient')
        plt.legend(prop = {'size': 14})
    plt.ylabel('log-likelihood (test)')
    if savefig:
        plt.savefig(savefig, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()
        
def visualize_best_lls(log_liks, mlls, names, info, savefig = None, show = 0, all = False):
    '''
    Visualizes log-likelihoods for different models for all 
    lasso-coefficients used in optimization.

    Args:
        log_liks (list) : dictionary of log_likelihoods 
        (all lasso-coefficients included)
        names (list) : list of strings
        savefig (string) : path in which figure is saved, 
        if None -> not saving anywhere
        show (bool) : wheter figure is shown or just closed
    '''
    zipped = sorted(zip(names, mlls, log_liks))
    zipped2 = zip(*zipped)
    names, mlls, log_liks = [list(elem) for elem in zipped2]
    best_log_liks = list(map(lambda x : x[0], best_mll(mlls)))
    plt.figure(figsize = (6,10))
    counter = 0
    new_names = []
    for i, log_lik in enumerate(log_liks):
        if not all:
            log_liks_as_list = log_lik[best_log_liks[i]]
            mean_ll = np.mean(log_liks_as_list)
            std_ll = np.std(log_liks_as_list)
            if 'SVI' in names[i]:
                color = NAME_COLORS['SVI']
            elif 'GPR' in names[i]:
                color = NAME_COLORS['GPR']
            else:
                color = NAME_COLORS['SGHMC']
            plt.plot(mean_ll, i, '.', markersize = 18, color = color)
            plt.errorbar(mean_ll, i, xerr=std_ll, fmt='-', 
                        color = color, elinewidth = 3)
        else:
            for coef, log_liks_as_list in log_lik.items():
                mean_ll = np.mean(log_liks_as_list)
                std_ll = np.std(log_liks_as_list)
                if 'SVI' in names[i]:
                    color = NAME_COLORS['SVI']
                elif 'GPR' in names[i]:
                    color = NAME_COLORS['GPR']
                else:
                    color = NAME_COLORS['SGHMC']
                    
                if 'ARD' in names[i]:
                    new_names.append(names[i])
                else:
                    new_names.append(names[i][:-3] + str(np.round(coef,2)))
                    
                plt.plot(mean_ll, counter, '.', markersize = 18, color = color)
                plt.errorbar(mean_ll, counter, xerr=std_ll, fmt='-', 
                            color = color, elinewidth = 3)
                counter += 1
    if not all:
        plt.yticks(np.arange(len(names)), names)
    else:
        plt.yticks(np.arange(len(new_names)), new_names)


    plt.grid(False)
    plt.title(info, fontweight = 'bold')
    if savefig:
        plt.savefig(savefig, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()

def visualize_best_rmse(mlls, rmses, names, info, savefig = None, show = 0):
    '''
    Visualizes rmse for different models for all 
    lasso-coefficients used in optimization.

    Args:
        log_liks (list) : dictionary of log_likelihoods 
        (all lasso-coefficients included)
        names (list) : list of strings
        savefig (string) : path in which figure is saved, 
        if None -> not saving anywhere
        show (bool) : wheter figure is shown or just closed
    '''
    
    zipped = sorted(zip(names, mlls, rmses))
    zipped2 = zip(*zipped)
    names, mlls, rmses = [list(elem) for elem in zipped2]
    best_log_liks = list(map(lambda x : x[0], best_mll(mlls)))
    plt.figure(figsize = (6,10))
    for i, rmse in enumerate(rmses):
        rmse_as_list = rmse[best_log_liks[i]]
        mean = np.mean(rmse_as_list)
        std = np.std(rmse_as_list)
        if 'SVI' in names[i]:
            color = NAME_COLORS['SVI']
        elif 'GPR' in names[i]:
            color = NAME_COLORS['GPR']
        else:
            color = NAME_COLORS['SGHMC']
        plt.plot(mean, i, '.', markersize = 18, color = color)
        plt.errorbar(mean, i, xerr=std, fmt='-', color = color, elinewidth = 3)
    
        plt.yticks(np.arange(len(names)), names)


    plt.grid(False)
    plt.title(info, fontweight = 'bold')
    if savefig:
        plt.savefig(savefig, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()

def visualize_errors(errors, names, error_type, kernels, num_runs, fro = False, 
                     savefig = None, show = 0):

    '''
    Visualizes rmse for different models for all 
    lasso-coefficients used in optimization.
    
    Args:
        errors (list) : dictionary of errors (all lasso-coefficients included)
        names (list) : list of strings
        error_type (str) : train/test 
        kernels (list) : dictionary of parameters 
        (all lasso-coefficients included)
        num_runs (list) : number of random initializations
        fro (bool) : whether log-likelihood is plotted against Frobenius 
        norm
        savefig (str) : path in which figure is saved, 
        if None -> not saving anywhere
        show (bool) : wheter figure is shown or just closed
    '''

    plt.figure(figsize = (10,6))
    for idx, model_errors in enumerate(errors):
        model_lassos = model_errors.keys()
        counter = 0
        errors_as_list = np.array([er for er in model_errors.values()])
        min_e_model = np.unravel_index(np.argmin(errors_as_list, axis=None), 
                                       errors_as_list.shape)
        for lasso_idx, l in enumerate(model_lassos):
            if fro:
                frobenius = average_frobenius(kernels[idx][lasso_idx], 
                                              num_runs[idx])
            min_e = np.argmin(model_errors[l])
            for jdx, e in enumerate(model_errors[l]):
                if fro:
                    x_value = frobenius
                    if (lasso_idx, jdx) == min_e_model:
                        plt.plot(x_value, e, '.', color = COLORS[idx % 10], 
                                 markersize = 15, label = names[idx])
                    else:
                        plt.plot(x_value, e, '.', color = COLORS[idx % 10], 
                                 alpha = 0.25, markersize = 5)
                else:
                    x_value = l
                    if jdx == min_e:
                        if counter == 0:
                            counter += 1
                            plt.plot(x_value, e, '.', color = COLORS[idx % 10], 
                                     markersize = 10, label = names[idx])
                        else:
                            plt.plot(x_value, e, '.', color = COLORS[idx % 10], 
                                     markersize = 10)
                    else:
                        plt.plot(x_value, e, '.', color = COLORS[idx % 10], 
                                 alpha = 0.2)

    plt.grid(True)
    if fro:
        plt.xlabel('Frobenius norm')
    else:
        plt.xlabel('Coefficient')
    plt.ylabel(f'{error_type} error')
    plt.legend(prop = {'size': 14})
    if savefig:
        plt.savefig(savefig, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()

def visualize_errors_mean(errors, names, error_type, kernels, num_runs, 
                          fro = False, savefig = None, show = 0):

    '''
    Visualizes mean rmse for different models for all 
    lasso-coefficients used in optimization.
    
    Args:
        errors (list) : dictionary of errors (all lasso-coefficients included)
        names (list) : list of strings
        error_type (str) : train/test 
        kernels (list) : dictionary of parameters (all lasso-coefficients included)
        num_runs (list) : number of random initializations
        fro (bool) : whether log-likelihood is plotted against Frobenius norm
        savefig (str) : path in which figure is saved, if None -> not saving anywhere
        show (bool) : wheter figure is shown or just closed
    '''

    plt.figure(figsize = (10,6))
    for idx, model_errors in enumerate(errors):
        model_lassos = model_errors.keys()
        counter = 0
        errors_as_list = np.array([er for er in model_errors.values()])
        mean_errors = np.mean(errors_as_list, axis = 1)
        std_errors = np.std(errors_as_list, axis = 1)
        min_e_model = np.argmin(mean_errors)
        for lasso_idx, l in enumerate(model_lassos):
            frobenius = average_frobenius(kernels[idx][lasso_idx], 
                                          num_runs[idx])
            x_value = frobenius
            if lasso_idx == min_e_model:
                plt.plot(x_value, mean_errors[lasso_idx], '.', 
                         color = COLORS[idx % 10], markersize = 15, 
                         label = names[idx])
                plt.errorbar(x_value, mean_errors[lasso_idx], 
                             yerr=std_errors[lasso_idx], fmt='-', 
                             color = COLORS[idx % 10], alpha = 0.5)
            else:
                plt.plot(x_value, mean_errors[lasso_idx], '.', 
                         color = COLORS[idx % 10], alpha = 0.25, markersize = 5)

    plt.grid(True)
    if fro:
        plt.xlabel('Frobenius norm')
    else:
        plt.xlabel('Coefficient')
    plt.ylabel(f'{error_type} error')
    plt.legend(prop = {'size': 14})
    if savefig:
        plt.savefig(savefig, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()

def visualize_loss_landscape(results, model, kernel, data, lasso, gradient, 
                             num_runs, savefig = None, show = 0):
    '''
    Visualize the loss landscape of the parameters using pca.

    Args:
        results (dict) : whole dictionary after optimization (df)
        model (str) : possible models in src.models.models
        kernel (str) : possible kernels in src.models.kernels
        data (tuple) : training data used during optimization
        lasso (float) : lasso coefficient
        gradient (bool) : wheter pca is calculated for gradient of 
        parameters or just parameters
        num_runs (int) : number of random initializations
        savefig (str) : path in which figure is saved, 
        if None -> not saving anywhere
        show (bool) : wheter figure is shown or just closed
    '''
    _, comp1, explained_variance, pca1 = pca_to_params(
        np.array(results['params'][lasso][0]), gradient)
    
    plt.figure(figsize = (8,8))
    maximum = -np.inf
    minimum = np.inf
    for i in range(num_runs):
        res, _, _, _ = pca_to_params(
            np.array(results['params'][lasso][i]), gradient)
        a, b = transform_M(pca1, res).T
        if np.max(a) > maximum:
            maximum = np.max(a) 
        if np.max(b) > maximum:
            maximum = np.max(b)
        if np.min(a) < minimum:
            minimum = np.min(a) 
        if np.min(b) < minimum:
            minimum = np.min(b)
    

    _range = np.linspace(minimum-1, maximum+1, 25)
    ll = loss_landscape(model, kernel, lasso, results['num_Z'], data, 
                        results['params'][lasso][0], 
                        results['variances'][lasso][0], 
                        results['likelihood_variances'][lasso][0], 
                        comp1, _range,_range, results['q_mu'], 
                        results['q_sqrt'], results['Z'], 
                        results['n'], results["rank"])
    im = plt.contourf(ll, extent=[minimum-1,maximum+1,minimum-1,maximum+1], 
                      levels=15, origin='lower')
    for i in range(num_runs):
        res, _, _, _ = pca_to_params(np.array(results['params'][lasso][i]), 
                                     gradient)
        a, b = transform_M(pca1, res).T
        plt.plot(a,b, color = COLORS[i % len(COLORS)], alpha = 0.7, 
                 linewidth = 2.5)
        plt.plot(a[-1], b[-1],'.', color = COLORS[i % len(COLORS)], 
                 markersize = 20, alpha = 0.7)
    plt.xlabel(f'PCA component 1: {round(explained_variance[0]*100,2)}%')
    plt.ylabel(f'PCA component 2: {round(explained_variance[1]*100,2)}%')
    if savefig:
        plt.savefig(savefig, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()

def visualize_eigen_threshhold(eigen_values, names, threshhold = 0.001, 
                               savefig = None, show = 0):
    '''
    Visualize the eigen_values with a threshhold

    Args:
        eigen_values (list) : dictionary of eigen_values of different 
        models
        names (list) : list of strings
        threshhold (float) : threshold for eigen values to be treated as 
        zeros
        savefig (str) : path in which figure is saved, 
        if None -> not saving anywhere
        show (bool) : wheter figure is shown or just closed
    '''
    plt.figure(figsize = (10,6))
    for idx, values in enumerate(eigen_values.values()):
        counter = 0
        for key,val in values.items():
            mean_val =  1/len(val)*len(np.where(np.array(val) > threshhold)[0])
            if counter == 0:
                counter += 1
                plt.plot(key, mean_val, '.', color = COLORS[idx], 
                         markersize = 15, label = names[idx])
            else:
                plt.plot(key, mean_val, '.', color = COLORS[idx], 
                         markersize = 15)
    
    plt.grid(True)
    plt.xlabel('Coefficient')
    plt.ylabel('pcs')
    plt.legend(prop = {'size': 14})
    if savefig:
        plt.savefig(savefig, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()

def combined_visualize_best(results, plot_log_lik = True, savefig = None, show = 0, all = False):
    '''
    Visualizes log-likelihoods for different models for all 
    lasso-coefficients used in optimization.

    Args:
        log_liks (list) : dictionary of log_likelihoods 
        (all lasso-coefficients included)
        names (list) : list of strings
        savefig (string) : path in which figure is saved, 
        if None -> not saving anywhere
        show (bool) : wheter figure is shown or just closed
    '''
    ticks = ['FULL W', 'FULL L', 'LR', 'ARD', 'FULL W', 'FULL L', 'ARD','FULL W', 'FULL L', 'LR', 'ARD']
    rand = 4*['SVI'] + 3*['SGHMC'] + 4*['GPR']
    
    ticks = list(reversed(ticks))
    rand = list(reversed(rand))
    
    num_datasets = len(results)
    
    NUM_COLS = 4
    num_rows = math.ceil(num_datasets / NUM_COLS)
    
    fig = plt.figure(figsize=(42, 24))
    fig.set_facecolor('#FAFAFA')
    for idx, key in enumerate(results.keys()):
        names = results[key]['names']
        mlls = results[key]['mlls']
        if plot_log_lik:
            log_liks = results[key]['log_likelihoods']
        else:
            log_liks = results[key]['test_errors']
        info = results[key]['info']
        plots = {}
        plt.subplot(num_rows, NUM_COLS, idx+1)
        plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
        ax = plt.gca()
        
        #print(names)
        #zipped = sorted(zip(names, mlls, log_liks))
        name_indices = sorted(range(len(names)), key=lambda k: names[k])
        #zipped2 = zip(*zipped)
        new_names = []
        new_mlls = []
        new_log_liks = []
        for ii in name_indices:
            new_names.append(names[ii])
            new_mlls.append(mlls[ii])
            new_log_liks.append(log_liks[ii])
        
        names = new_names
        mlls = new_mlls
        log_liks = new_log_liks
        
        #names, mlls, log_liks = [list(elem) for elem in zipped2]
        best_log_liks = list(map(lambda x : x[0], best_mll(mlls)))
        best_log_lik_values = list(map(lambda x : x[1], best_mll(mlls)))
        #print(names)
        #print(best_log_liks)
        #plt.figure(figsize = (6,10))
        counter = 0
        new_names = []
        for i, tick in enumerate(rand):
            if ticks[i] != 'LR':
                truth = ticks[i] in names[counter] and rand[i] in names[counter]
            else:
                truth = any(char.isdigit() for char in names[counter])
            if truth:
                #print(ticks[i], rand[i], names[counter])
                log_lik = log_liks[counter]
                log_liks_as_list = log_lik[best_log_liks[counter]]
                mean_ll = np.mean(log_liks_as_list)
                std_ll = np.std(log_liks_as_list)
                if 'SVI' in names[counter]:
                    color = NAME_COLORS['SVI']
                    #names[counter] = names[counter][4:]
                elif 'GPR' in names[counter]:
                    color = NAME_COLORS['GPR']
                    #names[counter] = names[counter][4:]
                else:
                    color = NAME_COLORS['SGHMC']
                    #names[counter] = names[counter][6:]
                
                if names[counter] not in plots:
                    plots[names[counter]] = {}
                    plots[names[counter]]['x'] = mean_ll 
                    plots[names[counter]]['y'] = i
                    plots[names[counter]]['color'] = color
                    plots[names[counter]]['std'] = std_ll
                    plots[names[counter]]['mll'] = best_log_lik_values[counter]
                    
                    current_name = names[counter]
                    counter += 1
                    if counter < len(names):
                        while current_name == names[counter]:
                            if plots[names[counter]]['mll'] < best_log_lik_values[counter]:
                                log_lik = log_liks[counter]
                                plots[names[counter]]['x'] = np.mean(log_lik[best_log_liks[counter]])
                                plots[names[counter]]['y'] = i
                                plots[names[counter]]['color'] = color
                                plots[names[counter]]['std'] = np.std(log_lik[best_log_liks[counter]])
                                plots[names[counter]]['mll'] = best_log_lik_values[counter]
                            counter += 1
                            if counter == len(names):
                                break
                        
                else:
                    #print(plots[names[counter]]['mll'], best_log_lik_values[counter])
                    if plots[names[counter]]['mll'] < best_log_lik_values[counter]:
                        plots[names[counter]]['x'] = mean_ll 
                        plots[names[counter]]['y'] = i
                        plots[names[counter]]['color'] = color
                        plots[names[counter]]['std'] = std_ll
                        plots[names[counter]]['mll'] = best_log_lik_values[counter]
                        
                        
        for key in plots.keys():
            x = plots[key]['x']     
            y = plots[key]['y']   
            color = plots[key]['color'] 
            std = plots[key]['std']
            plt.plot(x, y, '.', markersize = 36, color = color)
            plt.errorbar(x, y, xerr=std, fmt='-', 
                        color = color, elinewidth = 6)
        plt.yticks(np.arange(len(ticks)), ticks, fontweight = 'bold')
        ax.set_facecolor('#FAFAFA')

        def create_dummy_line(**kwds):
            return Line2D([], [], **kwds)

        # your code here

        # Create the legend
        lines = [
            ('SVI', {'color': NAME_COLORS['SVI'], 'linestyle': '-', 'marker': 'o', 'linewidth':3, 'markersize':12}),
            ('SGHMC', {'color': NAME_COLORS['SGHMC'], 'linestyle': '-', 'marker': 'o', 'linewidth':3, 'markersize':12}),
            ('GPR', {'color': NAME_COLORS['GPR'], 'linestyle': '-', 'marker': 'o', 'linewidth':3, 'markersize':12}),
        ]
        fig.legend(
            # Line handles
            [create_dummy_line(**l[1]) for l in lines],
            # Line titles
            [l[0] for l in lines],
            loc='lower center', 
            ncol = 3, 
            prop = {'size':34}
        )
        
        #plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.3))
        plt.grid(False)
        plt.title(info, fontweight = 'bold')
        plt.ylim([-0.5,10.5])
        
    if savefig:
        plt.savefig(savefig, bbox_inches='tight', facecolor = '#FAFAFA')
    if show:
        plt.show()
    else:
        plt.close()

def log_likelihood_mean(results, model, savefig = None, show = 0, fro = False,
                        mean = True):
    '''
    Visualizes log-likelihoods for different models for all 
    lasso-coefficients used in optimization.

    Args:
        log_liks (list) : dictionary of log_likelihoods 
        (all lasso-coefficients included)
        names (list) : list of strings
        kernels (list) : dictionary of parameters 
        (all lasso-coefficients included)
        num_runs (list) : number of random initializations
        fro (bool) : whether log-likelihood is plotted against Frobenius 
        norm
        savefig (string) : path in which figure is saved, 
        if None -> not saving anywhere
        show (bool) : wheter figure is shown or just closed
    '''
    GPR_DATASETS = ['Boston', 'Concrete', 'Yacht', 'Energy']
    if model == 'GPR':
        num_datasets = 4
        fig = plt.figure(figsize=(32, 24))
    else:
        num_datasets = len(results)
        #fig = plt.figure(figsize=(32, 32))
        fig = plt.figure(figsize=(32, 20))
        fig.set_facecolor('#FAFAFA')
        
    ax = plt.gca()
    
    NUM_COLS = 1
    num_rows = math.ceil(num_datasets / NUM_COLS)
    
    counter = 0
    for idx, key in enumerate(results.keys()):
        names = results[key]['names']
        log_liks = results[key]['log_likelihoods']
        info = results[key]['info']
        num_runs = results[key]['num_runs']
        kernels = results[key]['precisions']

        if model != 'GPR' or key in GPR_DATASETS:
            plots = {}
            counter += 1
            plt.subplot(num_rows, NUM_COLS, counter)
            for jdx, log_lik in enumerate(log_liks):
                if model in names[jdx]:
                    if any(char.isdigit() for char in names[jdx]):
                        if 'W' in names[jdx]:
                            color = KERNEL_COLORS['LowrankW']
                        else:
                            color = KERNEL_COLORS['LowrankL1']
                    elif 'FULL' in names[jdx]:
                        if 'W' in names[jdx]:
                            color = KERNEL_COLORS['FullW']
                        else:
                            color = KERNEL_COLORS['FullL1']
                    else:
                        color = KERNEL_COLORS['ARD']
                    lassos = log_lik.keys()
                    if fro:
                        new_lassos = {}
                        for lasso_idx, l in enumerate(lassos):
                            for nm in range(num_runs[jdx]):
                                if nm not in new_lassos:
                                    new_lassos[nm] = []
                                new_lassos[nm].append(tf.norm(kernels[jdx][lasso_idx][nm], 'euclidean'))
                                
                            #frobenius = average_frobenius(kernels[jdx][lasso_idx], num_runs)
                            #new_lassos.append(frobenius)
                        #lassos = new_lassos
                    log_liks_as_list = np.array([ll for ll in log_lik.values()])
                    mean_ll = np.mean(log_liks_as_list, axis = 1)
                    std_ll = np.std(log_liks_as_list, axis = 1)
                    
                    
                    if names[jdx] in plots:
                        plots[names[jdx]]['std'] += std_ll.tolist()
                        plots[names[jdx]]['mean'] += mean_ll.tolist()
                        plots[names[jdx]]['lassos'] += list(lassos)
                        if fro:
                            for nm in range(num_runs[jdx]):
                                plots[names[jdx]]['fro_y'][nm] += log_liks_as_list[:,nm].tolist()
                                plots[names[jdx]]['fro_x'][nm] += new_lassos[nm]
                    else:
                        plots[names[jdx]] = {}
                        if fro:
                            plots[names[jdx]]['fro_y'] = {}
                            plots[names[jdx]]['fro_x'] = {}
                            for nm in range(num_runs[jdx]):
                                plots[names[jdx]]['fro_y'][nm] = log_liks_as_list[:,nm].tolist()
                                plots[names[jdx]]['fro_x'][nm] = new_lassos[nm]
                        plots[names[jdx]]['std'] = std_ll.tolist()
                        plots[names[jdx]]['mean'] = mean_ll.tolist()
                        plots[names[jdx]]['lassos'] = list(lassos)
                        if model != 'SGHMC':
                            label = names[jdx][4:]
                        else:
                            label = names[jdx][6:]
                            
                        if any(char.isdigit() for char in names[jdx]):
                            label = 'Lowrank L'
                        plots[names[jdx]]['label'] = label
                        plots[names[jdx]]['color'] = color
                    
                    # if not idx:
                    #     if model != 'SGHMC':
                    #         label = names[jdx][4:]
                    #     else:
                    #         label = names[jdx][6:]
                            
                    #     if any(char.isdigit() for char in names[jdx]):
                    #         label = 'Lowrank L'
                    #     plt.plot(list(lassos), mean_ll, color = color, linestyle='-', marker='o', label=label, alpha = 0.7)
                    # else:
                    #     plt.plot(list(lassos), mean_ll, color = color, linestyle='-', marker='o', alpha = 0.7)
            for name in plots.keys():
                if model in name:
                    if not fro:
                        lassos = np.array(plots[name]['lassos'])
                        means = np.array(plots[name]['mean'])
                        stds = np.array(plots[name]['std'])
                        sort_index = np.argsort(lassos)
                        lassos = lassos[sort_index]
                        means = means[sort_index]
                        stds = stds[sort_index]
                        plt.plot(lassos, means, color = plots[name]['color'], linestyle='-', marker='o', label=plots[name]['label'], alpha = 0.7)          
                        plt.fill_between(lassos, means - 1.96*stds, means + 1.96*stds, color = plots[name]['color'], alpha = 0.20)
                    else:
                        for nm in range(num_runs[jdx]):
                            fro_x = np.array(plots[name]['fro_x'][nm])
                            fro_y = np.array(plots[name]['fro_y'][nm])
                            sort_index = np.argsort(fro_x)
                            fro_x = fro_x[sort_index]
                            fro_y = fro_y[sort_index]
                            plt.plot(fro_x, fro_y, color = plots[name]['color'], linestyle = '', marker='o', label=plots[name]['label'], alpha = 0.3, markersize = 12)
                    plt.title(info, fontweight = 'bold')


            plt.grid(True)
            if not fro:
                plt.xlabel('Coefficient')
            else:
                plt.xlabel('$\mathrm{Frobenius\;norm}$')
            plt.ylabel('$\mathrm{Log-likelihood\;(test)}$')
                
    def create_dummy_line(**kwds):
            return Line2D([], [], **kwds)
            
    lines = [
            ('$\mathrm{SE\;(L1)}$', {'color': KERNEL_COLORS['ARD'], 'linestyle': '', 'marker': 'o','markersize':12}),
            ('$\mathrm{Full\;(L1)}$', {'color': KERNEL_COLORS['FullL1'], 'linestyle': '', 'marker': 'o','markersize':12}),
            ('$\mathrm{Full\;(Wishart)}$', {'color': KERNEL_COLORS['FullW'], 'linestyle': '', 'marker': 'o','markersize':12}),
            #('$\mathrm{Full\;(Lowrank\;L1)}$', {'color': KERNEL_COLORS['LowrankL1'], 'linestyle': '', 'marker': 'o','markersize':12}),
            
        ]
    # fig.legend(
    #     # Line handles
    #     [create_dummy_line(**l[1]) for l in lines],
    #     # Line titles
    #     [l[0] for l in lines],
    #     loc='lower center', 
    #     ncol = 4, 
    #     prop = {'size':32}
    # )
    plt.legend(
        # Line handles
        [create_dummy_line(**l[1]) for l in lines],
        # Line titles
        [l[0] for l in lines],
        loc='lower right', 
        prop = {'size':50}
    )
    ax.set_facecolor('#FAFAFA')
    plt.xlim([0.1,1.6])
    if savefig:
        plt.savefig(savefig, bbox_inches='tight', facecolor='#FAFAFA')
    if show:
        plt.show()
    else:
        plt.close()


def log_likelihood_against_mll(results, model, savefig = None, show = 0):
    '''
    Visualizes log-likelihoods for different models for all 
    lasso-coefficients used in optimization.

    Args:
        log_liks (list) : dictionary of log_likelihoods 
        (all lasso-coefficients included)
        names (list) : list of strings
        kernels (list) : dictionary of parameters 
        (all lasso-coefficients included)
        num_runs (list) : number of random initializations
        fro (bool) : whether log-likelihood is plotted against Frobenius 
        norm
        savefig (string) : path in which figure is saved, 
        if None -> not saving anywhere
        show (bool) : wheter figure is shown or just closed
    '''
    GPR_DATASETS = ['Boston', 'Concrete', 'Yacht', 'Energy']
    if model == 'GPR':
        num_datasets = 4
        fig = plt.figure(figsize=(32, 24))
    else:
        num_datasets = len(results)
        fig = plt.figure(figsize=(32, 32))
    
    NUM_COLS = 2
    num_rows = math.ceil(num_datasets / NUM_COLS)
    
    counter = 0
    for idx, key in enumerate(results.keys()):
        names = results[key]['names']
        log_liks = results[key]['log_likelihoods']
        mlls = results[key]['mlls']
        info = results[key]['info']
        num_runs = results[key]['num_runs']
        
        plots = {}
        if model != 'GPR' or key in GPR_DATASETS:
            counter += 1
            # plt.subplot(num_rows, NUM_COLS, counter)
            for jdx, tup in enumerate(zip(log_liks, mlls)):
                log_lik = tup[0]
                _mll = tup[1]
                if model in names[jdx]:
                    if any(char.isdigit() for char in names[jdx]):
                        if 'W' in names[jdx]:
                            color = KERNEL_COLORS['FullW']
                        else:
                            color = KERNEL_COLORS['FullW']
                    elif 'FULL' in names[jdx]:
                        if 'W' in names[jdx]:
                            color = KERNEL_COLORS['FullW']
                        else:
                            color = KERNEL_COLORS['FullW']
                    else:
                        color = KERNEL_COLORS['ARD']
                    lassos = log_lik.keys()
                    log_liks_as_list = np.array([ll for ll in log_lik.values()])
                    mlls_as_list = [list(ll.values()) for ll in _mll.values()]
                    new_list = []
                    for mll in mlls_as_list:
                        ml = []
                        for m in mll:
                            ml.append(m[-1].numpy())
                        new_list.append(ml)
                    new_list = np.array(new_list)
                    mean_ll = np.mean(log_liks_as_list, axis = 1)
                    #std_ll = np.std(log_liks_as_list, axis = 1)
                    # if not idx:
                    #     if model != 'SGHMC':
                    #         label = names[jdx][4:]
                    #     else:
                    #         label = names[jdx][6:]
                    
                    if names[jdx] in plots:
                        #plots[names[jdx]]['mll'] += new_list
                        #plots[names[jdx]]['log_likelihood'] += mean_ll.tolist()
                        #plots[names[jdx]]['lassos'] += list(lassos)
                        for nm in range(num_runs[jdx]):
                            plots[names[jdx]]['log_likelihood'][nm] += log_liks_as_list[:,nm].tolist()
                            plots[names[jdx]]['mll'][nm] += new_list[:,nm].tolist()
                    else:
                        plots[names[jdx]] = {}
                        #plots[names[jdx]]['mll'] = new_list
                        #plots[names[jdx]]['log_likelihood'] = mean_ll.tolist()
                        #plots[names[jdx]]['lassos'] = list(lassos)
                        plots[names[jdx]]['log_likelihood'] = {}
                        plots[names[jdx]]['mll'] = {}
                        for nm in range(num_runs[jdx]):
                            plots[names[jdx]]['log_likelihood'][nm] = log_liks_as_list[:,nm].tolist()
                            plots[names[jdx]]['mll'][nm] = new_list[:,nm].tolist()
                        if model != 'SGHMC':
                            label = names[jdx][4:]
                        else:
                            label = names[jdx][6:]
                            
                        if any(char.isdigit() for char in names[jdx]):
                            label = 'Lowrank L'
                        plots[names[jdx]]['label'] = label
                        plots[names[jdx]]['color'] = color
                        
                    
                            
            # if any(char.isdigit() for char in names[jdx]):
            #     label = 'Lowrank L'
            plt.subplot(num_rows, NUM_COLS, counter)
            for ii, name in enumerate(plots.keys()):
                if model in name:
                    #lassos = np.array(plots[name]['lassos'])
                    for nm in range(num_runs[jdx]):
                        ll = np.array(plots[name]['log_likelihood'][nm])
                        mll = np.array(plots[name]['mll'][nm])
                        #sort_index = np.argsort(lassos)
                        #lassos = lassos[sort_index]
                        #ll = ll[sort_index]
                        #mll = mll[sort_index]
                        #plt.scatter(mll, ll, cmap = plots[name]['color'], label=plots[name]['label'], c = lassos)
                        f_color = plots[name]['color']#[:-3].lower()
                        plt.plot(mll,ll, color = f'{f_color}', alpha = 0.4, marker = 'o', linestyle = '')
            # else:
            #     plt.scatter(new_list, mean_ll, cmap = color, c = list(lassos))
                
            #plt.fill_between(new_list, mean_ll - 1.96*std_ll, mean_ll + 1.96*std_ll, alpha = 0.20, c = list(lassos), cmap = color)
            plt.title(info, fontweight = 'bold')


            plt.grid(True)
            if model == 'GPR':
                plt.xlabel('MLL')
            else:
                plt.xlabel('ELBO')
            plt.ylabel('log-likelihood (test)')
    
    def create_dummy_line(**kwds):
            return Line2D([], [], **kwds)
            
    lines = [
            ('ARD', {'color': KERNEL_COLORS['ARD'], 'linestyle': '', 'marker': 'o','markersize':12}),
            ('Full', {'color': KERNEL_COLORS['FullW'], 'linestyle': '', 'marker': 'o','markersize':12}),
            
        ]
    fig.legend(
        # Line handles
        [create_dummy_line(**l[1]) for l in lines],
        # Line titles
        [l[0] for l in lines],
        loc='lower center', 
        ncol = 4, 
        prop = {'size':18}
    )
                
    #fig.legend(loc='lower center', ncol = 5, prop = {'size':18})
    if savefig:
        plt.savefig(savefig, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()
        

def combined_visualize_mll(results, savefig = None, show = 0, all = False):
    '''
    Visualizes log-likelihoods for different models for all 
    lasso-coefficients used in optimization.

    Args:
        log_liks (list) : dictionary of log_likelihoods 
        (all lasso-coefficients included)
        names (list) : list of strings
        savefig (string) : path in which figure is saved, 
        if None -> not saving anywhere
        show (bool) : wheter figure is shown or just closed
    '''
    ticks = ['FULL W', 'FULL L', 'Lowrank L', 'ARD', 'FULL W', 'FULL L', 'ARD','FULL W', 'FULL L', 'Lowrank L', 'ARD']
    rand = 4*['SVI'] + 3*['SGHMC'] + 4*['GPR']
    
    ticks = list(reversed(ticks))
    rand = list(reversed(rand))
    
    num_datasets = len(results)
    
    NUM_COLS = 4
    num_rows = math.ceil(num_datasets / NUM_COLS)
    
    fig = plt.figure(figsize=(36, 12))
    for idx, key in enumerate(results.keys()):
        names = results[key]['names']
        mlls = results[key]['mlls']
        log_liks = results[key]['log_likelihoods']
        info = results[key]['info']
        plt.subplot(num_rows, NUM_COLS, idx+1)
        plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
        
        
        zipped = sorted(zip(names, mlls, log_liks))
        zipped2 = zip(*zipped)
        names, mlls, log_liks = [list(elem) for elem in zipped2]
        best_log_liks = best_coef(log_liks)
        #print(names)
        #print(best_log_liks)
        #plt.figure(figsize = (6,10))
        counter = 0
        new_names = []
        for i, tick in enumerate(rand):
            if not all:
                if ticks[i] != 'Lowrank L':
                    truth = ticks[i] in names[counter] and rand[i] in names[counter]
                else:
                    truth = any(char.isdigit() for char in names[counter])
                if truth and rand[i] != 'SGHMC':
                    #print(ticks[i], rand[i], names[counter])
                    _mll = mlls[counter]
                    mlls_as_list = [list(_mll[best_log_liks[counter]].values())]
                    #mlls_as_list = [list(ll.values()) for ll in _mll.values()]
                    new_list = []
                    new_list2 = []
                    for mll in mlls_as_list:
                        ml = []
                        svd = []
                        for m in mll:
                            ml.append(m[-1].numpy())
                            svd.append(m[-1].numpy())
                        new_list.append(np.mean(ml))
                        new_list2.append(np.std(svd))
                    
                    mean_ll = np.array(new_list)#np.mean(new_list, axis = 1)
                    std_ll = np.array(new_list2)#np.mean(new_list, axis = 1)
                    if 'SVI' in names[counter]:
                        color = NAME_COLORS['SVI']
                        #names[counter] = names[counter][4:]
                    elif 'GPR' in names[counter]:
                        color = NAME_COLORS['GPR']
                        #names[counter] = names[counter][4:]
                    else:
                        color = NAME_COLORS['SGHMC']
                        #names[counter] = names[counter][6:]
                    plt.plot(mean_ll, i, '.', markersize = 18, color = color)
                    plt.errorbar(mean_ll, i, xerr=std_ll, fmt='-', 
                                color = color, elinewidth = 3)
                    counter += 1
                elif rand[i] == 'SGHMC':
                    counter += 1
                #_mll = log_liks[0]
                #mlls_as_list = [list(_mll[best_log_liks[0]].values())]
                #new_list = []
                #for mll in mlls_as_list:
                #    ml = []
                #    for m in mll:
                #        ml.append(m[-1].numpy())
                #    new_list.append(np.mean(ml))
                #
                #mean_ll = np.mean(mlls_as_list, axis = 1)
                #std_ll = np.mean(mlls_as_list, axis = 1)
                #plt.plot(mean_ll, i, color = 'w')
                #counter += 1
        if not all:
            plt.yticks(np.arange(len(ticks)), ticks, fontweight = 'bold')
            plt.ylim((-0.5, 10.5))
        else:
            plt.yticks(np.arange(len(new_names)), new_names)

        def create_dummy_line(**kwds):
            return Line2D([], [], **kwds)

        # your code here

        # Create the legend
        lines = [
            ('SVI', {'color': NAME_COLORS['SVI'], 'linestyle': '-', 'marker': 'o', 'linewidth':3, 'markersize':12}),
            ('SGHMC', {'color': NAME_COLORS['SGHMC'], 'linestyle': '-', 'marker': 'o', 'linewidth':3, 'markersize':12}),
            ('GPR', {'color': NAME_COLORS['GPR'], 'linestyle': '-', 'marker': 'o', 'linewidth':3, 'markersize':12}),
        ]
        fig.legend(
            # Line handles
            [create_dummy_line(**l[1]) for l in lines],
            # Line titles
            [l[0] for l in lines],
            loc='lower center', 
            ncol = 3, 
            prop = {'size':18}
        )
        
        #plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.3))
        plt.grid(False)
        plt.title(info, fontweight = 'bold')
        
    if savefig:
        plt.savefig(savefig, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()
        
        