import matplotlib.pyplot as plt 
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import math 
from src.visuals.process_results import average_frobenius, pca_to_params, transform_M, loss_landscape, best_coef

# TODO : this needs cleaning up
COLORS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:olive', 'tab:cyan', 'tab:pink', 'tab:brown', 'tab:gray']

class MidpointNormalize(colors.Normalize):
    
    '''
    Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)
    e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
    '''
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, min(vmin,0), max(0,vmax), clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))

def show_kernel(kernel, title, cols, aspect = 'minmax', show_nums = -1, savefig = None, show = 0):
    '''
    Visualizes the kernel with colors.

    Args:
        kernel (numpy array) : D x D shaped numpy array or tensorflow tensor
        title (string)       : text for the title
        cols (list)          : D x 1 list for the covariates
        aspect (string)      : 'minmax' -> colors are scaled from minimum to maximum (center is not zero)
                               '<anything else>' -> centered to zero
        show_nums (int)      : -1 -> dont show numeric values of the kernel
                                > -1 -> show values of the kernel
        save_fig (string)    : path in which the figure is saved
        show (bool)          : wheter figure is shown or just closed
    
    Returns:
        Shows the figure and saves it to the specified path if <savefig not None>.
    '''
    maximum = max(abs(np.min(kernel)), np.max(kernel))
    fig = plt.figure(figsize = (6,6))
    ax = plt.gca()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.15)
    if aspect != 'minmax':
        im = ax.imshow(kernel, cmap = 'bwr', norm = MidpointNormalize(midpoint = 0, vmin = -maximum, vmax = maximum))
    else:
        im = ax.imshow(kernel, cmap = 'bwr', norm = MidpointNormalize(midpoint = 0, vmin = np.min(kernel), vmax = np.max(kernel)))
    
    ax.set_yticks(np.arange(len(cols)))
    ax.set_yticklabels(cols)
    ax.set_title(title)
    num_rows = kernel.shape[0]
    # Loop over data dimensions and create text annotations.
    if show_nums > -1:
        for i in range(num_rows):
            for j in range(num_rows):
                text = ax.text(j, i, np.around(kernel[i, j],show_nums),
                              ha='center', va='center', color='black')

    plt.colorbar(im, cax=cax)
    if savefig:
        plt.savefig(savefig, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()

def show_kernels(kernels, titles, cols, aspect = 'own', show_nums = -1, savefig = None, show = 0):
    '''
    Visualizes a list of kernel with colors.

    Args:
        kernels (list of numpy arrays) : [D x D] list of numpy arrays or tensorflow tensors
        title (list)                   : list of the title texts
        cols (list)                    : D x 1 list for the covariates
        aspect (string)                : 'own' -> centered to zero with respect to won values
                                         '<anything else>' -> centered to zero with respect to global values
        show_nums (int)                : -1 -> dont show numeric values of the kernel
                                         > -1 -> show values of the kernel
        savefig (string)               : path in which the figure is saved
        show (bool)                    : wheter figure is shown or just closed
    
    Returns:
        Shows the figure and saves it to the specified path if <save_fig == True>.
    '''
    num_rows = math.ceil(len(kernels) / 3)
    global_maximum = np.max([max(abs(np.min(k)), np.max(k)) for k in kernels])

    fig, axs = plt.subplots(num_rows,3, figsize = (24,num_rows*6))
    axs = axs.ravel()
    for j in range(num_rows):
      for i in range(3):
        kernel = kernels[j*3+i]
        maximum = max(abs(np.min(kernel)), np.max(kernel))
        ax = axs[j*3+i]
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.15)
        if aspect == 'own':
            pcm = ax.imshow(kernel,cmap='bwr', norm = MidpointNormalize(midpoint = 0, vmin = -maximum, vmax = maximum))
        else:
            pcm = ax.imshow(kernel,cmap='bwr', norm = MidpointNormalize(midpoint = 0, vmin = -global_maximum, vmax = global_maximum))
        if i == 0:
          ax.set_yticks(np.arange(len(cols)))
          ax.set_yticklabels(cols, fontsize = 16)
        num_rows = kernel.shape[0]
        # Loop over data dimensions and create text annotations.
        if show_nums > -1:
            for i in range(num_rows):
                for j in range(num_rows):
                    text = ax.text(j, i, np.around(kernel[i, j],show_nums),
                                ha='center', va='center', color='black')
        ax.set_title(titles[j*3+i], fontsize =16)
        fig.colorbar(pcm, cax=cax)
    if savefig:
        plt.savefig(savefig, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()

def visualize_mlls(mlls, names, savefig = None, show = 0):
    '''
    Visualizes marginal log likelihood through iterations for different models.

    Args:
        mlls (list)      : dictionary of different runs for specific model and lasso
        names (list)     : list of strings
        savefig (string) : path in which figure is saved, if None -> not saving anywhere
        show (bool)      : wheter figure is shown or just closed
    '''

    minimum = np.inf
    maximum = -np.inf 

    plt.figure(figsize = (10,6))
    for idx, log_lik in enumerate(mlls):
        iterations = log_lik.keys()
        counter = 0
        for jdx in iterations:
            current = log_lik[jdx][-1]
            if current < minimum:
                minimum = current
            if current > maximum:
                maximum = current 
            
            if 'SVI' in names[idx]:
                x_value = np.arange(len(log_lik[jdx]))*50
            else:
                x_value = np.arange(len(log_lik[jdx]))*5
            
            if counter == 0:
                counter += 1
                plt.plot(x_value, log_lik[jdx], color = COLORS[idx], label = names[idx], alpha = 0.8)
            else:
                plt.plot(x_value, log_lik[jdx], color = COLORS[idx], alpha = 0.8)
    
    plt.grid(True)
    plt.ylim([maximum - 2*(maximum - minimum), maximum + int(1/4*(maximum - minimum))])
    plt.xlabel('Iteration')
    plt.ylabel('MLL (train)')
    plt.legend(prop = {'size': 14})
    if savefig:
        plt.savefig(savefig, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()    

def visualize_log_likelihood(log_liks, names, kernels, num_runs, fro = False, savefig = None, show = 0):
    '''
    Visualizes log-likelihoods for different models for all lasso-coefficients used in optimization.

    Args:
        log_liks (list)  : dictionary of log_likelihoods (all lasso-coefficients included)
        names (list)     : list of strings
        kernels (list)   : dictionary of parameters (all lasso-coefficients included)
        num_runs (list)   : number of random initializations
        fro (bool)       : whether log-likelihood is plotted against Frobenius norm
        savefig (string) : path in which figure is saved, if None -> not saving anywhere
        show (bool)      : wheter figure is shown or just closed
    '''

    plt.figure(figsize = (10,6))
    for idx, log_lik in enumerate(log_liks):
        lassos = log_lik.keys()
        counter = 0
        log_liks_as_list = np.array([ll for ll in log_lik.values()])
        max_ll_model = np.unravel_index(np.argmax(log_liks_as_list, axis=None), log_liks_as_list.shape)
        for lasso_idx, l in enumerate(lassos):
            max_ll = np.argmax(log_lik[l])
            if fro:
                frobenius = average_frobenius(kernels[idx][lasso_idx], num_runs[idx])
            for jdx, ll in enumerate(log_lik[l]):
                if fro:
                    x_value = frobenius
                    if (lasso_idx, jdx) == max_ll_model:
                        plt.plot(x_value, ll, '.', color = COLORS[idx], markersize = 15, label = names[idx])
                    else:
                        plt.plot(x_value, ll, '.', color = COLORS[idx], alpha = 0.25, markersize = 5)
                else:
                    x_value = l
                    if jdx == max_ll:
                        if counter == 0:
                            counter += 1
                            plt.plot(x_value, ll, '.', color = COLORS[idx], markersize = 10, label = names[idx])
                        else:
                            plt.plot(x_value, ll, '.', color = COLORS[idx], markersize = 10)
                    else:
                        plt.plot(x_value, ll, '.', color = COLORS[idx], alpha = 0.2)

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

def visualize_log_likelihood_mean(log_liks, names, kernels, num_runs, fro = False, savefig = None, show = 0):
    '''
    Visualizes log-likelihoods for different models for all lasso-coefficients used in optimization.

    Args:
        log_liks (list)  : dictionary of log_likelihoods (all lasso-coefficients included)
        names (list)     : list of strings
        kernels (list)   : dictionary of parameters (all lasso-coefficients included)
        num_runs (list)   : number of random initializations
        fro (bool)       : whether log-likelihood is plotted against Frobenius norm
        savefig (string) : path in which figure is saved, if None -> not saving anywhere
        show (bool)      : wheter figure is shown or just closed
    '''

    plt.figure(figsize = (10,6))
    for idx, log_lik in enumerate(log_liks):
        lassos = log_lik.keys()
        log_liks_as_list = np.array([ll for ll in log_lik.values()])
        mean_ll = np.mean(log_liks_as_list, axis = 1)
        std_ll = np.std(log_liks_as_list, axis = 1)
        max_ll_model = np.argmax(mean_ll)
        for lasso_idx, l in enumerate(lassos):
            frobenius = average_frobenius(kernels[idx][lasso_idx], num_runs[idx])
            x_value = frobenius
            if lasso_idx == max_ll_model:
                plt.plot(x_value, mean_ll[lasso_idx], '.', color = COLORS[idx], markersize = 15, label = names[idx])
                plt.errorbar(x_value, mean_ll[lasso_idx], yerr=std_ll[lasso_idx], fmt='-', color = COLORS[idx])
            else:
                plt.plot(x_value, mean_ll[lasso_idx], '.', color = COLORS[idx], alpha = 0.25, markersize = 5)


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
        
def visualize_best_lls(log_liks, names, savefig = None, show = 0):
    '''
    Visualizes log-likelihoods for different models for all lasso-coefficients used in optimization.

    Args:
        log_liks (list)  : dictionary of log_likelihoods (all lasso-coefficients included)
        names (list)     : list of strings
        savefig (string) : path in which figure is saved, if None -> not saving anywhere
        show (bool)      : wheter figure is shown or just closed
    '''
    best_log_liks = best_coef(log_liks)

    plt.figure(figsize = (10,6))
    for i, log_lik in enumerate(log_liks):
        log_liks_as_list = log_lik[best_log_liks[i]]
        mean_ll = np.mean(log_liks_as_list)
        std_ll = np.std(log_liks_as_list)
        plt.plot(mean_ll, i, '.', markersize = 12, color = COLORS[i % 10])
        plt.errorbar(mean_ll, i, xerr=std_ll, fmt='-', color = COLORS[i % 10])
    
        plt.yticks(np.arange(len(names)), names)  # Set text labels and properties.


    plt.grid(False)
    if savefig:
        plt.savefig(savefig, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()

def visualize_best_rmse(log_liks, rmses, names, savefig = None, show = 0):
    '''
    Visualizes log-likelihoods for different models for all lasso-coefficients used in optimization.

    Args:
        log_liks (list)  : dictionary of log_likelihoods (all lasso-coefficients included)
        names (list)     : list of strings
        savefig (string) : path in which figure is saved, if None -> not saving anywhere
        show (bool)      : wheter figure is shown or just closed
    '''
    best_log_liks = best_coef(log_liks)

    plt.figure(figsize = (10,6))
    for i, rmse in enumerate(rmses):
        rmse_as_list = rmse[best_log_liks[i]]
        mean = np.mean(rmse_as_list)
        std = np.std(rmse_as_list)
        plt.plot(mean, i, '.', markersize = 12, color = COLORS[i % 10])
        plt.errorbar(mean, i, xerr=std, fmt='-', color = COLORS[i % 10])
    
        plt.yticks(np.arange(len(names)), names)  # Set text labels and properties.


    plt.grid(False)
    if savefig:
        plt.savefig(savefig, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()

def visualize_errors(errors, names, error_type, kernels, num_runs, fro = False, savefig = None, show = 0):

    '''
    errors (list)       : dictionary of errors (all lasso-coefficients included)
    names (list)        : list of strings
    error_type (string) : train/test 
    kernels (list)      : dictionary of parameters (all lasso-coefficients included)
    num_runs (list)      : number of random initializations
    fro (bool)          : whether log-likelihood is plotted against Frobenius norm
    savefig (string)    : path in which figure is saved, if None -> not saving anywhere
    show (bool)      : wheter figure is shown or just closed
    '''

    plt.figure(figsize = (10,6))
    for idx, model_errors in enumerate(errors):
        model_lassos = model_errors.keys()
        counter = 0
        errors_as_list = np.array([er for er in model_errors.values()])
        min_e_model = np.unravel_index(np.argmin(errors_as_list, axis=None), errors_as_list.shape)
        for lasso_idx, l in enumerate(model_lassos):
            if fro:
                frobenius = average_frobenius(kernels[idx][lasso_idx], num_runs[idx])
            min_e = np.argmin(model_errors[l])
            for jdx, e in enumerate(model_errors[l]):
                if fro:
                    x_value = frobenius
                    if (lasso_idx, jdx) == min_e_model:
                        plt.plot(x_value, e, '.', color = COLORS[idx], markersize = 15, label = names[idx])
                    else:
                        plt.plot(x_value, e, '.', color = COLORS[idx], alpha = 0.25, markersize = 5)
                else:
                    x_value = l
                    if jdx == min_e:
                        if counter == 0:
                            counter += 1
                            plt.plot(x_value, e, '.', color = COLORS[idx], markersize = 10, label = names[idx])
                        else:
                            plt.plot(x_value, e, '.', color = COLORS[idx], markersize = 10)
                    else:
                        plt.plot(x_value, e, '.', color = COLORS[idx], alpha = 0.2)

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

def visualize_errors_mean(errors, names, error_type, kernels, num_runs, fro = False, savefig = None, show = 0):

    '''
    errors (list)       : dictionary of errors (all lasso-coefficients included)
    names (list)        : list of strings
    error_type (string) : train/test 
    kernels (list)      : dictionary of parameters (all lasso-coefficients included)
    num_runs (list)      : number of random initializations
    fro (bool)          : whether log-likelihood is plotted against Frobenius norm
    savefig (string)    : path in which figure is saved, if None -> not saving anywhere
    show (bool)         : wheter figure is shown or just closed
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
            frobenius = average_frobenius(kernels[idx][lasso_idx], num_runs[idx])
            x_value = frobenius
            if lasso_idx == min_e_model:
                plt.plot(x_value, mean_errors[lasso_idx], '.', color = COLORS[idx], markersize = 15, label = names[idx])
                plt.errorbar(x_value, mean_errors[lasso_idx], yerr=std_errors[lasso_idx], fmt='-', color = COLORS[idx])
            else:
                plt.plot(x_value, mean_errors[lasso_idx], '.', color = COLORS[idx], alpha = 0.25, markersize = 5)

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

def visualize_loss_landscape(results, model, kernel, data, lasso, gradient, num_runs, savefig = None, show = 0):
    '''
    Visualize the loss landscape of the parameters using pca.

    Args:
        results (dict)   : whole dictionary after optimization (df)
        model (string)   : possible models in src.models.models
        kernel (string)  : possible kernels in src.models.kernels
        data (tuple)     : training data used during optimization
        lasso (float)    : lasso coefficient
        gradient (bool)  : wheter pca is calculated for gradient of parameters or just parameters
        num_runs (int)   : number of random initializations
        savefig (string) : path in which figure is saved, if None -> not saving anywhere
        show (bool)      : wheter figure is shown or just closed
    '''
    _, comp1, explained_variance, pca1 = pca_to_params(np.array(results['params'][lasso][0]), gradient)
    
    plt.figure(figsize = (8,8))
    maximum = -np.inf
    minimum = np.inf
    for i in range(num_runs):
        res, _, _, _ = pca_to_params(np.array(results['params'][lasso][i]), gradient)
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
    ll = loss_landscape(model, kernel, lasso, results['num_Z'], data, results['params'][lasso][0], results['variances'][lasso][0], results['likelihood_variances'][lasso][0], comp1, _range,_range, results['q_mu'], results['q_sqrt'], results['Z'], results['n'], results["rank"])
    im = plt.contourf(ll, extent=[minimum-1,maximum+1,minimum-1,maximum+1], levels=15, origin='lower')
    for i in range(num_runs):
        res, _, _, _ = pca_to_params(np.array(results['params'][lasso][i]), gradient)
        a, b = transform_M(pca1, res).T
        plt.plot(a,b, color = COLORS[i % len(COLORS)], alpha = 0.7, linewidth = 2.5)
        plt.plot(a[-1], b[-1],'.', color = COLORS[i % len(COLORS)], markersize = 20, alpha = 0.7)
    plt.xlabel(f'PCA component 1: {round(explained_variance[0]*100,2)}%')
    plt.ylabel(f'PCA component 2: {round(explained_variance[1]*100,2)}%')
    if savefig:
        plt.savefig(savefig, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()

def visualize_eigen_threshhold(eigen_values, names, threshhold = 0.001, savefig = None, show = 0):
    '''
    Visualize the eigen_values with a threshhold

    Args:
        eigen_values (list) : dictionary of eigen_values of different models
        names (list)        : list of strings
        threshhold (float)  : threshold for eigen values to be treated as zeros
        savefig (string)    : path in which figure is saved, if None -> not saving anywhere
        show (bool)         : wheter figure is shown or just closed
    '''
    plt.figure(figsize = (10,6))
    for idx, values in enumerate(eigen_values.values()):
        counter = 0
        for key,val in values.items():
            mean_val =  1/len(val)*len(np.where(np.array(val) > threshhold)[0])
            if counter == 0:
                counter += 1
                plt.plot(key, mean_val, '.', color = COLORS[idx], markersize = 15, label = names[idx])
            else:
                plt.plot(key, mean_val, '.', color = COLORS[idx], markersize = 15)
    
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
        