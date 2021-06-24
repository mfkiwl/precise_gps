from src.visuals.plot_kwargs import DEFAULT_KWARGS
import matplotlib.pyplot as plt 
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import math 

COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:olive", "tab:cyan", "tab:pink"]

class MidpointNormalize(colors.Normalize):
    """
    Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)
    e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
    """
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, min(vmin,0), max(0,vmax), clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))

def show_kernel(kernel, title, cols, aspect = "minmax", show_nums = -1, savefig = None):
    """
    Visualizes the kernel with colors.

    Args:
        kernel (numpy array) : D x D shaped numpy array or tensorflow tensor
        title (string)       : text for the title
        cols (list)          : D x 1 list for the covariates
        aspect (string)      : "minmax" -> colors are scaled from minimum to maximum (center is not zero)
                               "<anything else>" -> centered to zero
        show_nums (int)      : -1 -> dont show numeric values of the kernel
                                > -1 -> show values of the kernel
        save_fig (string)    : path in which the figure is saved
    
    Returns:
        Shows the figure and saves it to the specified path if <savefig not None>.
    """
    maximum = max(abs(np.min(kernel)), np.max(kernel))
    fig = plt.figure(figsize = (6,6))
    ax = plt.gca()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.15)
    if aspect != "minmax":
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
                              ha="center", va="center", color="black")

    plt.colorbar(im, cax=cax)
    if savefig:
        plt.savefig(savefig, bbox_inches='tight')
    plt.show()

def show_kernels(kernels, titles, cols, aspect = "own", show_nums = -1, savefig = None):
    """
    Visualizes a list of kernel with colors.

    Args:
        kernels (list of numpy arrays) : [D x D] list of numpy arrays or tensorflow tensors
        title (list)                   : list of the title texts
        cols (list)                    : D x 1 list for the covariates
        aspect (string)                : "own" -> centered to zero with respect to won values
                                         "<anything else>" -> centered to zero with respect to global values
        show_nums (int)                : -1 -> dont show numeric values of the kernel
                                         > -1 -> show values of the kernel
        savefig (string)               : path in which the figure is saved
    
    Returns:
        Shows the figure and saves it to the specified path if <save_fig == True>.
    """
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
        cax = divider.append_axes("right", size="5%", pad=0.15)
        if aspect == "own":
            pcm = ax.imshow(kernel,mcmap='bwr', norm = MidpointNormalize(midpoint = 0, vmin = -maximum, vmax = maximum))
        else:
            pcm = ax.imshow(kernel,mcmap='bwr', norm = MidpointNormalize(midpoint = 0, vmin = -global_maximum, vmax = global_maximum))
        if i == 0:
          ax.set_yticks(np.arange(len(cols)))
          ax.set_yticklabels(cols)
        num_rows = kernel.shape[0]
        # Loop over data dimensions and create text annotations.
        if show_nums > -1:
            for i in range(num_rows):
                for j in range(num_rows):
                    text = ax.text(j, i, np.around(kernel[i, j],show_nums),
                                ha="center", va="center", color="black")
        ax.set_title(titles[j*3+i])
        fig.colorbar(pcm, cax=cax)
    if savefig:
        plt.savefig(savefig, bbox_inches='tight')
    plt.show()

def visualize_mlls(mlls, names, savefig = None):
    plt.figure(figsize = (10,6))
    for idx, log_lik in enumerate(mlls):
        lassos = log_lik.keys()
        for l in lassos:
            counter = 0
            for mll in log_lik[l]:
                if not counter:
                    plt.plot(l, mll, color = COLORS[idx], label = names[idx], alpha = 0.8)
                else:
                    plt.plot(l, mll, color = COLORS[idx], alpha = 0.8)
                counter += 1
    
    plt.grid(True)
    plt.xlabel("Lasso coefficient")
    plt.ylabel("MLL (train)")
    plt.legend(prop = {'size': 14})
    if savefig:
        plt.savefig(savefig, bbox_inches='tight')
    plt.show()


def visualize_log_likelihood(log_liks, names, savefig = None):

    plt.figure(figsize = (10,6))
    for idx, log_lik in enumerate(log_liks):
        lassos = log_lik.keys()
        counter = 0
        for l in lassos:
            max_ll = np.argmax(log_lik[l])
            for jdx, ll in enumerate(log_lik[l]):
                if jdx == max_ll:
                    if counter == 0:
                      counter += 1
                      plt.plot(l, ll, '.', color = COLORS[idx], markersize = 10, label = names[idx])
                    else:
                      plt.plot(l, ll, '.', color = COLORS[idx], markersize = 10)
                else:
                    plt.plot(l, ll, '.', color = COLORS[idx], alpha = 0.2)

    plt.grid(True)
    plt.xlabel("Lasso coefficient")
    plt.ylabel("log-likelihood (test)")
    plt.legend(prop = {'size': 14})
    if savefig:
        plt.savefig(savefig, bbox_inches='tight')
    plt.show()

def visualize_errors(errors, names, lassos, error_type, savefig = None):

    plt.figure(figsize = (10,6))
    for idx, model_errors in enumerate(errors):
        model_lassos = lassos[idx]
        counter = 0
        for lasso_idx, ers in enumerate(model_errors):
            min_e = np.argmin(ers)
            for jdx, e in enumerate(ers):
                if jdx == min_e:
                    if counter == 0:
                      counter += 1
                      plt.plot(model_lassos[lasso_idx], e, '.', color = COLORS[idx], markersize = 10, label = names[idx])
                    else:
                      plt.plot(model_lassos[lasso_idx], e, '.', color = COLORS[idx], markersize = 10)
                else:
                    plt.plot(model_lassos[lasso_idx], e, '.', color = COLORS[idx], alpha = 0.2)

    plt.grid(True)
    plt.xlabel("Lasso coefficient")
    plt.ylabel(f"{error_type} error")
    plt.legend(prop = {'size': 14})
    if savefig:
        plt.savefig(savefig, bbox_inches='tight')
    plt.show()

def visualize_loss_landscape(params, gradient):
    """
    Visualize the loss landscape of the parameters using pca.

    Args:
        params (list)   : parameters through iteration steps
        gradient (bool) : wheter pca is calculated for gradient of parameters or just parameters
    """
    pass 

