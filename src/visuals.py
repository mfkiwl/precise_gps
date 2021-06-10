import matplotlib.pyplot as plt 
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

# set the colormap and centre the colorbar
class MidpointNormalize(colors.Normalize):
    """
    Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)

    e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
    """
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, min(vmin,0), max(0,vmax), clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))

def show_kernel(kernel, title, cols, save, aspect = "minmax", show_nums = -1, save_fig = False):
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
    ax.set_yticklabels(cols);
    ax.set_title(title)
    num_rows = kernel.shape[0]
    # Loop over data dimensions and create text annotations.
    if show_nums > -1:
        for i in range(num_rows):
            for j in range(num_rows):
                text = ax.text(j, i, np.around(kernel[i, j],show_nums),
                              ha="center", va="center", color="black")

    plt.colorbar(im, cax=cax)
    if save_fig:
        plt.savefig("{}.pdf".format(save), bbox_inches='tight')
    plt.show()