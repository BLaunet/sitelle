import matplotlib.pyplot as plt
import numpy as np

## Plot a 2D map
def plot_map(map, region=None, projection=None,
                colorbar=False, figsize=(7,7), facecolor='w', title='',
                xlims=None, ylims=None,
                **kwargs):
    """
    Helper function to plot 2D maps

    :param map: the 2d map to plot
    :param region: a region (obtained with np.where() for example) to plot on top of the image
    :param projection: a WCS projection to plot the map on
    :param colorbar: if True (non default), a color bar is associated to the plot
    :param figsize: size of the figure (x,y) in inches (Default (7,7))
    :param facecolor: background color, Default = 'w'
    :param title: (optional) a title for the plot
    :param xlims: limits to apply on the x axis (in pixels)
    :param ylims: limits to apply on the y axis (in pixels)
    :param kwargs: kwargs passed to imshow() function (e.g vmin, cmap etc..)
    """

    fig = plt.figure(figsize=figsize, facecolor=facecolor)
    if projection:
        ax = fig.add_subplot(111, projection=projection)
        ax.coords[0].set_major_formatter('hh:mm:ss.s')
        ax.coords[1].set_major_formatter('dd:mm:ss')
        ax.set_xlabel('RA')
        ax.set_ylabel('DEC')
    else:
        ax = fig.add_subplot(111)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
    im = ax.imshow(map.T, origin='bottom-left', **kwargs)
    ax.set_title(title)
    if region:
        if isinstance(region[0], tuple): #We have a list of regions:
            for r in region:
                mask = np.zeros_like(map)
                mask[r] = True
                ax.contour(mask.T, 1, colors='r')
        else:
            mask = np.zeros_like(map)
            mask[region] = True
            ax.contour(mask.T, 1, colors='r')

    if xlims:
        ax.set_xlim(xlims)
    if ylims:
        ax.set_ylim(ylims)
    if colorbar:
        fig.colorbar(im)
    if projection:
        plt.grid()
    plt.show()


def plot_spectra(*args, **kwargs):
    """
    Helper function to plot spectra

    :param args: args accepted by plt.plot()
                e.g : plot(ax1, spec1, 'r', ax2, spec2, 'g')
    :param xlims: limits to apply on the x axis (in cm-1)
    :param ylims: limits to apply on the y axis (in cm-1)
    :param label: list of labels of same size as number of plots
    :param figsize:
    :param facecolor:

    """
    xlims = kwargs.pop('xlims', None)
    ylims = kwargs.pop('ylims', None)
    label = kwargs.pop('label', '')
    fs = kwargs.pop('figsize', (12,6))
    fc = kwargs.pop('facecolor', 'w')
    fig, ax1 = plt.subplots(1,1, figsize=fs, facecolor=fc)
    lineObjects = ax1.plot(*args)
    if xlims:
        ax1.set_xlim(xlims)
    if ylims:
        ax1.set_ylim(ylims)



    #Cosmetics
    ax1.set_xlabel('Wavenumber [cm -1]')
    #Angstroms x axis
    ax2=ax1.twiny()
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks(ax1.get_xticks()[1:-1])
    ax2.set_xticklabels(["%.f" % (1e8/wn) for wn in ax1.get_xticks()[1:-1]])
    ax2.set_xlabel("Wavelength [Angstroms]")

    if label != '':
        if type(label) == str:
            ax1.legend(iter(lineObjects), [label])
        else:
            ax1.legend(iter(lineObjects), label)
    ax1.grid()
    plt.show()
