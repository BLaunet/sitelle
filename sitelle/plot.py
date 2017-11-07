import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

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
    return fig, ax


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
    :param vlines: list of vertical lines x position
    :param annotations: list of annonations to add to the plot. Each annotation should be a tuple (text, position)
    :param legendloc: keyword to locate the lmegend (default = 'best')
    :param title: title to add
    :param ax: ax where to plot
    :param return: returns the created figure
    """
    xlims = kwargs.pop('xlims', None)
    ylims = kwargs.pop('ylims', None)
    label = kwargs.pop('label', '')
    annotations = kwargs.pop('annotations', None)
    vlines = kwargs.pop('vlines', None)
    legendloc = kwargs.pop('legendloc', 'best')
    title = kwargs.pop('title', '')

    fs = kwargs.pop('figsize', (12,6))
    fc = kwargs.pop('facecolor', 'w')

    ax1 = kwargs.pop('ax', None)
    return_figure = kwargs.pop('return_figure', True)
    if ax1 is None:
        fig, ax1 = plt.subplots(1,1, figsize=fs, facecolor=fc)
    else:
        return_figure = False

    lineObjects = ax1.plot(*args)
    if xlims:
        ax1.set_xlim(xlims)
    if ylims:
        ax1.set_ylim(ylims)
    ymin, ymax = ax1.get_ylim()
    if annotations:
        for an in annotations:
            try:
                scale = an[2]
            except IndexError:
                scale = 0.95
            ax1.annotate(an[0], xy=(an[1], ymax*scale))
    if vlines:
        for line in vlines:
            ax1.axvline(line, ymin=0.98,c='k', ls='-', lw=1.)

    #Cosmetics
    ax1.set_xlabel('Wavenumber [cm$^{-1}$]')
    ax1.set_ylabel('Flux [erg/cm$^2$/s/A]')
    #Angstroms x axis
    ax2=ax1.twiny()
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks(ax1.get_xticks()[1:-1])
    ax2.set_xticklabels(["%.f" % (1e8/wn) for wn in ax1.get_xticks()[1:-1]])
    ax2.set_xlabel("Wavelength [Angstroms]")

    if label != '':
        if type(label) == str:
            ax1.legend(iter(lineObjects), [label], loc=legendloc)
        else:
            ax1.legend(iter(lineObjects), label, loc=legendloc)
    ax1.grid()
    ax1.set_title(title,position=(0.1,1.1))
    if return_figure:
        return fig

class InteractivePlotter:
    def __init__(self, axis, ordinate_cube, plot_axis, title=''):
        self.axis = axis
        self.ordinate_cube = ordinate_cube
        self.plot_axis = plot_axis

        self.patch = None
        self.annotation = None
        self.title = title

    def connect(self, figure):
        'connect to all the events we need'
        self.figure = figure
        self.cidpress = self.figure.canvas.mpl_connect(
            'button_press_event', self.on_press)
        self.cidmotion = self.figure.canvas.mpl_connect(
            'motion_notify_event', self.on_motion)

    def on_press(self, event):
        'on button press we will see if the mouse is over us and store some data'
        if event.inaxes is None: return
        x, y = map(int, map(round, (event.xdata, event.ydata)))
        self.plot_axis.clear()
        self.plot_axis.get_figure().show()


        self.plot_axis.plot(self.axis, self.ordinate_cube[x,y])
        self.plot_axis.set_title(self.title)
        self.plot_axis.get_figure().show()

    def on_motion(self, event):
        if event.inaxes is None: return

        x, y = map(int,map(round, (event.xdata, event.ydata)))

        if self.patch is None:
            ax = self.figure.axes[0]
            self.patch = patches.Rectangle((x-0.5,y-0.5),1,1,fill=False, ec='w')
            ax.add_patch(self.patch)
            data = ax.images[0].get_array()
            self.annotation = ax.annotate('V = %f km/s'%(data.T[x,y]) ,(0,1.1), xycoords='axes fraction' )
        elif self.patch.xy == (x-0.5,y-0.5): return
        else:
            self.patch.remove()
            self.patch = None
            self.annotation.remove()


        self.figure.canvas.draw()


class SpectraPlotter:
    def __init__(self, axis, original_cube, fit_cube, plot_axis, projection=None, residual = True, **kwargs):
        self.axis = axis
        self.original_cube = original_cube
        self.fit_cube = fit_cube
        self.residual = residual
        self.plot_axis = plot_axis
        self.projection = projection
        self.kwargs = kwargs

        self.patch = None
        self.annotation = None
        self.annotation2 = None
    def connect(self, figure):
        'connect to all the events we need'
        self.figure = figure
        self.cidpress = self.figure.canvas.mpl_connect(
            'button_press_event', self.on_press)
        self.cidmotion = self.figure.canvas.mpl_connect(
            'motion_notify_event', self.on_motion)

    def on_press(self, event):
        'on button press we will see if the mouse is over us and store some data'
        if event.inaxes is None: return
        x, y = map(int, map(round, (event.xdata, event.ydata)))
        self.plot_axis.clear()
        self.plot_axis.get_figure().show()

        args = (self.axis, self.original_cube[x,y,:],self.axis, self.fit_cube[x,y,:])
        if self.residual:
            args += (self.axis, self.original_cube[x,y,:]-self.fit_cube[x,y,:])
        plot_spectra(*args, ax = self.plot_axis, **self.kwargs)
        self.plot_axis.get_figure().show()

    def on_motion(self, event):
        if event.inaxes is None: return

        x, y = map(int,map(round, (event.xdata, event.ydata)))

        if self.patch is None:
            ax = self.figure.axes[0]
            self.patch = patches.Rectangle((x-0.5,y-0.5),1,1,fill=False, ec='w')
            ax.add_patch(self.patch)
            data = ax.images[0].get_array()
            self.annotation = ax.annotate('V = %f km/s'%(data.T[x,y]) ,(0,1.1), xycoords='axes fraction' )
            if self.projection is not None:
                ra, dec = self.projection.pix2world((x*48+24, y*48+24))[0]
                self.annotation2 = ax.annotate('RA = %s \nDEC = %s'%(ra,dec) ,(0,1.15), xycoords='axes fraction' )
        elif self.patch.xy == (x-0.5,y-0.5): return
        else:
            self.patch.remove()
            self.patch = None
            self.annotation.remove()
            if self.projection is not None:
                self.annotation2.remove()




        self.figure.canvas.draw()

    def disconnect(self):
        'disconnect all the stored connection ids'
        self.figure.canvas.mpl_disconnect(self.cidpress)
