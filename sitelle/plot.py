import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from orb.fit import Lines
from photutils import CircularAperture

def customize_axes(axes, **kwargs):
    for k,v in kwargs.items():
        try:
            getattr(axes, 'set_%s'%k)(v)
        except AttributeError:
            raise AttributeError(str(type(axes))+' has no method set_%s'%k)
def make_wavenumber_axes(ax, **kwargs):
    customize_axes(ax, **kwargs)
    _make_spectral_axes(ax, False)
def make_wavelength_axes(ax, **kwargs):
    customize_axes(ax, **kwargs)
    _make_spectral_axes(ax, True)

def _make_spectral_axes(ax, wavelength):
    wl_label = 'Wavelength [Angstroms]'
    wn_label = "Wavenumber [cm$^{-1}$]"
    if wavelength:
        bottom_label = wl_label
        top_label = wn_label
    else:
        bottom_label = wn_label
        top_label = wl_label
    customize_axes(ax, xlabel= bottom_label,
                       ylabel='Flux [erg/cm$^2$/s/A]')
    #Angstroms x axis
    ax2=ax.twiny()
    customize_axes(ax2, xlim=ax.get_xlim(),
                        xticks = ax.get_xticks()[1:-1],
                        xticklabels = ["%.f" % (1e8/wn) for wn in ax.get_xticks()[1:-1]],
                        xlabel = top_label)

def lines_pos(lines_name, v, wavenumber=False):
    if wavenumber is True:
        return [(wn*(1+v/3e5)) for wn in Lines().get_line_cm1(lines_name)]

    else:
        return [1e8/(wn*(1-v/3e5)) for wn in Lines().get_line_cm1(lines_name)]

def add_lines_label(ax, filter, velocity, wavenumber=False,offset=15):
    if type(velocity) != tuple and type(velocity) != list:
        velocity = [velocity]
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    if filter == 'SN2':
        lines_names = ['Hbeta','[OIII]4959', '[OIII]5007']
    elif filter == 'SN3':
        lines_names = ['[NII]6548', 'Halpha', '[NII]6583', '[SII]6716', '[SII]6731']
    pos = lines_pos(lines_names, velocity[0], wavenumber)

    for i, name in enumerate(lines_names) :
        #ax.annotate(name, ((pos[i]-offset-xmin)/(xmax-xmin), 0.99), xycoords='axes fraction', rotation=90.)
        ax.annotate(name, (pos[i]-offset, 0.99*ymax), rotation=90.,  annotation_clip = False)
        #ax.text((pos-10-xmin)/(xmax-xmin), 0.94, name, rotation = 45.)

        color = iter(['k', 'r', 'g'])
        for v in velocity:
            ax.axvline(lines_pos(lines_names, v, wavenumber)[i], ymin=0.95, c=next(color), ls='-', lw=1.)

## Plot a 2D map
def plot_map(data, ax=None, region=None, projection=None,
                colorbar=False,
                xlims=None, ylims=None,
                **kwargs):
    """
    Helper function to plot 2D maps

    :param map: the 2d map to plot
    :param region: a region (obtained with np.where() for example) to plot on top of the image
    :param projection: a WCS projection to plot the map on
    :param colorbar: if True (non default), a color bar is associated to the plot
    :param pmin: (Optional) if passed, vmin set to np.nanpercentile(data, pmin)
    :param pmax: (Optional) if passed, vmax set to np.nanpercentile(data, pmax)
    :param kwargs: kwargs passed to imshow() function (e.g vmin, cmap etc..)
    """
    if ax is None: #No ax has been given : we have to create a new one
        if projection:
            fig, ax = plt.subplots(subplot_kw={'projection':projection})
        else:
            fig,ax = plt.subplots()
    else:
        fig = ax.get_figure()
        if projection: #We have to remove the current ax and replace it with one handling the projection
            pos = ax.get_position()
            ax.remove()
            ax = fig.add_axes(pos, projection=projection)

    if projection:
        ax.coords[0].set_major_formatter('hh:mm:ss.s')
        ax.coords[1].set_major_formatter('dd:mm:ss')
        ax.set_xlabel('RA')
        ax.set_ylabel('DEC')
    else:
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

    pmin = kwargs.pop('pmin', None)
    pmax = kwargs.pop('pmax', None)
    if pmin is not None or pmax is not None:
        if pmin is None:
            pmin = 10
        if pmax is None:
            pmax = 90
        kwargs['vmin'] = np.nanpercentile(data, pmin)
        kwargs['vmax'] = np.nanpercentile(data, pmax)

    cmap = kwargs.pop('cmap', 'gray_r')
    origin = kwargs.pop('origin', 'lower')
    im = ax.imshow(data.T, origin=origin, cmap=cmap, **kwargs)

    if region:
        if isinstance(region[0], tuple): #We have a list of regions:
            for r in region:
                mask = np.zeros_like(data)
                mask[r] = True
                ax.contour(mask.T, 1, colors='r')
        else:
            mask = np.zeros_like(data)
            mask[region] = True
            ax.contour(mask.T, 1, colors='r')
    if colorbar:
        fig.colorbar(im)
    if projection:
        ax.grid()
    return fig, ax

def plot_scatter(x,y,ax=None, **kwargs):
    """
    Helper function to plot scattered data
    :param x: x coordinates
    :param y: y coordinates
    :param ax: the matplotlib ax on which to plot. If None, a new one is created
    :param label: (Optional) labels of the data
    :param color: (Default 'red') color of the markers
    :param marker: (Default '+') Marker to use
    :param **kwargs: any keyword argument accepted by plt.scatter()
    :return fig, ax:
    """
    if ax is None: #No ax has been given : we have to create a new one
        fig,ax = plt.subplots()
    else:
        fig = ax.get_figure()
    label = kwargs.pop('label', None)
    color = kwargs.pop('c', 'red')
    color = kwargs.pop('color', color)
    marker = kwargs.pop('marker', '+')
    ax.scatter(x,y,label = label, color = color, marker = marker, **kwargs)
    if label is not None:
        ax.legend()
    return fig, ax
def plot_spectra(axis, spectrum, ax=None, wavenumber=True, **kwargs):
    """
    Helper function to plot a spectrum
    :param axis: the spectrum axis
    :param spectrum: the spectrum itself
    :param label: (Optional) label of the plot
    :param wavenumber: (Default True) if the axis is in wavenumber ([cm-1]) or wavelength ([Angstroms])
    :param ax: ax where to plot. if none, a new one is created
    :param kwargs: Any kwargs accepted by plt.plot()

    """
    if ax is None: #No ax has been given : we have to create a new one
        fig,ax = plt.subplots()
    else:
        fig = ax.get_figure()
    label = kwargs.pop('label', None)
    ax.plot(axis, spectrum, label = label, **kwargs)
    if label is not None:
        ax.legend()

    if wavenumber:
        make_wavenumber_axes(ax)
    else:
        make_wavelength_axes(ax)
    return fig, ax

def plot_hist(map, ax=None, log = False, pmin = None, pmax=None, **kwargs):
    """
    Helper function to plot an histogram.
    Especially helpful when dealing with 2d values (map) containing NaN (they are excluded from the analysis)
    :param map: the data from which the histogram is taken
    :param ax: ax where to plot. If None, a new one is created
    :param pmin: (Optional) if passed, data is cut to np.nanpercentile(map, pmin)
    :param pmax: (Optional) if passed, vmin set to np.nanpercentile(map, pmax)
    :param kwargs: Any kwargs accepted by np.histogram
    """
    _map = np.copy(map)
    if type(map) is np.ma.MaskedArray:
        _map = _map[~map.mask]
    if ax is None:
        f,ax = plt.subplots()
    else:
        f = ax.get_figure()
    if pmin is not None:
        min = np.nanpercentile(_map, pmin)
    else:
        min = np.nanmin(_map)
    if pmax is not None:
        max = np.nanpercentile(_map, pmax)
    else:
        max = np.nanmax(_map)
    _map[_map > max] = np.nan
    _map[_map < min] = np.nan
    h = np.histogram(_map[~np.isnan(_map)], **kwargs)
    X = h[1][:-1]
    Y = h[0]
    ax.bar(h[1][:-1], h[0], align='edge', width = h[1][1]-h[1][0], log=log)
    ax.set_title('Median : %.2e, Std : %.2e'%(np.nanmedian(_map), np.nanstd(_map)))
    return f,ax

def plot_sources(sources, ax, **kwargs):
    """
    Helper function to plot sources nicely over a map
    WARNING : sources should respect astropy convention : x and y reversed
    :param sources: a Pandas Dataframe containing at least columns 'xcentroid' and 'ycentroid'
    :param ax: the ax on which to plot (should be containing the map on top of which we are going to plot the sources)
    :param kwargs: any kwargs accepted by CircularAperture.plot
    """
    f = ax.get_figure()
    positions=(sources['ycentroid'], sources['xcentroid'])
    apertures = CircularAperture(positions, r=4.)
    color = kwargs.pop('c', 'red')
    color = kwargs.pop('color', color)
    lw = kwargs.pop('lw', 1.5)
    alpha = kwargs.pop('alpha', 0.5)
    apertures.plot(color=color, lw=lw, alpha=alpha, ax=ax, **kwargs)
    return f,ax

class Interactive1DPlotter:
    def __init__(self, axes, cube_axis, cube, *args, **kwargs):
        self.axes = axes
        self.cube_axis = cube_axis
        self.cube = cube
        self.args = args
        self.kwargs = kwargs

        self.patch = None
        self.annotation = None
        self.artist = []
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
        if self.artist is not []:
            for a in self.artist:
                a.remove()
        self.axes.get_figure().show()
        self.artist = self.axes.plot(self.cube_axis, self.cube[x,y,...], *self.args, **self.kwargs)
        # for ax in self.figure.get_axes():
        #     ax.relim()
        #     ax.autoscale_view()
        self.axes.relim()
        self.axes.autoscale_view()
        self.axes.legend()
        self.axes.get_figure().show()

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
