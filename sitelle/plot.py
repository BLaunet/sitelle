import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from orb.fit import Lines
from photutils import CircularAperture
from orb.utils.spectrum import line_shift
from sitelle.constants import SN2_LINES, SN3_LINES
import scipy

__all__ = ['customize_axes', 'make_wavenumber_axes', 'make_wavelength_axes', 'lines_pos', 'add_lines_label', 'add_colorbar', 'plot_map', 'plot_hist', 'plot_scatter', 'plot_density_scatter', 'plot_sources', 'plot_spectra', 'Interactive1DPlotter', 'SpectraPlotter']

def customize_axes(axes, **kwargs):
    """
    Wrapper meant to help the customization of :class:`~plt:matplotlib.axes.Axes`.

    Parameters
    ----------
    axes : :class:`~plt:matplotlib.axes.Axes`
        The matplotlib axes to modify
    kwargs : dict
        A dictionnary {key : value} to customize **axes**. It is used as ``axes.set_$key$ = value``
    """
    for k,v in kwargs.items():
        try:
            getattr(axes, 'set_%s'%k)(v)
        except AttributeError:
            raise AttributeError(str(type(axes))+' has no method set_%s'%k)
def make_wavenumber_axes(ax, **kwargs):
    """
    Create an axes suitable to plot spectra in wavenumber.

    Parameters
    ----------
    ax : :class:`~plt:matplotlib.axes.Axes`
        The matplotlib axes to modify
    kwargs : dict
        A dictionnary {key : value} to customize **axes**. It is used as ``axes.set_$key$ = value``
    See Also
    --------
    `customize_axes`
    """
    customize_axes(ax, **kwargs)
    _make_spectral_axes(ax, False)
def make_wavelength_axes(ax, **kwargs):
    """
    Create an axes suitable to plot spectra in wavelength.

    Parameters
    ----------
    ax : :class:`~plt:matplotlib.axes.Axes`
        The matplotlib axes to modify
    kwargs : dict
        A dictionnary {key : value} to customize **axes**. It is used as ``axes.set_$key$ = value``
    See Also
    --------
    `customize_axes`
    """
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
    """
    From a list of emission line names and a velocity, compute the position (in cm-1 or Angstroms) of their position.

    Parameters
    ----------
    lines_name : list of str
        Names of the lines (as defined `here <http://celeste.phy.ulaval.ca/orcs-doc/introduction.html#list-of-available-lines>`_).
    v : float
        The velocity
    wavenumber : bool, Default = False
        (Optional) If True, the position in cm-1 is returned, else in Angstroms.
    See Also
    --------
    **SN2_LINES** and **SN3_LINES** in :mod:`sitelle.constants` for line names
    """
    if wavenumber is True:
        return [(wn*(1+v/3e5)) for wn in Lines().get_line_cm1(lines_name)]

    else:
        return [1e8/(wn*(1-v/3e5)) for wn in Lines().get_line_cm1(lines_name)]

def add_lines_label(ax, filter, velocity, wavenumber=True,offset=15, **kwargs):
    """
    Decorates an axis with lines labels.
    It puts a tick at a given position of the top axes of the plot, and displays the name of the line above.

    Parameters
    ----------
    ax : :class:`~plt:matplotlib.axes.Axes`
        The matplotlib axes to modify
    filter : string
        'SN2' or 'SN3'.
    velocity : list of float
        The velocities at wich we want to display the lines.
    wavenumber : bool, Default True
        (Optional) If True, compute the position in cm-1, else' in Angstroms.
    offset : integer
        (Optioanl) Pixel offset of the tick. Default = 15
    kwargs : dict
        Additional keyword arguments accepted by :func:`plt:matplotlib.axes.Axes.annotate`
    """
    if type(velocity) != tuple and type(velocity) != list:
        velocity = [velocity]
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    if filter == 'SN2':
        lines_names = SN2_LINES
    elif filter == 'SN3':
        lines_names = SN3_LINES
    rest_lines = Lines().get_line_cm1(lines_names)
    pos = rest_lines + line_shift(velocity[0], rest_lines, wavenumber)
    for i, name in enumerate(lines_names) :
        # ax.annotate(name, ((pos[i]-offset-xmin)/(xmax-xmin), 0.99), xycoords='axes fraction', rotation=90.)
        ax.annotate(name, (pos[i]-offset, 0.99*ymax), rotation=90.,  annotation_clip = False, **kwargs)
        #ax.text((pos-10-xmin)/(xmax-xmin), 0.94, name, rotation = 45.)

        color = iter(['k', 'r', 'g'])
        for v in velocity:
            pos_v = rest_lines + line_shift(v, rest_lines, wavenumber)
            ax.axvline(pos_v[i], ymin=0.97, c=next(color), ls='-', lw=1.)

## Plot a 2D map
from mpl_toolkits import axes_grid1

def add_colorbar(im, ax, fig, aspect=20, pad_fraction=0.5, **kwargs):
    """
    DEPRECATED
    Add a vertical color bar to an image plot.
    """
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = ax
    cax = divider.append_axes("right", size=width, pad=pad)

    return fig.colorbar(im, cax=cax, **kwargs)

def plot_map(data, ax=None, region=None, projection=None,
                colorbar=False,
                xlims=None, ylims=None,
                **kwargs):
    """
    Helper function to plot 2D maps

    Parameters
    ----------
    data : 2D :class:`~numpy:numpy.ndarray`
        The 2D map to be plotted
    ax : :class:`~plt:matplotlib.axes.Axes`
        (Optional) The matplotlib axes on which to plot. If None, a new one will be created.
    region : tuple, Optional
        a region in pixel (obtained with :func:`numpy:numpy.where` for example) to plot on top of the image
    projection : :class:`astropy:astropy.wcs.WCS`, Optional
        a WCS projection to plot the map on
    colorbar : bool, Default = False
        if True , a color bar is associated to the plot
    pmin : integer between 0 and 100
        (Optional) if passed, vmin set to ``np.nanpercentile(data, pmin)``
    pmax : integer between 0 and 100
        (Optional) if passed, vmax set to ``np.nanpercentile(data, pmax)``
    kwargs: dict
        Additional keyword arguments passed to :func:`plt:matplotlib.pyplot.imshow` function (e.g vmin, cmap etc..)

    Returns
    -------
    fig : :class:`~plt:matplotlib.figure.Figure`
        The figure containing the plot.
    ax : :class:`~plt:matplotlib.axes.Axes`
        The axes containing the plot.
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
        #   add_colorbar(im, ax, fig)
        fig.colorbar(im, ax=ax)
    if projection:
        ax.grid()
    return fig, ax

def plot_scatter(x,y,ax=None, **kwargs):
    """
    Helper function to plot scattered data

    Parameters
    ----------
    x : 1D :class:`~numpy:numpy.ndarray`
        X coordinates of the data
    y : 1D :class:`~numpy:numpy.ndarray`
        Y coordinates of the data
    ax : :class:`~plt:matplotlib.axes.Axes`
        (Optional) The matplotlib axes on which to plot. If None, a new one will be created.
    label : str
        (Optional) labels of the data, to be displayed in the legend.
    color : str, Default = 'red'
        color of the markers
    marker : str, Default = '+'
        Style of marker to use
    kwargs : dict
        any keyword argument accepted by :func:`plt:matplotlib.pyplot.scatter`

    Returns
    -------
    fig : :class:`~plt:matplotlib.figure.Figure`
        The figure containing the plot.
    ax : :class:`~plt:matplotlib.axes.Axes`
        The axes containing the plot.
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
    Helper function to plot a spectrum.

    Parameters
    ----------
    axis : 1D :class:`~numpy:numpy.ndarray`
        the spectrum axis
    spectrum : 1D :class:`~numpy:numpy.ndarray`
        the spectrum to plot
    ax : :class:`~plt:matplotlib.axes.Axes`
        (Optional) The matplotlib axes on which to plot. If None, a new one will be created.
    wavenumber : bool, Default True
        (Optional) If True, the axis is in wavenumber ([cm-1]) else in wavelength ([Angstroms])
    label : str
        (Optional) labels of the plot, to be displayed in the legend.
    build_ax : bool, Default = False
        (Optional) Used to force the customization of the axes. Should be False if we plot a new spectrum on an existing plot.
    kwargs: dict
        Additional keyword arguments passed to :func:`plt:matplotlib.pyplot.plot` function

    Returns
    -------
    fig : :class:`~plt:matplotlib.figure.Figure`
        The figure containing the plot.
    ax : :class:`~plt:matplotlib.axes.Axes`
        The axes containing the plot.
    """
    if ax is None: #No ax has been given : we have to create a new one
        fig,ax = plt.subplots()
        build_ax = True
    else:
        fig = ax.get_figure()
        build_ax = kwargs.pop('build_ax', False)
    label = kwargs.pop('label', None)
    ax.plot(axis, spectrum, label = label, **kwargs)
    if label is not None:
        ax.legend()

    if build_ax:
        if wavenumber:
            make_wavenumber_axes(ax)
        else:
            make_wavelength_axes(ax)
    return fig, ax

def plot_hist(map, ax=None, log = False, pmin = None, pmax=None, step=True,**kwargs):
    """
    Helper function to plot an histogram.
    Especially helpful when dealing with 2d values (map) containing NaN (they are excluded from the analysis)

    Parameters
    ----------
    map : 1D or 2D :class:`~numpy:numpy.ndarray`
        The data from which the histogram is taken
    ax : :class:`~plt:matplotlib.axes.Axes`
        (Optional) The matplotlib axes on which to plot. If None, a new one will be created.
    log
    pmin : integer between 0 and 100
        (Optional) if passed, vmin set to ``np.nanpercentile(data, pmin)``
    pmax : integer between 0 and 100
        (Optional) if passed, vmax set to ``np.nanpercentile(data, pmax)``
    step : bool, Default = True
        (Optional) If True, displays a step histogram, else a bar histogram.
    label : str
        (Optional) labels of the data, to be displayed in the legend.
    cumulative : bool, Default = False
        (Optional) If True, the culmulative histogram is computed and plotted.
    kwargs: dict
        Additional keyword arguments passed to :func:`numpy:numpy.histogram` function

    Returns
    -------
    fig : :class:`~plt:matplotlib.figure.Figure`
        The figure containing the plot.
    ax : :class:`~plt:matplotlib.axes.Axes`
        The axes containing the plot.
    """
    label = kwargs.pop('label', None)
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
    cumulative = kwargs.pop('cumulative', False)
    h = np.histogram(_map[~np.isnan(_map)], **kwargs)
    X = h[1][:-1]
    if cumulative:
        Y=np.cumsum(h[0])
    else:
        Y = h[0]
    if step:
        ax.step(X,Y, where='post', label=label)
    else:
        ax.bar(X, Y, align='edge', width = h[1][1]-h[1][0], log=log, label=label)
    ax.set_title('Median : %.3f, Std : %.3f'%(np.nanmedian(_map), np.nanstd(_map)))
    if label is not None:
        ax.legend()
    return f,ax

def plot_sources(sources, ax, **kwargs):
    """
    DEPRECATED
    Helper function to plot sources nicely over a map
    WARNING : sources should respect astropy convention : x and y reversed

    Parameters
    ----------
    sources : :class:`~pandas:pandas.DataFrame`
        a Pandas Dataframe containing at least columns 'xcentroid' and 'ycentroid'
    ax : :class:`~plt:matplotlib.axes.Axes`
        The matplotlib axes on which to plot (should be containing the map on top of which we are going to plot the sources)
    kwargs : dict
        any keyword argument accepted by :func:`photutils:photutils.CircularAperture.plot`
    Returns
    -------
    fig : :class:`~plt:matplotlib.figure.Figure`
        The figure containing the plot.
    ax : :class:`~plt:matplotlib.axes.Axes`
        The axes containing the plot.
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

def plot_density_scatter(xdat,ydat,xlims=None, ylims=None, ax=None, bins=[100,100], colorbar=True):
    """
    Helper to plot a density scatter plot.

    Parameters
    ----------
    xdat : 1D :class:`~numpy:numpy.ndarray`
        X coordinates of the data
    ydat : 1D :class:`~numpy:numpy.ndarray`
        Y coordinates of the data
    xlims : tuple of float
        (Optional) Limits to use on ``xdat``
    ylims : tuple of float
        (Optional) Limits to use on ``ydat``
    ax : :class:`~plt:matplotlib.axes.Axes`
        (Optional) The matplotlib axes on which to plot. If None, a new one will be created.
    bins : list of int
        (Optional). The binning to use on each axis. Default = [100,100]
    colorbar : bool, Default = True
        (Optional) If True, the colorbar is displayed.
    Returns
    -------
    fig : :class:`~plt:matplotlib.figure.Figure`
        The figure containing the plot.
    ax : :class:`~plt:matplotlib.axes.Axes`
        The axes containing the plot.

    Example
    -------
    xlims = [-18.5, -15]
    ylims = [-18.5, -15]
    f,ax = density_scatter(np.log10(X), np.log10(Y), xlims=xlims, ylims=ylims)
    ax.plot(np.linspace(-20, -15), np.linspace(-20, -15), c='k')
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    """
    if xlims is None:
        xlims = [np.nanmin(xdat), np.nanmax(xdat)]
    if ylims is None:
        ylims = [np.nanmin(ydat), np.nanmax(ydat)]
    xyrange = [xlims, ylims]
    thresh = 3  #density threshold

    # histogram the data
    hh, locx, locy = scipy.histogram2d(xdat, ydat, range=xyrange, bins=bins)
    posx = np.digitize(xdat, locx)
    posy = np.digitize(ydat, locy)
    #select points within the histogram
    ind = (posx > 0) & (posx <= bins[0]) & (posy > 0) & (posy <= bins[1])
    hhsub = hh[posx[ind] - 1, posy[ind] - 1] # values of the histogram where the points are
    xdat1 = xdat[ind][hhsub < thresh] # low density points
    ydat1 = ydat[ind][hhsub < thresh]
    hh[hh < thresh] = np.nan # fill the areas with low density by NaNs

    if ax is None:
        f,ax = plt.subplots()
    else:
        f = ax.get_figure()
    im = ax.imshow(np.flipud(hh.T),cmap='jet', interpolation='none', origin='upper', extent=np.array(xyrange).flatten())
    x0,x1 = xlims
    y0,y1 = ylims
    ax.set_aspect((x1-x0)/(y1-y0))
    if colorbar:
        #   add_colorbar(im, ax, fig)
        f.colorbar(im, ax=ax)
    ax.plot(xdat1, ydat1, '+',color='darkblue')
    return f,ax

class Interactive1DPlotter:
    """
    An interactive 1D Plotter.
    A map is displayed, and clicking on a pixel displays the corresponding features contained inside.

    Attributes
    ----------
    axes : :class:`~plt:matplotlib.axes.Axes`
        The matplotlib axes on which to plot. If None, a new one will be created.
    axis : 1D :class:`~numpy:numpy.ndarray`
        The spectra axis
    cube : 3D :class:`~numpy:numpy.ndarray`
        The datacube where the features are stored
    args :
        Arguments to be passed to :func:`plt:matplotlib.pyplot.plot` when plotting 1D data.
    kwargs : dict
        Keyword arguments to be passed to :func:`plt:matplotlib.pyplot.plot` when plotting 1D data.
    figure : :class:`~plt:matplotlib.figure.Figure`
        The figure on which we listen for event (the 2D map)
    """
    def __init__(self, axes, cube_axis, cube, *args, **kwargs):
        """
        Parameters
        ----------
        axes : :class:`~plt:matplotlib.axes.Axes`
            (Optional) The matplotlib axes on which to plot. If None, a new one will be created.
        axis : 1D :class:`~numpy:numpy.ndarray`
            The spectra axis
        cube : 3D :class:`~numpy:numpy.ndarray`
            The datacube where the features are stored
        args :
            Arguments to be passed to :func:`plt:matplotlib.pyplot.plot` when plotting 1D data.
        kwargs : dict
            Keyword arguments to be passed to :func:`plt:matplotlib.pyplot.plot` when plotting 1D data.
        """
        self.axes = axes
        self.cube_axis = cube_axis
        self.cube = cube
        self.args = args
        self.kwargs = kwargs

        self.patch = None
        self.annotation = None
        self.artist = []
    def connect(self, figure):
        """
        Connect the 1D plotter to a given figure (the 2D map), to listen for specific events (press and motion)

        Parameters
        ----------
        figure : :class:`~plt:matplotlib.figure.Figure`
            The figure containing the plot.
        """
        self.figure = figure
        self.cidpress = self.figure.canvas.mpl_connect(
            'button_press_event', self.on_press)
        self.cidmotion = self.figure.canvas.mpl_connect(
            'motion_notify_event', self.on_motion)

    def on_press(self, event):
        """
        Defines the behavior for a press event.
        Here, we display the data contained in the z dimesion of the cube at the [x,y] position corresponding to the press.
        """
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
        """
        Defines the behavior for a motion event.
        Here, we display a rectangular patch around the pixel above which the mouse stands, as well as the value of the pixel in the 2D map.
        """
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
    """
    An interactive Spectra Plotter.
    A map is displayed, and clicking on a pixel displays the corresponding spectra contained inside, as well as the fit and the residual.
    Its an augmentation of :class:`Interactive1DPlotter`.

    Attributes
    ----------
    plot_axis : :class:`~plt:matplotlib.axes.Axes`
        The matplotlib axes on which to plot. If None, a new one will be created.
    axis : 1D :class:`~numpy:numpy.ndarray`
        The spectra axis
    original_cube : 3D :class:`~numpy:numpy.ndarray`
        The datacube where the original spectra are stored
    fit_cube : 3D :class:`~numpy:numpy.ndarray`
        The datacube where the fitted spectra are stored
    residual : bool
        If True, residuals will be displayed
    projection : :class:`astropy:astropy.wcs.WCS`, Optional
        a WCS projection to plot the map on
    args :
        Arguments to be passed to :func:`plt:matplotlib.pyplot.plot` when plotting 1D data.
    kwargs : dict
        Keyword arguments to be passed to :func:`plt:matplotlib.pyplot.plot` when plotting 1D data.
    figure : :class:`~plt:matplotlib.figure.Figure`
        The figure on which we listen for event (the 2D map)
    Note
    ----
    The code is far from optimized, and is in parts outdated (not updated since v0.1)

    """
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
        """
        Connect the 1D plotter to a given figure (the 2D map), to listen for specific events (press and motion)

        Parameters
        ----------
        figure : :class:`~plt:matplotlib.figure.Figure`
            The figure containing the plot.
        """
        self.figure = figure
        self.cidpress = self.figure.canvas.mpl_connect(
            'button_press_event', self.on_press)
        self.cidmotion = self.figure.canvas.mpl_connect(
            'motion_notify_event', self.on_motion)

    def on_press(self, event):
        """
        Defines the behavior for a press event.
        Here, we display the data contained in the z dimesion of the cube at the [x,y] position corresponding to the press.
        """
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
        """
        Defines the behavior for a motion event.
        Here, we display a rectangular patch around the pixel above which the mouse stands, the value of the pixel in the 2D map as well as the physical position (RA/DEC) if ``projection`` has been provided.
        """
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
        """
        Disconnect all the stored connection ids
        """
        self.figure.canvas.mpl_disconnect(self.cidpress)
