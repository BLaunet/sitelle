import numpy as np
from sitelle.region import centered_square_region
from sitelle.plot import *
from photutils import CircularAperture, CircularAnnulus, EllipticalAperture
from photutils import DAOStarFinder, IRAFStarFinder
from photutils import find_peaks, detect_sources, deblend_sources, source_properties
from scipy.interpolate import UnivariateSpline
from astropy.stats import sigma_clipped_stats
from orb.astrometry import Astrometry
from astropy.convolution import Gaussian2DKernel, Box2DKernel
from astropy.table import Table
from orb.utils import vector
from numbers import Number
from orcs.utils import fit_lines_in_spectrum
from astropy.units.quantity import Quantity
import warnings
import logging

def mask_sources(sources, annulus):
    x,y = sources['ycentroid'], sources['xcentroid']
    return sources[annulus[np.round(x).astype(int), np.round(y).astype(int)].astype(bool)]

def filter_frame(frame, annulus, val=0):
    _frame = np.copy(frame)
    _frame[np.where(annulus == 0)] = val
    return _frame

def extract_max_frame(x,y, spectral_cube, id_max_detection_frame):
    """
    For a given position, extracts the frames around the maxium of detection in the cube and returns the sum
    :param x: abscissa of the source
    :param y: ordinate of the source
    :param spectral_cube: SpectralCube instance where we are looking at the source
    :param id_max_detection_frame: the id of the max frame, or the detection_pos_frame
    :return: a cutout of the sum of the frames around the max of detection
    """
    if isinstance(id_max_detection_frame, Number):
        iframe = int(id_max_detection_frame)
    else:
        raise TypeError('Non valid type for id_max_detection_frame : %s'%type(id_max_detection_frame))
    data = spectral_cube.get_data(x-10, x+11, y-10,y+11, iframe-2,iframe+3)
    return np.sum(data, axis=2)

def estimate_local_background(x,y,cube, small_bin = 3, big_bin = 30):
    big_box = centered_square_region(x,y, b=big_bin)
    small_box = centered_square_region(x,y, b=small_bin)
    mask = np.zeros((cube.dimx, cube.dimy))
    mask[big_box]=1
    mask[small_box]=0
    _, bkg_spec = cube.extract_integrated_spectrum(np.nonzero(mask), median=True, mean_flux=True, silent=True)
    return bkg_spec

def extract_point_source(x,y, cube, small_bin=3, big_bin = 30):
    """
    Basic way to extract a point source spectra with the local background subtracted.
    For a given position xy, we sum the spectra extracted in a squared region of size small_bin**2 centered on xy,
    and subtract from it the median spectra from a squared region of size big_bin**2 centered on xy excluding the central area
    :param xy: tuple of the position
    :param cube: the SpectralCube in which we extract the spectra
    :param small_bin: (Default 3) the binsize of the region of extraction of the source
    :param big_bin: (Default 30) the binsize of the region of extraction of the background
    """
    small_box = centered_square_region(x,y, b=small_bin)
    bkg_spec = estimate_local_background(x,y, cube, small_bin, big_bin)
    a,s, n = cube.extract_integrated_spectrum(small_box, silent=True, return_spec_nb = True)
    return a, s-n*bkg_spec

def check_source(x,y, spectral_cube, frame=None, smooth_factor = None):
    """
    Helper function to quickly look at a source
    We extract the source at positon (x,y) with extract_point_source and plot the resulting spectra.
    We also plot a map around the source, to check if we actually detect something
    :param x: abscissa of the source
    :param y: ordinate of the source
    :param spectral_cube: SpectralCube instance where we look at the source
    :param frame: (Optional) Frame to plot the detection on. If None, the deep_frame is used.
    If frame is a 2d array, we plot in this. If frame in as index, we plot on the sum of the frames around this index in the cube
    :param smooth_factor: (Optional) Factor used to smooth the spectrum
    """
    a,s = extract_point_source(x,y, spectral_cube)
    if smooth_factor is not None:
        s = vector.smooth(s, smooth_factor)
    f, ax = plot_spectra(a,s)
    ax.set_xlim(spectral_cube.get_filter_range())
    if frame is not None:
        if isinstance(frame, Number):
            f,ax = plot_map(extract_max_frame(x,y, spectral_cube, frame))
            wl = 1e8/a[int(frame)]
            ax.set_title('Frame at %.1f Angstroms'%wl)
        else:
            try:
                if frame.shape == (spectral_cube.dimx, spectral_cube.dimy):
                    f,ax = plot_map(frame[x-10:x+11, y-10:y+11])
                else:
                    raise ValueError('Invalid shape for the frame : %s. Cube shape : %s'%(frame.shape,(spectral_cube.dimx, spectral_cube.dimy)))
            except:
                raise TypeError('Non valid type for frame : %s'%type(frame))
    else:
        f,ax = plot_map(spectral_cube.get_deep_frame()[x-10:x+11, y-10:y+11])
    ax.scatter(10,10, marker='+', color='red')

def measure_coherence(source, detection_pos_frame):
    """
    Coherence is a measure of the credibility of a source as an emission line source.
    It checks if the hot pixels of a source are coming from ~the same frames of the cube
    :param source: pandas Dataframe with 'xcentroid', 'ycentroid' columns
    :param detection_pos_frame: map of the id of the max pixels
    """
    y,x = np.round(source[['xcentroid', 'ycentroid']]).astype(int)
    return 1/np.nanstd(detection_pos_frame[x-1:x+2, y-1:y+2])

def measure_source_fwhm(detection, data, rmax=10):
    x,y = np.array(detection[['xcentroid', 'ycentroid']])
    photo_flux = np.zeros(rmax)
    for r in range(rmax):
        if r == 0:
            aper = CircularAperture((x,y), 1.)
        else:
            aper = CircularAnnulus((x,y), r, r+1)
        photo_flux[r] = aper.do_photometry(data)[0]/aper.area()


    def get_fwhm(flux):
        #We assume max is on 0. If not, source is probably contaminated
        flux = flux - flux.min()
        spline = UnivariateSpline(np.arange(rmax), flux-flux[0]/2., s=0)
        if spline.roots().shape != (1,):
            return np.nan
        return spline.roots()[0]
    return (get_fwhm(photo_flux))


def get_sources(detection_frame, mask=False, sigma = 5.0, mode='DAO', fwhm = 2.5, threshold = None):
    if mask is False:
        mask = np.ones_like(detection_frame)
    if threshold is None:
        mean, median, std = sigma_clipped_stats(detection_frame, sigma=3.0, iters=5,
                                        mask=~mask.astype(bool) )#On masque la region hors de l'anneau
        threshold = median+sigma*std
    #On detecte sur toute la frame, mais on garde que ce qui est effectivement dans l'anneau
    if mode == 'DAO':
        daofind = DAOStarFinder(fwhm=fwhm, threshold=threshold)
        sources = daofind(detection_frame)
    elif mode == 'IRAF':
        irafind = IRAFStarFinder(threshold=threshold, fwhm=fwhm)
        sources = irafind(detection_frame)
    elif mode == 'PEAK':
        sources = find_peaks(detection_frame, threshold=threshold )
        sources.rename_column('x_peak', 'xcentroid')
        sources.rename_column('y_peak', 'ycentroid')
    elif mode == 'ORB':
        astro = Astrometry(detection_frame, instrument='sitelle')
        path, fwhm_arc = astro.detect_stars(min_star_number=5000, r_max_coeff=1., filter_image=False)
        star_list = astro.load_star_list(path)
        sources = Table([star_list[:,0], star_list[:,1]], names=('ycentroid', 'xcentroid'))
    elif mode == 'SEGM':
        logging.debug('Detecting')
        segm = detect_sources(detection_frame, threshold, npixels=4)
        deblend = True
        if deblend:
            try:
                logging.debug('Deblending')
                # fwhm = 3.
                # s = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
                # kernel = Gaussian2DKernel(s, x_size = 3, y_size = 3)
                # kernel = Box2DKernel(3, mode='integrate')
                deblended = deblend_sources(detection_frame, segm, npixels=4)#, filter_kernel=kernel)
            except ValueError as e:
                warnings.warn('Deblend was not possible.\n %s'%e)
                deblended = segm
            logging.debug('Retieving properties')
            sources = source_properties(detection_frame, deblended).to_table()
        else:
            logging.debug('Retieving properties')
            sources = source_properties(detection_frame, segm).to_table()
        logging.debug('Filtering Quantity columns')
        for col in sources.colnames:
            if type(sources[col]) is Quantity:
                sources[col] = sources[col].value
    sources = mask_sources(sources, mask) # On filtre
    df = sources.to_pandas()
    if 'id' in df:
       df.pop('id')
    return df
