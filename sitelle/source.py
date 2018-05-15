"""
This module groups convenience methods used when dealing with point-like sources.
"""
import numpy as np
import matplotlib.pyplot as plt
from numbers import Number
import warnings
import logging
import pandas as pd

from astropy.convolution import Gaussian2DKernel, Box2DKernel
from astropy.table import Table
from astropy.stats import sigma_clipped_stats
from astropy.units.quantity import Quantity

from sitelle.region import centered_square_region
from sitelle.plot import *

from photutils import CircularAperture, CircularAnnulus, EllipticalAperture
from photutils import DAOStarFinder, IRAFStarFinder
from photutils import find_peaks, detect_sources, deblend_sources, source_properties

from scipy.interpolate import UnivariateSpline

from orb.astrometry import Astrometry
from orb.utils import vector
from orcs.utils import fit_lines_in_spectrum


__all__ = ['mask_sources', 'filter_frame', 'extract_max_frame', 'estimate_local_background', 'extract_point_source', 'check_source', 'measure_coherence', 'measure_source_fwhm', 'get_sources', 'analyse_source']
def mask_sources(sources, annulus):
    """
    Mask sources not contained in a given 2D mask.

    Parameters
    ----------
    sources : :class:`~pandas:pandas.DataFrame`
        DataFrame containing detected sources. Should have columns ``ycentroid`` and ``xcentroid``. WARNING : astropy convention : y and x are inversed.
    annulus : 2D :class:`~numpy:numpy.ndarray`
        2D array containing 1 where we want to keep sources, 0 where we don't

    Returns
    -------
    sources : :class:`~pandas:pandas.DataFrame`
        Subset of the original sources containing only the ones whose position is in the annulus.
    """
    x,y = sources['ycentroid'], sources['xcentroid']
    return sources[annulus[np.round(x).astype(int), np.round(y).astype(int)].astype(bool)]

def filter_frame(frame, annulus, val=0):
    """
    DEPRECATED
    """
    _frame = np.copy(frame)
    _frame[np.where(annulus == 0)] = val
    return _frame

def extract_max_frame(x,y, spectral_cube, id_max_detection_frame):
    """
    For a given spatial position, extracts a few frames around the maximum of detection in the cube and returns their sum.

    Parameters
    ----------
    x : int
        Abscisse of the source, in pixels
    y : int
        Ordinate of the source, in pixels
    spectral_cube : :class:`~ORCS:orcs.process.SpectralCube`
        SpectralCube instance where we are looking at the source
    id_max_detection_frame : int
        Index of the max frame along the z dimension

    Returns
    -------
    out : 2D :class:`~numpy:numpy.ndarray`
        A cutout of the sum of the frames around the max of detection
    """
    if isinstance(id_max_detection_frame, Number):
        iframe = int(id_max_detection_frame)
    else:
        raise TypeError('Non valid type for id_max_detection_frame : %s'%type(id_max_detection_frame))
    data = spectral_cube.get_data(x-10, x+11, y-10,y+11, iframe-2,iframe+3)
    return np.sum(data, axis=2)

def estimate_local_background(x,y,cube, small_bin = 3, big_bin = 30):
    """
    Estimation of a local background spectrum. For a given position, the background spectrum is defined as the median taken in a ``big_bin x big_bin`` pixels box from which the ``small_bin x small_bin`` center pixels have been excluded.

    Parameters
    ----------
    x : int
        Abscisse position, in pixels
    y : int
        Ordinate position, in pixels
    cube : :class:`~ORCS:orcs.process.SpectralCube`
        SpectralCube instance where we are looking at the source
    small_bin : int
        (Optional) Size of the inner region. Default = 3
    big_bin : int
        (Optional) Size of the outter region. Default = 30

    Returns
    -------
    bkg_spec : 1D :class:`~numpy:numpy.ndarray`
        The corresponding spectrum
    """
    big_box = centered_square_region(x,y, b=big_bin)
    small_box = centered_square_region(x,y, b=small_bin)
    mask = np.zeros((cube.dimx, cube.dimy))
    mask[big_box]=1
    mask[small_box]=0
    _, bkg_spec = cube.extract_integrated_spectrum(np.nonzero(mask), median=True, mean_flux=True, silent=True)
    return bkg_spec

def extract_point_source(x,y, cube, small_bin=3, medium_bin = None, big_bin = 30):
    """
    Basic way to extract a point source spectra with the local background subtracted.
    For a given position xy, we sum the spectra extracted in a squared region of size small_bin**2 centered on x,y, and subtract from it the median spectra from a squared region of size big_bin**2 centered on x,y excluding the medium_bin**2 central area.

    Parameters
    ----------
    x : int
        Abscisse position, in pixels
    y : int
        Ordinate position, in pixels
    cube : :class:`~ORCS:orcs.process.SpectralCube`
        SpectralCube instance where we are looking at the source
    small_bin : int
        (Optional) Size of the inner region. Default = 3
    medium_bin : int
        (Optional) Size of the middle region. Default = small_bin
    big_bin : int
        (Optional) Size of the outter region. Default = 30

    Returns
    -------
    a : 1D :class:`~numpy:numpy.ndarray`
        The axis of the spectrum
    spec : 1D :class:`~numpy:numpy.ndarray`
        The source spectrum, background subtracted
    """
    small_box = centered_square_region(x,y, b=small_bin)
    if medium_bin is None:
        medium_bin = small_bin
    bkg_spec = estimate_local_background(x,y, cube, medium_bin, big_bin)
    a,s, n = cube.extract_integrated_spectrum(small_box, silent=True, return_spec_nb = True)
    return a, s-n*bkg_spec

def check_source(x,y, spectral_cube, frame=None, smooth_factor = None):
    """
    Helper function to quickly look at a source
    We extract the source at positon (x,y) with :func:`extract_point_source` and plot the resulting spectra.
    We also plot a map around the source, to check if we actually detect something.

    Parameters
    ----------
    x : int
        Abscisse position, in pixels
    y : int
        Ordinate position, in pixels
    spectral_cube : :class:`~ORCS:orcs.process.SpectralCube`
        SpectralCube instance where we are looking at the source
    frame : 2D :class:`~numpy:numpy.ndarray`, int
        (Optional) Frame to plot the detection on. If None, the deep_frame is used. If frame in as integer, we plot on the sum of the frames around this index in the cube
    smooth_factor : int
        (Optional) Factor used to smooth the spectrum (see :func:`ORB:orb.utils.vector.smooth`)
    """
    a,s = extract_point_source(x,y, spectral_cube)
    if smooth_factor is not None:
        s = vector.smooth(s, smooth_factor)
    f, ax = plot_spectra(a,s)
    ax.set_xlim(spectral_cube.params.filter_range)
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

def measure_coherence(source, argmax_map, segm_image = None):
    """
    Coherence is a measure of the credibility of a source as an emission line source.
    It checks if the hot pixels of a source are coming from ~the same frames of the cube by measuring the inverse of the variance around teh source in the detection pos frame. If the source is coherent, the max along the z axis is obtained in the same frame for every pixels of the source (i.e. the velocity is the same) and then the coherence measure is infinite.

    Parameters
    ----------
    source : :class:`~pandas:pandas.Series`
        Row of a DataFrame containing detected sources. Should have columns ``xpos`` and ``ypos``, in classic convention (not astropy)
    detection_pos_frame : 2D :class:`~numpy:numpy.ndarray`
        map of the id along z axis of the max pixels
    """
    if segm_image is not None:
        source_pos = argmax_map[np.nonzero(segm_image == source['id'])]
    else:
        x,y = source[['xpos', 'ypos']].astype(int)
        source_pos = argmax_map[x-1:x+2, y-1:y+2].flatten()
    var = np.nanstd(np.sort(source_pos))
    if var == 0:
        return 10
    else:
        return 1/var

def measure_source_fwhm(detection, data, rmax=10):
    """
    TO USE CAREFULLY
    Function used to estimate the FWHM of a source.
    It performs aperture photometry with inscreasing radius from the source position, and then, tries to find where half of the maximum of flux is reached.

    Parameters
    ----------
    detection : :class:`~pandas:pandas.Series`
        Row of a DataFrame containing detected sources. Should have columns ``xcentroid`` and ``ycentroid``, in classic convention (not astropy)
    data : 2D :class:`~numpy:numpy.ndarray`
        Flux map used for the photometry
    """
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


def get_sources(detection_frame, mask=False, sigma = 5.0, mode='DAO', fwhm = 2.5, threshold = None, npix=4, return_segm_image = False):
    """
    Main method used to identify sources in a detection frame and estimate their position.
    Different modes are available, accesible through the ``mode`` keyword :

    * DAO : uses the :class:`photutils:photutils.DAOStarFinder` method, adapted from DAOPHOT.
    * IRAF : uses the :class:`photutils:photutils.IRAFStarFinder` method, adapted from IRAF.
    * PEAK : uses the :func:`photutils:photutils.find_peaks` method, looking for local peaks above a given threshold.
    * ORB : uses the :func:`ORB:orb.utils.astrometry.detect_stars` method, fitting stars in the frame
    * SEGM : uses the :func:`photutils:photutils.detect_sources` method, segmenting the image.

    The most reliable is SEGM.

    Parameters
    ----------
    detection_frame : 2D :class:`~numpy:numpy.ndarray`
        Map on which the sources should be visible.
    mask : 2D :class:`~numpy:numpy.ndarray` or bool,  Default = False
        (Optional) If passed, only sources inside the mask are detected.
    sigma : float
        (Optional) Signal to Noise of the detections we want to keep. Only used if threshold is None. In this case, the signal and the noise are computed with sigma-clipping on the deteciton frame. Default = 5
    threshold : float or 2D :class:`~numpy:numpy.ndarray` of floats
        (Optional) Threshold above which we consider having a detection. Default is None
    mode : str
        (Optional) One of the detection mode listed above. Dafault = 'DAO'
    fwhm : float
        (Optional) Expected FWHM of the sources. Default : 2.5
    npix : int
        (Optional) Only used by the 'SEGM' method : minimum number of connected pixels with flux above the threshold to make a credible source. Default = 4
    return_segm_image : bool, Default = False
        (Optional) Only used in the 'SEGM' mode. If True, returns the obtained segmentation image.

    Returns
    -------
    sources : :class:`~pandas:pandas.DataFrame`
        A DataFrame where each row represents a detection, with at least the positions named as ``xcentroid``, ``ycentroid`` (WARNING : using astropy convention). The other columns depend on the mode used.

    """
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
        logging.info('Detecting')
        segm = detect_sources(detection_frame, threshold, npixels=npix)
        deblend = True
        labels = segm.labels
        if deblend:
            # while labels.shape != (0,):
            #     try:
            #         #logging.info('Deblending')
            #         # fwhm = 3.
            #         # s = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
            #         # kernel = Gaussian2DKernel(s, x_size = 3, y_size = 3)
            #         # kernel = Box2DKernel(3, mode='integrate')
            #         deblended = deblend_sources(detection_frame, segm, npixels=npix, labels=labels)#, filter_kernel=kernel)
            #         success = True
            #     except ValueError as e:
            #         #warnings.warn('Deblend was not possible.\n %s'%e)
            #         source_id = int(e.args[0].split('"')[1])
            #         id = np.argwhere(labels == source_id)[0,0]
            #         labels = np.concatenate((labels[:id], labels[id+1:]))
            #         success = False
            #     if success is True:
            #         break
            try:
                logging.info('Deblending')
                # fwhm = 3.
                # s = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
                # kernel = Gaussian2DKernel(s, x_size = 3, y_size = 3)
                # kernel = Box2DKernel(3, mode='integrate')
                deblended = deblend_sources(detection_frame, segm, npixels=npix)#, filter_kernel=kernel)
            except ValueError as e:
                warnings.warn('Deblend was not possible.\n %s'%e)
                deblended = segm
            logging.info('Retieving properties')
            sources = source_properties(detection_frame, deblended).to_table()
        else:
            deblended = segm
            logging.info('Retieving properties')
            sources = source_properties(detection_frame, deblended).to_table()
        logging.info('Filtering Quantity columns')
        for col in sources.colnames:
            if type(sources[col]) is Quantity:
                sources[col] = sources[col].value
    sources = mask_sources(sources, mask) # On filtre
    df = sources.to_pandas()
    if return_segm_image:
        return deblended.array, df
    else:
        return df

def analyse_source(source, cube, plot=False, return_fit_params=False):
    """
    Convenience method to spatially analyse a source.
    A 30x30 pixels 'flux map' is build from the sum of a few frames around each detected lines for the source.

    Two analysis are then performed:

    * Aperture photometry from the center of the source, to estimate a flux growth function and fit it with a custom erf function.
    * A Gaussian 2D fit of the PSF on the flux map



    This method can be used in a parallel process.
    Parameters
    ----------
    source : :class:`~pandas:pandas.Series`
        A row from a :class:`~pandas:pandas.DataFrame` containing detected sources. Should have columns ``xpos``, ``ypos`` (Not astropy convention), ``velocity``, ``*_detected`` whare * is a line name, containing True or False for each line.
    cube : :class:`~ORCS:orcs.process.SpectralCube`
        SpectralCube instance where we are looking at the source
    plot : bool, Default = False
        (Optional) If True, the two fits are plotted
    return_fit_params : bool, Default = False
        (Optional) If True, returns the full fits parameters

    Returns
    -------
    res : dict
        A dictionnary containing all the relevant fitted quantities.

        +-----------------------+---------------------------------------------------------------------------------------------------+
        |Parameter              |Description                                                                                        |
        +=======================+===================================================================================================+
        |flux_map_ks_pvalue     |Estimates the 'randomness' of the flux map, i.e if it's just noise or if we actually have a signal |
        +-----------------------+---------------------------------------------------------------------------------------------------+
        |flux_r                 |Flux at different radius *r*                                                                       |
        +-----------------------+---------------------------------------------------------------------------------------------------+
        |flux_err_r             |Flux error varying with *r*                                                                        |
        +-----------------------+---------------------------------------------------------------------------------------------------+
        |erf_amplitude          |Amplitude estimated from erf fit                                                                   |
        +-----------------------+---------------------------------------------------------------------------------------------------+
        |erf_amplitude_err      |Amplitude error                                                                                    |
        +-----------------------+---------------------------------------------------------------------------------------------------+
        |erf_xfwhm              |x-axis fwhm from erf fit                                                                           |
        +-----------------------+---------------------------------------------------------------------------------------------------+
        |erf_yfwhm              |y-axis fwhm from erf fit                                                                           |
        +-----------------------+---------------------------------------------------------------------------------------------------+
        |erf_fwhm               |Fwhm defined as *r* at which half of the max flux is reached                                       |
        +-----------------------+---------------------------------------------------------------------------------------------------+
        |flux_fraction_3        |Ratio between flux measured at 3 pixels from the center and max flux                               |
        +-----------------------+---------------------------------------------------------------------------------------------------+
        |model_flux_fraction_15 |Ratio between estimated flux at 15 pixels from the center and max flux                             |
        +-----------------------+---------------------------------------------------------------------------------------------------+
        |modeled_flux_r         |Modeled flux varying with *r*                                                                      |
        +-----------------------+---------------------------------------------------------------------------------------------------+
        |psf_snr                |Ratio between amplitude of the 2D fit and noise in the flux map                                    |
        +-----------------------+---------------------------------------------------------------------------------------------------+
        |psf_amplitude          |Amplitude of the 2D fit                                                                            |
        +-----------------------+---------------------------------------------------------------------------------------------------+
        |psf_xfwhm              |x-axis fwhm from 2D fit                                                                            |
        +-----------------------+---------------------------------------------------------------------------------------------------+
        |psf_yfwhm              |y-axis fwhm from 2D fit                                                                            |
        +-----------------------+---------------------------------------------------------------------------------------------------+
        |psf_ks_pvalue          |Randomness of the residuals map                                                                    |
        +-----------------------+---------------------------------------------------------------------------------------------------+



    """
    result = {}
    try:
        from astropy.stats import sigma_clipped_stats, gaussian_sigma_to_fwhm
        from sitelle.constants import SN2_LINES, SN3_LINES
        from sitelle.region import centered_square_region
        from orb.utils.spectrum import line_shift
        from orb.core import Lines

        filter_name = cube.params.filter_name

        if filter_name == 'SN2':
            LINES = SN2_LINES
        elif filter_name == 'SN3':
            LINES = SN3_LINES
        else:
            raise ValueError(filter_name)

        ## We build a flux map of the detected lines
        try:
            detected_lines = [line_name for line_name in LINES if source['%s_detected'%line_name.lower().replace('[', '').replace(']', '')]]
        except KeyError as e:
            raise ValueError('No columns *_detected in the source')
        if detected_lines == []:
            return pd.Series(result)

        x,y = source.as_matrix(['xpos', 'ypos']).astype(int)
        big_box = centered_square_region(x,y,30)
        medium_box = centered_square_region(15,15,5)
        small_box = centered_square_region(15,15, 3)
        data = cube._extract_spectra_from_region(big_box, silent=True)
        mask = np.ones((30, 30))
        mask[medium_box] = 0
        bkg_spec = np.nanmedian(data[np.nonzero(mask)], axis=0)
        data -= bkg_spec

        axis = cube.params.base_axis
        spec = np.nansum(data[small_box], axis=0)

        line_pos = np.atleast_1d(Lines().get_line_cm1(detected_lines) + line_shift(source['velocity'], Lines().get_line_cm1(detected_lines), wavenumber=True))
        pos_min = line_pos - cube.params.line_fwhm
        pos_max = line_pos + cube.params.line_fwhm
        pos_index = np.array([[np.argmin(np.abs(axis-pos_min[i])), np.argmin(np.abs(axis-pos_max[i]))] for i in range(pos_min.shape[0])])

        bandpass_size = 0
        flux_map = np.zeros(data.shape[:-1])
        for line_detection in pos_index:
            bandpass_size += line_detection[1]-line_detection[0]
            flux_map += np.nansum(data[:,:,line_detection[0]:line_detection[1]], axis=-1)

        _,_,std_map = sigma_clipped_stats(data, axis=-1)
        flux_noise_map = np.sqrt(bandpass_size)*std_map

        #Test for randomness of the flux_map
        from scipy import stats
        result['flux_map_ks_pvalue'] = stats.kstest((flux_map/flux_noise_map).flatten(), 'norm').pvalue

        #Fit of the growth function
        from photutils import RectangularAperture
        from scipy.special import erf
        from scipy.optimize import curve_fit


        try:
            _x0 = source['xcentroid'] - x + 15.
            _y0 = source['ycentroid'] - y + 15.
        except:
            _x0 = source['xpos'] - x + 15.
            _y0 = source['ypos'] - y + 15.

        flux_r = [0.]
        flux_err_r = [np.nanmin(flux_noise_map)]

        r_max = 15
        r_range = np.arange(1, r_max+1)
        for r in r_range:
#             aper = CircularAperture((_x0,_y0), r)
            aper = RectangularAperture((_x0,_y0), r,r,0)
            flux_r.append(aper.do_photometry(flux_map)[0][0])
            flux_err_r.append(np.sqrt(aper.do_photometry(flux_noise_map**2)[0][0]))

        flux_r = np.atleast_1d(flux_r)
        flux_err_r = np.atleast_1d(flux_err_r)

        result['flux_r'] = flux_r
        result['flux_err_r'] = flux_err_r
        try:
            def model(r, x0, y0, sx, sy, A):
                return A*erf((r/2.-x0)/(2*sx*np.sqrt(2)))*erf((r/2.-y0)/(2*sy*np.sqrt(2)))
            R = np.arange(r_max+1)
            p, cov = curve_fit(model, R, flux_r,
                               p0=[0,0,1.5,1.5,flux_map.max()],
                               bounds=([-2, -2, -np.inf, -np.inf, -np.inf], [2,2,np.inf, np.inf, np.inf]),
                               sigma= flux_err_r, absolute_sigma=True,
                               maxfev=10000)
            if (p[2] < 0) != (p[3] < 0):
                if p[-1] < 0:
                    p[-1] = -p[-1]
                    if p[2]<0:
                        p[2] = - p[2]
                    elif p[3] < 0:
                        p[3] = -p[3]
            if plot:
                f,ax = plt.subplots()
                ax.plot(R,model(R,*p), label='Fit')
                ax.errorbar(R, flux_r, flux_err_r, label='Flux')
                ax.set_ylabel('Flux')
                ax.set_xlabel('Radius from source')
                ax.legend()

            from scipy.optimize import bisect
            fwhm = bisect(lambda x:model(x, *p) -p[-1]/2, 0.1, 10)
            result['erf_amplitude'] = p[-1]
            result['erf_amplitude_err'] = np.sqrt(np.diag(cov))[-1]
            result['erf_xfwhm'] = gaussian_sigma_to_fwhm*p[2]
            result['erf_yfwhm'] = gaussian_sigma_to_fwhm*p[3]
            result['erf_ks_pvalue'] = stats.kstest((flux_r-model(R,*p))/flux_err_r, 'norm').pvalue
            result['erf_fwhm'] =fwhm

            result['flux_fraction_3'] = flux_r[3]/p[-1]
            result['model_flux_fraction_15'] = model(R,*p)[r_range[-1]] / p[-1]

            result['modeled_flux_r'] = model(R,*p)

        except Exception as e:
            print(e)
            pass

        ## 2D fit of the PSF
        from astropy.modeling import models, fitting

        fitter = fitting.LevMarLSQFitter()
        X,Y = np.mgrid[:30, :30]

        flux_std = np.nanmean(flux_noise_map)

        gauss_model = models.Gaussian2D(amplitude = np.nanmax(flux_map/flux_std),x_mean = _y0, y_mean = _x0)
        gauss_model.bounds['x_mean'] = (14, 16)
        gauss_model.bounds['y_mean'] = (14, 16)
        gauss_fit = fitter(gauss_model, X,Y, flux_map/flux_std)

        if plot is True:
            f, ax = plt.subplots(1,3, figsize=(8,3))
            v_min = np.nanmin(flux_map)
            v_max = np.nanmax(flux_map)
            plot_map(flux_map, ax=ax[0], cmap='RdBu_r', vmin=v_min, vmax=v_max)
            ax[0].set_title("Data")
            plot_map(gauss_fit(X, Y)*flux_std, ax=ax[1], cmap='RdBu_r', vmin=v_min, vmax=v_max)
            ax[1].set_title("Model")
            plot_map(flux_map - gauss_fit(X, Y)*flux_std, ax=ax[2], cmap='RdBu_r', vmin=v_min, vmax=v_max)
            ax[2].set_title("Residual")

        result['psf_snr'] = gauss_fit.amplitude[0]
        result['psf_amplitude'] = flux_std*gauss_fit.amplitude[0]*2*np.pi*gauss_fit.x_stddev*gauss_fit.y_stddev
        result['psf_xfwhm'] = gauss_fit.x_fwhm
        result['psf_yfwhm'] = gauss_fit.y_fwhm
        normalized_res = (flux_map - gauss_fit(X, Y)*flux_std)/flux_noise_map
        result['psf_ks_pvalue'] = stats.kstest(normalized_res.flatten(),'norm').pvalue
        if return_fit_params:
            return pd.Series(result), p, gauss_fit
        else:
            return pd.Series(result)
    except Exception as e:
        print e
        return pd.Series(result)
