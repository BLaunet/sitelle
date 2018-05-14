from orb.fit import Lines
import pandas as pd
import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit
from orcs.utils import fit_lines_in_spectrum
from orb.utils.spectrum import line_shift, compute_radial_velocity
import gvar
from sitelle.region import centered_square_region
import logging
from sitelle.source import extract_point_source
from sitelle import constants
from orb.utils.vector import smooth
from astropy.stats import sigma_clipped_stats
import matplotlib.pyplot as plt
from sitelle.utils import stats_without_lines
from sitelle.plot import *

__all__ = ['sky_model_to_remove', 'parse_line_params', 'chi2_func', 'remove_OH_line', 'fit_gas_lines', 'guess_line_velocity', 'guess_source_velocity', 'refine_velocity_guess', 'fit_spectrum', 'fit_source', 'check_fit', 'fit_SN2', 'fit_SN3']

def sky_model_to_remove(mean_spectrum, axis, sky_axis, sky_model):
    """
    This function shifts the velocity of a skymodel to match at best the skylines in a spectrum.
    It returns the fitted vector, that should be then subtracted from the original spectrum.

    Parameters
    ----------
    mean_spectrum : 1D :class:`~numpy:numpy.ndarray`
        The spectrum containing skylines.
        If it comes from an integrated region, pay attention to give the **mean** spectrum, i.e divided by the number of integrated pixels
    axis : 1D :class:`~numpy:numpy.ndarray`
        Axis of the spectrum [cm-1]
    sky_axis : 1D :class:`~numpy:numpy.ndarray`
        Axis on which the skymodel is known [cm-1]
    skymodel : 1D :class:`~numpy:numpy.ndarray`
        Sky spectrum to be fitted
    Returns
    -------
    shifted_spectrum : :class:`~numpy:numpy.ndarray`
        The shifted sky spectrum, to be removed
    """
    c = 299792.458
    axis = np.log10(axis) #This way, a velocity shift correspond to a constant
                                      #wavenumber offset over the whole axis
    sky_axis = np.log10(sky_axis)

    sky_interpolator = UnivariateSpline(sky_axis, sky_model, s=0)
    fit_vector = np.zeros_like(mean_spectrum)
    scale = np.nanmax(mean_spectrum) - np.nanmedian(mean_spectrum)

    def model(x, h, v):
        shift = -np.log10(1-v/c) # minus sign because it's in wavenumber
        sky_shifted = sky_interpolator(axis.astype(float)+shift)
        return h + UnivariateSpline(axis, sky_shifted, s=0)(x)
    popt, pcov = curve_fit(model, axis, mean_spectrum, p0=[scale,-75.])
    fit_vector = model(axis, *popt)-popt[0]
    return fit_vector

def parse_line_params(rest_lines, fit, error = True, wavenumber = True):
    """
    Parse the fitted parameters to display them as a nice :class:`~pandas:pandas.DataFrame`

    Parameters
    ----------
    rest_lines: list of str or list of float
        Names (as defined `here <http://celeste.phy.ulaval.ca/orcs-doc/introduction.html#list-of-available-lines>`_) or positions of the lines at rest.
    fit : dict
        the ``fit_params`` in the output of a fit from orcs (e.g. output from :func:`~ORCS:orcs.core.HDFCube.fit_lines_in_spectrum`)
    error : bool
        if True, the errors are also parsed (Default = True)
    wavenumber : bool
        if True, output is in wavenumber (else in wavelength). Default = True
    Returns
    -------
    table : :class:`~pandas:pandas.DataFrame`
        Table containing all information about the fitted lines, easier to read
    """
    #Potential names
    lines_cm1 = list()
    names = False
    for iline in rest_lines:
        if isinstance(iline, str):
            names = True
            iline = Lines().get_line_cm1(iline)
        lines_cm1.append(iline)
    lines_cm1 = np.array(lines_cm1)

    lines_list = []
    for i in range(len(lines_cm1)):
        line = {}
        if names:
            line['name'] = rest_lines[i]
        line['rest_pos'] = lines_cm1[i]

        line['height'], line['amp'], line['pos'], line['fwhm'], line['sigma'] = fit['lines_params'][i]
        line['velocity'] = fit['velocity'][i]
        line['snr'] = fit['snr'][i]
        line['flux'] = fit['flux'][i]
        line['broadening'] = fit['broadening'][i]

        if error:
            line['height_err'], line['amp_err'], line['pos_err'], line['fwhm_err'], line['sigma_err'] = fit['lines_params_err'][i]
            line['velocity_err'] = fit['velocity_err'][i]
            line['broadening_err'] = fit['broadening_err'][i]
        lines_list.append(line)

    lines_df = pd.DataFrame(lines_list)
    if not wavenumber:
        lines_df['pos'] = 1e8/lines_df['pos']
        lines_df['rest_pos'] = 1e8/lines_df['rest_pos']
        lines_df['sigma'] = lines_df['sigma']*lines_df['pos']**2/1e8
        if error:
            lines_df['pos_err'] = lines_df['pos_err']*lines_df['pos']**2/1e8
            lines_df['sigma_err'] = lines_df['sigma_err']*lines_df['pos']**2/1e8
    return lines_df

def chi2_func(axis, spectrum, fit_params, delta=8):
    """
    Computes a chi square value only around fitted lines.

    axis : 1D :class:`~numpy:numpy.ndarray`
        Axis of the spectrum [cm-1]
    spectrum : 1D :class:`~numpy:numpy.ndarray`
        Spectrum on which the fit has been performed
    fit_params: dict
        fit parameters containing information about the lines (output from an orcs fit, e.g. :func:`~ORCS:orcs.core.HDFCube.fit_lines_in_spectrum`)
    delta : int
        Delta in FWHM around line peak that should be used to compute the chi2. Default = 8

    Returns
    -------
    float
        a chi square value, computed only around +- delta * fwhm each line

    """

    #We keep only indices around the fitted lines
    indices_to_keep = []
    for i,line in enumerate(fit_params['fitted_models']['Cm1LinesModel']):
        #indices_to_keep.append(np.argwhere(np.abs(line)>=line.std()/2)[:,0])
        peak_id = np.argmax(line)
        fwhm_pix = fit_params['lines_params'][i][3]/(axis[1]-axis[0])
        imin = max(0, int(round(peak_id-delta*fwhm_pix)))
        imax = min(len(spectrum)-1, int(round(peak_id+delta*fwhm_pix)))
        indices_to_keep.append(np.arange(imin,imax))
    to_keep = np.unique(np.concatenate(indices_to_keep))
    sigma = np.std(spectrum)
    residual = (spectrum-fit_params['fitted_vector'])[to_keep]
    return np.sum(np.square(residual/sigma))

def remove_OH_line(spectrum, theta, cube, **kwargs):
    """
    Function used to remove the strange Halpha line around -400km/s (probably OH line).

    DEPRECATED
    To be used with care, especially when there is a real Halpha signal there.
    By default, the line is modeled as a sinc at the position of a Halpha line offseted at a velocity of -400 km/s

    Parameters
    ----------
    spectrum : 1D :class:`~numpy:numpy.ndarray`
        the spectrum to fit
    theta : float
        incident angle on the detector corresponding to the spectrum (obtained by passing ``return_mean_theta = True`` in :func:`~ORCS:orcs.core.HDFCube.extract_spectrum`)
    cube : :class:`~ORCS:orcs.process.SpectralCube`
        Spectral cube instance from which the spectrum has been extracted.
    kwargs :
        Additional arguments defining the line model (same as :func:`~ORB:orb.fit._prepare_input_params`).

    Returns
    -------
    1D :class:`~numpy:numpy.ndarray`
        The spectrium whithout this line (hopefully)
    """
    lines = ['Halpha']

    if 'fmodel' not in kwargs:
        kwargs['fmodel'] = 'sinc'
    if 'pos_def' not in kwargs:
        kwargs['pos_def'] = '1'
    # if 'sigma_cov' not in kwargs:
    #     kwargs['sigma_cov'] = [gvar.gvar(30, 10)]
    # if 'sigma_def' not in kwargs:
    #     kwargs['sigma_def'] = ['1']

    cube._prepare_input_params(lines, **kwargs)
    inputparams = cube.inputparams.convert()
    params = cube.params.convert()
    V_range  = [gvar.gvar(-400, 2)]
    snr_guess = 1

    fit = fit_gas_lines([spectrum, theta], inputparams, params, lines, V_range, snr_guess, silent=True)
    return spectrum - fit['line_spectra']

def fit_gas_lines(z_dim, inputparams, params, lines, V_range, snr_guess = None, silent=False):
    """
    This function fit lines in a given spectrum.
    DEPRECATED
    It has been deisgned to be parallelized, hence the apparent complexity.

    Parameters
    ----------
    z_dim: 2D :class:`~numpy:numpy.ndarray`
        z_dim[0] contains the spectrum to fit, z_dim[1] contains ``theta`` , the incident angle at which the spectrum was recorded
    inputparams : :class:`ORB:orb.fit.InputParams`
        Output from :func:`~ORCS:orcs.core._prepare_input_params`
    params : dict
        Parameters of the cube, obtained with :func:`ORCS:orcs.HDFCube.params.convert`.
    lines : list of str
        Names of the lines to fit (as defined `here <http://celeste.phy.ulaval.ca/orcs-doc/introduction.html#list-of-available-lines>`_)
    V_range : 1D :class:`~numpy:numpy.ndarray`
        The range of velocity to test.s
    snr_guess : float
        Signal to Noise ratio guess (Default = None)

    Returns
    -------
    3D :class:`~numpy:numpy.ndarray`
        A cube containing at each cell a dict with :
        - 'line_spectra': the fitted line spectra
        - 'fit_params': the fitetd parameters of the lines
        - 'chi2': chi2 list, each element corresponds to a velocity in V_range
    """
    spectrum = z_dim[0]
    theta = z_dim[1]
    chi2_list = []
    spectra_list = []
    params_list = []
    line_spectra, fit_params, chi2 = None, None, None
    for v in V_range:
        fit = fit_lines_in_spectrum(params, inputparams, 1e10, spectrum,
                              theta, pos_cov=v, snr_guess=snr_guess)
        if fit != []:
            hChi2 = chi2_func(params['base_axis'], spectrum, fit)
            chi2_list.append(hChi2)
            spectra_list.append(np.sum(fit['fitted_models']['Cm1LinesModel'], 0))
            params_list.append(parse_line_params(lines,fit))
            if not silent:
                print 'Init V = ', v, 'Fitted V = ', fit['velocity'][0], 'Chi2 = ', hChi2
    if chi2_list != list():
        imin = np.argmin(chi2_list)
        chi2 = chi2_list[imin]#{'chi2': chi2_list, 'v': V_range[imin]}
        line_spectra = spectra_list[imin]
        fit_params = params_list[imin]
    return {'line_spectra':line_spectra, 'fit_params':fit_params, 'chi2':chi2}

def guess_line_velocity(max_pos, v_min, v_max, lines = None, debug=False, return_line=False):
    """
    Estimation of the velocity shift from a line position.
    Because we don't have a priori knowledge on which line we are looking at, we try different rest lines while the estimation is not in a given velocity range. If None of the lines are compatible with a velocity shift in the given range, we ouput NaN.


    Parameters
    ----------
    max_pos : float
        Observed line position, in cm-1.
    v_min : float
        Constraint on the lower bound of the velocity range.
    v_max : float
        Constraint on the upper bound of the velocity range.
    lines : list of str
        Names of the rest line candidates for the observed line.
    debug : bool
        (Optional) If True, a debug id displayed (default = False)
    return_line : bool
        (Optional) If True, the name of the line that produced the estimation is output.

    Returns
    -------
    v_guess : float
        The guess on the velocity (NaN if not included in the velocity range)
    line_name : str
        (Only if ``return_line = True``) the name of the line.
    """
    if debug:
        logging.info('Max line position : %.2f'%max_pos)
    if lines is None:
        return None
    while True:
        rest_lines = np.atleast_1d(Lines().get_line_cm1(lines))
        #We find the nearest line
        idx = (np.abs(rest_lines-max_pos)).argmin()

        #We compute the corresponding velocity shift
        v_guess = compute_radial_velocity(max_pos, rest_lines[idx], wavenumber=True)
        if debug:
            logging.info('Guessed line : %s'%lines[idx])
            logging.info('Velocity : %.2f'%v_guess)
        if not (v_min <= v_guess <= v_max):
            lines.remove(lines[idx])
            if lines != []:
                continue
            else:
                v_guess = None
                break
        else:
            break
    if return_line and lines != []:
        return v_guess, lines[idx]
    elif return_line and lines == []:
        return v_guess, None
    else:
        return v_guess

def guess_source_velocity(spectrum, cube, v_min = -800., v_max = 50., debug=False, lines = None, force=False, return_line = False):
    """
    Estimation of the velocity shift of a spectrum.
    The position of the max of emission in the spectrum is found, and compared to different lines rest position until a compatible velocity (i.e. included in a given velocity range) is found.
    The obtained guess is discrete in the sense it corresponds to the frame in which the max was found. For better accuracy see :func:`refine_velocity_guess`.

    Parameters
    ----------
    spectrum : 1D :class:`~numpy:numpy.ndarray`
        Spectrum to analyze
    cube : :class:`~ORCS:orcs.process.SpectralCube`
        Spectral cube instance from which the spectrum has been extracted.
    v_min : float
        Constraint on the lower bound of the velocity range.
    v_max : float
        Constraint on the upper bound of the velocity range.
    debug : bool
        (Optional) If True, a debug id displayed (default = False)
    lines : list of str
        (Optional) Names of the rest line candidates for the observed line. If None, the lines in the filter of the cube are used (see `~sitelle.constants.SN2_LINES` and `~sitelle.constants.SN3_LINES`)
    force : bool
        (Optional) If True, a velocity guess is outputed no matter it's compatibility with the velocity range input. Should be used only when ``lines`` is restricted to one element.
    return_line : bool
        (Optional) If True, the name of the line that produced the estimation is output.

    Returns
    -------
    v_guess : float
        The guess on the velocity (NaN if not included in the velocity range)
    line_name : str
        (Only if ``return_line = True``) the name of the line.

    See Also
    --------
    :func:`guess_line_velocity`
    """
    if lines is None:
        if cube.params.filter_name =='SN2':
            lines = ['Hbeta', '[OIII]5007']
        elif cube.params.filter_name == 'SN3':
            lines = ['[NII]6583', 'Halpha']
        else:
            raise NotImplementedError()
    if force is True:
        #print ('Forced detection of velocity in the range [%.1f, %.1f] km/s'%(v_min, v_max))
        line_pos = np.atleast_1d(Lines().get_line_cm1(lines))
        pos_min = line_pos + line_shift(v_max, line_pos, wavenumber=True)
        pos_max = line_pos + line_shift(v_min, line_pos, wavenumber=True)
        pos_index = np.searchsorted(cube.params.base_axis, [pos_min, pos_max]).T

        s = np.concatenate([spectrum[pos[0]:pos[1]] for pos in pos_index])
        a = np.concatenate([cube.params.base_axis[pos[0]:pos[1]] for pos in pos_index])

        max_pos = a[np.nanargmax(s)]
    else:
        imin, imax = np.searchsorted(cube.params.base_axis, cube.params.filter_range)
        max_pos = cube.params.base_axis[imin+5:imax-5][np.nanargmax(spectrum[imin+5:imax-5])]
    return guess_line_velocity(max_pos, v_min, v_max, debug=debug, lines=lines, return_line=return_line)

def refine_velocity_guess(spectrum, axis, v_guess, detected_line, return_fit = False):
    """
    Refines a velocity guess with better accuracy, avoiding to have discrete guess corresponding to pixel numbers.
    A gaussian function is fitted to estimate the line position.

    Parameters
    ----------
    spectrum : 1D :class:`~numpy:numpy.ndarray`
        The spectrum containing emission lines
    axis : 1D :class:`~numpy:numpy.ndarray`
        The corresponding axis, in wavenumber
    v_guess : float
        A first guess on the velocity, typically obtained from :func:`guess_source_velocity`
    detected_line : str
        Names of the main line present in the spectrum (as defined `here <http://celeste.phy.ulaval.ca/orcs-doc/introduction.html#list-of-available-lines>`_)
    return_fit : bool, Default = False
        (Optional) If True, returns the fit parameters, for further investigation
    Returns
    -------
    v : float
        The updated velocity guess
    """
    from orb.utils.spectrum import line_shift, compute_radial_velocity
    from orb.core import Lines
    from astropy.modeling import models, fitting
    line_rest = Lines().get_line_cm1(detected_line)

    mu =  line_rest + line_shift(v_guess, line_rest, wavenumber=True)
    G0 = models.Gaussian1D(amplitude=np.nanmax(spectrum), mean=mu, stddev=11)
    G0.mean.max = line_rest + line_shift(v_guess-25, line_rest, wavenumber=True)
    G0.mean.min = line_rest + line_shift(v_guess+25, line_rest, wavenumber=True)
    C = models.Const1D(amplitude = np.nanmedian(spectrum))

    model = C+G0
    fitter = fitting.LevMarLSQFitter()
    fit = fitter(model, axis, spectrum)
    if return_fit:
        return compute_radial_velocity(fit.mean_1,line_rest, wavenumber=True), fit
    else:
        return compute_radial_velocity(fit.mean_1,line_rest, wavenumber=True)

def fit_spectrum(spec, theta, v_guess, cube, lines = None, **kwargs):
    """
    Helper function to simplify fitting, with some default parameters being already set.
    The underlying method is :func:`~ORCS:orcs.core.HDFCube.fit_lines_in_spectrum`.

    Parameters
    ----------
    spec : 1D :class:`~numpy:numpy.ndarray`
        The spectrum containing emission lines
    theta : float
        Theta value corresponding to the ``spec``
    v_guess : float
        Velocity guess on the spectrum
    cube : :class:`~ORCS:orcs.process.SpectralCube`
        The cube from which ``spec`` has been extracted
    lines : str
        (Optional) The lines to fit. Default is set to the main lines contained in the filter of the cube. See **sitelle.constants.SN2_LINES** and **sitelle.constants.SN3_LINES**
    kwargs : dict
        Any additional argumebnts to be passed to ``_prepare_input_params``.
        Some are set to default values :

        +---------------+---------------+
        | Parameter     | Value         |
        +===============+===============+
        | fmodel        |'sinc'         |
        +---------------+---------------+
        | pos_def       | '1'           |
        +---------------+---------------+
        | signal_range  | filter_range  |
        +---------------+---------------+
        | pos_cov       | v_guess       |
        +---------------+---------------+
        | snr_guess     | 'auto'        |
        +---------------+---------------+
        | nofilter      | True          |
        +---------------+---------------+
    Returns
    -------
    fit :
        the performed fit
    """
    if lines is None:
        if cube.params.filter_name =='SN2':
            lines = constants.SN2_LINES
        elif cube.params.filter_name == 'SN3':
            lines = constants.SN3_LINES
        else:
            raise NotImplementedError()

    # if 'fmodel' not in kwargs:
    #     kwargs['fmodel'] = 'sincgauss'
    # if 'sigma_cov' not in kwargs:
    #     kwargs['sigma_cov'] = [gvar.gvar(10, 30)]
    # if 'sigma_def' not in kwargs:
    #     kwargs['sigma_def'] = ['1']

    if 'fmodel' not in kwargs:
        kwargs['fmodel'] = 'sinc'
    if 'pos_def' not in kwargs:
        kwargs['pos_def']='1'
    if 'signal_range' not in kwargs:
        kwargs['signal_range'] = cube.params.filter_range
    if 'pos_cov' not in kwargs:
        kwargs['pos_cov'] = v_guess
    snr_guess = kwargs.pop('snr_guess', 'auto')
    nofilter = kwargs.pop('nofilter', True)
    cube._prepare_input_params(lines, nofilter=nofilter, **kwargs)
    inputparams = cube.inputparams.convert()
    params = cube.params.convert()
    return fit_lines_in_spectrum(params, inputparams, 1e10, spec, theta, snr_guess=snr_guess, debug=False)

def fit_source(xpos, ypos, cube, v_guess = None, return_v_guess=False, v_min=-800., v_max=100., **kwargs):
    """
    Fit a spectrum from a position in the cube.
    The spectrum is extracted inside a 3x3 pixels box, a guess of the velocity if performed, and the spectrum is fitted using :func:`fit_spectrum`

    Parameters
    ----------
    xpos : int
        abscisse of the source in the cube
    ypos : int
        ordonate of the source in the cube
    cube : :class:`~ORCS:orcs.process.SpectralCube`
        The cube in which we want to extract and fit the spectrum.
    v_guess : float
        (Optional) If None, a guess is performed using :func:`guess_source_velocity`
    return_v_guess : bool, Default = False
        (Optional) If True, returns the guessed velocity value.
    v_min : float
        (Optional) v_min used by :func:`guess_source_velocity`. Default = -800
    v_max : float
        (Optional) v_max used by :func:`guess_source_velocity`. Default = 100
    kwargs : dict
        Additional keyword arguments to be passed to :func:`fit_spectrum`
    """
    xpos = int(xpos)
    ypos = int(ypos)
    a,s = extract_point_source(xpos, ypos, cube)
    if v_guess is None:
        v_guess = guess_source_velocity(s, cube, v_min = v_min, v_max = v_max)
    if v_guess != None:
        small_box = centered_square_region(xpos, ypos, 3)
        theta_orig = np.nanmean(cube.get_theta_map()[small_box])
        fit_params = fit_spectrum(s, theta_orig, v_guess, cube, **kwargs)
        if return_v_guess:
            return [v_guess, fit_params]
        else:
            return fit_params
    else:
        return []


def check_fit(source, SN2_ORCS, SN3_ORCS, SN2_detection_frame, SN3_detection_frame, **kwargs):
    """
    Convenience function to display a fit and visually check if it is meaningful. The fit is recomputed.

    Parameters
    ----------
    source : :class:`~pandas:pandas.Series`
        A row from a :class:`~pandas:pandas.DataFrame` containing detected sources. Should have columns ``xpos_SN2``, ``ypos_SN2``, ``xpos_SN3``, ``ypos_SN3``.
    SN2_ORCS : :class:`~ORCS:orcs.process.SpectralCube`
        The cube taken in SN2 filter.
    SN3_ORCS : :class:`~ORCS:orcs.process.SpectralCube`
        The cube taken in SN3 filter.
    SN2_detection_frame : 2D :class:`~numpy:numpy.ndarray`
        The SN2 frame on which the source position has been found
    SN3_detection_frame : 2D :class:`~numpy:numpy.ndarray`
        The SN3 frame on which the source position has been found
    kwargs : dict
        Additional keyword arguments to be passed to :func:`fit_source`

    Returns
    -------
    fit_SN2 : dict
        Fit parameters in SN2 filter
    fit_SN3 : dict
        Fit parameters in SN3 filter
    Note
    ----
    The signature is not well constructed yet, and works only with sources defined as a DataFrame, and has to be performed on both cubes (no possibility to check a fit from a single cube)
    """
    x_SN2, y_SN2 = map(int, source[['xpos_SN2', 'ypos_SN2']])
    x_SN3, y_SN3 = map(int, source[['xpos_SN3', 'ypos_SN3']])
    a_SN2,s_SN2 = extract_point_source(x_SN2, y_SN2,cube = SN2_ORCS)
    a_SN3, s_SN3 = extract_point_source(x_SN3, y_SN3,cube = SN3_ORCS)
    v_guess = guess_source_velocity(s_SN2, SN2_ORCS, debug=True, v_min = -1500.)
    guess_source_velocity(s_SN3, SN3_ORCS, debug=True)
    fit_SN2 = fit_source(x_SN2, y_SN2,cube = SN2_ORCS, v_guess = v_guess, **kwargs)
    fit_SN3 = fit_source(x_SN3, y_SN3,cube = SN3_ORCS, v_guess =fit_SN2['velocity'][0], **kwargs)

    f,ax = plt.subplots(1,2, figsize=(10,6), sharex=True, sharey=True)
    plot_map(SN2_detection_frame, ax=ax[0], pmin=1, pmax=99)#, projection=SN2_ORCS.get_wcs())
    ax[0].scatter(x_SN2, y_SN2, marker='x', c='r')
    ax[0].set_xlim(x_SN2-20, x_SN2+21)
    ax[0].set_ylim(y_SN2-20, y_SN2+21)

    plot_map(SN3_detection_frame, ax=ax[1], pmin=1, pmax=99)#, projection=SN3_ORCS.get_wcs())
    ax[1].scatter(x_SN3, y_SN3, marker='x', c='r')
    ax[1].scatter(*map(int, source[['x_guess', 'y_guess']]), marker='x', c='blue')
    ax[1].set_xlim(x_SN3-20, x_SN3+21)
    ax[1].set_ylim(y_SN3-20, y_SN3+21)

    f.tight_layout()
    f,ax = plt.subplots(1,2, figsize=(10,6))
    plot_spectra(a_SN2, s_SN2, ax=ax[0])
    if fit_SN2 != []:
        plot_spectra(a_SN2, fit_SN2['fitted_vector'], ax=ax[0])
    ax[0].set_xlim(SN2_ORCS.params.filter_range)
    make_wavenumber_axes(ax[0])
    add_lines_label(ax[0], 'SN2', -300., wavenumber=True)

    plot_spectra(a_SN3, s_SN3, ax=ax[1])
    if fit_SN3 != []:
        plot_spectra(a_SN3, fit_SN3['fitted_vector'], ax=ax[1])
    ax[1].set_xlim(SN3_ORCS.params.filter_range)
    make_wavenumber_axes(ax[1])
    add_lines_label(ax[1], 'SN3', -300., wavenumber=True)

    print 'SN2 fit velocity : %.2f +- %.2f'%(fit_SN2['velocity'][0], fit_SN2['velocity_err'][0])
    print 'SN3 fit velocity : %.2f +- %.2f'%(fit_SN3['velocity'][0], fit_SN3['velocity_err'][0])

    print source.filter(regex='detected')
    return fit_SN2, fit_SN3


def fit_SN2(source, cube, v_guess = None, v_min = -800., v_max = 0., lines=None, return_fit_params = False, kwargs_spec={}, kwargs_bkg = {}, debug=False):
    """
    Function specialized to fit sources found in the SN2 cube. The background spectrum is fitted as well.
    Can be used in a parallel process.
    The SNR of the spec is estimated using the :func:`stats_without_lines` method.
    Only the velocity of the background is estimated, as the flux is biased by an unkown amount of absorption.

    Parameters
    ---------
    source : :class:`~pandas:pandas.Series`
        A row from a :class:`~pandas:pandas.DataFrame` containing detected sources. Should have columns ``xpos``, ``ypos``, assumed to correspond to the SN2 pixel coordinates.
    cube : :class:`~ORCS:orcs.process.SpectralCube`
        The cube taken in SN2 filter.
    v_guess : float
        (Optional) If None, a guess is performed using :func:`guess_source_velocity` and :func:`refine_velocity_guess`
    v_min : float
        (Optional) v_min used by :func:`guess_source_velocity`. Default = -800
    v_max : float
        (Optional) v_max used by :func:`guess_source_velocity`. Default = 0
    lines : list of str
        (Optinal) Names of the lines to fit. If None, **SN2_LINES** are used, except if no v_guess has been found; then we assume it's only a Hbeta line at very fast velocity.
    return_fit_params : bool, Default = False
        (Optional) If True, returns the full parameters of the fit.
    kwargs_spec : dict
        (Optional) Additional keyword arguments to be used by :func:`fit_spectrum` when fitting the source spectrum.
    kwargs_bkg : dict
        (Optional) Additional keyword arguments to be used by :func:`fit_spectrum` when fitting the background spectrum.
    debug : bool, Default = False
        (Optional) If True, the velocity guess is verbose.

    Returns
    -------
    fit_res : dict
        A dict containg a lot of information about the fit.

        +--------------------+--------------------------------------------------------------------+
        | Parameter          | Description                                                        |
        +====================+====================================================================+
        | err                | estimated noise value on the spectra                               |
        +--------------------+--------------------------------------------------------------------+
        | guess_snr          | SNR guess                                                          |
        +--------------------+--------------------------------------------------------------------+
        | exit_status        | code to identify cause of crash                                    |
        +--------------------+--------------------------------------------------------------------+
        | v_guess            | guessed velocity in km/s                                           |
        +--------------------+--------------------------------------------------------------------+
        | chi2               | chi2 computed on the residual                                      |
        +--------------------+--------------------------------------------------------------------+
        | rchi2              | reduced chi2 computed on the residual                              |
        +--------------------+--------------------------------------------------------------------+
        | ks_pvalue          | ks test computed on the residuals                                  |
        +--------------------+--------------------------------------------------------------------+
        | logGBF             | log Gaussian Bayes Factor on the residuals                         |
        +--------------------+--------------------------------------------------------------------+
        | rchi2              | reduced chi2 computed on the residual                              |
        +--------------------+--------------------------------------------------------------------+
        | broadening         | broadening estimation of the lines                                 |
        +--------------------+--------------------------------------------------------------------+
        | broadening_err     | error on the brodeaning estimation                                 |
        +--------------------+--------------------------------------------------------------------+
        | velocity           | estimated velocity                                                 |
        +--------------------+--------------------------------------------------------------------+
        | velocity_err       | error on the fitted velocity                                       |
        +--------------------+--------------------------------------------------------------------+
        | flux_*             | flux estimation for * line, where * is the line name               |
        +--------------------+--------------------------------------------------------------------+
        | flux_*_err         | error on the flux estimation for * line, where * is the line name  |
        +--------------------+--------------------------------------------------------------------+
        | snr_*              | Estimated SNR of the * line                                        |
        +--------------------+--------------------------------------------------------------------+
        | bkg_v_guess        | guess on the background spectrum velocity                          |
        +--------------------+--------------------------------------------------------------------+
        | bkg_exit_status    | code to identify cause of crash                                    |
        +--------------------+--------------------------------------------------------------------+
        | bkg_velocity       | estimated velocity of the background spectrum                      |
        +--------------------+--------------------------------------------------------------------+
        | bkg_velocity_err   | error on the background velocity estimation                        |
        +--------------------+--------------------------------------------------------------------+


    See Also
    --------
    :func:`fit_SN3`
    """
    fit_res = {}
    try: # Try catch to avoid a stupid crash during a long loop over many sources
        x, y = map(int, source[['xpos', 'ypos']])

        big_box = centered_square_region(x,y,30)
        medium_box_bkg = centered_square_region(15,15,15)
        small_box = centered_square_region(15,15, 3)
        data = cube._extract_spectra_from_region(big_box, silent=True)
        mask = np.ones((30, 30))
        mask[medium_box_bkg] = 0
        bkg_spec = np.nanmedian(data[np.nonzero(mask)], axis=0)

        medium_box = centered_square_region(15,15,5)
        mask = np.ones((30, 30))
        mask[medium_box] = 0
        data -= np.nanmedian(data[np.nonzero(mask)], axis=0)

        a = cube.params.base_axis
        imin,imax = np.searchsorted(a, cube.params.filter_range)
        s = np.nansum(data[small_box], axis=0)

        theta_orig = np.nanmean(cube.get_theta_map()[centered_square_region(x,y,3)])


        try: # Spec FIT
            spec = s[imin+5:imax-5]
            axis = a[imin+5:imax-5]
            mean,_,err = stats_without_lines(spec, axis,
                                          SN2_LINES, -1300., 700.)
            fit_res['err'] = err
            fit_res['guess_snr'] = np.nanmax((spec - mean) / err)

            if lines is None:
                lines = SN2_LINES
            if v_guess is None:
                if 'v_guess' in source.keys():
                    v_guess = source['v_guess']
                    if (v_guess > v_max) | (v_guess < v_min):
                        lines = ['Hbeta']
            if v_guess is None:
                v_guess, l = guess_source_velocity(s, cube, v_min = v_min, v_max = v_max, debug=debug, return_line=True)
            if v_guess is None:
                fit_res['exit_status'] = 1
                v_guess, l = guess_source_velocity(s, cube, v_min = -np.inf, v_max = np.inf, lines=['Hbeta'], debug=debug, return_line=True)
                lines = ['Hbeta']
            try:
                logging.info('V_guess before refine %s'%v_guess)
                coeff = np.nanmax(spec)
                v_guess = refine_velocity_guess(spec/coeff, axis, v_guess, l)
                logging.info('V_guess after refine %s'%v_guess)
            except Exception as e:
                pass
            fit_res['v_guess'] = v_guess
            if 'fmodel' not in kwargs_spec:
                kwargs_spec['fmodel'] = 'sinc'
            kwargs_spec['pos_def']=['1']
            if 'signal_range' not in kwargs_spec:
                kwargs_spec['signal_range'] = cube.params.filter_range
            kwargs_spec['pos_cov'] = v_guess

            cube._prepare_input_params(lines, nofilter=True, **kwargs_spec)
            fit_params = fit_lines_in_spectrum(cube.params, cube.inputparams, cube.fit_tol,
                                                s, theta_orig,
                                                snr_guess=err,  debug=debug)
            # fit_params = cube._fit_lines_in_spectrum(s, theta_orig, snr_guess=err)
#             _,_,fit_params = cube.fit_lines_in_integrated_region(centered_square_region(x,y,3), SN2_LINES,
#                                                               nofilter=True, snr_guess=err,
#                                                               subtract_spectrum=sub_spec, **kwargs_spec)

            if fit_params == []:
                fit_res['exit_status'] = 2
            else:
                fit_res['exit_status'] = 0
                keys_to_keep = ['chi2', 'rchi2', 'ks_pvalue', 'logGBF']
                fit_res.update({k:v for (k,v) in fit_params.items() if k in keys_to_keep})
                fit_res['broadening'] = fit_params['broadening'][0]
                fit_res['broadening_err'] = fit_params['broadening_err'][0]
                fit_res['velocity'] = fit_params['velocity'][0]
                fit_res['velocity_err'] = fit_params['velocity_err'][0]

                # !! has to be in the same order than in fit_spectrum function
                for j, l in enumerate(lines):
                    line_name = l.lower().replace('[', '').replace(']', '')
                    fit_res['flux_%s'%line_name] = fit_params['flux'][j]
                    fit_res['flux_%s_err'%line_name] = fit_params['flux_err'][j]
                    fit_res['snr_%s'%line_name] = fit_params['snr'][j]
        except Exception as e:
            print e
            pass
        try: #BKG velocity
            bkg_err = np.nanstd(np.concatenate([bkg_spec[:imin-40], bkg_spec[imax+40:]]))
            kwargs_bkg.update({'fmodel':'gaussian'})
            kwargs_bkg.update({'fwhm_def':['1']})
            kwargs_bkg.update({'signal_range':(19500,20500)})

            lines = ['[OIII]5007']
            v_guess, l = guess_source_velocity(bkg_spec, cube, lines=lines, force=True, return_line=True)
            try:
                logging.info('V_guess before refine %s'%v_guess)
                coeff = np.nanmax(bkg_spec)
                v_guess = refine_velocity_guess(bkg_spec/coeff, a, v_guess, l)
                logging.info('V_guess after refine %s'%v_guess)
            except Exception as e:
                pass
            fit_res['bkg_v_guess'] = v_guess
            kwargs_bkg['pos_cov'] = v_guess
            kwargs_bkg['pos_def'] = ['1']
            cube.inputparams = {}
            cube._prepare_input_params(lines, nofilter = True, **kwargs_bkg)
            fit_params_bkg = fit_lines_in_spectrum(cube.params, cube.inputparams, cube.fit_tol,
                                                bkg_spec, theta_orig,
                                                snr_guess=bkg_err,  debug=debug)

            if fit_params_bkg == []:
                fit_res['bkg_exit_status'] = 2
            else:
                fit_res['bkg_exit_status'] = 0
                fit_res['bkg_velocity'] = fit_params_bkg['velocity'][0]
                fit_res['bkg_velocity_err'] = fit_params_bkg['velocity_err'][0]

        except Exception as e:
            print e
            pass
    except Exception as e:
        print e
        fit_res['exit_status'] = 3
    if return_fit_params:
        return pd.Series(fit_res), fit_params
    else:
        return pd.Series(fit_res)

def fit_SN3(source, cube, v_guess = None, lines=None, return_fit_params = False, kwargs_spec={}, kwargs_bkg = {}, debug=False):
    """
    Function specialized to fit sources found in the SN3 cube. The background spectrum is not fitted, due to the presence of sky lines that would bias the estimation of the velocity.
    It looks very similar to :func:`fit_SN2`and the code should be refactored.
    It differs however in the philosophy behind the velocity estimation : this method has been designed to be performed after a SN2 fit, from which we already estimated the velocity of the source. Thus no guess is performed here.

    The method can be used in a parallel process.
    The SNR of the spec is estimated using the :func:`stats_without_lines` method.

    Parameters
    ---------
    source : :class:`~pandas:pandas.Series`
        A row from a :class:`~pandas:pandas.DataFrame` containing detected sources. Should have columns ``xpos``, ``ypos``, assumed to correspond to the SN2 pixel coordinates.
    cube : :class:`~ORCS:orcs.process.SpectralCube`
        The cube taken in SN3 filter.
    v_guess : float
        (Optional) If None, looking for the element ``source['v_guess']``
    lines : list of str
        (Optinal) Names of the lines to fit. If None, **SN3_LINES** are used, except if no v_guess has been found; then we assume it's only a Hbeta line at very fast velocity.
    return_fit_params : bool, Default = False
        (Optional) If True, returns the full parameters of the fit.
    kwargs_spec : dict
        (Optional) Additional keyword arguments to be used by :func:`fit_spectrum` when fitting the source spectrum.
    kwargs_bkg : dict
        (Optional) Additional keyword arguments to be used by :func:`fit_spectrum` when fitting the background spectrum.
    debug : bool, Default = False
        (Optional) If True, the velocity guess is verbose.

    Returns
    -------
    fit_res : dict
        A dict containg a lot of information about the fit.

        +--------------------+--------------------------------------------------------------------+
        | Parameter          | Description                                                        |
        +====================+====================================================================+
        | err                | estimated noise value on the spectra                               |
        +--------------------+--------------------------------------------------------------------+
        | guess_snr          | SNR guess                                                          |
        +--------------------+--------------------------------------------------------------------+
        | exit_status        | code to identify cause of crash                                    |
        +--------------------+--------------------------------------------------------------------+
        | v_guess            | guessed velocity in km/s                                           |
        +--------------------+--------------------------------------------------------------------+
        | chi2               | chi2 computed on the residual                                      |
        +--------------------+--------------------------------------------------------------------+
        | rchi2              | reduced chi2 computed on the residual                              |
        +--------------------+--------------------------------------------------------------------+
        | ks_pvalue          | ks test computed on the residuals                                  |
        +--------------------+--------------------------------------------------------------------+
        | logGBF             | log Gaussian Bayes Factor on the residuals                         |
        +--------------------+--------------------------------------------------------------------+
        | rchi2              | reduced chi2 computed on the residual                              |
        +--------------------+--------------------------------------------------------------------+
        | broadening         | broadening estimation of the lines                                 |
        +--------------------+--------------------------------------------------------------------+
        | broadening_err     | error on the brodeaning estimation                                 |
        +--------------------+--------------------------------------------------------------------+
        | velocity           | estimated velocity                                                 |
        +--------------------+--------------------------------------------------------------------+
        | velocity_err       | error on the fitted velocity                                       |
        +--------------------+--------------------------------------------------------------------+
        | flux_*             | flux estimation for * line, where * is the line name               |
        +--------------------+--------------------------------------------------------------------+
        | flux_*_err         | error on the flux estimation for * line, where * is the line name  |
        +--------------------+--------------------------------------------------------------------+
        | snr_*              | Estimated SNR of the * line                                        |
        +--------------------+--------------------------------------------------------------------+


    See Also
    --------
    :func:`fit_SN2`
    """
    fit_res = {}
    try:
        x, y = map(int, source[['xpos', 'ypos']])

        big_box = centered_square_region(x,y,30)
        medium_box_bkg = centered_square_region(15,15,15)
        data = cube._extract_spectra_from_region(big_box, silent=True)
        mask = np.ones((30, 30))
        mask[medium_box_bkg] = 0
        bkg_spec = np.nanmedian(data[np.nonzero(mask)], axis=0)

        medium_box = centered_square_region(15,15,5)
        small_box = centered_square_region(15,15, 3)
        mask = np.ones((30, 30))
        mask[medium_box] = 0
        data -= np.nanmedian(data[np.nonzero(mask)], axis=0)

        a = cube.params.base_axis
        imin,imax = np.searchsorted(a, cube.params.filter_range)
        s = np.nansum(data[small_box], axis=0)

        theta_orig = np.nanmean(cube.get_theta_map()[centered_square_region(x,y,3)])

        try: # Spec FIT
            mean,_,err = stats_without_lines(s[imin+5:imax-5], a[imin+5:imax-5],
                                          SN3_LINES, -1300., 700.)
            fit_res['err'] = err
            fit_res['guess_snr'] = np.nanmax((s[imin+5:imax-5] - mean) / err)

            if lines is None:
                lines = SN3_LINES
            if v_guess is None:
                if 'v_guess' in source.keys():
                    v_guess = source['v_guess']
            fit_res['v_guess'] = v_guess
            if 'fmodel' not in kwargs_spec:
                kwargs_spec['fmodel'] = 'sinc'
            if 'pos_def' not in kwargs_spec:
                kwargs_spec['pos_def']=['1']
            if 'signal_range' not in kwargs_spec:
                kwargs_spec['signal_range'] = cube.params.filter_range
            kwargs_spec['pos_cov'] = v_guess

            cube._prepare_input_params(lines, nofilter=True, **kwargs_spec)
            fit_params = fit_lines_in_spectrum(cube.params, cube.inputparams, cube.fit_tol,
                                                s, theta_orig,
                                                snr_guess=err,  debug=debug)
#             _,_,fit_params = cube.fit_lines_in_integrated_region(centered_square_region(x,y,3), SN2_LINES,
#                                                               nofilter=True, snr_guess=err,
#                                                               subtract_spectrum=sub_spec, **kwargs_spec)
            if fit_params == []:
                fit_res['exit_status'] = 2
            else:
                fit_res['exit_status'] = 0
                keys_to_keep = ['chi2', 'rchi2', 'ks_pvalue', 'logGBF']
                fit_res.update({k:v for (k,v) in fit_params.items() if k in keys_to_keep})
                fit_res['broadening'] = fit_params['broadening'][0]
                fit_res['broadening_err'] = fit_params['broadening_err'][0]
                fit_res['velocity'] = fit_params['velocity'][0]
                fit_res['velocity_err'] = fit_params['velocity_err'][0]

                # !! has to be in the same order than in fit_spectrum function
                for j, l in enumerate(lines):
                    line_name = l.lower().replace('[', '').replace(']', '')
                    fit_res['flux_%s'%line_name] = fit_params['flux'][j]
                    fit_res['flux_%s_err'%line_name] = fit_params['flux_err'][j]
                    fit_res['snr_%s'%line_name] = fit_params['snr'][j]
        except Exception as e:
            pass
    except Exception as e:
        print e
        fit_res['exit_status'] = 3
    if return_fit_params:
        return pd.Series(fit_res), fit_params
    else:
        return pd.Series(fit_res)
