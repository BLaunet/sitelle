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

import matplotlib.pyplot as plt

from sitelle.plot import *
def sky_model_to_remove(mean_spectrum, axis, sky_axis, sky_model):
    """
    This function shifts the velocity of a skymodel to match at best the skylines in a spectrum.
    It returns the fitted vector, that shoukld be then subtracted from the original spectrum.
    :param mean_spectrum: The spectrum containing skylines.
    If it comes from an integrated region, pay attention to give the **mean** spectrum,
    i.e divided by the number of integrated pixels
    :param axis: Axis of the spectrum
    :param sky_axis: axis on which the skymodel is known
    :param skymodel: Sky spectrum to be fitted
    :return: the shifted skymodel, to be removed
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
    Parse the fitted parameters to display them as a nice pandas DataFrame
    :param rest_lines: Name or position of the lines at rest
    :param fit: the 'fit_params' in the output of a fit from orcs
    :param error: if True, the errors are also parsed (Default = True)
    :param wavenumber: if True, output is in wavenumber (else in wavelength). Default = True
    :return lines_df: a pandas Dataframe containing all information about the fitted lines
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
    Computes a chi square only around fitted lines
    :param axis: axis of the spectrum
    :param spectrum: spectrum on whoch the fit has been performed
    :param fit_params: fit parameters containing information about the lines (output from an orcs fit)
    :param delta: delta around line peak that should be used to compute the chi2. Default = 8
    :return chi2: a chi square value, computed only around +- sigma * fwhm each line
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
    To be used with care, especially when there is a real Halpha signal there.
    :param spectrum: the spectrum to fit
    :param theta: theta value corresponding to the spectrum
    :param cube: Spectral cube instance from which the spctrum is taken
    :return: the spectrium whithout this line (hopefully)
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
    :param z_dim: 2D-array. z_dim[0] contains the spectrum to fit, z_dim[1] contains theta, the incident angle at which the spectrum was recorded
    :param inputparams: output from cube._prepare_input_params()
    :param params: cube.params.convert()
    :param lines: name of the lines to fit  (understanble by ORCS)
    :param V_range: the range of velocity to test
    :param snr_guess: snr_guess (Default = None)
    :return out: a cube containing at each cell a dict with 'line_spectra': the fitted line spectra,
    'fit_params': the fitetd parameters of the lines,
    'chi2': chi2 list, each elemen,t corresponds to a velocity in V_range
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

def guess_line_velocity(max_pos, v_min, v_max, lines = None,debug=False, return_line=False):
    """
    From a line position, we estimate the velocity shift
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
    ax[0].set_xlim(SN2_ORCS.get_filter_range())
    make_wavenumber_axes(ax[0])
    add_lines_label(ax[0], 'SN2', -300., wavenumber=True)

    plot_spectra(a_SN3, s_SN3, ax=ax[1])
    if fit_SN3 != []:
        plot_spectra(a_SN3, fit_SN3['fitted_vector'], ax=ax[1])
    ax[1].set_xlim(SN3_ORCS.get_filter_range())
    make_wavenumber_axes(ax[1])
    add_lines_label(ax[1], 'SN3', -300., wavenumber=True)

    print 'SN2 fit velocity : %.2f +- %.2f'%(fit_SN2['velocity'][0], fit_SN2['velocity_err'][0])
    print 'SN3 fit velocity : %.2f +- %.2f'%(fit_SN3['velocity'][0], fit_SN3['velocity_err'][0])

    print source.filter(regex='detected')
    return fit_SN2, fit_SN3
