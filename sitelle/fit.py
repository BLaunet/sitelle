from orb.fit import Lines
import pandas as pd
import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit
from orcs.utils import fit_lines_in_spectrum
from orb.utils.spectrum import line_shift

def sky_model_to_remove(spectrum, input_axis, skymodel, nb_pixels, xlims=[14700,15430]):
    """
    This function shifts the velocity of a skymodel to match at best the skylines in a spectrum.
    It returns the fitted vector, that shoukld be then subtracted from the original spectrum.
    :param input_axis: Axis on which is known the skymodel
    :param spectrum: The spectrum containing skylines
    :param skymodel: Sky spectrum to be fitted_models
    :param nb_pixels: Number of pixels on which the original spectrum is integrated. Used to scale the skymodel
    :return fit_vector: the shifted skymodel, to be removed
    """
    c = 299792.458
    input_axis = np.log10(input_axis) #This way, a velocity shift correspond to a constant
                                      #wavenumber offset over the whole axis

    sky_interpolator = UnivariateSpline(input_axis, skymodel, s=0)
    fit_vector = np.zeros_like(spectrum)

    imin, imax = np.searchsorted(input_axis, np.log10(xlims))
    a = input_axis[imin:imax]
    s = spectrum[imin:imax]/nb_pixels
    scale = np.nanmax(s) - np.nanmedian(s)
    def model(x, h, v):
        shift = -np.log10(1-v/c) # minus sign because it's in wavenumber
        sky_shifted = sky_interpolator(a.astype(float)+shift)
        return h + UnivariateSpline(a, sky_shifted, s=0)(x)
    popt, pcov = curve_fit(model, a, s, p0=[scale,-75.])
    fit_vector[imin:imax] = model(a, *popt)-popt[0]
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

def fit_gas_lines(z_dim, inputparams, params, lines, V_range, snr_guess = None):
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
            print 'Init V = ', v, 'Fitted V = ', fit['velocity'][0], 'Chi2 = ', hChi2
    if chi2_list != list():
        imin = np.argmin(chi2_list)
        chi2 = chi2_list[imin]#{'chi2': chi2_list, 'v': V_range[imin]}
        line_spectra = spectra_list[imin]
        fit_params = params_list[imin]
    return {'line_spectra':line_spectra, 'fit_params':fit_params, 'chi2':chi2}
