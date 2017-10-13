from orb.fit import Lines
import pandas as pd
import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit

def sky_model_to_remove(spectrum, input_axis, skymodel, nb_pixels):

    sky_interpolator = UnivariateSpline(input_axis, skymodel, s=0)
    fit_vector = np.zeros_like(spectrum)

    imin, imax = np.searchsorted(input_axis, [14700,15430])
    a = input_axis[imin:imax]
    s = spectrum[imin:imax]/nb_pixels
    scale = np.nanmax(s) - np.nanmedian(s)
    def model(x, h, v):
        shift = line_shift(v, 15000)
        sky_shifted = sky_interpolator(a.astype(float)+shift)
        return h + UnivariateSpline(a, sky_shifted, s=0)(x)

    popt, pcov = curve_fit(model, a, s, p0=[scale,-75.])
    fit_vector[imin:imax] = model(a, *popt)-popt[0]

    return fit_vector

def parse_line_params(rest_lines, fit, error = True, wavenumber = True):
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
