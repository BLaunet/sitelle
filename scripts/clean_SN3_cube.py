# -*- coding: utf-8 -*-
from orb.utils import io
import multiprocessing
import numpy as np
from orb.utils.parallel import init_pp_server, close_pp_server
from orcs.process import SpectralCube
from orcs.utils import fit_lines_in_spectrum
import gvar
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit
from sitelle.parallel import parallel_apply_along_axis
from orb.utils.spectrum import line_shift

SN3 = SpectralCube('../fits/orig/M31_SN3.merged.cm1.1.0.hdf5')
SN3.correct_wavelength('../fits/M31_SN3.1.0.ORCS/M31_SN3.1.0.skymap.fits')
theta_binned = io.read_fits('../fits/maps/M31_SN3_thetamap_48x48.fits')
cube_binned = io.read_fits('../fits/wavenumber_rebinned/M31_SN3_rebinned_48x48.fits')



velocity_guess_map_binned = np.load('../fits/maps/velocity_guess_map_48x48.npy')

##Smaller
#cube_binned = cube_binned[:10, :10, :]
#theta_binned = theta_binned[:10,:10]
#velocity_guess_map_binned = velocity_guess_map_binned[:10,:10]

### We remove sky lines
skymodel = io.read_fits('../fits/orig/sky_model_-75.7kms.fits')
axis = SN3.params.base_axis.astype(float)

def sky_model_to_remove(spectrum, input_axis, skymodel, nb_pixels):

    sky_interpolator = UnivariateSpline(axis, skymodel, s=0)
    fit_vector = np.zeros_like(spectrum)

    #Right part
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

to_subtract_cube = parallel_apply_along_axis(sky_model_to_remove, 2, cube_binned, axis, skymodel, 2304 )*2304.
#Correction pour la dernière colonne où on n'a intégré que 1536 spectres
to_subtract_cube[42, :, : ] = to_subtract_cube[42,:,:]*1536./2304.

cube_sky_clean = cube_binned - to_subtract_cube

#We fit the lines

lines=['[NII]6548','[NII]6583', 'Halpha', '[SII]6716', '[SII]6731']
kwargs={'fmodel':'sincgauss', 'pos_def':'1', 'snr':'auto', 'sigma_cov':1}

SN3._prepare_input_params(lines, nofilter=False, **kwargs)
inputparams = SN3.inputparams
params = SN3.params

def _func_vec(sub_cube, sub_theta_map, sub_V_map, inputparams, params, lines):
    def homemade_chi2(axis, spectrum, fit_params):
        indices_to_keep = []
        for line in fit_params['fitted_models']['Cm1LinesModel']:
            indices_to_keep.append(np.argwhere(np.abs(line)>=line.std()/2)[:,0])
        to_keep = np.unique(np.concatenate(indices_to_keep))
        sigma = np.std(spectrum)
        residual = (spectrum-fit_params['fitted_vector'])[to_keep]
        return np.sum(np.square(residual/sigma))

    line_spectra = np.zeros(sub_cube.shape, dtype=float)
    fit_params = np.zeros(sub_cube.shape[:2], dtype=object)
    for x, y in np.ndindex(sub_cube.shape[:2]):
        for v in np.linspace(0,-600,6):
            chi2_list = []
            spectra_list = []
            params_list = []

            fit = fit_lines_in_spectrum(params, inputparams, 1e10, sub_cube[x, y, :],
                                  sub_theta_map[x, y], pos_cov=gvar.gvar(sub_V_map[x,y], 50), snr_guess=None)
            if fit != []:
                hChi2 = homemade_chi2(params['base_axis'], sub_cube[x, y, :], fit)
                if hChi2 < 1:
                    line_spectra[x, y, :] = np.sum(fit['fitted_models']['Cm1LinesModel'], 0)
                    fit_params[x,y] = parse_line_params(lines,fit)
                    break
                else:
                    print 'Skipping chi2 = ', hChi2
                    chi2_list.append(hChi2)
                    spectra_list.append(np.sum(fit['fitted_models']['Cm1LinesModel'], 0))
                    params_list.append(parse_line_params(lines,fit))
            else:
                continue
        if v == -600: #On n'a pas vraiment converg
            if chi2_list != list():
                imin = np.argmin(chi2_list)
                line_spectra[x, y, :] = spectra_list[imin]
                fit_params[x,y] = params_list[imin]
    return line_spectra, fit_params

job_server, ncpus = init_pp_server(multiprocessing.cpu_count(),silent=False)
jobs = [job_server.submit(
            _func_vec,
            args=(sub_cube, sub_theta_map, sub_V_map, inputparams.convert(), params.convert(), lines),
            modules=('import numpy as np',
                     'import orb.utils.spectrum',
                     'import orb.utils.vector',
                     'import orb.fit',
                     'import gvar',
                     'import logging',
                     'import warnings',
                     'from sitelle.fit import parse_line_params'),
            depfuncs=((fit_lines_in_spectrum,)))
                for sub_cube, sub_theta_map, sub_V_map in zip(np.array_split(cube_binned, ncpus),
                                                              np.array_split(theta_binned, ncpus),
                                                              np.array_split(velocity_guess_map_binned, ncpus))]
job_output = [job() for job in jobs]
close_pp_server(job_server)

line_spectra = np.concatenate([j[0] for j in job_output])
fit_params = np.concatenate([j[1] for j in job_output])
#for job in jobs:
#    individual_results = job()
#    print individual_results.shape
#    res = np.concatenate(individual_results, 0)
#    print res.shape
np.save('gas_after_sky_model_fit.npy', line_spectra)
np.save('gas_after_sky_model_param.npy', fit_params)
