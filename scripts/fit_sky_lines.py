#Classical fit of the skylines : can be very long and very wrong

from orb.utils import io
import multiprocessing
import numpy as np
from orb.utils.parallel import init_pp_server, close_pp_server
from orcs.process import SpectralCube
from orcs.utils import fit_lines_in_spectrum
import gvar

SN3 = SpectralCube('../fits/orig/M31_SN3.merged.cm1.1.0.hdf5')
SN3.correct_wavelength('../fits/M31_SN3.1.0.ORCS/M31_SN3.1.0.skymap.fits')
theta_binned = io.read_fits('../fits/maps/M31_SN3_thetamap_48x48.fits')
cube_binned = io.read_fits('../fits/wavenumber_rebinned/M31_SN3_rebinned_48x48.fits')

##Smaller
#cube_binned = cube_binned[:10, :10, :]
#theta_binned = theta_binned[:10,:10]
#velocity_guess_map_binned = velocity_guess_map_binned[:10,:10]

lines=SN3.get_sky_lines()
kwargs={'fmodel':'sinc', 'pos_def':'1', 'pos_cov':0,
        'fwhm_def':'fixed'}

SN3._prepare_input_params(lines, nofilter=False, **kwargs)
inputparams = SN3.inputparams
params = SN3.params

def _func_vec(sub_cube, sub_theta_map, inputparams, params, lines):
    line_spectra = np.zeros(sub_cube.shape, dtype=float)
    fit_params = np.zeros(sub_cube.shape[:2], dtype=object)
    for x, y in np.ndindex(sub_cube.shape[:2]):
        fit = fit_lines_in_spectrum(params, inputparams, 1e10, sub_cube[x, y, :],
                              sub_theta_map[x, y], snr_guess=None)
        if fit != []:
            line_spectra[x, y, :] = np.sum(fit['fitted_models']['Cm1LinesModel'], 0)
            fit_params[x,y] = parse_line_params(lines,fit)
    return line_spectra, fit_params

job_server, ncpus = init_pp_server(multiprocessing.cpu_count(),silent=False)
jobs = [job_server.submit(
            _func_vec,
            args=(sub_cube, sub_theta_map, inputparams.convert(), params.convert(), lines),
            modules=('import numpy as np',
                     'import orb.utils.spectrum',
                     'import orb.utils.vector',
                     'import orb.fit',
                     'import gvar',
                     'import logging',
                     'import warnings',
                     'from sitelle.fit import parse_line_params'),
            depfuncs=((fit_lines_in_spectrum,)))
                for sub_cube, sub_theta_map in zip(np.array_split(cube_binned, ncpus),
                                                              np.array_split(theta_binned, ncpus))]
job_output = [job() for job in jobs]
close_pp_server(job_server)

line_spectra = np.concatenate([j[0] for j in job_output])
fit_params = np.concatenate([j[1] for j in job_output])
#for job in jobs:
#    individual_results = job()
#    print individual_results.shape
#    res = np.concatenate(individual_results, 0)
#    print res.shape
np.save('skylines_fit.npy', line_spectra)
np.save('skylines_param.npy', fit_params)
