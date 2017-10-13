import multiprocessing
import numpy as np
from orb.utils.parallel import init_pp_server, close_pp_server

# def parallel_apply_over_map(func1d, map, *args, **kwargs):
#     """
#     Apply a function over each cell of a 3D cube
#     :param func_spectrum: function to apply to each spectrum
#     :param cube: cube; should be [x, y, z]
#     :param map_2: secondary map [x, y] from which arguments have to be taken for the func_spectrum
#     :param args: args of func_spectrum
#     :params kwargs: kwargs of func_spectrum
#     """
#
#     ncpus = multiprocessing.cpu_count()
#     chunks = [(func1d, sub_map, args, kwargs)
#               for sub_map in zip(np.array_split(map, ncpus))]
#
# def _func_vec(sub_cube, sub_theta_map, sub_V_map, inputparams, params, lines):
#     def homemade_chi2(axis, spectrum, fit_params):
#         indices_to_keep = []
#         for line in fit_params['fitted_models']['Cm1LinesModel']:
#             indices_to_keep.append(np.argwhere(np.abs(line)>=line.std()/2)[:,0])
#         to_keep = np.unique(np.concatenate(indices_to_keep))
#         sigma = np.std(spectrum)
#         residual = (spectrum-fit_params['fitted_vector'])[to_keep]
#         return np.sum(np.square(residual/sigma))
#
#     line_spectra = np.zeros(sub_cube.shape, dtype=float)
#     fit_params = np.zeros(sub_cube.shape[:2], dtype=object)
#     for x, y in np.ndindex(sub_cube.shape[:2]):
#         chi2_list = []
#         fit = fit_lines_in_spectrum(params, inputparams, 1e10, sub_cube[x, y, :],
#                               sub_theta_map[x, y], pos_cov=gvar.gvar(sub_V_map[x,y], 50), snr_guess=None)
#         if fit != []:
#
#             line_spectra[x, y, :] = np.sum(fit['fitted_models']['Cm1LinesModel'], 0)
#             fit_params[x,y] = parse_line_params(lines,fit)
#     return line_spectra, fit_params
#
# job_server, ncpus = init_pp_server(multiprocessing.cpu_count(),silent=False)
# jobs = [job_server.submit(
#             _func_vec,
#             args=(sub_cube, sub_theta_map, sub_V_map, inputparams.convert(), params.convert(), lines),
#             modules=('import numpy as np',
#                      'import orb.utils.spectrum',
#                      'import orb.utils.vector',
#                      'import orb.fit',
#                      'import gvar',
#                      'import logging',
#                      'import warnings',
#                      'from sitelle.fit import parse_line_params'),
#             depfuncs=((fit_lines_in_spectrum,)))
#                 for sub_cube, sub_theta_map, sub_V_map in zip(np.array_split(cube_binned, ncpus),
#                                                               np.array_split(theta_binned, ncpus),
#                                                               np.array_split(velocity_guess_map_binned, ncpus))]
# job_output = [job() for job in jobs]
# close_pp_server(job_server)
#
# line_spectra = np.concatenate([j[0] for j in job_output])
# fit_params = np.concatenate([j[1] for j in job_output])
#
# def _func_vec(func1d, sub_cube, sub_map, args, kwargs):
#     result = np.empty(sub_cube.shape)
#     for x, y in np.ndindex(sub_cube.shape[:2]):
#         result[x, y, :] = func1d(sub_cube[x, y, :], sub_map[x, y], *args, **kwargs)
#     return result
def parallel_apply_along_axis(func1d, axis, arr, *args, **kwargs):
    """
    Like numpy.apply_along_axis(), but takes advantage of multiple
    cores.
    """

    # Chunks for the mapping (only a few chunks):
    chunks = [(func1d, axis, sub_arr, args, kwargs)
              for sub_arr in np.array_split(arr, multiprocessing.cpu_count())]

    pool = multiprocessing.Pool()
    individual_results = pool.map(_unpacking_apply_along_axis, chunks)
    # Freeing the workers:
    pool.close()
    pool.join()

    return np.concatenate(individual_results)
def _unpacking_apply_along_axis((func1d, axis, arr, args, kwargs)):
    """
    Like numpy.apply_along_axis(), but and with arguments in a tuple
    instead.

    This function is useful with multiprocessing.Pool().map(): (1)
    map() only handles functions that take a single argument, and (2)
    this function can generally be imported from a module, as required
    by map().
    """
    return np.apply_along_axis(func1d, axis, arr, *args, **kwargs)
