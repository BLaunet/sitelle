from orb.utils import io
theta_map = io.read_fits()
    secondary_map= kwargs.pop('secondary_map', None)
    if secondary_map is None:
        return parallel_apply_along_axis(func1d, 2, cube, *args, **kwargs)

    ncpus = multiprocessing.cpu_count()
    chunks = [(func1d, sub_cube, sub_map, args, kwargs)
              for sub_cube, sub_map in zip(np.array_split(cube, ncpus), np.array_split(secondary_map, ncpus))]

    #pool = multiprocessing.Pool()
    #individual_results = pool.map(_func_vec, chunks)
    # Freeing the workers:
    #pool.close()
    #pool.join()

    #return np.concatenate(individual_results)

    job_server, ncpus = init_pp_server(ncpus,silent=False)
    jobs = [job_server.submit(
        _func_vec,
        args=(sub_cube, sub_map, args, kwargs),
        modules=('import orb.utils.spectrum',
                'import orcs.utils',
                'import orb.fit',
                'import numpy as np',
                'import logging',
                'import warnings',
                'import gvar)')
        depfuncs=(func1d,))
            for sub_cube, sub_map in zip(np.array_split(cube, ncpus), np.array_split(secondary_map, ncpus))]

    for job in jobs:
        individual_results = job()
    close_pp_server(job_server)
    return individual_results

def _func_vec((func1d, sub_cube, sub_map, args, kwargs)):
    result = np.empty(sub_cube.shape)
    for x, y in np.ndindex(sub_cube.shape[:2]):
        result[x, y, :] = func1d(sub_cube[x, y, :], sub_map[x, y], *args, **kwargs)
    return result
utils.fit_lines_in_spectrum(
    self.params, self.inputparams, self.fit_tol,
    spectrum, theta_orig, snr_guess=snr_guess, **kwargs)
