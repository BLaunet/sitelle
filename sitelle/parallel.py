import multiprocessing
import numpy as np
from orb.utils.parallel import init_pp_server, close_pp_server

def parallel_apply_over_map(func1d, cube, *args, **kwargs):
    """
    Apply a function over each cell of a 3D cube
    :param func_spectrum: function to apply to each spectrum
    :param cube: cube; should be [x, y, z]
    :param map_2: secondary map [x, y] from which arguments have to be taken for the func_spectrum
    :param args: args of func_spectrum
    :params kwargs: kwargs of func_spectrum
    """
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
        modules=('import numpy as np',
                 'import orb.utils.spectrum',
                 'import orb.utils.vector'),
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
