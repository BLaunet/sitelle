import multiprocessing
import numpy as np
from orb.utils.parallel import init_pp_server, close_pp_server

def parallel_apply_along_axis(func1d, axis, arr, *args, **kwargs):
    """
    Like numpy.apply_along_axis(), but takes advantage of multiple
    cores.
    -------------
    Additional keywords :
    :param modules: modules to be imported so that the function works correctly. Example : 'import numpy as np'
    :param depfuncs: functions used by the main function but defined outside of its body
    """
    job_server, ncpus = init_pp_server(multiprocessing.cpu_count(),silent=False)

    chunks = [(func1d, axis, sub_arr, args, kwargs)
              for sub_arr in np.array_split(arr, ncpus)]

    def helper(func1d, axis, arr, args, kwargs):
        return np.apply_along_axis(func1d, axis, arr, *args, **kwargs)

    modules = kwargs.pop('modules', tuple())
    modules += ('import numpy as np',)

    depfuncs = kwargs.pop('depfuncs', tuple())
    depfuncs += (func1d,)
    jobs = [job_server.submit(
                helper,
                args=(c),
                modules=(modules),
                depfuncs=(depfuncs))
                for c in chunks]

    job_output = [job() for job in jobs]
    close_pp_server(job_server)

    return np.concatenate(job_output)
