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
              for sub_arr in np.array_split(arr, ncpus, axis=((axis + 1 )% 3))]

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

def parallel_apply_over_frames(func2d, cube, *args, **kwargs):
    """
    Iterates over the z dimension of a cube to apply a function on each frame,
    taking advantage of multiple cores.

    :param func2d: the function to apply to each frame
    :param cube: the data cube
    :param modules: modules to be imported so that the function works correctly. Example : 'import numpy as np'
    :param depfuncs: functions used by the main function but defined outside of its body
    :param args: arguments to be passed to func2d
    :param kwargs: keyword arguments to be passed to func2d
    """
    try:
        if len(cube.shape) != 3:
            raise ValueError('Shape of the cube must be 3 dimensional; got %s instead'%(cube.shape))
    except AttributeError as e:
        raise TypeError('Incorrect type for the data cube : %s'%type(a))


    job_server, ncpus = init_pp_server(multiprocessing.cpu_count(),silent=False)
    if cube.shape[-1]<ncpus:
        ncpus=cube.shape[-1]
    chunks = [(func2d, sub_cube, args, kwargs)
              for sub_cube in np.array_split(cube, ncpus, axis=2)]

    def helper(func2d, cube, args, kwargs):
        #We have to determine dimension of what is returned
        res0 = np.array(func2d(cube[:,:,0], *args, **kwargs))
        return cube
        # return res0.shape
        # res_cube = np.zeros((res0.shape + (cube.shape[-1], )))
        # for i in range(1,(cube.shape[-1])):
        #     res_cube[...,i] = func2d(cube[:,:,i], *args, **kwargs)
        # return res_cube

    modules = kwargs.pop('modules', tuple())
    modules += ('import numpy as np',)

    depfuncs = kwargs.pop('depfuncs', tuple())
    depfuncs += (func2d,)
    jobs = [job_server.submit(
                helper,
                args=(c),
                modules=(modules),
                depfuncs=(depfuncs))
                for c in chunks]

    job_output = [job() for job in jobs]
    close_pp_server(job_server)
    return job_output
    try:
        return np.concatenate(job_output, axis=-1)
    except Exception as e:
        print 'Could not concatenate'
        return job_output




# def parallel_loop(func, iterator, *args, **kwargs):
#     job_server, ncpus = init_pp_server(multiprocessing.cpu_count(),silent=False)
#     it_list = list(iterator)
#     it_chunks = [it_chunk for it_chunk in np.array_split(it_list, ncpus)]
#
#     def helper(func, iterator, args, kwargs):
#         out = []
#         for i in iterator:
#             out.append(func(i, *args, **kwargs))
#         return out
#
#     modules = kwargs.pop('modules', tuple())
#     modules += ('import numpy as np',)
#
#     depfuncs = kwargs.pop('depfuncs', tuple())
#     depfuncs += (func,)
#     jobs = [job_server.submit(
#                 helper,
#                 args=(func, it_chunk, args, kwargs),
#                 modules=(modules),
#                 depfuncs=(depfuncs))
#                 for it_chunk in it_chunks]
#
#     job_output = []
#     for job in jobs:
#         job_output.append(job())
#
#     close_pp_server(job_server)
#     return np.array(job_output).flatten()
