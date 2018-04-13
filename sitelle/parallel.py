import multiprocessing
import numpy as np
from orb.utils.parallel import init_pp_server, close_pp_server
import pandas as pd
import dill
import os
import re
import subprocess


def available_cpu_count():
    """ Number of available virtual or physical CPUs on this system, i.e.
    user/real as output by time(1) when called with an optimally scaling
    userspace-only program"""

    # cpuset
    # cpuset may restrict the number of *available* processors
    try:
        m = re.search(r'(?m)^Cpus_allowed:\s*(.*)$',
                      open('/proc/self/status').read())
        if m:
            res = bin(int(m.group(1).replace(',', ''), 16)).count('1')
            if res > 0:
                return res
    except IOError:
        pass

    # Python 2.6+
    try:
        import multiprocessing
        return multiprocessing.cpu_count()
    except (ImportError, NotImplementedError):
        pass

    # https://github.com/giampaolo/psutil
    try:
        import psutil
        return psutil.cpu_count()   # psutil.NUM_CPUS on old versions
    except (ImportError, AttributeError):
        pass

    # POSIX
    try:
        res = int(os.sysconf('SC_NPROCESSORS_ONLN'))

        if res > 0:
            return res
    except (AttributeError, ValueError):
        pass

    # Windows
    try:
        res = int(os.environ['NUMBER_OF_PROCESSORS'])

        if res > 0:
            return res
    except (KeyError, ValueError):
        pass

    # jython
    try:
        from java.lang import Runtime
        runtime = Runtime.getRuntime()
        res = runtime.availableProcessors()
        if res > 0:
            return res
    except ImportError:
        pass

    # BSD
    try:
        sysctl = subprocess.Popen(['sysctl', '-n', 'hw.ncpu'],
                                  stdout=subprocess.PIPE)
        scStdout = sysctl.communicate()[0]
        res = int(scStdout)

        if res > 0:
            return res
    except (OSError, ValueError):
        pass

    # Linux
    try:
        res = open('/proc/cpuinfo').read().count('processor\t:')

        if res > 0:
            return res
    except IOError:
        pass

    # Solaris
    try:
        pseudoDevices = os.listdir('/devices/pseudo/')
        res = 0
        for pd in pseudoDevices:
            if re.match(r'^cpuid@[0-9]+$', pd):
                res += 1

        if res > 0:
            return res
    except OSError:
        pass

    # Other UNIXes (heuristic)
    try:
        try:
            dmesg = open('/var/run/dmesg.boot').read()
        except IOError:
            dmesgProcess = subprocess.Popen(['dmesg'], stdout=subprocess.PIPE)
            dmesg = dmesgProcess.communicate()[0]

        res = 0
        while '\ncpu' + str(res) + ':' in dmesg:
            res += 1

        if res > 0:
            return res
    except OSError:
        pass

    raise Exception('Can not determine number of CPUs on this system')

def parallel_apply_along_axis(func1d, axis, arr, *args, **kwargs):
    """
    Like numpy.apply_along_axis(), but takes advantage of multiple
    cores.
    -------------
    Additional keywords :
    :param modules: modules to be imported so that the function works correctly. Example : 'import numpy as np'
    :param depfuncs: functions used by the main function but defined outside of its body
    """
    job_server, ncpus = init_pp_server(available_cpu_count(),silent=False)

    axis_to_split_on = (axis + 1 )% 3
    if arr.shape[axis_to_split_on] < ncpus:
        ncpus = arr.shape[axis_to_split_on]

    chunks = [(func1d, axis, sub_arr, args, kwargs)
              for sub_arr in np.array_split(arr, ncpus, axis=axis_to_split_on)]

    def helper(func1d, axis, arr, args, kwargs):
        return np.apply_along_axis(func1d, axis, arr, *args, **kwargs)

    modules = tuple(kwargs.pop('modules', tuple()))
    modules += ('import numpy as np',)

    depfuncs = tuple(kwargs.pop('depfuncs', tuple()))
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

def apply_over_frames(func2d, cube, *args, **kwargs):
    #We have to determine dimension of what is returned
    res0 = np.array(func2d(cube[:,:,0], *args, **kwargs))
    res_cube = np.zeros((res0.shape + (cube.shape[-1], )))
    res_cube[...,0] = res0
    for i in range(1,(cube.shape[-1])):
        # print i
        res_cube[...,i] = func2d(cube[:,:,i], *args, **kwargs)
    return res_cube

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


    job_server, ncpus = init_pp_server(available_cpu_count(),silent=False)
    if cube.shape[-1]<ncpus:
        ncpus=cube.shape[-1]
    chunks = [(func2d, sub_cube, args, kwargs)
              for sub_cube in np.array_split(cube, ncpus, axis=2)]

    def helper(func2d, cube, args, kwargs):
        return apply_over_frames(func2d, cube, *args, **kwargs)

    modules = kwargs.pop('modules', tuple())
    modules += ('import numpy as np',)

    depfuncs = kwargs.pop('depfuncs', tuple())
    depfuncs += (func2d,apply_over_frames)
    jobs = [job_server.submit(
                helper,
                args=(c),
                modules=(modules),
                depfuncs=(depfuncs))
                for c in chunks]

    job_output = [job() for job in jobs]
    close_pp_server(job_server)
    try:
        return np.concatenate(job_output, axis=-1)
    except Exception as e:
        print 'Could not concatenate'
        return job_output

def parallel_apply_over_df(df, func, axis=1, broadcast=False, raw=False, reduce=None, args=(),  **kwargs):
    """
    Iterates over a pandas Dataframe to apply a function on each line,
    taking advantage of multiple cores.
    Arguments are the same as pd.DataFrame.apply()
    :param df: the dataframe
    :param func: the function to apply to each line (should take a pandas Series as input)
    :param modules: modules to be imported so that the function works correctly. Example : 'import numpy as np'
    :param depfuncs: functions used by the main function but defined outside of its body
    :param args: arguments to be passed to func
    :param kwargs: keyword arguments to be passed to func
    """
    modules = kwargs.pop('modules', tuple())
    modules += ('import numpy as np','import pandas as pd')

    depfuncs = kwargs.pop('depfuncs', tuple())
    depfuncs += (func,)

    job_server, ncpus = init_pp_server(available_cpu_count(),silent=False)
    if df.shape[0]<ncpus:
        ncpus=df.shape[0]
    chunks = [ (dill.dumps(df[i::ncpus]), func, axis, broadcast,
                raw, reduce, args, kwargs) for i in xrange(ncpus) ]

    def helper(df, func, axis, broadcast, raw, reduce, args, kwargs):
        import dill
        df = dill.loads(df)
        return df.apply(func, axis, broadcast, raw, reduce, args=args, **kwargs)

    jobs = [job_server.submit(
                helper,
                args=(c),
                modules=(modules),
                depfuncs=(depfuncs))
                for c in chunks]

    job_output = [job() for job in jobs]
    close_pp_server(job_server)
    try:
        return pd.concat(job_output).sort_index()
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
