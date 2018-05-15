# coding: utf-8

import multiprocessing
import numpy as np
from orb.utils.parallel import init_pp_server, close_pp_server
import pandas as pd
import dill
import os
import re
import subprocess

__all__ = ['available_cpu_count', 'parallel_apply_along_axis', 'apply_over_frames', 'parallel_apply_over_frames', 'parallel_apply_over_df']

def available_cpu_count():
    """
    Number of available virtual or physical CPUs on this system, i.e.
    user/real as output by time(1) when called with an optimally scaling userspace-only program
    """

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
    Apply a function to 1-D slices along the given axis.
    Like :func:`numpy:numpy.apply_along_axis()`, but takes advantage of multiple
    cores.

    Parameters
    ----------
    func1d : callable
        This function should accept 1-D arrays. It is applied to 1-D slice of **arr** along the specified axis.
    axis : integer
        Axis along which **arr** is sliced.
    arr : :class:`~numpy:numpy.ndarray`
        Input array
    modules : tuple of strings
        The modules to be imported so that **func1d** works correctly. Example : ('import numpy as np',)
    depfuncs : tuple of string
        The functions used by **func1d** but defined outside of its body
    args :
        Additional arguments to be passed  to **func1d**
    kwargs :
        Additional keywords arguments to be passed  to **func1d**

    Returns
    -------
    out : :class:`~numpy:numpy.ndarray`
        The output array. The shape of out is identical to the shape of **arr**, except along the **axis** dimension. This axis is removed, and replaced with new dimensions equal to the shape of the return value of **func1d**. So if **func1d** returns a scalar out will have one fewer dimensions than **arr**.

    See Also
    --------
    :func:`numpy:numpy.apply_along_axis()`
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
    """
    Iterates over the z dimension of a cube to apply a function on each frame (i.e. 2D images)

    Parameters
    ----------
    func2d : callable
        This function should accept 2-D arrays. It is applied to 2-D slices (frames) of **cube**.
    cube : 3-D :class:`~numpy:numpy.ndarray`
        The data cube. Dimensions should be [x,y,z]. **func2d** will be applied along the z dimension
    args :
        Additional arguments to be passed  to **func2d**
    kwargs :
        Additional keywords arguments to be passed  to **func2d**

    Returns
    -------
    out : :class:`~numpy:numpy.ndarray`
        The output array. The first 2 dimensions are identical to **cube**. the rest depends on what **func2d** returns; it it returns a vector, **out** will be 3-D.
    """
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
    Iterates over the z dimension of a cube to apply a function on each frame (i.e. 2D images), taking advantage of multiple cores.

    Parameters
    ----------
    func2d : callable
        This function should accept 2-D arrays. It is applied to 2-D slices (frames) of **cube**.
    cube : 3-D :class:`~numpy:numpy.ndarray`
        The data cube. Dimensions should be [x,y,z]. **func2d** will be applied along the z dimension
    modules : tuple of strings
        The modules to be imported so that **func2d** works correctly. Example : ('import numpy as np',)
    depfuncs : tuple of string
        The functions used by **func2d** but defined outside of its body
    args :
        Additional arguments to be passed  to **func2d**
    kwargs :
        Additional keywords arguments to be passed  to **func2d**

    Returns
    -------
    out : :class:`~numpy:numpy.ndarray`
        The output array. The first 2 dimensions are identical to **cube**. the rest depends on what **func2d** returns; it it returns a vector, **out** will be 3-D.

    See Also
    --------
    `apply_over_frames`
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
    The signature is similar to :func:`pandas:pandas.DataFrame.apply()`

    Parameters
    ----------
    df : :class:`~pandas:pandas.DataFrame`
        The input dataframe
    func : callable
        The function to apply to each line (should accept a :class:`~pandas:pandas.Series` as input)
    axis : {0 or 'index', 1 or 'columns'}, default 0
        | 0 or 'index': apply function to each column
        | 1 or 'columns': apply function to each row
    broadcast : boolean, default False
        For aggregation functions, return object of same size with values propagated
    raw : boolean, default False
        If False, convert each row or column into a Series. If raw=True the passed function will receive ndarray objects instead. If you are just applying a NumPy reduction function this will achieve much better performance
    reduce : boolean or None, default None
        Try to apply reduction procedures. If the DataFrame is empty, apply will use reduce to determine whether the result should be a Series or a DataFrame. If reduce is None (the default), apply’s return value will be guessed by calling func an empty Series (note: while guessing, exceptions raised by func will be ignored). If reduce is True a Series will always be returned, and if False a DataFrame will always be returned
    modules : tuple of strings
        The modules to be imported so that **func** works correctly. Example : ('import numpy as np',)
    depfuncs : tuple of string
        The functions used by **func** but defined outside of its body
    args :
        Additional arguments to be passed  to **func**
    kwargs :
        Additional keywords arguments to be passed  to **func**
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
