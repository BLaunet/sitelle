import numpy as np
from scipy.interpolate import UnivariateSpline

def gen_wavelength_header(h, axis):
    h['NAXIS3'] = len(axis)
    h['CRPIX3'] = 1
    h['CRVAL3'] = axis[0]
    h['CDELT3'] = axis[1]-axis[0]
    h['CUNIT3'] = 'Angstroms'
    h['CTYPE3'] = 'LINEAR'
    return h

def read_wavelength_axis(header, axis):
    offset = 1-header['CRPIX'+str(axis)]
    grid =  np.arange(offset,header['NAXIS'+str(axis)] + offset)*header['CDELT'+str(axis)]
    return header['CRVAL'+str(axis)] + grid

def rebin(map, binsize, type):
    xsize, ysize = map.shape[:2]
    new_xsize = xsize//binsize if xsize % binsize == 0 else xsize//binsize + 1
    new_ysize = ysize//binsize if ysize % binsize == 0 else ysize//binsize + 1
    rebinned = np.zeros((new_xsize, new_ysize)+ map.shape[2:])

    for x,y in np.ndindex((new_xsize, new_ysize)):
        data = map[x*binsize:(x+1)*binsize, y*binsize:(y+1)*binsize, ...]
        if type == 'ERR':
            rebinned[x,y, ...] = np.sqrt( np.nansum(  np.square( data )))
        else:
            rebinned[x, y, ...] = np.nanmean(data, (0,1))
    return rebinned
def wavelength_regrid(cube, rebinned, only_bandpass, type, header=None):
    ## CREATION OF THE NEW AXIS
    base_axis = cube.params.base_axis.astype(float)
    irreg_axis = 1e8/base_axis
    reg_axis = np.linspace(irreg_axis[0], irreg_axis[-1], base_axis.shape[0] )
    #We flip it to make it wavelength ascending
    reg_axis = np.flip(reg_axis, 0)

    if only_bandpass:
        #We get extremas of the filter bandpass
        wl_min, wl_max = list(reversed([1e8/x for x in cube.params.filter_range]))
        #We conserve only spectrum inside this bandpass
        i_min = np.searchsorted(reg_axis, wl_min)
        i_max = np.searchsorted(reg_axis, wl_max)
        reg_axis = reg_axis[i_min:i_max]
    new_axis = 1e8/reg_axis #back in cm-1 to evaluate the spectrum function on it


    ##INTERPOLATION OF THE CUBE
    wl_cube = np.zeros((rebinned.shape[0], rebinned.shape[1], len(new_axis)))

    def interpolator(spectrum, old_axis, new_axis, type):
        f = UnivariateSpline(old_axis, spectrum, s=0)
        res = f(new_axis)
        if type == 'FWHM':
            res = np.array([r*1e8/n**2 for r,n in zip(res,new_axis)])
        return res

    if type != 'ERR':
        wl_cube = np.apply_along_axis(interpolator, 2, rebinned,
                                                    old_axis=base_axis,
                                                    new_axis=new_axis,
                                                    type = type)
    else:
        wl_cube = rebinned[:,:,:len(new_axis)]

    ## header
    if header is None:
        header = cube.get_header()
    h = gen_wavelength_header(header, reg_axis)

    return wl_cube, h

def nm_to_cm1(spectrum, nm_axis, out_axis):
    cm1_axis = np.flip(1e8/nm_axis.astype(float),0)
    spectrum = np.flip(spectrum, 0)
    cm1_spectrum = UnivariateSpline(cm1_axis, spectrum.astype(float), s=0, k=1,ext=1)(out_axis.astype(float))
    return cm1_spectrum

def swap_header_axis(h, a0, a1):
    keywords = ['NAXIS', 'CRPIX', 'CRVAL', 'CDELT', 'CUNIT', 'CTYPE']
    tmp = {}
    for k in keywords:
        tmp[k] = h[k+str(a0)]
        h[k+str(a0)] = h[k+str(a1)]
        h[k+str(a1)] = tmp[k]

    for k in h.keys():
        if 'PC' in k:
            if str(a0) in k:
                h[k.replace(str(a0), str(a1))] = h.pop(k)
    return h
