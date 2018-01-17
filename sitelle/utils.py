import numpy as np
from scipy.interpolate import UnivariateSpline
import orb
def estimate_noise(full_axis, full_spectrum, filter_lims, side='both'):
    '''
    Estimates the noise in an integrated spectra from sitelle, by measuring it outside of the filter range.
    :param full_axis: full wavenumber axis of the cube
    :param full_spectrum: spectrum on this full axis
    :param filter_lims: limits of the filter range (in wavenumber)
    :param side: (optional) if 'right' ('left'), only the right (left) side of the filter is considered. Default to 'both'
    :return: the standard deviaition of the noise outside the filter range
    '''
    imin,imax = np.searchsorted(full_axis,filter_lims)
    if side is 'right':
        noise = full_spectrum[imax+40:]
    elif side is 'left':
        noise = full_spectrum[:imin-40]
    elif side is 'both':
        noise = np.concatenate((full_spectrum[:imin-40], full_spectrum[imax+40:]))
    else:
        raise ValueError(side)
    return np.std(noise)



def gen_wavelength_header(h, axis, ax_num = 3):
    """
    Generates a FITS wavelength header in Ansgtroms
    :param h: the header to update
    :param axis: the regular wavelength axis (numpy array or list)
    :param ax_num: (Default 3) The axis number to use in the header
    :return: the updated header
    """
    h['NAXIS%d'%ax_num] = len(axis)
    h['CRPIX%d'%ax_num] = 1
    h['CRVAL%d'%ax_num] = axis[0]
    h['CDELT%d'%ax_num] = axis[1]-axis[0]
    h['CUNIT%d'%ax_num] = 'Angstroms'
    h['CTYPE%d'%ax_num] = 'LINEAR'
    return h

def read_wavelength_axis(header, axis):
    """
    Generates the right wavelength axis from a FITS header
    :param header: the header to consider
    :param axis: the number of the axis on which the wavelength axis is stored in the header
    :return: a numpy array describing the waveklength axis
    """
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

def regular_wl_axis(axis, xlims=None):
    """
    Converts a wavenumber axis ([cm-1]) into a regular (==equally spaced) wavelength axis ([Angstroms])
    :param axis: the input axis in cm-1
    :param xlims: optional : limits in cm-1 to reduce the range of the axis (typically : filter range)
    :return reg_axis: a regular axis in Angstroms
    """
    irreg_axis = 1e8/axis
    reg_axis = np.linspace(irreg_axis[0], irreg_axis[-1], axis.shape[0] )
    #We flip it to make it wavelength ascending
    reg_axis = np.flip(reg_axis, 0)

    if xlims:
        #We get extremas of the filter bandpass
        wl_min, wl_max = list(reversed([1e8/x for x in xlims]))
        #We conserve only spectrum inside this bandpass
        i_min = np.searchsorted(reg_axis, wl_min)
        i_max = np.searchsorted(reg_axis, wl_max)
        reg_axis = reg_axis[i_min:i_max]
    return reg_axis

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
        try:
            tmp[k] = h[k+str(a0)]
        except KeyError:
            tmp[k] = ''
        try:
            h[k+str(a0)] = h[k+str(a1)]
        except KeyError:
            h[k+str(a0)] = ''
        h[k+str(a1)] = tmp[k]

    for k in h.keys():
        if 'PC' in k:
            if str(a0) in k:
                h[k.replace(str(a0), str(a1))] = h.pop(k)
    return h

def filter_star_list(_star_list):
    """
    Very basic filter to remove stars position which are out of the image.
    Hard coded to an image of 2048*2064 pix
    :param _star_list: the list of XY pixel coordinates
    :return: the filtered list
    """
    _star_list = np.copy(_star_list)
    for istar in range(_star_list.shape[0]):
        if (_star_list[istar,0] < 0
            or _star_list[istar,0] > 2048
            or _star_list[istar,1] < 0
            or _star_list[istar,1] > 2064):
            _star_list[istar,:] = np.nan
    return _star_list
def measure_dist(pos1,pos2):
    """
    Measure the distance (norm 2) between two positions, in units of the position
    :param pos1: the first position
    :param pos2: the second position
    :return: the distance (norm 2)
    """
    return np.sqrt((pos1[:,0] - pos2[:,0])**2+(pos1[:,1] - pos2[:,1])**2)
def get_star_list(star_list_deg, im, hdr, dxmap, dymap):
    """
    Computes the pixel positions of a list of positions in degrees.
    :param star_list_deg: a 2d array of object positions, in degree
    :param im: the image on which this objects are supposed to be
    :param hdr: the header of this image
    :param dxmap: the dxmap correction to apply
    :param dymap: the dymap correction to apply
    :return: two lists of pixel poition : one that includes dxdymaps in the copmputation, one that doesn't
    
    """
    #Star positions without dxdymaps
    dxmap_null = np.copy(dxmap)
    dxmap_null.fill(0.)
    dymap_null = np.copy(dymap)
    dymap_null.fill(0.)
    star_list_pix1 = orb.utils.astrometry.world2pix(
        hdr, im.shape[0], im.shape[1], np.copy(star_list_deg), dxmap_null, dymap_null)
    star_list_pix1 = filter_star_list(star_list_pix1)

    #Star positions with dxdymaps
    star_list_pix2 = orb.utils.astrometry.world2pix(
        hdr, im.shape[0], im.shape[1], np.copy(star_list_deg), dxmap, dymap)

    star_list_pix2 = filter_star_list(star_list_pix2)

    return star_list_pix1, star_list_pix2
