"""
Module grouping a bunch of utility functions used throughout the package.
"""
import numpy as np
from scipy.interpolate import UnivariateSpline
import orb

__all__ = ['estimate_noise', 'gen_wavelength_header', 'read_wavelength_axis', 'rebin', 'regular_wl_axis', 'wavelength_regrid', 'nm_to_cm1', 'swap_header_axis', 'filter_star_list', 'measure_dist', 'get_star_list', 'stats_without_lines']
def estimate_noise(full_axis, full_spectrum, filter_lims, side='both'):
    '''
    Estimates the noise in an integrated spectra from sitelle, by measuring it outside of the filter range.

    Parameters
    ----------
    full_axis : 1D :class:`~numpy:numpy.ndarray`
        complete wavenumber axis of the cube
    full_spectrum : 1D :class:`~numpy:numpy.ndarray`
        spectrum on this full axis
    filter_lims : tuple of floats
        limits of the filter range (in wavenumber)
    side : str
        (Optional) if 'right' ('left'), only the right (left) side of the filter is considered. Default to 'both'

    Returns
    -------
    std : float
        the standard deviaition of the signal outside the filter range, i.e. basically noise
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

    Parameters
    ----------
    h : dict
        The header to update
    axis : 1D :class:`~numpy:numpy.ndarray`
        The regular wavelength axis
    ax_num : int
        (Optional) The axis number to use in the header for the spectral dimmension. Default = 3

    Returns
    -------
    h : dict
        Updated header
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

    Parameters
    ----------
    header : dict
        The header to update
    axis: int
        Index of the axis on which the wavelength axis is stored in the header

    Returns
    -------
    axis : 1D :class:`~numpy:numpy.ndarray`
        The parsed wavelength axis
    """
    offset = 1-header['CRPIX'+str(axis)]
    grid =  np.arange(offset,header['NAXIS'+str(axis)] + offset)*header['CDELT'+str(axis)]
    return header['CRVAL'+str(axis)] + grid

def rebin(map, binsize, type):
    """
    Method used to rebin a map. For a regular map, the mean is used. For error maps, the RMS is preffered

    Parameters
    ----------
    map : 2D :class:`~numpy:numpy.ndarray`
        The original map
    binsize : int
        Binning to use.
    type : str
        if type == 'ERR', the RMS is used; else the mean is used.

    Returns
    -------
    binned_map : 2D :class:`~numpy:numpy.ndarray`
        The binned data
    """
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

    Parameters
    ----------
    axis : 1D :class:`~numpy:numpy.ndarray`
        Input axis in cm-1
    xlims : tuple of floats
        (Optional) limits in cm-1 to reduce the range of the axis (typically : filter range)

    Returns
    -------
    reg_axis : 1D :class:`~numpy:numpy.ndarray`
        a regular axis in Angstroms
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
    """
    Interpolation of a 3D datacube on a regular wavelength axis.

    Parameters
    ----------
    cube : :class:`~ORCS:orcs.process.SpectralCube`
        Input SpectralCube
    rebinned : 3D :class:`~numpy:numpy.ndarray`
        A cube of data taken from cube (can be fwhm or error or flux)
    only_bandpass : bool, Default = False
        If True, data is cut outside of the bandpass of the filter
    type : 'FWHM' or 'ERR'
        Defines different behaviors depending on the type of data
    header : dict
        (Optional) Default is SpectraLCube header

    """
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
    """
    Interpolates a spectrum from a wavelength axis to a wavenulber axis

    Parameters
    ----------
    spectrum : 1D :class:`~numpy:numpy.ndarray`
        Input spectrum
    nm_axis : 1D :class:`~numpy:numpy.ndarray`
        Wavelength axis in Angstroms on which the spectrum is defined
    out_axis : 1D :class:`~numpy:numpy.ndarray`
        Wavenumber axis in cm-1 on which we want the spectrum to be interpolated

    Returns
    -------
    out_spec : 1D :class:`~numpy:numpy.ndarray`
        Interpolated spectrum

    """
    cm1_axis = np.flip(1e8/nm_axis.astype(float),0)
    spectrum = np.flip(spectrum, 0)
    cm1_spectrum = UnivariateSpline(cm1_axis, spectrum.astype(float), s=0, k=1,ext=1)(out_axis.astype(float))
    return cm1_spectrum

def swap_header_axis(h, a0, a1):
    """
    Swaps header keywords between dimensions.
    Affects only keywords 'NAXIS', 'CRPIX', 'CRVAL', 'CDELT', 'CUNIT', 'CTYPE'.

    Parameters
    ----------
    h : dict
        The header to update
    a0 : int
        First index to swap from
    a1 : int
        Second index to swap to

    Returns
    -------
    updated_h : dict
        Updated header where dimensions have been interchanged
    """
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

    Parameters
    ----------
    _star_list : :class:`~numpy:numpy.ndarray`
        List of XY pixel coordinates
    Returns
    -------
    filtered_list : :class:`~numpy:numpy.ndarray`
        The filtered list
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
    Measure the distance (norm 2) between two positions, in units of the position.
    Can be used with vectors where each line is a [x,y] position.

    Parameters
    ----------
    pos1 : :class:`~numpy:numpy.ndarray`
        First position
    pos2 : :class:`~numpy:numpy.ndarray`
        Second position

    Returns
    -------
    dist : :class:`~numpy:numpy.ndarray`
        Distance (norm 2), same dimension as pos1
    """
    return np.sqrt((pos1[:,0] - pos2[:,0])**2+(pos1[:,1] - pos2[:,1])**2)
def get_star_list(star_list_deg, im, hdr, dxmap, dymap):
    """
    Computes the pixel positions of a list of positions in degrees.

    Parameters
    ----------
    star_list_deg : :class:`~numpy:numpy.ndarray`
        A 2d array of object positions, in degrees
    im : 2D :class:`~numpy:numpy.ndarray`
        The image on which this objects are supposed to be
    hdr : dict
        The header of this image
    dxmap : 2D :class:`~numpy:numpy.ndarray`
        The dxmap correction to apply
    dymap : 2D :class:`~numpy:numpy.ndarray`
        The dymap correction to apply

    Returns
    -------
    star_list_pix1 : :class:`~numpy:numpy.ndarray`
        2d array of pixel positions where dx and dy maps **have not been used** in the computation
    star_list_pix2 : :class:`~numpy:numpy.ndarray`
        2d array of pixel positions where dx and dy maps **have been used** in the computation
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

def stats_without_lines(spec, cube_axis, lines, v_min, v_max):
    """
    Computes statistics on a spectrum when region around lines are excluded.
    We translate a velocity range into a position range around the lines, and remove these regions to compute the mean, median and standard deviation of the spectrum.

    Parameters
    ----------
    spec : 1D :class:`~numpy:numpy.ndarray`
        Input spectrum
    cube_axis : 1D :class:`~numpy:numpy.ndarray`
        Wavenumber axis ([cm-1]) on which the spectrum is evaluated
    lines : list of str
        Names of the lines to exclude
    v_min : float
        Minimum velocity at which we expect the line to be present
    v_max : float
        Maximum velocity at which we expect the line to be present

    Returns
    -------
    mean : float
    median : float
    std : float
    """
    from orb.utils.spectrum import line_shift
    from orb.core import Lines
    rest_lines = Lines().get_line_cm1(lines)
    pos_min = rest_lines + line_shift(v_max, rest_lines, wavenumber=True)
    pos_max = rest_lines + line_shift(v_min, rest_lines, wavenumber=True)
    lines_lims = np.searchsorted(cube_axis, [pos_min, pos_max]).T
    for lims in lines_lims:
        spec = np.concatenate((spec[..., :lims[0]], spec[...,lims[1]:]), -1)
    return np.nanmean(spec), np.nanmedian(spec), np.nanstd(spec)
