import numpy as np
import pandas as pd
import orb
import logging
from astropy.coordinates import SkyCoord

__all__ = ['filter_star_list', 'fit_stars_from_list', 'compute_precision', 'world2pix', 'pix2world', 'match_star_list']

def filter_star_list(_star_list):
    """
    Very basic filter to remove stars position which are out of the image.

    It replaces with NaN pixel positions where x < 10 or y < 10 or x > 2037 or y > 2053.

    Parameters
    ----------
    _star_list : :class:`~numpy:numpy.ndarray`
        A 2D array containg **pixel** positions of detections. The array should be column oriented ([[x0, y0], [x1, y1], [x2, y2], ...])

    Returns
    -------
    :class:`~numpy:numpy.ndarray`
        A 2D array of the same dimension but where positions out of the frame have been replaced with NaN.
    """
    _star_list = np.copy(_star_list)
    for istar in range(_star_list.shape[0]):
        if (_star_list[istar,0] < 10
            or _star_list[istar,0] > 2037
            or _star_list[istar,1] < 10
            or _star_list[istar,1] > 2053):
            _star_list[istar,:] = np.nan
    return _star_list

def fit_stars_from_list(im, star_list_pix, box_size=10, **kwargs):
    """
    Wrapper method around :func:`ORB:orb.utils.astrometry.fit_stars_in_frame`.

    The function as the same signature as :func:`ORB:orb.utils.astrometry.fit_stars_in_frame` but returns a :class:`~pandas:pandas.DataFrame` object.

    Parameters
    ----------
    im : :class:`~numpy:numpy.ndarray`
        A 2D image containing the stars to fit.
    star_list_pix : :class:`~numpy:numpy.ndarray`
        A 2D array containg **pixel** positions of detections. The array should be column oriented ([[x0, y0], [x1, y1], [x2, y2], ...])
    box_size : int
        The side length of the box in which the star is fitted.
    kwargs :
        See :func:`ORB:orb.utils.astrometry.fit_stars_in_frame`.

    Returns
    -------
    :class:`~pandas:pandas.DataFrame`
        A table where rows represent a star and columns represent fit output quantities. See :func:`ORB:orb.utils.astrometry.fit_stars_in_frame` for details about the columns.
    """
    _fit_results = orb.utils.astrometry.fit_stars_in_frame(im, star_list_pix, box_size, **kwargs)
    return pd.DataFrame(f for f in _fit_results if f is not None)

def compute_precision(list0, list1, scale):
    """
    Displays the astrometric mean difference between two position lists of the same objects.

    Parameters
    ----------
    list0 : :class:`~numpy:numpy.ndarray`
        A 2D array containg **pixel** positions of objects. The array should be column oriented ([[x0, y0], [x1, y1], [x2, y2], ...])
    list1 : :class:`~numpy:numpy.ndarray`
        A 2D array containg **pixel** positions of the same objects than in list0 but obtained with another method.
    scale : float
        The pixel scale in arcsec.

    Returns
    -------
        None
    """
    dx = list0[:,0] - list1[:,0]
    dy = list0[:,1] - list1[:,1]
    precision = np.sqrt(dx**2. + dy**2.)
    precision_mean = np.sqrt(np.nanmedian(np.abs(dx))**2.
                             + np.nanmedian(np.abs(dy))**2.)
    precision_mean_err = np.sqrt(
        (np.nanpercentile(dx, 84) - precision_mean)**2.
         + (np.nanpercentile(dy, 84) - precision_mean)**2.)

    logging.info("Astrometrical precision [in arcsec]: {:.3f} [+/-{:.3f}] computed over {} stars".format(
            precision_mean * scale,
            precision_mean_err * scale, np.size(dx)))

def world2pix(star_list_deg, wcs, dxmap=None, dymap=None):
    """
    Conversion from physical coordinates to pixel coordinates using WCS and eventually dxdymaps.
    It's the mathematical inverse of :func:`pix2world`.

    This supplements the :func:`~astropy:astropy.wcs.WCS.all_world2pix` by adding the possibility to use dxmaps and dymaps. For details see :func:`ORB:orb.utils.astrometry.world2pix`.
    The resulting pixel positions are then filtered using :func:`filter_star_list`.

    Parameters
    ----------
    star_list_deg : 2D :class:`~numpy:numpy.ndarray`
        A 2D array containg **RA/DEC** positions of objects. The array should be column oriented ([[ra0, dec0], [ra1, dec1], [ra2, dec2], ...])
    wcs : :class:`~astropy:astropy.wcs.WCS`
        The :class:`~astropy:astropy.wcs.WCS` transformation we want to use. See `here <http://docs.astropy.org/en/stable/wcs/index.html>`_ for more details.
    dxmap : 2D :class:`~numpy:numpy.ndarray`
        2D array containing the astrometric third order correction on the x axis. If None (default value), the third order correction is not considered.
    dymap : 2D :class:`~numpy:numpy.ndarray`
        2D array containing the astrometric third order correction on the y axis. If None (default value), the third order correction is not considered.

    Return
    -------
    :class:`~numpy:numpy.ndarray`
        A 2D array of the same dimension than ``star_list_deg`` with pixel positions.
    """

    if dxmap is None:
        dxmap = np.zeros((200,200))
    if dymap is None:
        dymap = np.zeros((200,200))
    return filter_star_list(orb.utils.astrometry.world2pix(
        wcs.to_header(relax=True), 2048,2064, np.copy(star_list_deg), dxmap, dymap))

def pix2world(star_list_pix, wcs, dxmap=None, dymap=None):
    """
    Conversion from pixel coordinates to physical coordinates using WCS and eventually dxdymaps.
    It's the mathematical inverse of :func:`pix2world`.

    This supplements the :func:`~astropy:astropy.wcs.WCS.all_pix2world` by adding the possibility to use dxmaps and dymaps. For details see :func:`ORB:orb.utils.astrometry.pix2world`.
    The resulting pixel positions are then filtered using :func:`filter_star_list`.

    Parameters
    ----------
    star_list_pix : :class:`~numpy:numpy.ndarray`
        A 2D array containg **pixel** positions of objects. The array should be column oriented ([[x0, y0], [x1, y1], [x2, y2], ...])
    wcs : :class:`~astropy:astropy.wcs.WCS`
        The :class:`~astropy:astropy.wcs.WCS` transformation than we want to use. See `here <http://docs.astropy.org/en/stable/wcs/index.html>`_ for more details.
    dxmap : :class:`~numpy:numpy.ndarray`
        2D array containing the astrometric third order correction on the x axis. If None (default value), the third order correction is not considered.
    dymap : :class:`~numpy:numpy.ndarray`
        2D array containing the astrometric third order correction on the y axis. If None (default value), the third order correction is not considered.

    Return
    -------
    :class:`~numpy:numpy.ndarray`
        A 2D array of the same dimension than ``star_list_pix`` with RA/DEC positions.
    """
    if dxmap is None:
        dxmap = np.zeros((200,200))
    if dymap is None:
        dymap = np.zeros((200,200))
    return orb.utils.astrometry.pix2world(wcs.to_header(relax=True), 2048,2064, np.copy(star_list_pix), dxmap, dymap)

def match_star_list(_list0, _list1):
    """
    Astrometric cross-match method between two list of objects.

    The underlying method is :func:`~astropy:astropy.coordinates.SkyCoord.match_to_catalog_sky`

    Parameters
    ----------
    _list0 : :class:`~numpy:numpy.ndarray`
        A 2D array containg **RA/DEC** positions of objects. The array should be column oriented ([[ra0, dec0], [ra1, dec1], [ra2, dec2], ...])
    _list1 : :class:`~numpy:numpy.ndarray`
        A second list of **RA/DEC** positions of objects.

    Returns
    -------
    idx : interger array
        Indices in _list1 that match _list0 positions (same dimension as _list0).
    sep2d : float array
        On-sky separation of the matches, in arcsec (same dimesion as _list1).

    """
    true_list = SkyCoord(ra=_list0[:,0], dec=_list0[:,1], unit=['deg', 'deg'])
    gaia_list = SkyCoord(ra=_list1[:,0], dec=_list1[:,1], unit=['deg', 'deg'])
    idx, d2d, d3d = true_list.match_to_catalog_sky(gaia_list)
    return idx, d2d.arcsec
