import numpy as np
import pandas as pd
import orb
import logging
# this is a very basic filter to remove stars position which are out of the image
def filter_star_list(_star_list):
    _star_list = np.copy(_star_list)
    for istar in range(_star_list.shape[0]):
        if (_star_list[istar,0] < 10
            or _star_list[istar,0] > 2037
            or _star_list[istar,1] < 10
            or _star_list[istar,1] > 2053):
            _star_list[istar,:] = np.nan
    return _star_list

def fit_stars_from_list(im, star_list_pix, box_size=10, **kwargs):
    _fit_results = orb.utils.astrometry.fit_stars_in_frame(im, star_list_pix, box_size, **kwargs)
    return pd.DataFrame(f for f in _fit_results if f is not None)

def compute_precision(list0, list1, scale):
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
    if dxmap is None:
        dxmap = np.zeros((200,200))
    if dymap is None:
        dymap = np.zeros((200,200))
    return filter_star_list(orb.utils.astrometry.world2pix(
        wcs.to_header(relax=True), 2048,2064, np.copy(star_list_deg), dxmap, dymap))

def pix2world(star_list_pix, wcs, dxmap=None, dymap=None):
    if dxmap is None:
        dxmap = np.zeros((200,200))
    if dymap is None:
        dymap = np.zeros((200,200))
    return orb.utils.astrometry.pix2world(wcs.to_header(relax=True), 2048,2064, np.copy(star_list_pix), dxmap, dymap)

from astropy.coordinates import SkyCoord
def match_star_list(_list0, _list1):
    """Match on coords in deg"""
    true_list = SkyCoord(ra=_list0[:,0], dec=_list0[:,1], unit=['deg', 'deg'])
    gaia_list = SkyCoord(ra=_list1[:,0], dec=_list1[:,1], unit=['deg', 'deg'])
    idx, d2d, d3d = true_list.match_to_catalog_sky(gaia_list)
    return idx, d2d.arcsec
