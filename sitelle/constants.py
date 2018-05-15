"""
File defining constants that are heavily used throughout the project.

Attributes:
-----------
    M31_CENTER : :class:`~astropy:astropy.coordinates.SkyCoord`
        Center of the galaxy, as defined in Crane et al. (1992).
        See `here <http://docs.astropy.org/en/stable/coordinates/index.html>`_ how to use this :class:`~astropy:astropy.coordinates.SkyCoord` object.
    SN2_LINES : list of str
        Main line names of SN2 filter : ``Hbeta, [OIII]4959, [OIII]5007``. To get the lines rest positions, use `orb.core.Lines.get_line_cm1 <http://132.203.11.199/orb-doc/orb.html#orb.core.Lines.get_line_cm1>`_.
    SN3_LINES : list of str
        Main line names of SN3 filter : ``[NII]6548, Halpha,[NII]6583, [SII]6716, [SII]6731``
    FITS_DIR : :class:`~path:path.Path`
        The path of the main directory hosting all the fits files.
        This is really a convenience attribute for B. Launet, but can be ignored
        for a regular use.
"""

from astropy.coordinates import SkyCoord
import astropy.units as u
from path import Path
import socket


M31_CENTER = SkyCoord(ra='00h42m44.371s', dec='41d16m08.34s')#, distance=778*u.kpc)
SN2_LINES = ['Hbeta', '[OIII]4959', '[OIII]5007']
SN3_LINES = ['[NII]6548', 'Halpha','[NII]6583', '[SII]6716', '[SII]6731']

if 'MacBookAirdeBarthelemy' in socket.gethostname():
    FITS_DIR = Path('/home/blaunet/Documents/M31/fits/')

if 'johannes' in socket.gethostname() or 'tycho' in socket.gethostname():
    FITS_DIR = Path('/data/blaunet/fits/')
if 'lp-alm-stg' in socket.gethostname():
    FITS_DIR = Path('/Users/blaunet/Documents/M31/fits')
if 'celeste' in socket.gethostname():
    FITS_DIR = Path('/home/blaunet/fits')
