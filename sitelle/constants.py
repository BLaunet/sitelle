from astropy.coordinates import SkyCoord
import astropy.units as u
from path import Path
import socket

M31_CENTER = SkyCoord(ra='00h42m44.371s', dec='41d16m08.34s')#, distance=778*u.kpc)

if 'MacBookAirdeBarthelemy' in socket.gethostname():
    FITS_DIR = Path('/home/blaunet/Documents/M31/fits/')

if 'johannes' in socket.gethostname() or 'tycho' in socket.gethostname():
    FITS_DIR = Path('/data/blaunet/fits/')
if 'lp-alm-stg' in socket.gethostname():
    FITS_DIR = Path('/Users/blaunet/Documents/M31/fits')
if 'celeste' in socket.gethostname():
    FITS_DIR = Path('/home/blaunet/fits')

SN2_LINES = ['Hbeta', '[OIII]4959', '[OIII]5007']
SN3_LINES = ['[NII]6548', 'Halpha','[NII]6583', '[SII]6716', '[SII]6731']
