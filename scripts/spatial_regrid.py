from astropy.io import fits
from astropy import wcs
import numpy as np
from scipy import interpolate
import argparse

def regrid():

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", help="file to process")
    args = parser.parse_args()

    path=args.file

    M31 = fits.open(path)
    h = M31[0].header
    d = M31[0].data

    w = wcs.WCS(h, naxis=(1,2))

    #We build a grid of pixel values
    pix_grid_i, pix_grid_j = np.meshgrid(np.arange(2048), np.arange(2064))

    #We compute the RA and DEC over all the grid points
    world = w.all_pix2world(pix_grid_i.flatten(), pix_grid_j.flatten(),0)
    ra_real =  world[0]
    dec_real = world[1]

    #We build our own flat grid, equally space in RA and DEC
    ra_step = (ra_real.min()-ra_real.max())/float(2048)
    dec_step = (dec_real.max()-dec_real.min())/float(2064)
    ra_grid, dec_grid = np.meshgrid(np.arange(ra_real.max(), ra_real.min(), ra_step ), np.arange(dec_real.min(), dec_real.max(), dec_step))

    ra_axis=ra_grid[0,:]
    dec_axis=dec_grid[:,0]

    new_data = np.zeros((840, 2064, 2048))

    for wn in range(840):
        new_data[wn, :, :] = interpolate.griddata( (dec_real, ra_real) , d[wn,:,:].flatten(), (dec_grid.flatten(), ra_grid.flatten()))

    new_data[np.isnan(new_data)] = 0

    nmin = h.keys().index('BP_0_0')
    nmax = h.keys().index('AP_0_2')
    for kw in h.keys()[nmin:nmax+1]:
        h.pop(kw)
    h.pop('PC1_1')
    h.pop('PC1_2')
    h.pop('PC2_1')
    h.pop('PC2_2')
    h['CTYPE1'] = 'RA---CAR'
    h['CTYPE2'] = 'DEC--CAR'
    h['CRVAL1'] = ra_axis[0]
    h['CRPIX1'] = 1
    h['CDELT1'] = ra_step
    h['CRVAL2'] = dec_axis[0]
    h['CRPIX2'] = 1
    h['CDELT2'] = dec_step

    hdu = fits.PrimaryHDU(new_data, header=h)
    hdu.writeto('/cluster/scratch/blaunet/M31_regrided.fits')
