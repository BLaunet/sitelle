from __future__ import division
import numpy as np
import argparse
from astropy.io import fits
import sys
import os
from scipy.interpolate import UnivariateSpline

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--out_prefix",
                        help="prefix for output path",
                        default='./')
    parser.add_argument("-t", "--type",
                        help="type of the cube",
                        default = "CUBE")
    return parser

def sew(SN2_filename, SN3_filename, type, out_prefix, on_rebinned=True):

    hdu2 = fits.open(SN2_filename)[0]
    hdu3 = fits.open(SN3_filename)[0]
    SN2_wl_grid = hdu2.data.T
    SN3_wl_grid = hdu3.data.T

    SN2_h = hdu2.header
    SN3_h = hdu3.header

    SN2_axis = SN2_h['CRVAL1'] + SN2_h['CDELT1']*np.arange(0,SN2_h['NAXIS1'])
    SN3_axis = SN3_h['CRVAL1'] + SN3_h['CDELT1']*np.arange(0,SN3_h['NAXIS1'])

    step = SN2_h['CDELT1']
    new_axis = np.arange(SN2_axis.min(), SN3_axis.max(), step=step) #We choose the largest step (lower resolution)

    SN2_max_id = np.searchsorted(new_axis,SN2_axis.max(), side='right')
    SN3_min_id = np.searchsorted(new_axis,SN3_axis.min())


    new_cube = np.zeros( (len(new_axis), SN2_wl_grid.shape[1], SN2_wl_grid.shape[2]) )

    print(SN2_wl_grid.shape)
    print(new_cube.shape)

    new_cube[:SN2_max_id, :, :] = SN2_wl_grid


    def interpolator(spectrum, old_axis, new_axis):
        f = UnivariateSpline(old_axis, spectrum, s=0)
        return f(new_axis)

    new_cube[SN3_min_id:, :, :] = np.apply_along_axis(interpolator, 0, SN3_wl_grid,
                                                        old_axis=SN3_axis,
                                                        new_axis=new_axis[SN3_min_id:])


    hdu = fits.PrimaryHDU(new_cube.T, header=SN2_h)
    del new_cube

    hdu.writeto(out_prefix+'M31_SN2-3_fwhm_cube.fits', overwrite=True)
    del hdu

    # if type == 'CUBE':
    #     hdu2 = fits.open(files_dir+'M31_SN2_rebinned_48_err.fits')[0]
    #     hdu3 = fits.open(files_dir+'M31_SN3_rebinned_48_err.fits')[0]
    #     #hdu2 = fits.open(files_dir+'M31_SN2_errcube.fits')[0]
    #     #hdu3 = fits.open(files_dir+'M31_SN3_errcube.fits')[0]
    #
    #     new_err_cube = np.zeros((len(new_axis), SN2_wl_grid.shape[1], SN2_wl_grid.shape[2]))
    #
    #
    #     if on_rebinned:
    #         SN2_err = hdu2.data.T
    #         SN3_err = hdu3.data.T
    #     else:
    #         SN2_err = hdu2.data
    #         SN3_err = hdu3.data
    #
    #     new_err_cube[:SN2_max_id, :, :] = SN2_err
    #     new_err_cube[SN3_min_id:, :, :] = SN3_err[:len(new_axis[SN3_min_id:]),:,:]
    #
    #     if on_rebinned:
    #         new_err_cube = new_err_cube.T
    #     hdu = fits.PrimaryHDU(new_err_cube, header=SN2_h)
    #     del new_err_cube
    #     hdu.writeto(out_prefix+'M31_SN2-3_errcube.fits', overwrite=True)

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    out_prefix = args.out_prefix
    type = str.upper(args.type)

    SN2 = '/Users/blaunet/Documents/M31/nburst/SN2/cube/M31_SN2_fwhm_cube_48x48.fits'
    SN3 = '/Users/blaunet/Documents/M31/nburst/SN3/cube/M31_SN3_fwhm_cube_48x48.fits'
    sew(SN2, SN3, type, out_prefix)
