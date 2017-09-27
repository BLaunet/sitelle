from __future__ import division
import numpy as np
import argparse
from astropy.io import fits
import sys
import os

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dir",
                        help="path to files directory")
    parser.add_argument("-o", "--out_prefix",
                        help="prefix for output path",
                        default='.')
    parser.add_argument("-t", "--type",
                        help="type of the cube",
                        default = "CUBE")
    return parser

def sew(files_dir, type, out_prefix):
    if type == 'CUBE':
        hdu2 = fits.open(files_dir+'M31_SN2_wl_grid_bandpass.fits')[0]
        hdu3 = fits.open(files_dir+'M31_SN3_wl_grid_bandpass.fits')[0]
    elif type == 'FWHM':
        hdu2 = fits.open(files_dir+'M31_SN2_fwhm_cube_bandpass.fits')[0]
        hdu3 = fits.open(files_dir+'M31_SN3_fwhm_cube_bandpass.fits')[0]

    SN2_wl_grid = hdu2.data
    SN2_h = hdu2.header

    SN3_wl_grid = hdu3.data
    SN3_h = hdu3.header

    SN2_axis = SN2_h['CRVAL3'] + SN2_h['CDELT3']*np.arange(0,SN2_h['NAXIS3'])
    SN3_axis = SN3_h['CRVAL3'] + SN3_h['CDELT3']*np.arange(0,SN3_h['NAXIS3'])

    new_axis = np.arange(SN2_axis.min(), SN3_axis.max(), step=SN2_h['CDELT3']) #We choose the largest step (lower resolution)

    SN2_max_id = np.searchsorted(new_axis,SN2_axis.max(), side='right')
    SN3_min_id = np.searchsorted(new_axis,SN3_axis.min())

    new_cube = np.zeros( (len(new_axis), SN2_wl_grid.shape[1], SN2_wl_grid.shape[2]) )

    new_cube[:SN2_max_id, :, :] = SN2_wl_grid


    def interpolator(spectrum, old_axis, new_axis):
        f = scipy.interpolate.UnivariateSpline(old_axis, spectrum, s=0)
        return f(new_axis)

    new_cube[SN3_min_id:, :, :] = np.apply_along_axis(interpolator, 0, SN3_wl_grid,
                                                        old_axis=SN3_axis,
                                                        new_axis=new_axis[SN3_min_id:])

    hdu = fits.PrimaryHDU(new_cube, header=SN2_h)
    del new_cube

    hdu.writeto(out_prefix+'M31_SN2-3_%s.fits'%str.lower(type), overwrite=True)
    del hdu

    if type == 'CUBE':
        hdu2 = fits.open(files_dir+'M31_SN2_errcube.fits')[0]
        hdu3 = fits.open(files_dir+'M31_SN3_errcube.fits')[0]

        new_err_cube = np.zeros((len(new_axis), SN2_wl_grid.shape[1], SN2_wl_grid.shape[2]))
        new_err_cube[:SN2_max_id, :, :] = hdu2.data

        err_map_SN3 = hdu3.data[0,:,:]
        dim_sup = len(new_axis[SN3_min_id:]) - hdu3.data.shape[0]
        to_add = np.repeat(err_map_SN3[np.newaxis, :, :], dim_sup, axis=0)

        new_err_cube[SN3_min_id:, :, :] = np.append(hdu3.data, to_add)

        hdu = fits.PrimaryHDU(new_err_cube, header=SN2_h)
        del new_err_cube
        hdu.writeto(out_prefix+'M31_SN2-3_errcube.fits', overwrite=True)

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    files_dir = args.dir
    out_prefix = args.out_prefix
    type = str.upper(args.type)
    sew(files_dir, type, out_prefix)
