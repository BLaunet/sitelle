from __future__ import division
from orcs.process import SpectralCube
import orb
import numpy as np
import logging
from orb.utils import io
import argparse
import orb.core

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cube",
                        help="path to the cube")
    parser.add_argument("-f", "--fwhm",
                        help="path to the fwhm cube")
    parser.add_argument("-e", "--error",
                        help="path to the error cube")
    parser.add_argument("-o", "--out_prefix",
                        help="prefix for output path",
                        default='.')
    parser.add_argument("-xmin", "--xmin",
                        help="Min index on the x axis, Default = 0",
                        default=0)
    parser.add_argument("-xmax", "--xmax",
                        help="Max index on the x axis, Default = 2047",
                        default=2047)
    parser.add_argument("-ymin", "--ymin",
                        help="Min index on the y axis, Default = 0",
                        default=0)
    parser.add_argument("-ymax", "--ymax",
                        help="Max index on the y axis, Default = 2063",
                        default=2063)
    parser.add_argument("-bin", "--binsize",
                        help="If a value is passed, the cubes are binned with this value as binsize",
                        default=None)
    return parser

def transpose(cubefile, errfile, fwhmfile, out_prefix='.', xmin=0, xmax = 2047, ymin = 0, ymax = 2063, binsize=None):
    hdu = fits.open(cubefile)
    cube, cube_h = [hdu[0].data, hdu[0].header]
    err = fits.open(errfile)[0].data
    fwhm = fits.open(fwhmfile)[0].data

    ext=''
    if xmin != 0 or xmax != 2047 or ymin !=0 or ymax != 2063:
        cube = cube[xmin:xmax, ymin:ymax]
        err = err[xmin:xmax, ymin, ymax]
        fwhm = fwhm[xmin, xmax]
        ext='_{}_{}_{}_{}'.format(xmin, xmax, ymin, ymax)

    if binsize:
        cube = rebin(cube, binsize)
        err = rebin(cube, binsize)
        fwhm = rebin(cube, binsize)
        ext+='_rebinned'
    cube_h = transpose_header(cube_h)

    path = "{}/{}_{}".format(out_prefix,'M31', 'SN2', ext)
    io.write_fits('{}_cube.fits'.format(path), fits_data=cube, fits_header=cube_h, overwrite=True)
    io.write_fits('{}_err.fits'.format(path), fits_data=err, fits_header=cube_h, overwrite=True)
    io.write_fits('{}_fwhm.fits'.format(path), fits_data=fwhm, fits_header=cube_h, overwrite=True)

def rebin(cube, binsize):
    ysize, xsize = cube.shape[1:]
    if ysize % binsize == 0:
        new_ysize = ysize/binsize
    else:
        new_ysize = ysize/binsize + 1
    if xsize % binsize == 0:
        new_xsize = xsize/binsize
    else:
        new_xsize = xsize/binsize + 1

    rebinned = np.zeros(shape=(cube.shape[0], new_ysize, new_xsize))
    ix = 0
    for x in range(new_xsize-1):
        iy = 0
        for y in range(new_ysize-1):
            rebinned[:, y, x] = np.apply_along_axis(np.sum, 1, np.apply_along_axis(np.sum, 1, cube[:, (ix*binsize):(ix+1)*binsize,(iy*binsize):(iy+1)*binsize]))
            iy+=1
        ix+=1
    rebinned[:, -1, -1] =  np.apply_along_axis(np.sum, 1, np.apply_along_axis(np.sum, 1, cube[:, (ix*binsize):,(iy*binsize):]))
    return rebinned

def transpose_header(h):
    tmp = {}
    tmp['NAXIS1'] = h['NAXIS1']
    tmp['CRPIX1'] = h['CRPIX1']
    tmp['CRVAL1'] = h['CRVAL1']
    tmp['CDELT1'] = h['CDELT1']
    tmp['CUNIT1'] = h['CUNIT1']
    tmp['CTYPE1'] = h['CTYPE1']

    h['NAXIS1'] = h['NAXIS3']
    h['CRPIX1'] = h['CRPIX3']
    h['CRVAL1'] = h['CRVAL3']
    h['CDELT1'] = h['CDELT3']
    h['CUNIT1'] = h['CUNIT3']
    h['CTYPE1'] = h['CTYPE3']

    h['NAXIS3'] = tmp['NAXIS1']
    h['CRPIX3'] = tmp['CRPIX1']
    h['CRVAL3'] = tmp['CRVAL1']
    h['CDELT3'] = tmp['CDELT1']
    h['CUNIT3'] = tmp['CUNIT1']
    h['CTYPE3'] = tmp['CTYPE1']
    return h
if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    cubefile = args.cube
    errfile = args.error
    fwhmfile = args.fwhm
    out_prefix = args.out_prefix
    binsize = args.binsize
    xmin, xmax, ymin, ymax = [args.xmin, args.xmax, args.ymin, args.ymax]

    transpose(cubefile, errfile, fwhmfile, out_prefix, xmin, xmax, ymin, ymax, binsize)
