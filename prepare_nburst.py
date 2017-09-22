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
                        help="Min index on the x axis, Default = 1000",
                        default=1000)
    parser.add_argument("-xmax", "--xmax",
                        help="Max index on the x axis, Default = 1100",
                        default=1100)
    parser.add_argument("-ymin", "--ymin",
                        help="Min index on the y axis, Default = 1000",
                        default=1000)
    parser.add_argument("-ymax", "--ymax",
                        help="Max index on the y axis, Default = 1100",
                        default=1100)
    return parser

def transpose(cubefile, errfile, fwhmfile, out_prefix='.'):
    cube, cube_h = io.read_fits(cubefile, return_header=True)
    err = io.read_fits(errfile)
    fwhm = io.read_fits(fwhmfile)

    cube = cube.T
    err = err.T
    fwhm = fwhm.T

    cube_h = transpose_header(cube_h)

    path = "{}/{}_{}_".format(out_prefix,
                                    'M31',
                                    'SN2')
    io.write_fits('{}_cube.fits', fits_data=cube, fits_header=cube_h, overwrite=True)
    io.write_fits('{}_err.fits', fits_data=err, fits_header=cube_h, overwrite=True)
    io.write_fits('{}_fwhm.fits', fits_data=fwhm, fits_header=cube_h, overwrite=True)


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

    transpose(cubefile, errfile, fwhmfile, out_prefix)
