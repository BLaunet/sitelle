from __future__ import division
import numpy as np
import argparse
from astropy.io import fits
from orb.utils import io
import sys
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cube",
                        help="path to the cube",
                        default=None)
    parser.add_argument("-f", "--fwhm",
                        help="path to the fwhm cube",
                        default=None)
    parser.add_argument("-e", "--error",
                        help="path to the error cube",
                        default=None)
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



def transpose(file, type, out_prefix='.', xmin=0, xmax = 2047, ymin = 0, ymax = 2063, binsize=None):

    hdu = fits.open(file)[0]
    cube = hdu.data
    if type=='CUBE':
        cube_h = hdu.header
        cube_h = transpose_header(cube_h)
    else:
        cube_h = None

    cube = cube[xmin:xmax, ymin:ymax]

    ext=''
    if xmin != 0 or xmax != 2047 or ymin !=0 or ymax != 2063:
        ext='_{}-{}_{}-{}'.format(xmin, xmax, ymin, ymax)

    if binsize:
        cube = rebin(cube, binsize, type)
        ext+='_rebinned_{}'.format(binsize)

    path = "{}/{}_{}{}".format(out_prefix,'M31', 'SN2', ext)
    io.write_fits('{}_{}.fits'.format(path, str.lower(type)), fits_data=cube, fits_header=cube_h, overwrite=True)

def rebin(cube, binsize, type):
    def special(data, type):
        if type=='CUBE':
            return np.sum( binned_data , (1,2)) #We sum the spectra
        elif type=='ERR':
            return np.sqrt( np.sum( np.power(binned_data,2) , (1,2))) #sum of the square
        elif type=='FWHM':
            return np.mean(binned_data, axis=(1,2))
    ysize, xsize = cube.shape[1:]
    if ysize % binsize == 0:
        new_ysize = ysize//binsize
    else:
        new_ysize = ysize//binsize + 1
    if xsize % binsize == 0:
        new_xsize = xsize//binsize
    else:
        new_xsize = xsize//binsize + 1

    rebinned = np.zeros(shape=(cube.shape[0], new_ysize, new_xsize))
    ix = 0
    for x in range(new_xsize-1):
        iy = 0
        for y in range(new_ysize-1):
            rebinned[:, y, x] = special(cube[:, (ix*binsize):(ix+1)*binsize,(iy*binsize):(iy+1)*binsize], type)
            iy+=1
        ix+=1
    rebinned[:, -1, -1] =  special(cube[:, (ix*binsize):,(iy*binsize):], type)
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

    if not cubefile and not errfile and not fwhmfile:
        print('No file path has been provided !')
        sys.exit()

    out_prefix = args.out_prefix
    binsize = args.binsize
    if binsize:
        binsize = int(binsize)
    xmin, xmax, ymin, ymax = map(int,[args.xmin, args.xmax, args.ymin, args.ymax])

    if cubefile:
        transpose(cubefile, 'CUBE', out_prefix, xmin, xmax, ymin, ymax, binsize)
    if errfile:
        transpose(errfile, 'ERR', out_prefix, xmin, xmax, ymin, ymax, binsize)
    if fwhmfile:
        transpose(fwhmfile, 'FWHM', out_prefix, xmin, xmax, ymin, ymax, binsize)
