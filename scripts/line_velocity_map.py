from __future__ import division
from orcs.process import SpectralCube
from astropy.io import fits
import numpy as np
import logging
from orb.utils import io
import argparse
import orb.core
import scipy
from multiprocessing import Pool
from functools import partial
import sys

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cube",
                        help="Original hdf5 cube")
    parser.add_argument('-u', "--usage", action="store_true",
                        help="Usage of the script")
    parser.add_argument('-n', "--nburst",
                        help="Nburst fit")
    parser.add_argument("-o", "--out_prefix",
                        help="Prefix for output path")

    return parser

def subtract_fitted_spectra(nburst_axis, nburst_fit, cube, binMap, binNumber, silent=False):
    #We interpolate the fit on a cm-1 axis
    cm1_axis = np.flip(1e8/nburst_axis.astype(float),0)
    nburst_fit = np.flip(nburst_fit, 0)
    f = scipy.interpolate.UnivariateSpline(cm1_axis, nburst_fit.astype(float), s=0, k=1,ext=1)
    cm1_fit = f(cube.params.base_axis.astype(float))

    #We redefine the region corresponding to the bin in the global image
    region = (np.where(binMap == binNumber))
    mask = np.zeros((cube.dimx, cube.dimy), dtype=np.uint8)
    mask[region] = 1

    #We extract the spectra from the corresponding region
    a, s = cube.extract_integrated_spectrum(region, silent=silent)

    cm1_fit[np.where(cm1_fit == 0)] = s[np.where(cm1_fit == 0)] #fit is perfect outside of the filter bandpass

    residual = s-cm1_fit
    to_subtract = cm1_fit/np.sum(mask)

    return a, s-cm1_fit, to_subtract, region

def fit_over_bin_range(cube, fit_table, binMap, binRange):
    fit_map = np.full(binMap.shape, np.nan,  dtype=object)
    for b in range(*binRange):
        print('Bin : %s'%b)
        nburst_axis = fit_table['WAVE'][b,:]
        nburst_spectrum = fit_table['FLUX'][b,:]
        nburst_fit = fit_table['FIT'][b,:]
        if np.all(np.isnan(nburst_fit)):
            continue
        axis, res, to_subtract, region = subtract_fitted_spectra(nburst_axis, nburst_fit, cube, binMap, b, silent=True)
        for pos_cov in np.linspace(0,-800,6):
            a, s, fit = cube.fit_lines_in_integrated_region(region,
                                                            ['[OIII]4959', '[OIII]5007'],
                                                            fmodel='sincgauss',
                                                            pos_def=['1'],
                                                            pos_cov=pos_cov,
                                                            subtract_spectrum=to_subtract,
                                                            silent = True)
            if fit != [] and fit['chi2'] < 2:
                fit_map[np.where(binMap==b)] = fit
                print('Found V = %s'%fit['velocity'][0])
                break

    np.save('fit_map_%s_%s.npy'%(binRange[0], binRange[1]), fit_map)
    return fit_map


def line_velocity_map(cubefile, nburstfile, out_prefix, silent=False):
    cube = SpectralCube(cubefile)
    hdu = fits.open(nburstfile)
    bin_table = hdu[1].data
    fit_table = hdu[2].data

    if 'rebinned' in nburstfile: #We get the bin size
        binsize_id = nburstfile.split('_').index('rebinned')+1
        binsize = int(nburstfile.split('_')[binsize_id])

    tmp_binMap = bin_table['BINNUM'].reshape(43,43).astype(np.int32).T
    b=48
    binMap = np.zeros((2048,2064), np.int32)
    for x in range(43):
        for y in range(43):
            binMap[x*b:(x+1)*b, y*b:(y+1)*b] = tmp_binMap[x,y]

    ncpus = 8
    binRange = np.linspace(0, len(fit_table) - len(fit_table)%(ncpus-1) ,ncpus-1, endpoint=False)
    binRange = np.append(binRange, len(fit_table))

    fit_map = np.zeros_like(binMap, dtype=object)

    for i in range(len(binRange)-1):
        sub_map = fit_over_bin_range(cube, fit_table, binMap, map(int, (binRange[i], binRange[i+1])))
        fit_map[np.where(~np.isnan(sub_map))] = sub_map[np.where(~np.isnan(sub_map))]

    np.save('velmap.npy', fit_map)




if __name__ == '__main__':

    parser = get_parser()
    args = parser.parse_args()
    cubefile = args.cube
    nburstfile = args.nburst
    out_prefix = args.out_prefix
    if args.usage:
        print("""
        Usage examples:
        python line_velocity_map.py -c ../../fits/orig/M31_SN2.merged.cm1.1.0.hdf5 -n ../../nburst/M31_SN2_rebinned_48_fitted.fits -o ../../Notebooks/
        """)
    else:
        line_velocity_map(cubefile, nburstfile, out_prefix)