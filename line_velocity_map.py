from __future__ import division
from orcs.process import SpectralCube
from astropy.io import fits
import numpy as np
import logging
from orb.utils import io
import argparse
import orb.core

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cube",
                        help="Original hdf5 cube")
    parser.add_argument('-n', "--nburst",
                        help="Nburst fit")
    parser.add_argument("-o", "--out_prefix",
                        help="Prefix for output path")

    return parser

def line_velocity_map(cubefile, nburstfile, out_prefix, silent=False):
    def subtract_fitted_spectra(nburst_axis, nburst_fit, cube, binMap, biNumber, silent=False):
        #We interpolate the fit on a cm-1 axis
        cm1_axis = np.flip(1e8/nburst_axis.astype(float),0)
        nburst_fit = np.flip(nburst_fit, 0)
        f = scipy.interpolate.UnivariateSpline(cm1_axis, nburst_fit.astype(float), s=0, k=1,ext=1)
        cm1_fit = f(cube.params.base_axis.astype(float))

        #We redefine the region corresponding to the bin in the global image
        region = [x for x in (np.where(binMap == b))]
        mask = np.zeros((cube.dimx, cube.dimy), dtype=np.uint8)
        mask[region] = 1

        #We extract the spectra from the corresponding region
        a, s = cube.extract_integrated_spectrum(region, silent=silent)

        cm1_fit[np.where(cm1_fit == 0)] = s[np.where(cm1_fit == 0)] #fit is perfect outside of the filter bandpass

        residual = s-cm1_fit
        to_subtract = cm1_fit/np.sum(mask)

        return a, s-cm1_fit, to_subtract, region

        V_map = np.full(binMap.shape, np.nan)
        for b in range(*binRange):
            nburst_axis = fit_table['WAVE'][b,:]
            nburst_spectrum = fit_table['FLUX'][b,:]
            nburst_fit = fit_table['FIT'][b,:]
            if np.all(np.isnan(nburst_fit)):
                continue
            axis, res, to_subtract, region = subtract_fitted_spectra(nburst_axis, nburst_fit, cube, binMap, b, silent=True)
            a, s, fit = cube.fit_lines_in_integrated_region(region,
                                                            ['[OIII]4959', '[OIII]5007'],
                                                            fmodel='sincgauss',
                                                            pos_def=['1'],
                                                            pos_cov=-300.0,
                                                            subtract_spectrum=to_subtract,
                                                            silent = True)
            if fit != []:
                V_map[np.where(binMap==b)] = fit['velocity'][0]
        return V_map
    def fit_over_bin_range(cube, fit_table, binMap, binRange):
        import numpy as np
        import scipy

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

    job_server, ncpus = cube._init_pp_server(silent=silent)

    binRange = np.linspace(0, len(fit_table) - len(fit_table)%(ncpus-1) ,ncpus-1, endpoint=False)
    binRange = np.append(binRange, len(fit_table))

    jobs = [(ijob, job_server.submit(fit_over_bin_range,
                                    args=(cube, fit_table, binMap, map(int,(binRange[ijob], binRange[ijob+1]))),
                                    depfuncs=(subtract_fitted_spectra,)))
                                            for ijob in range(ncpus)]

    V_map = np.zeros_like(binMap, dtype=float)
    for ijob, job in jobs:
        sub_map = job()
        V_map[np.where(~np.isnan(sub_map))] = sub_map[np.where(~np.isnan(sub_map))]
    cube._close_pp_server(job_server)


    np.save('velmap.npy', V_map)




if __name__ == '__main__':

    parser = get_parser()
    args = parser.parse_args()
    cubefile = args.cube
    nburstfile = args.nburst
    out_prefix = args.out_prefix

    line_velocity_map(cubefile, nburstfile, out_prefix)
