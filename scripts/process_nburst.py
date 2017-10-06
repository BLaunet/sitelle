from __future__ import division
from orcs.process import SpectralCube
import numpy as np
from orb.utils import io
import argparse
import os
from scipy.interpolate import UnivariateSpline
#from sitelle.plot import *

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path",
                        help = "Path to the cube")
    parser.add_argument("-c", "--cube",action="store_true",
                        help="Rebin the cube",
                        default=None)
    parser.add_argument("-f", "--fwhm",action="store_true",
                        help="Rebin the fwhm cube",
                        default=None)
    parser.add_argument("-e", "--error",action="store_true",
                        help="Rebin the error cube",
                        default=None)
    parser.add_argument("-o", "--out_prefix",
                        help="prefix for output path",
                        default='.')
    parser.add_argument("-b", "--binsize",
                        help="If a value is passed, the cubes are binned with this value as binsize",
                        default=int(48))
    return parser

def transform(cube, x, y, binsize, sky_lines, type):
    """
    Function to apply to each bin
    """
    if type=='CUBE':
        axis, spectra, fit = cube.fit_lines_in_spectrum_bin(x, y, binsize,
                                                sky_lines,
                                                fmodel='sinc',
                                                pos_def='1',
                                                fwhm_def='fixed',
                                                nofilter=False,
                                                pos_cov=0 )
        if fit != []:
            for line in fit['fitted_models']['Cm1LinesModel']:
                spectra -= line
        return spectra
    elif type=='ERR':
        flux_err = cube[x*binsize:(x+1)*binsize, y*binsize:(y+1)*binsize]
        return np.sqrt( np.nansum( np.square(flux_err) ))*np.ones(840) #sum of the square
    elif type=='FWHM':
        return np.mean(cube[x*binsize:(x+1)*binsize, y*binsize:(y+1)*binsize])*np.ones(840)

def wl_regrid(cube, rebinned, only_bandpass, type):
    ## CREATION OF THE NEW AXIS
    base_axis = cube.params.base_axis.astype(float)
    irreg_axis = 1e8/base_axis
    reg_axis = np.linspace(irreg_axis[0], irreg_axis[-1], base_axis.shape[0] )
    #We flip it to make it wavelength ascending
    reg_axis = np.flip(reg_axis, 0)

    if only_bandpass:
        #We get extremas of the filter bandpass
        wl_min, wl_max = list(reversed([1e8/x for x in cube.params.filter_range]))
        #We conserve only spectrum inside this bandpass
        i_min = np.searchsorted(reg_axis, wl_min)
        i_max = np.searchsorted(reg_axis, wl_max)
        reg_axis = reg_axis[i_min:i_max]
    new_axis = 1e8/reg_axis #back in cm-1 to evaluate the spectrum function on it


    ##INTERPOLATION OF THE CUBE
    wl_cube = np.zeros((rebinned.shape[0], rebinned.shape[1], len(new_axis)))

    def interpolator(spectrum, old_axis, new_axis, type):
        f = UnivariateSpline(old_axis, spectrum, s=0)
        res = f(new_axis)
        if type == 'FWHM':
            res = np.array([r*1e8/n**2 for r,n in zip(res,new_axis)])
        return res

    if type != 'ERR':
        wl_cube = np.apply_along_axis(interpolator, 2, rebinned,
                                                    old_axis=base_axis,
                                                    new_axis=new_axis,
                                                    type = type)
    else:
        wl_cube = rebinned[:,:,:len(new_axis)]

    ## header
    h = gen_wavelength_header(cube, reg_axis)

    return wl_cube, h

def gen_wavelength_header(cube, axis):
    h = cube.get_header()
    h['NAXIS3'] = len(axis)
    h['CRPIX3'] = 1
    h['CRVAL3'] = axis[0]
    h['CDELT3'] = axis[1]-axis[0]
    h['CUNIT3'] = 'Angstroms'
    h['CTYPE3'] = 'LINEAR'
    return h

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

    h['PC3_3'] = h.pop('PC1_1')
    h['PC3_2'] = h.pop('PC1_2')
    h['PC2_3'] = h.pop('PC2_1')
    return h


def process(cubefile, binsize, type, only_bandpass=True):
    cube = SpectralCube(cubefile)
    new_xsize = cube.dimx//binsize if cube.dimx % binsize == 0 else cube.dimx//binsize + 1
    new_ysize = cube.dimy//binsize if cube.dimy % binsize == 0 else cube.dimy//binsize + 1
    rebinned = np.zeros((new_xsize, new_ysize, cube.dimz))

    if type == 'CUBE':
        #We generate the sky_lines
        sky_lines = np.flip(cube.get_sky_lines(), 0)
        imin = np.searchsorted(sky_lines, 14610)
        imax = np.searchsorted(sky_lines, 15430)
        sky_lines = sky_lines[imin:imax]
        vsys = -700.0
        c = 3e5
        lines_max = [6548,6562, 6583,6678, 6716,6731] #  #Dans l'ordre : NII, Halpha, NII, HeI, SII, SII
        lines_min = [x*(1+vsys/c) for x in lines_max]
        for lmin, lmax in zip(lines_min, lines_max):
            imax = np.searchsorted(sky_lines, 1e8/lmin)
            imin = np.searchsorted(sky_lines, 1e8/lmax)
            sky_lines = np.hstack((sky_lines[:imin],sky_lines[imax:]))
    else:
        sky_lines = None

    print('Extracting...')
    if type == 'ERR':
        data = cube.get_flux_uncertainty()
    elif type == 'FWHM':
        data = cube.get_fwhm_map()
    elif type == 'CUBE':
        data = cube
    for x in range(new_xsize):
        for y in range(new_ysize):
            print '({},{})'.format(x,y)
            rebinned[x, y, :] = transform(data, x, y, binsize, sky_lines, type)
        np.save('bak.npy', rebinned)
    #plot_map(rebinned[:,:,200], xlims=(0,new_xsize), ylims=(0, new_ysize))

    print('Regridding...')
    regrided, header = wl_regrid(cube, rebinned, only_bandpass, type)

    print('Converting...')
    nburst_cube = regrided.T
    nburst_header = transpose_header(header)

    ext='_new'
    if binsize:
        ext+='_rebinned_{}'.format(binsize)
    filter = os.path.basename(cubefile).split('_')[1]
    path = "{}/{}_{}{}".format(out_prefix,'M31', filter, ext)
    io.write_fits('{}_{}.fits'.format(path, str.lower(type)), fits_data=nburst_cube, fits_header=nburst_header, overwrite=True)


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    cubefile = args.path

    out_prefix = args.out_prefix
    binsize = args.binsize
    if binsize:
        binsize = int(binsize)

    if args.cube:
        process(cubefile, binsize, 'CUBE')
    if args.error:
        process(cubefile, binsize, 'ERR')
    if args.fwhm:
        process(cubefile, binsize, 'FWHM')
