from orcs.process import SpectralCube
from orb.utils.vector import interpolate_axis
from orb.utils import io
import argparse
import numpy as np
from orb.core import ProgressBar

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file",
                        help="file to process")
    return parser

def gen_spectrum_header(cube, axis):
    h = cube.get_header()
    for k in cube.get_wcs_header():
        if k in h.keys() and k not in ['LONPOLE', 'LATPOLE', 'RADESYS', 'EQUINOX', 'MJD-OBS', 'DATE-OBS']:
            h.pop(k)
    h.pop('NAXIS2')

    h.pop('WAVTYPE')
    h.pop('CTYPE3')
    h.pop('CRVAL3')
    h.pop('CUNIT3')
    h.pop('CRPIX3')
    h.pop('CDELT3')
    h.pop('CROTA3')

    h['NAXIS'] = 1
    h['NAXIS1'] = len(axis)
    #h['NAXIS2'] = 2
    h['CRPIX1'] = 1
    h['CRVAL1'] = axis[0]
    h['CDELT1'] = axis[1]-axis[0]
    h['CUNIT1'] = 'Angstroms'
    h['CTYPE1'] = 'LINEAR'
    return h

def regrid_wavelength(cube):
    axis, spectrum = cube.extract_spectrum_bin(0, 0, 1)

    #We  build a regular axis
    irreg_axis = 1e8/axis
    reg_axis = np.linspace(irreg_axis[0], irreg_axis[-1],axis.shape[0] )
    #We flip it to make it wavelength ascending
    reg_axis = np.flip(reg_axis, 0)
    #We get extremas of the filter bandpass
    wl_min, wl_max = list(reversed([1e8/x for x in cube.params.filter_range]))
    #We conserve only spectrum inside this bandpass
    i_min = np.searchsorted(reg_axis, wl_min)
    i_max = np.searchsorted(reg_axis, wl_max)
    reg_axis = reg_axis[i_min:i_max]

    #This is the new header
    h = gen_spectrum_header(cube, reg_axis)

    new_cube = np.zeros((cube.shape[0], cube.shape[1], len(reg_axis)))
    k = 0
    k_max = cube.shape[0]*cube.shape[1]
    p = ProgressBar(k_max)
    for x in range(cube.shape[0]):
        for y in range(cube.shape[1]):
            k = k+1
            if k%1000 == 0:
                pass
                p.update(k)

            _, spectrum = cube.extract_spectrum_bin(x, y, 1, silent=True)
            #We interpolate on the regular axis
            new_cube[x, y, :] = interpolate_axis(spectrum, reg_axis, 5, old_axis=irreg_axis)
    p.end()

    path = "../nburst/{}_{}_wl_grid".format(cube.params['object_name'], cube.params['filter_name'],)
    io.write_fits('%s.fits'%path, new_cube, fits_header=h, overwrite=True)
    #return reg_axis, reg_spectrum



if __name__ == '__main__':

    parser = get_parser()
    args = parser.parse_args()
    path = args.file
    cube = SpectralCube(path)
    regrid_wavelength(cube)
