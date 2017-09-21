from __future__ import division
from orcs.process import SpectralCube
import orb
import numpy as np
import logging
from orb.utils import io
import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file",
                        help="file to process")
    parser.add_argument("-o", "--out_prefix",
                        help="prefix for output path")
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

def generate_fwhm_cube(cube, out_prefix):

    fwhm_cm1_map = cube.get_fwhm_map()
    base_axis = cube.params.base_axis.astype(float)
    irreg_axis = 1e8/base_axis
    reg_axis = np.linspace(irreg_axis[0], irreg_axis[-1], base_axis.shape[0] )
    #We flip it to make it wavelength ascending
    reg_axis = np.flip(reg_axis, 0)
    fwhm_cube = np.repeat(fwhm_cm1_map[:, :, np.newaxis], cube.dimz, axis=2)
    for z in range(fwhm_cube.shape[2]):
        print(z)
        fwhm_cube[:, :, z] =  fwhm_cm1_map * reg_axis[z]**2/1e8

    h = cube.get_header()
    path = "{}/{}_{}_fwhm_cube".format(out_prefix,
                                    cube.params['object_name'],
                                    cube.params['filter_name'])
    io.write_fits('%s.fits'%path, fwhm_cube, fits_header=h, overwrite=True)



if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    path = args.file
    cube = SpectralCube(path)

    out_prefix = args.out_prefix
    generate_fwhm_cube(cube, out_prefix)
