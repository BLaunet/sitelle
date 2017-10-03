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
    parser.add_argument("-f", "--file",
                        help="file to process")
    parser.add_argument("-o", "--out_prefix",
                        help="prefix for output path",
                        default='.')
    parser.add_argument("-z", "--zdim",
                        help="Dimension along spectral axis. Default = 221 = size for SN2 cube bandpass",
                        default=221)
    return parser

def generate_errmap_cube(flux_err, z_size, out_prefix='.'):

    errcube = np.repeat(flux_err[:, :, np.newaxis], z_size, axis=2)
    path = "{}/M31_SN3_errcube".format(out_prefix)
    io.write_fits('%s.fits'%path, errcube, overwrite=True)

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    path = args.file
    zdim = args.zdim
    out_prefix = args.out_prefix

    errmap = io.read_fits(path)
    generate_errmap_cube(errmap, zdim, out_prefix)
