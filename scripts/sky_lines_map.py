from orcs.process import SpectralCube
from sitelle.region import *

cube = SpectralCube('/Users/blaunet/Documents/M31/fits/orig/M31_SN3.merged.cm1.1.0.hdf5')
cube.correct_wavelength('/Users/blaunet/Documents/M31/fits/M31_SN3.1.0.ORCS/M31_SN3.1.0.skymap.fits')

lines = cube.get_sky_lines()
full_region = square_region(0,0,2064)

cube.fit_lines_in_region(full_region, lines, binning=48,
                        pos_cov=0,
                        pos_def='1',
                        fwhm_def='fixed',
                        fmodel='sinc',
                        nofilter=False)
