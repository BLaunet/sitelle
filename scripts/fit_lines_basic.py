from orb.utils import io
import numpy as np
from orcs.process import SpectralCube
from orcs.utils import fit_lines_in_spectrum
import gvar

SN3 = SpectralCube('../fits/orig/M31_SN3.merged.cm1.1.0.hdf5')
SN3.correct_wavelength('../fits/M31_SN3.1.0.ORCS/M31_SN3.1.0.skymap.fits')
theta_binned = io.read_fits('../fits/M31_SN3_thetamap_48x48.fits')
cube_binned = io.read_fits('../fits/M31_SN3_rebinned_48.fits')
velocity_guess_map_binned = np.load('../fits/velocity_guess_map_48x48.npy')

##Smaller
cube_binned = cube_binned[:10, :10, :]
theta_binned = theta_binned[:10,:10]
velocity_guess_map_binned = velocity_guess_map_binned[:10, :10]

result = np.zeros((cube_binned.shape[:2]), dtype=object)
lines=['[NII]6548','[NII]6583', 'Halpha', '[SII]6716', '[SII]6731']
kwargs={'fmodel':'sincgauss', 'pos_def':'1'}
for x, y in np.ndindex(cube_binned.shape[:2]):
    print x,y
    kwargs['pos_cov'] = velocity_guess_map_binned[x,y]
    SN3._prepare_input_params(lines, nofilter=False, **kwargs)
    inputparams = SN3.inputparams
    params = SN3.params

    result[x, y] = fit_lines_in_spectrum(params, inputparams, 1e10, cube_binned[x, y, :],
                          theta_binned[x, y], snr_guess=None)



np.save('LinesFit.npy', result)
