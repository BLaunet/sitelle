from orcs.process import SpectralCube
import numpy as np
from sitelle.region import *
from orb.utils import io

SN2 = SpectralCube('../fits/orig/M31_SN2.merged.cm1.1.0.hdf5')
SN3 = SpectralCube('../fits/orig/M31_SN3.merged.cm1.1.0.hdf5')

binsize = 48
xsize, ysize = SN2.shape[:2]
new_xsize = xsize//binsize if xsize % binsize == 0 else xsize//binsize + 1
new_ysize = ysize//binsize if ysize % binsize == 0 else ysize//binsize + 1
SN2_binned = np.zeros((new_xsize, new_ysize)+ SN2.shape[2:])

for x, y in np.ndindex(SN2_binned.shape[:2]):
    print x,y
    SN3_region = square_region(x*binsize,y*binsize,binsize)
    SN2x = []
    SN2y = []
    for xy in zip(SN3_region[0], SN3_region[1]):
        xi,yi =  SN2.world2pix(SN3.pix2world(xy))[0]
        if xi < 0 or xi > SN2.dimx-1 or yi <0 or yi > SN2.dimy-1:
            continue
        SN2x.append(int(round(xi)))
        SN2y.append(int(round(yi)))
    SN2_region = tuple([np.array(SN2x), np.array(SN2y)])
    a, s = SN2.extract_integrated_spectrum(SN2_region)
    SN2_binned[x,y, ...] = s
io.write_fits('M31_SN2_rebinned_on_SN3_48x48_cube.fits', SN2_binned, SN2.get_header())
