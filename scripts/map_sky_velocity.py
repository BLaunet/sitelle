from orcs.process import SpectralCube

SN2 = SpectralCube('../../fits/orig/M31_SN2.merged.cm1.1.0.hdf5')
SN2.map_sky_velocity(-80., div_nb = 40)
