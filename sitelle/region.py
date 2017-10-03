import numpy as np
import orb


def circular_region(x, y, r):
    if r < 0: r = 0.001
    X, Y = np.mgrid[0:2048, 0:2064]
    R = np.sqrt(((X-x)**2 + (Y-y)**2))
    return np.nonzero(R <= r)
def square_region(x, y, b):
    mask = np.zeros((2048, 2064))
    mask[x:x+b, y:y+b] = 1
    return np.where(mask == 1)
def physical_region(cube, ra, dec, r = 2, circle=True):
    ra_sex = map(float, ra.split(':'))
    dec_sex = map(float, dec.split(':'))
    ra_deg = orb.utils.astrometry.ra2deg(ra_sex)
    dec_deg = orb.utils.astrometry.dec2deg(dec_sex)
    x, y = map(int, map(round, cube.world2pix((ra_deg, dec_deg))[0]))
    if circle:
        return circular_region(x, y, r)
    else:
        return square_region(x, y, int(round(r)))
