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

def remap(map, binMap=None, binsize=None, original_shape = None):
    """
    Used to remap a binned map on a full map
    User has to provide either a binMap or a binsize and an original shape
    :param binMap: a map of original size filled with indices corresponding to the bin number in the binned map
    """
    if binMap is None and (binsize is None or original_shape is None):
        raise ValueError("If no binMap is provided, both binsize and original_shape should be")
    if binMap is not None:
        raise  NotImplementedError('binMap not implemented yet')
    else:
        if len(map.shape) > 2:
            original_shape = original_shape + map.shape[2:]
        full_map = np.zeros(original_shape)
        for x,y in np.ndindex(map.shape[:2]):
            full_map[x*binsize:(x+1)*binsize, y*binsize:(y+1)*binsize, ...] = map[x, y, ...]

    return full_map
