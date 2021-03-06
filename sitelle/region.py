"""
File containing helper functions to defines *regions* in the cube, in the format of tuple of indexes.
"""
import numpy as np
import orb
from scipy.interpolate import UnivariateSpline

__all__ = ['circular_region', 'square_region', 'centered_square_region', 'physical_region', 'remap', 'get_contour_boundaries', 'fill_contour', 'smooth_contour', 'convert_contour']

def circular_region(x, y, r):
    """
    Computes a circular region of radius r centered on (x,y)

    Parameters
    ----------
    x : int
        abscissse of the center, in pixels
    y : int
        ordonate of the center, in pixels
    r : float
        radius of the region, in pixels
    Returns
    -------
    region : tuple
        The corresponding region
    """
    if r < 0: r = 0.001
    X, Y = np.mgrid[0:2048, 0:2064]
    R = np.sqrt(((X-x)**2 + (Y-y)**2))
    return np.nonzero(R <= r)
def square_region(x, y, b):
    """
    Computes a square region of size b*b

    Parameters
    ----------
    x : int
        abscisse of the bottom left corner of the square
    y : int
        ordonate of the bottom left corner of the square
    b : int
        size of the square
    Returns
    -------
    region : tuple
        The corresponding region
    """
    mask = np.zeros((2048, 2064))
    mask[x:x+b, y:y+b] = 1
    return np.where(mask == 1)
def centered_square_region(x,y,b):
    """
    Computes a square region of size b*b centered on x,y

    Parameters
    ----------
    x : int
        abscisse of the center of the square
    y : int
        rdonate of the center of the square
    b : int
        size of the square
    Returns
    -------
    region : tuple
        The corresponding region
    """
    mask = np.zeros((2048, 2064))
    if b%2 == 0:
        xmin, xmax = x-b/2, x+b/2
        ymin, ymax = y-b/2, y+b/2
    else:
        xmin, xmax = x-b/2, x+b/2+1
        ymin, ymax = y-b/2, y+b/2+1
    if xmin < 0:
        xmin = 0
    if ymin < 0:
        ymin = 0
    mask[xmin:xmax, ymin:ymax] = 1
    return np.where(mask == 1)

def physical_region(cube, ra, dec, r = 2, circle=True, sex = True, centered=True):
    """
    Computes a physical region on a sitelle cube.
    If circle = True (Default), it's a circulare region centered on ra,dec. Otherwise, (ra,dec) are the coordinates of the bottom left corner of a square region

    Parameters
    ----------
    cube : :class:`~ORCS:orcs.process.SpectralCube`
        The cube under consideration
    ra : str or float
        right ascension in sexagesimal in format 'xx:xx:xx' if sex = True, else in degrees
    dec : str or float
        declinaison in sexagesimal in format 'xx:xx:xx' if sex = True, else in degrees
    r : float
        radius in pixels of the circle if circle=True, else size of the square
    circle : bool, Default = True
        (Optional) If True, circular region, otherwise square
    sex : bool, Default = True
        (Optional) If True, ra,dec expected to be in sexagesimal, otherwise in degrees
    centered : bool, Default = True
        (Optional) If True, the square is a centered square
    Returns
    -------
    region : tuple
        The corresponding region
    """
    if sex:
        ra_sex = map(float, ra.split(':'))
        dec_sex = map(float, dec.split(':'))
        ra_deg = orb.utils.astrometry.ra2deg(ra_sex)
        dec_deg = orb.utils.astrometry.dec2deg(dec_sex)
    else:
        ra_deg = ra
        dec_deg = dec
    x, y = map(int, map(round, cube.world2pix((ra_deg, dec_deg))[0]))
    if circle:
        return circular_region(x, y, r)
    elif centered:
        return centered_square_region(x, y, int(round(r)))
    else:
        return square_region(x, y, int(round(r)))

def remap(map, binMap=None, binsize=None, original_shape = None):
    """
    Used to remap a binned map on a full map
    User has to provide either a binMap, or a binsize and an original shape

    Parameters
    ----------
    map : 2D :class:`~numpy:numpy.ndarray`
        The binned data map
    binMap : 2D :class:`~numpy:numpy.ndarray`
        a map of original size filled with indices corresponding to the bin number in the binned map. Not needed if  ``binsize`` and ``original_shape`` are provided.
    binsize : int
        The binning size used. ``original_shape`` must be given as well
    original_shape : tuple of int
        The original shape (x,y).

    Returns
    -------
    full_map : 2D :class:`~numpy:numpy.ndarray`
        The full map containing the unbinned data
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

def get_contour_boundaries(region):
    '''
    DEPRECATED

    Computes the contours of a regions.
    Only works for reasonnably smooth regions (convex, ideally square or circle).
    Boundaries are not necessary smooth (some indices may be missing)

    Parameters
    ----------
    region : tuple of array of indices
        The region from which we want boundaries

    Returns
    -------
    contour : dict
        A dict of 4 1-D regions defining the contour : top, bottom, left, right

    See Also
    --------
    :func:`smooth_contour` to complete missing values
    '''
    X, Y = region[0], region[1]
    ymin = []
    ymax = []
    for x in np.unique(X):
        x_indices = np.argwhere(X == x)
        ymin.append(Y[x_indices[0]][0])
        ymax.append(Y[x_indices[-1]][0])
    ymin = np.array(ymin)
    ymax = np.array(ymax)

    xmin = []
    xmax = []
    for y in np.unique(Y):
        y_indices = np.argwhere(Y == y)
        xmin.append(X[y_indices[0]][0])
        xmax.append(X[y_indices[-1]][0])
    xmin = np.array(xmin)
    xmax = np.array(xmax)
    contour = {'left': tuple([xmin, np.unique(Y)]),
            'right': tuple([xmax, np.unique(Y)]),
            'top' : tuple([np.unique(X),ymax]),
            'bottom': tuple([np.unique(X),ymin])}

    return contour

def fill_contour(mask, contour):
    """
    Fill the inside of a contour, to go back to the original region.

    Parameters
    ----------
    mask : 2D :class:`~numpy:numpy.ndarray`
        full map on wich the contour is defined
    contour : dict
        The dict of 4 1D regions : top, bottom, left, right (returned by :func:`get_contour_boundaries`)

    Returns
    -------
    mask : the mask filled inside the contour
    """
    top_x = contour['top'][0]
    bottom_x = contour['top'][0]
    right_y = contour['right'][1]
    left_y = contour['left'][1]
    X_range = top_x if len(top_x) >= len(bottom_x) else bottom_x
    Y_range = right_y if len(right_y) >= len(left_y) else left_y
    for i,x in enumerate(X_range):
        for j, y in enumerate(Y_range):
            if x >= contour['left'][0][j] and x <= contour['right'][0][j]:
                if y >= contour['bottom'][1][i] and y <= contour['top'][1][i]:
                    mask[x,y] = 1
    return mask

def smooth_contour(contour):
    """
    Smooth discontinuous contours

    Parameters
    ----------
    contour : dict
        Contours of a region (see :func:`get_contour_boundaries`)
    Returns
    -------
    contour : dict
        A continuous contour
    """
    new_contour = {}
    _, a_left = contour['left']
    _, a_right = contour['right']
    a_h = np.arange(min(a_left.min(), a_right.min()), max(a_left.max(), a_right.max())+1)
    a_top, _ = contour['top']
    a_bottom, _ = contour['bottom']
    a_v = np.arange(min(a_top.min(), a_bottom.min()), max(a_top.max(), a_bottom.max())+1)
    for k in contour:
        if k in ['left', 'right']:
            O, A = contour[k]
        else:
            A, O = contour[k]
        #Make sure that abscisse is unique and sorted
        A, unique_id = np.unique(A, return_index=True)
        O = O[unique_id]
        argsort_A = np.argsort(A)
        smoother = UnivariateSpline(A[argsort_A],O[argsort_A], s=0)
        if k in ['left', 'right']:
            new_A = a_h
        else:
            new_A = a_v
        #new_A = np.arange(A.min(), A.max()+1)
        new_O = np.apply_along_axis(lambda x: map(int, map(round, x)), 0, smoother(new_A))

        if k in ['left', 'right']:
            new_contour[k] = tuple([new_O, new_A])
        else:
            new_contour[k] = tuple([new_A, new_O])

    return new_contour

def convert_contour(original_cube,  contour, new_cube):
    """
    Convert a contour computed on a given cube to a new cube, using astrometric calibration. It allows to plot the same contoured region on different datacubes.

    Parameters
    ----------
    original_cube : :class:`~ORCS:orcs.process.SpectralCube`
        Cube on which the contour has been computed
    new_cube : :class:`~ORCS:orcs.process.SpectralCube`
        Cube on which we want to project the contour
    contour : dict
        the contour, i.e. a dict of 4 1D  regions : top, bottom, left, right

    Returns
    -------
    contour : dict
        the contour projected on the new cube
    """
    new_contour = {}
    for k in contour.keys():
        newx = []
        newy = []
        for xy in zip(contour[k][0], contour[k][1]):
            xi,yi =  new_cube.world2pix(original_cube.pix2world(xy))[0]
            if xi < 0:
                xi = 0
            elif xi > new_cube.dimx-1:
                xi = new_cube.dimx-1
            if yi <0:
                yi = 0
            elif yi > new_cube.dimy-1:
                yi = new_cube.dimy-1
            newx.append(int(round(xi)))
            newy.append(int(round(yi)))
        new_contour[k] = tuple([np.array(newx), np.array(newy)])
    return new_contour
