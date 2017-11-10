from astropy.io import fits
from sitelle.utils import *
import numpy as np

def parameter_map(table, param, binMap):
    """
    Return a 2D map of the given parameter
    :param table: the BINTable in which the parameters are found
    :param param: The parameter we want to plot. See table.columns to see the list
    :param binMap: the 2D map of the binning scheme
    """
    def f(binNumber):
        return table[param][binNumber]
    return f(binMap)

def read(filename):
    hdu = fits.open(filename)
    bin_table = hdu[1].data
    fit_table = hdu[2].data
    return bin_table, fit_table

def extract_spectrum(fit_table, binNumber):
    axis = fit_table['WAVE'][binNumber,:]
    fit = fit_table['FIT'][binNumber,:]
    return axis,fit

class NburstCube():
    def __init__(self, axis, cube, error, fwhm, ORCS_cube):
        """
        all in wavenulber (== sortie de orcs)
        error et fwhm = meme dim spatiale que cube
        cube peut etre 3d, 2d (list de spectre), ou 1d
        cube [x,y,z] ou cube[x,z] ou cube[z]
        """
        self.original_axis = axis
        if len(cube.shape) == 1:
            self.original_cube
        self.original_cube = cube

        if type(error) is not np.ndarray:
            error = np.array([error])
        if
                raise ValueError("error shape %s doesn't match cube shape %s : "%(error.shape, cube.shape)
                raise ValueError("error shape %s doesn't match cube shape %s : "%(error.shape, cube.shape)
        else:
            raise ValueError("error shape %s not OK"%(error.shape)

        self.original_error = error
        self.original_fwhm = fwhm
        self.ORCS_cube = ORCS_cube
        self.filter = ORCS.params['filter_name']

        self.nburst_working_dir = ''
    def _check_shape(cube_shape, error_shape):

    def prepare_input(filename=None, xlims=None):
        if xlims is None:
            xlims = self.ORCS_cube.get_filter_range()
        self.wl_axis = regular_wl_axis(self.original_axis, xlims)

        err_vect = self.original_error*np.ones(len(self.wl_axis))
        fwhm_vect = self.original_fwhm*np.ones(len(self.wl_axis))
        fwhm_vect = np.array([fwhm*lam**2/1e8 for fwhm,lam in zip(fwhm_vect, wl_axis)])

        wl_spec = UnivariateSpline(a,s,s=0)(1e8/reg_axis)

    def configure_fit():
        pass
    def run_fit():
        pass
    def convert_result():
        pass

class Spectrum():
    def __init__(self, axis, spectrum, error, fwhm, cube):
        self.axis = axis
        self.spectrum = spectrum
        self.error = error
        self.fwhm = fwhm
        self.cube = cube
