from astropy.io import fits
from sitelle.utils import *
import numpy as np
from scipy.interpolate import UnivariateSpline
from orb.utils import io
import subprocess
import os
import copy

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

class NburstFitter():
    def __init__(self, axis, cube, error, fwhm, ORCS_cube):
        """
        all in wavenumber (== sortie de orcs)
        error et fwhm = meme dim spatiale que cube
        cube peut etre 3d, 2d (list de spectre), ou 1d
        cube [x,y,z] ou cube[x,z] ou cube[z]
        """
        self.original_axis = axis
        if len(cube.shape) == 1:
            self.original_cube = cube.reshape(1, cube.shape[0])
        else:
            self.original_cube = cube

        self.original_error = self._check_shape(error)
        self.original_fwhm = self._check_shape(fwhm)
        self.ORCS_cube = ORCS_cube
        self.filter = self.ORCS_cube.params['filter_name']

        self.nburst_working_dir = ''
    def set_original(original_axis, original_cube, original_error, original_fwhm, ORCS_cube):
        self.original_axis = axis
        if len(cube.shape) == 1:
            self.original_cube = cube.reshape(1, cube.shape[0])
        else:
            self.original_cube = cube

        self.original_error = self._check_shape(error)
        self.original_fwhm = self._check_shape(fwhm)
        self.ORCS_cube = ORCS_cube
        self.filter = self.ORCS_cube.params['filter_name']

    @classmethod
    def from_sitelle_data(axis, spectra, error, fwhm, ORCS_cube):
        pass


    def _check_shape(self, q):
        if type(q) is not np.ndarray:
            q = np.array([q])
        if self.original_cube.shape[:-1] != q.shape:
            raise ValueError("shape %s doesn't match cube shape %s : "%(q.shape, self.original_cube.shape))
        return q

    def prepare_input(self, filedir, name, xlims=None):
        #creation of the regular wavelength axis
        if xlims is None:
            xlims = self.ORCS_cube.get_filter_range()
        self.wl_axis = regular_wl_axis(self.original_axis, xlims).astype(float)

        #creation of an error cube
        self.nburst_err_cube = np.repeat(self.original_error.T[np.newaxis, ...],len(self.wl_axis), 0)

        #creation of a fwhm cube
        fwhm_cube = np.repeat(self.original_fwhm[..., np.newaxis],len(self.wl_axis), -1)
        for xy in np.ndindex(fwhm_cube.shape[:-1]):
            fwhm_cube[xy] = np.array([fwhm*lam**2/1e8 for fwhm,lam in zip(fwhm_cube[xy], self.wl_axis)])
        self.nburst_fwhm_cube = fwhm_cube.T

        #creation of the data cube interpolated on wl axis
        wl_cube = np.zeros(self.original_cube.shape[:-1]+self.wl_axis.shape)
        for xy in np.ndindex(self.original_cube.shape[:-1]):
            wl_cube[xy] = UnivariateSpline(self.original_axis.astype(float),self.original_cube[xy],s=0)(1e8/self.wl_axis)
        self.nburst_data_cube = wl_cube.T

        #Header
        wl_header = gen_wavelength_header(self.ORCS_cube.get_header(), self.wl_axis)
        self.nburst_header = swap_header_axis(wl_header, 1,3)

        self._save_input(filedir, name)
    def _save_input(self, filedir, name):

        if filedir[-1] != '/':
            filedir+='/'
        self.filedir = filedir
        self.fit_name = name

        io.write_fits(self.filedir+self.fit_name+'_data.fits', self.nburst_data_cube, self.nburst_header, overwrite=True)
        io.write_fits(self.filedir+self.fit_name+'_fwhm.fits', self.nburst_fwhm_cube, self.nburst_header, overwrite=True)
        io.write_fits(self.filedir+self.fit_name+'_error.fits', self.nburst_err_cube, self.nburst_header, overwrite=True)

    def configure_fit(self, **kwargs):
        self.template_path = '../nburst/fit_procedure_template.pro'
        kwargs['galfile'] = self.filedir+self.fit_name+ '_data.fits'
        kwargs['errfile'] = self.filedir+self.fit_name+ '_error.fits'
        kwargs['fwhmfile'] = self.filedir+self.fit_name+ '_fwhm.fits'
        kwargs['result_file'] = self.filedir+self.fit_name+ '_fitted.fits'
        try:
            nz, ny, nx = self.nburst_data_cube.shape
            kwargs['nx'] = nx
            kwargs['nz'] = nz
            kwargs['ny'] = ny
            kwargs['read3d'] = True
            kwargs['xsize'] = ny
        except ValueError:
            nz, nx = self.nburst_data_cube.shape
            kwargs['nx'] = nx
            kwargs['nz'] = nz
            kwargs['ny'] = 0
            kwargs['read3d'] = False
            kwargs['xsize'] = False


        if 'lmin' not in kwargs:
            kwargs['lmin'] = 4825.0 if self.filter == 'SN2' else 6480.
        if 'lmax' not in kwargs:
            kwargs['lmax'] = 5132.0 if self.filter == 'SN2' else 6800.

        if ('vorbin' in kwargs and 'SN' not in kwargs) or ('SN' in kwargs and 'vorbin' not in kwargs):
            raise ValueError('Please provide a signal to noise ratio value (SN=) for voronoi binning')
        if 'vorbin' in kwargs and kwargs['read3d'] is False:
            raise ValueError('Voronoi binning can only be performed on a 3d cube')
        if 'vorbin' not in kwargs:
            kwargs['vorbin'] = False
            kwargs['SN'] = False

        if 'exclreg' in kwargs:
            if kwargs['exclreg'] == 'default':
                kwargs.pop('exclreg')

        self.nburst_fit_params = copy.deepcopy(kwargs)

        with open(self.template_path, 'r') as temp:
            template = temp.readlines()
        for i,line in enumerate(template):
            for k,v in self.nburst_fit_params.items():
                if line.startswith(k):
                    template[i] = self._assign(k,v)
                    j = i
                    while '$' in line:
                        j += 1
                        line = template[j]
                        print line
                        template[j] = self._assign(k,False)

                    kwargs.pop(k)

        with open(self.filedir+self.fit_name+'_procedure.pro', 'w') as out:
            out.writelines(template)

    def _assign(self, key, value):
        if type(value) == bool and value == False:
            return '\n'
        if type(value) == bool and value == True:
            value = 1
        if type(value) == str:
            value = "'"+value+"'"
        return '%s = %s\n'%(key,value)

    def run_fit(self):
        env = os.environ
        env['PATH'] = "/usr/local/idl/idl/bin:/usr/local/idl/idl/bin/bin.darwin.x86_64:" + env['PATH']
        env['IDL_STARTUP'] = "~/.idl/start.pro"

        script_path = self.filedir+self.fit_name+'_procedure.pro'
        p = subprocess.Popen('idl %s'%script_path, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
        for line in p.stdout.readlines():
            print line,
        retval = p.wait()

        self._read(self.filedir+self.fit_name+'_fitted.fits')

    def parameter_map(self, param, binMap):
        """
        Return a 2D map of the given parameter
        :param table: the BINTable in which the parameters are found
        :param param: The parameter we want to plot. See table.columns to see the list
        :param binMap: the 2D map of the binning scheme
        """
        def f(binNumber):
            return table[param][binNumber]
        return f(binMap)

    def _read(self, filename):
        hdu = fits.open(filename)
        self.bin_table = hdu[1].data
        self.fit_table = hdu[2].data
        return self.bin_table, self.fit_table

    def extract_spectrum(self, binNumber):
        axis = self.fit_table['WAVE'][binNumber,:]
        fit = self.fit_table['FIT'][binNumber,:]
        return axis,fit

    # def extract_spectrum(self, binNumber, wn_axis):
    #     wl_axis, fit = self.extract_wl_spectrum(binNumber)
    #     return nm_to_cm1(fit, wl_axis, wn_axis)
