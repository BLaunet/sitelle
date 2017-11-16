from astropy.io import fits
from sitelle.utils import *
import numpy as np
from scipy.interpolate import UnivariateSpline
from orb.utils import io
import subprocess
import os
import copy
import sys
from path import Path
import socket

<<<<<<< HEAD
=======
if 'johannes' in socket.gethostname() or 'tycho' in socket.gethostname():
    NburstFitter.set_env('tycho')
>>>>>>> db19c795f14ee1710c3612710554f1e8b7713d34

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

def sew_spectra(left, right):
    if left.spectra.shape[1:] != right.spectra.shape[1:]:
        raise ValueError('Left and right spectra should have the same spatial dimension. Got %s and %s'%(left.spectra.shape[1:],right.spectra.shape[1:]))

    step = max(left.header['CDELT1'], right.header['CDELT1'])#We choose the largest step (lower resolution)
    axis = np.arange(left.axis.min(), right.axis.max(), step=step)
    spectra = np.zeros( axis.shape + left.spectra.shape[1:] )
    error = np.zeros( axis.shape + left.spectra.shape[1:] )
    fwhm = np.zeros( axis.shape + left.spectra.shape[1:] )

    left_max_id = np.searchsorted(axis, left.axis.max(), side='right')
    right_min_id = np.searchsorted(axis, right.axis.min())

    def interpolator(spectrum, old_axis, new_axis):
        f = UnivariateSpline(old_axis, spectrum, s=0)
        return f(new_axis)

    spectra[:left_max_id, ...] = np.apply_along_axis(interpolator, 0, left.spectra, left.axis, axis[:left_max_id])
    error[:left_max_id, ... ] = np.apply_along_axis(interpolator, 0, left.error, left.axis, axis[:left_max_id])
    fwhm[:left_max_id, ...] = np.apply_along_axis(interpolator, 0, left.fwhm, left.axis, axis[:left_max_id])

    spectra[right_min_id:, ...] = np.apply_along_axis(interpolator, 0, right.spectra, right.axis, axis[right_min_id:])
    error[right_min_id:, ...] = np.apply_along_axis(interpolator, 0, right.error, right.axis, axis[right_min_id:])
    fwhm[right_min_id:, ...] = np.apply_along_axis(interpolator, 0, right.fwhm, right.axis, axis[right_min_id:])

    # What do we put in between ?
    # Mean for fwhm & data cube
    # 10 x worst error for error
    def spec_func(vect, left_id, right_id):
        left = np.nanmean(vect[:left_id])
        right = np.nanmean(vect[right_id:])
        return np.linspace(left, right, right_id - left_id)
    def fwhm_func(vect, left_id, right_id):
        return np.linspace(vect[left_id-1], vect[right_id], right_id - left_id)
    def err_func(vect, left_id, right_id):
        worst = 10*max(vect[left_id-1], vect[right_id])
        return np.ones(right_id - left_id)*worst

    spectra[left_max_id:right_min_id, ...] = np.apply_along_axis(spec_func, 0, spectra, left_max_id, right_min_id)
    error[left_max_id:right_min_id, ...] = np.apply_along_axis(err_func, 0, error, left_max_id, right_min_id)
    fwhm[left_max_id:right_min_id, ...] = np.apply_along_axis(fwhm_func, 0, fwhm, left_max_id, right_min_id)

    return NburstFitter(axis, spectra, error, fwhm, left.header)

class NburstFitter():
    nburst_working_dir = Path('/Users/blaunet/Documents/M31/nburst/')
    idl_binary_path = "/usr/local/idl/idl/bin:/usr/local/idl/idl/bin/bin.darwin.x86_64:"
    def __init__(self, axis, spectra, error, fwhm, header, filedir, prefix):
        """
        all in wavenumber (== sortie de orcs)
        error et fwhm = meme dim spatiale que cube
        cube peut etre 3d, 2d (list de spectre), ou 1d
        cube [x,y,z] ou cube[x,z] ou cube[z]
        """
        self.axis = axis
        if len(spectra.shape) == 1:
            self.spectra = spectra.reshape(1, spectra.shape[0])
        else:
            self.spectra = spectra
        self.error = self._check_shape(self.spectra, error)
        self.fwhm = self._check_shape(self.spectra,fwhm)
        self.header = header

        self.fit_params={}
        self.filedir = Path(filedir).abspath()
        self.prefix = prefix
        self.idl_result = None
        self.fitted_spectra = None
        self.fit_name = None

        self._save_input()
        ## Environment variables

        self.idl_startup_script = "~/.idl/start.pro"
        self.template_path = self.nburst_working_dir / 'template.pro'
    @classmethod
    def set_env(cls, machine):
        if machine == 'tycho':
            cls.nburst_working_dir = Path('~/nburst/')
            cls.idl_binary_path = "/usr/local/idl/bin:/usr/local/idl/bin/bin.linux.x86_64:"
        else:
            pass

    @classmethod
    def from_sitelle_data(cls, axis, spectra, error, fwhm, ORCS_cube, filedir, prefix):
        if len(spectra.shape) == 1:
            original_spectra = spectra.reshape(1, spectra.shape[0])
        else:
            original_spectra = spectra

        original_error = cls._check_shape(original_spectra, error)
        original_fwhm = cls._check_shape(original_spectra,fwhm)
        xlims = ORCS_cube.get_filter_range()
        wl_axis = regular_wl_axis(axis, xlims).astype(float)

        #creation of an error cube
        err_cube = np.repeat(original_error.T[np.newaxis, ...],len(wl_axis), 0)

        #creation of a fwhm cube
        fwhm_cube = np.repeat(original_fwhm[..., np.newaxis],len(wl_axis), -1)
        for xy in np.ndindex(fwhm_cube.shape[:-1]):
            fwhm_cube[xy] = np.array([fwhm*lam**2/1e8 for fwhm,lam in zip(fwhm_cube[xy], wl_axis)])
        fwhm_cube = fwhm_cube.T

        #creation of the data cube interpolated on wl axis
        wl_spectra = np.zeros(original_spectra.shape[:-1]+wl_axis.shape)
        for xy in np.ndindex(original_spectra.shape[:-1]):
            wl_spectra[xy] = UnivariateSpline(axis.astype(float),original_spectra[xy],s=0)(1e8/wl_axis)
        wl_spectra = wl_spectra.T

        #Header
        wl_header = gen_wavelength_header(ORCS_cube.get_header(), wl_axis)
        nburst_header = swap_header_axis(wl_header, 1,3)

        nburst_fitter =  cls(wl_axis, wl_spectra, err_cube, fwhm_cube, nburst_header, filedir, prefix)
        nburst_fitter.fit_params['lmin'] = 4825.0 if ORCS_cube.params.filter_name == 'SN2' else 6480.
        nburst_fitter.fit_params['lmax'] = 5132.0 if ORCS_cube.params.filter_name == 'SN2' else 6800.
        return nburst_fitter

    @classmethod
    def from_previous(cls, filedir, prefix):
        spectra, header = io.read_fits(filedir+prefix+'_data.fits', return_header = True)
        errors = io.read_fits(filedir+prefix+'_error.fits')
        fwhms = io.read_fits(filedir+prefix+'_fwhm.fits')
        axis = read_wavelength_axis(header, 1)
        return cls(axis, spectra, errors, fwhms, header, filedir, prefix)

    @classmethod
    def from_single_spectra(cls, axis, spectra, error, fwhm, header, filedir, prefix):
        spectra = spectra.reshape(spectra.shape[0], 1)
        error = error.reshape(error.shape[0], 1)
        fwhm = fwhm.reshape(fwhm.shape[0], 1)
        return cls(axis, spectra, error, fwhm, header, filedir, prefix)

    @staticmethod
    def _check_shape(spectra,q):
        if type(q) is not np.ndarray:
            q = np.array([q])
        if spectra.shape[:-1] != q.shape[:-1]:
            raise ValueError("shape %s doesn't match spectra shape %s : "%(q.shape, spectra.shape))
        return q

    def _save_input(self, filedir=None, prefix=None):
        if filedir is None:
            if self.filedir is None:
                raise ValueError('You have to provide a valid directory for the data')
            else:
                filedir = self.filedir
        if prefix is None:
            if self.prefix is None:
                raise ValueError('You have to provide a valid prefix for the data')
            else:
                prefix = self.prefix

        self.filedir = Path(filedir).abspath()
        self.prefix = prefix

        self.filedir.makedirs_p()

        io.write_fits(self.filedir / self.prefix+'_data.fits', self.spectra, self.header, overwrite=True)
        io.write_fits(self.filedir / self.prefix+'_fwhm.fits', self.fwhm, self.header, overwrite=True)
        io.write_fits(self.filedir / self.prefix+'_error.fits', self.error, self.header, overwrite=True)

    def configure_fit(self, fit_name, **kwargs):
        self.fit_name = fit_name

        self.fit_params['galfile'] = str(self.filedir / self.prefix+ '_data.fits')
        self.fit_params['errfile'] = str(self.filedir / self.prefix+ '_error.fits')
        self.fit_params['fwhmfile'] = str(self.filedir / self.prefix+ '_fwhm.fits')
        self.fit_params['result_file'] = str(self.filedir / self.prefix+'_'+self.fit_name+'_fitted.fits')
        try:
            nz, ny, nx = self.spectra.shape
            self.fit_params['nx'] = nx
            self.fit_params['nz'] = nz
            self.fit_params['ny'] = ny
            self.fit_params['read3d'] = True
            self.fit_params['xsize'] = ny
        except ValueError:
            nz, nx = self.spectra.shape
            self.fit_params['nx'] = nx
            self.fit_params['nz'] = nz
            self.fit_params['ny'] = 0
            self.fit_params['read3d'] = False
            self.fit_params['xsize'] = False

        self.fit_params.update(kwargs)
        if 'lmin' not in self.fit_params:
            self.fit_params['lmin'] = False
        if 'lmax' not in self.fit_params:
            self.fit_params['lmax'] = False

        if ('vorbin' in self.fit_params and self.fit_params['vorbin'] is True) or ('SN' in self.fit_params and self.fit_params['SN'] is True):
            if ('vorbin' in self.fit_params and 'SN' not in self.fit_params) or ('SN' in self.fit_params and 'vorbin' not in self.fit_params):
                raise ValueError('Please provide a signal to noise ratio value (SN=) for voronoi binning')
            if 'vorbin' in self.fit_params and self.fit_params['read3d'] is False:
                raise ValueError('Voronoi binning can only be performed on a 3d cube')
        if 'vorbin' not in self.fit_params:
            self.fit_params['vorbin'] = False
            self.fit_params['SN'] = False

        if 'exclreg' in self.fit_params:
            if self.fit_params['exclreg'] == 'default':
                self.fit_params.pop('exclreg')

        if 'silent' in self.fit_params and self.fit_params['silent'] is True:
            self.fit_params.pop('silent')
            self.fit_params['plot'] = 0
            self.fit_params['window'] = False


        fit_params = copy.deepcopy(self.fit_params)

        def _assign(key, value):
            if type(value) == bool and value == False:
                return '\n'
            if type(value) == bool and value == True:
                value = 1
            if type(value) == str:
                value = "'"+value+"'"
            return '%s = %s\n'%(key,value)

        with open(self.template_path, 'r') as temp:
            template = temp.readlines()
        for i,line in enumerate(template):
            for k,v in fit_params.items():
                if line.startswith(k):
                    template[i] = _assign(k,v)
                    j = i
                    while '$' in line:
                        j += 1
                        line = template[j]
                        template[j] = _assign(k,False)

                    fit_params.pop(k)

        with open(self.filedir / self.prefix+'_'+self.fit_name+'.pro', 'w') as out:
            out.writelines(template)

    def run_fit(self):
        env = os.environ
        env['PATH'] = self.idl_binary_path + env['PATH']
        env['IDL_STARTUP'] = self.idl_startup_script

        script_path = self.filedir / self.prefix+'_'+self.fit_name+'.pro'
        p = subprocess.Popen('idl %s'%script_path, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
        for line in p.stdout.readlines():
            print line,
            sys.stdout.flush()
        retval = p.wait()

        self.read_result(self.filedir / self.prefix+'_'+self.fit_name+'_fitted.fits')

    def read_result(self, filename=None):
        if filename is None:
            filename = self.filedir / self.prefix+'_'+self.fit_name+'_fitted.fits'
        hdu = fits.open(filename)
        self.bin_table = hdu[1].data[0][0].astype(int).reshape(self.fwhm.shape[1:])
        self.idl_result = hdu[2].data

    def extract_spectrum(self, binNumber):
        axis = self.idl_result['WAVE'][binNumber,:]
        fit = self.idl_result['FIT'][binNumber,:]
        return axis,fit
    def get_fitted_spectra(self):
        if self.fitted_spectra is None:
            if self.idl_result is None:
                self.read_result()
            specs = self.idl_result['FIT'][self.bin_table].T
            idl_axis = self.idl_result['WAVE'][0]
            def interpolate(spec, old_axis, new_axis):
                return UnivariateSpline(old_axis, spec, s=0, ext='zeros')(new_axis)
            self.fitted_spectra = np.apply_along_axis(interpolate, 0, specs, idl_axis, self.axis)
        return self.fitted_spectra

if 'johannes' in socket.gethostname() or 'tycho' in socket.gethostname():
    NburstFitter.set_env('tycho')

    # def extract_spectrum(self, binNumber, wn_axis):
    #     wl_axis, fit = self.extract_wl_spectrum(binNumber)
    #     return nm_to_cm1(fit, wl_axis, wn_axis)
