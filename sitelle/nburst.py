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

__all__ = ['parameter_map', 'read', 'extract_spectrum', 'sew_spectra', 'NburstFitter', 'NburstFitterList']

def parameter_map(table, param, binMap):
    """
    Displays an estimated parameter on a 2D map
    Parameters
    ----------
    table : :class:`~astropy:astropy.io.fits.BinTableHDU`
        The table in which the parameters are found (returned by ``NBurst``)
    param : str
        The parameter we want to extract. See table.columns to see the list
    binMap: 2D :class:`~numpy:numpy.ndarray`
        The 2D map of the binning scheme (e.g. a voronoi map)

    Returns
    -------
    `~numpy:numpy.ndarray`
        The 2D map of the parameter
    """
    def f(binNumber):
        return table[param][binNumber]
    return f(binMap)

def read(filename):
    """
    Reads results from NBurst

    Parameters
    ----------
    filename : str
        the file path to read
    Return
    ------
    tuple
        the bin_table and fit_table created by ``Nburst``
    """
    hdu = fits.open(filename)
    bin_table = hdu[1].data
    fit_table = hdu[2].data
    hdu.close()
    return bin_table, fit_table

def extract_spectrum(fit_table, binNumber):
    """
    Get the ``NBurst`` fitted spectrum for a given bin number.
    Parameters
    ----------
    fit_table :
        the fit_table containg the fitted spectra
    binNumber : int
        the binNumber
    Returns
    -------
    tuple of :class:`~numpy:numpy.ndarray`
        the wavelength axis (in Angstroms) and fitted spectrum
    """
    axis = fit_table['WAVE'][binNumber,:]
    fit = fit_table['FIT'][binNumber,:]
    return axis,fit

def sew_spectra(left, right):
    """
    Sew 2 spectra fitter together.
    This is used to fit the data from two filters simultaneously.
    In between the two, we put a mean value for data & fwhm, and 10x the worst error for the error.

    Parameters
    ----------
    left : :class:`NburstFitter`
        left spectra fitter
    right : :class:`NburstFitter`
        right spectra fitter
    Returns
    -------
    :class:`NburstFitter`
        a NburstFitter instance containing the 2 spectra
    """
    if left.spectra.shape[:-1] != right.spectra.shape[:-1]:
        raise ValueError('Left and right spectra should have the same spatial dimension. Got %s and %s'%(left.spectra.shape[:-1],right.spectra.shape[:-1]))

    step = max(left.header['CDELT1'], right.header['CDELT1'])#We choose the largest step (lower resolution)
    axis = np.arange(left.axis.min(), right.axis.max(), step=step)
    spectra = np.zeros( left.spectra.shape[:-1] + axis.shape)
    error = np.zeros( left.spectra.shape[:-1] + axis.shape )
    fwhm = np.zeros( left.spectra.shape[:-1] + axis.shape)

    left_max_id = np.searchsorted(axis, left.axis.max(), side='right')
    right_min_id = np.searchsorted(axis, right.axis.min())

    def interpolator(spectrum, old_axis, new_axis):
        f = UnivariateSpline(old_axis, spectrum, s=0)
        return f(new_axis)

    spectra[..., :left_max_id] = np.apply_along_axis(interpolator, -1, left.spectra, left.axis, axis[:left_max_id])
    error[..., :left_max_id] = np.apply_along_axis(interpolator, -1, left.error, left.axis, axis[:left_max_id])
    fwhm[..., :left_max_id] = np.apply_along_axis(interpolator, -1, left.fwhm, left.axis, axis[:left_max_id])

    spectra[..., right_min_id:] = np.apply_along_axis(interpolator, -1, right.spectra, right.axis, axis[right_min_id:])
    error[..., right_min_id:] = np.apply_along_axis(interpolator, -1, right.error, right.axis, axis[right_min_id:])
    fwhm[..., right_min_id:] = np.apply_along_axis(interpolator, -1, right.fwhm, right.axis, axis[right_min_id:])

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

    spectra[..., left_max_id:right_min_id] = np.apply_along_axis(spec_func, -1, spectra, left_max_id, right_min_id)
    error[..., left_max_id:right_min_id] = np.apply_along_axis(err_func, -1, error, left_max_id, right_min_id)
    fwhm[..., left_max_id:right_min_id] = np.apply_along_axis(fwhm_func, -1, fwhm, left_max_id, right_min_id)

    return NburstFitter(axis, spectra, error, fwhm, left.header)

class NburstFitter():
    """
    This class is a helper to fit spectra with Nburst directly from python.
    It stores data as expected by Nburst, configure a script with user-defined parameters, runs it and reads the results.

    Attributes
    ----------
    axis : 1D :class:`~numpy:numpy.ndarray`
        a regular wavelength axis in Angstroms
    spectra : 2 or 3D :class:`~numpy:numpy.ndarray`
        spectra of shape [x,z] or [x,y,z], interpolated on the ``axis``
    error : 2 or 3D :class:`~numpy:numpy.ndarray`
        an error array of the *same shape* as ``spectra``
    fwhm : 2 or 3D :class:`~numpy:numpy.ndarray`
        an fwhm array of the *same shape* as ``spectra``
    header : str
        a header describing the data. Should match spectra shape : [NAXIS1, NAXIS2, NAXIS3] <=> [x,y,z]
    filedir : str
        the folder where the NburstFitter data should be stored
    prefix : str
        a prefix for this fitter.
    fit_name : str
        (Optional) the fit_name, if already performed.
    fit_params : dict
        Parameters for the nburst fit
    force : bool
        (Optional) If set to True, the Nburst input data will be recomputed even if the ``filedir`` and ``prefix`` already exists. Default = False, which means than the function can have no effect if the ``filedir`` and ``prefix`` already exist.
    idl_result : :class:`~astropy:astropy.io.fits.BinTableHDU`
        The table in which the parameters are found (returned by ``NBurst``)
    fitted_spectra :  1, 2 or 3D :class:`~numpy:numpy.ndarray`
        The Nburst-fitted spectra. Same dimension as input ``spectra``
    bin_table : 2D :class:`~numpy:numpy.ndarray`
        The binning scheme used by NBurst

    nburst_working_dir : :class:`~path:path.Path`
        Class attribute. The working directory for nburst, where the library are stored.
        This is machine dependent and can be updated with the :func:`set_env` function.
    idl_binary_path : str
        Class attribute. Location of the idl binary. This is machine dependent and can be updated with the :func:`set_env` function.
    idl_startup_script : str
        Class attribute. Path of the script to be executed at startup of IDL, to compile the right libraries. This script should always be stored at ``~/.idl/start.pro``.

    """

    nburst_working_dir = Path('/Users/blaunet/Documents/M31/nburst/')
    idl_binary_path = "/usr/local/idl/idl/bin:/usr/local/idl/idl/bin/bin.darwin.x86_64:"
    idl_startup_script = "~/.idl/start.pro"

    @classmethod
    def set_env(cls, machine, user):
        '''
        Sets the right value for ``nburst_working_dir`` and ``idl_binary_path``, depending on the machine. This has been specifically implemented for B. Launet and should be modified when used by others.

        Parameters
        ----------
        machine : str
            hostname of the machine
        '''
        if machine == 'tycho':
            cls.nburst_working_dir = Path('/obs/%s/nburst/'%user)
            cls.idl_binary_path = "/usr/local/idl/bin:/usr/local/idl/bin/bin.linux.x86_64:"
        elif machine == 'barth':
            cls.nburst_working_dir = Path('/Volumes/TOSHIBA/M31/nburst/')
            cls.idl_binary_path = "/usr/local/idl/idl/bin:/usr/local/idl/idl/bin/bin.darwin.x86_64:"
        else:
            pass

    @classmethod
    def from_sitelle_data(cls, axis, spectra, error, fwhm, ORCS_cube, filedir, prefix, force = False):
        """
        This function converts sitelle data to be fitted by Nburst.
        It interpolates a spectra onto a regular wavelength grid, and creates error and fwhm "spectra" as well (by extending maps in the spectral direction).
        Spectra, error and fwhm should have the same **spatial** dimensions (x or x,y)

        Parameters
        ----------
        axis : 1D :class:`~numpy:numpy.ndarray`
            Axis of the spectrum [cm-1]
        spectra : 1,2 3D :class:`~numpy:numpy.ndarray`
            The spectra to be fitted. It can be a one dimesion ([z]), two dimensions ([x,z]) or three dimensions ([x,y,z]).
        error : 1 or 2D :class:`~numpy:numpy.ndarray`
            an error array of the same spatial dim as the spectra ([x,y] if spectra is [x,y,z]), i.e one error per pixel
        fwhm : 1 or 2D :class:`~numpy:numpy.ndarray`
            a fwhm array of the same spatial dim as the spectra ([x,y] if spectra is [x,y,z]), i.e one fwhm per pixel
        ORCS_cube : :class:`~ORCS:orcs.process.SpectralCube`
            The SpectralCube from which the spectra are exatracted
        filedir : str
            the folder where the NburstFitter data should be stored
        prefix : str
            a prefix for this fitter.
        force : bool
            (Optional) If set to True, the Nburst input data will be recomputed even if the ``filedir`` and ``prefix`` already exists. Default = False, which means than the function can have no effect if the ``filedir`` and ``prefix`` already exist.

        Returns
        -------
        :class:`NburstFitter`
            A :class:`NburstFitter` on which the fit can be performed.
        """
        if len(spectra.shape) == 1:
            original_spectra = spectra.reshape(1, spectra.shape[0])
        else:
            original_spectra = spectra

        if type(error) is not np.ndarray:
            original_error = np.array([error])
        else:
            original_error = error

        if type(fwhm) is not np.ndarray:
            original_fwhm = np.array([fwhm])
        else:
            original_fwhm = fwhm

        if original_spectra.shape[:-1] != original_error.shape:
            raise ValueError("spatial dimensions don't match : error : %s, spectra : %s"%(original_error.shape, original_spectra.shape))
        if original_spectra.shape[:-1] != original_fwhm.shape:
            raise ValueError("spatial dimensions don't match : fwhm : %s, spectra : %s"%(original_fwhm.shape, original_spectra.shape))

        xlims = ORCS_cube.get_filter_range()
        wl_axis = regular_wl_axis(axis, xlims).astype(float)

        #creation of an error cube
        err_cube = np.repeat(original_error[..., np.newaxis],len(wl_axis), -1)

        #creation of a fwhm cube
        fwhm_cube = np.repeat(original_fwhm[..., np.newaxis],len(wl_axis), -1)
        for xy in np.ndindex(fwhm_cube.shape[:-1]):
            fwhm_cube[xy] = np.array([fwhm*lam**2/1e8 for fwhm,lam in zip(fwhm_cube[xy], wl_axis)])

        #creation of the data cube interpolated on wl axis
        wl_spectra = np.zeros(original_spectra.shape[:-1]+wl_axis.shape)
        for xy in np.ndindex(original_spectra.shape[:-1]):
            wl_spectra[xy] = UnivariateSpline(axis.astype(float),original_spectra[xy],s=0)(1e8/wl_axis)

        #Header
        wl_header = gen_wavelength_header(ORCS_cube.get_header(), wl_axis, len(original_spectra.shape))
        nburst_fitter =  cls(wl_axis, wl_spectra, err_cube, fwhm_cube, wl_header, filedir, prefix, force=force)
        nburst_fitter.fit_params['lmin'] = 4825.0 if ORCS_cube.params.filter_name == 'SN2' else 6480.
        nburst_fitter.fit_params['lmax'] = 5132.0 if ORCS_cube.params.filter_name == 'SN2' else 6842.
        return nburst_fitter

    @classmethod
    def from_sitelle_region(cls, region, ORCS_cube, filedir, prefix, force = False):
        """
        Generate the right parameters (spectra, error, fwhm) to create a Nburst fittable cube from an *integrated* region in a SITELLE datacube. It's a wrapper around `from_sitelle_data`.

        Parameters
        ----------
        region : tuple of :class:`~numpy:numpy.ndarray`.
            The region to study, in the same format understood by ORCS i.e. the results from ``np.nonzero(mask)`` for example.
        ORCS_cube : :class:`~ORCS:orcs.process.SpectralCube`
            The SpectralCube from which the spectra are exatracted
        filedir : str
            the folder where the NburstFitter data should be stored
        prefix : str
            a prefix for this fitter.
        force : bool
            (Optional) If set to True, the Nburst input data will be recomputed even if the ``filedir`` and ``prefix`` already exists. Default = False, which means than the function can have no effect if the ``filedir`` and ``prefix`` already exist.
        Returns
        -------
        :class:`NburstFitter`
            A :class:`NburstFitter` on which the fit can be performed.
        """
        axis, spec = ORCS_cube.extract_integrated_spectrum(region)
        error = estimate_noise(axis, spec, ORCS_cube.get_filter_range())
        fwhm = ORCS_cube.get_fwhm_map()[region].mean()
        return cls.from_sitelle_data(axis, spec, error, fwhm, ORCS_cube, filedir, prefix, force)
    @classmethod
    def from_previous(cls, filedir, prefix, fit_name = None):
        """
        Reload a NburstFitter that had been created previously, based on its filedir and prefix.
        Parameters
        ----------
        filedir : str
            the folder where the NburstFitter data should be stored
        prefix : str
            a prefix for this fitter.
        fit_name : str
            (Optional) the fit_name, if already performed.
        Returns
        -------
        :class:`NburstFitter`
            A :class:`NburstFitter` on which the fit can be performed.
        """
        spectra, header = io.read_fits(Path(filedir) / prefix+'_data.fits', return_header = True)
        spectra = spectra.T
        if len(spectra.shape) == 1:
            spectra = spectra.reshape(1, spectra.shape[0])
        errors = io.read_fits(Path(filedir) / prefix+'_error.fits').T
        if len(errors.shape) == 1:
            errors = errors.reshape(1, errors.shape[0])
        fwhms = io.read_fits(Path(filedir)  / prefix+'_fwhm.fits').T
        if len(fwhms.shape) == 1:
            fwhms = fwhms.reshape(1, fwhms.shape[0])
        axis = read_wavelength_axis(header, 1)
        header = swap_header_axis(header, 1, len(spectra.shape))

        return cls(axis, spectra, errors, fwhms, header, filedir, prefix, fit_name)


    def __init__(self, axis, spectra, error, fwhm, header, filedir, prefix, fit_name=None, force = False):
        """
        Initialize a NburstFitter instance.
        Data should already be in the right format : spectral axis, in a regular wavelength grid, errors and fwhms maps converted to cube, header properly written.
        We recommend to use specified method like `from_sitelle_data` or `from_sitelle_region` instead.

        Parameters
        ----------
        axis : 1D :class:`~numpy:numpy.ndarray`
            a regular wavelength axis in Angstroms
        spectra : 2 or 3D :class:`~numpy:numpy.ndarray`
            spectra of shape [x,z] or [x,y,z], interpolated in the ``axis``
        error : 2 or 3D :class:`~numpy:numpy.ndarray`
            an error array of the *same shape* as ``spectra``
        fwhm : 2 or 3D :class:`~numpy:numpy.ndarray`
            an fwhm array of the *same shape* as ``spectra``
        header : str
            a header describing the data. Should match spectra shape : [NAXIS1, NAXIS2, NAXIS3] <=> [x,y,z]
        filedir : str
            the folder where the NburstFitter data should be stored
        prefix : str
            a prefix for this fitter.
        fit_name : str
            (Optional) the fit_name, if already performed.
        force : bool
            (Optional) If set to True, the Nburst input data will be recomputed even if the ``filedir`` and ``prefix`` already exists. Default = False, which means than the function can have no effect if the ``filedir`` and ``prefix`` already exist.
        Returns
        -------
        :class:`NburstFitter`
            A :class:`NburstFitter` on which the fit can be performed.
        """
        self.axis = axis
        self.spectra = np.atleast_2d(spectra)
        error = np.atleast_2d(error)
        if error.shape != self.spectra.shape:
            raise ValueError("Error shape %s doesn't match spectra shape %s : "%(error.shape, self.spectra.shape))
        else:
            self.error = error
        fwhm = np.atleast_2d(fwhm)
        if fwhm.shape != self.spectra.shape:
            raise ValueError("Fwhm shape %s doesn't match spectra shape %s : "%(fwhm.shape, self.spectra.shape))
        else:
            self.fwhm = fwhm
        self.header = header

        self.fit_params={}
        self.filedir = Path(filedir).abspath()
        self.prefix = prefix
        self.idl_result = None
        self.fitted_spectra = None
        self.fit_name = fit_name

        if not (self.filedir/self.prefix+'_data.fits').isfile() or force is True:
            self._save_input()
        ## Environment variables
        self.template_path = self.nburst_working_dir / 'template.pro'

    def _save_input(self, filedir=None, prefix=None):
        """
        Save the data in the proper directory structure and naming.
        """
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

        swaped_header = swap_header_axis(copy.deepcopy(self.header), 1, len(self.spectra.shape))

        io.write_fits(self.filedir / self.prefix+'_data.fits', self.spectra.T, swaped_header, overwrite=True)
        io.write_fits(self.filedir / self.prefix+'_fwhm.fits', self.fwhm.T, swaped_header, overwrite=True)
        io.write_fits(self.filedir / self.prefix+'_error.fits', self.error.T, swaped_header, overwrite=True)

    def configure_fit(self, fit_name=None, **kwargs):
        """
        Configurator for the Nburst fit.
        These function writes a procedure file for IDL containing all the fitting parameters we want to be set.
        They are all optionals, in which case default value are used.

        Parameters
        ----------
        silent : bool
            If True, nothing is displayed during the fit (Default = False)
        lmin : int
            The lower wavelength limit on which the fit has to be performed
        lmax : int
            The upper wavelength limit on which the fit has to be performed
        vorbin : bool
            If True, a voronoi tessellation is performed on the data. Only works with 3D datacubes. (Default = False) If True, ``SN`` has to be defined as well.
        SN : int
            The signal to noise ratio to use for the voronoi tessellation. ``vorbin`` has to be set to True.
        exclreg : 'default' or list of float
            Exclusion regions to be ommitted during the fit, in Angstroms. Typically concern emission lines region. 'default' uses the main emission lines as defined in :py:data:`sitelle.constants.SN2_LINES` and :py:data:`sitelle.constants.SN3_LINES`
        mdegree : int
            Degree of the polynomial to be used (Default = 7)
        vsys : float
            Velocity guess in km/s. Default : -300 km/s
        sig_guess : float
            Velocity dispersion guess in km/s. Default = 100 km/s
        age_guess : float
            Age guess in Myr. Default = 15000
        met_guess : float
            Metallicity guess. Default = 0.
        fixpar : list of int
            Defines if parameters are fixed or not. The order is [vsys, sigma, age, met]. 0 standas for False (i.e. free) and 1 for True (i.e. fixed). For exmaple, to fix the velocity while letting the rest free, fixpar=[1,0,0,0]
        lsf : str
            Line shape function to use. If 'SINC', a cardinal sine is used. Else, the defaulkt gauss-hermite function is used.
        stell_lib : {'ELODIE', 'MILES'}
            The stellar library to use to create the SSP. Default = 'ELODIE'
        """
        if fit_name is None:
            if self.fit_name is None:
                raise ValueError('Please provide a name for this fit configuration')
            else:
                fit_name = self.fit_name
        self.fit_name = fit_name

        self.fit_params['galfile'] = str(self.filedir / self.prefix+ '_data.fits')
        self.fit_params['errfile'] = str(self.filedir / self.prefix+ '_error.fits')
        self.fit_params['fwhmfile'] = str(self.filedir / self.prefix+ '_fwhm.fits')
        self.fit_params['result_file'] = str(self.filedir / self.prefix+'_'+self.fit_name+'_fitted.fits')
        try: # If we are dealing with a 3D cube
            nx, ny, nz = self.spectra.shape
            self.fit_params['nx'] = nx
            self.fit_params['nz'] = nz
            self.fit_params['ny'] = ny
            self.fit_params['read3d'] = True
            self.fit_params['xsize'] = ny
        except ValueError: # Else it's 2D
            nx, nz = self.spectra.shape
            self.fit_params['nx'] = nx
            self.fit_params['nz'] = nz
            self.fit_params['ny'] = 0
            self.fit_params['read3d'] = False
            self.fit_params['xsize'] = False
        if 'silent' not in kwargs:
            kwargs['silent']=True
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

    def run_fit(self, silent = False):
        """
        Runs the NburstFitter, once properly configured.
        It spawns a subprocess and call IDL on the previously configured script.

        Parameters
        ----------
        silent : bool
            (Optional) Default False. If True, the output of the subporocess is displayed.
        """
        env = os.environ
        env['PATH'] = self.idl_binary_path + env['PATH']
        env['IDL_STARTUP'] = self.idl_startup_script

        script_path = self.filedir / self.prefix+'_'+self.fit_name+'.pro'
        p = subprocess.Popen('idl %s'%script_path, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
        if not silent:
            for line in p.stdout.readlines():
                print line,
                sys.stdout.flush()
        retval = p.wait()

        self.read_result()

    def read_result(self, filename=None, force = False):
        """
        Read the raw NBurst result and convert it to a more natural format, by populating the ``fitted_spectra``, ``idl_result`` and ``bin_table`` attributes of the :class:`NburstFitter`.

        Parameters
        ----------
        filename : str
            (Optional) The full path to the fitted result file. Default to 'filedir'/'prefix'_'fit_name'_fitted.fits
        force : bool
            (Optional) If True, we recompute the ``fitted_spectra``, ``idl_result`` and ``bin_table`` attributes even if they are already populated. Default = False.
        """
        if self.fitted_spectra is not None and force is False:
            return None
        if filename is None:
            try:
                filename = self.filedir / self.prefix+'_'+self.fit_name+'_fitted.fits'
            except TypeError:
                filename = self.filedir / self.prefix+'_fitted.fits'
        hdu = fits.open(filename)
        try:
            self.bin_table = hdu[1].data[0][0].astype(np.uint).reshape(self.spectra.shape[:-1])
            self.idl_result = hdu[2].data
        except IndexError:
            print "Can not load results for %s; the fit probably didn't converge"%(filename)
            hdu.close()
            self.idl_result = None
            self.fitted_spectra = None
            return None
        hdu.close()
        specs = self.idl_result['FIT'][self.bin_table]
        idl_axis = self.idl_result['WAVE'][0]
        def interpolate(spec, old_axis, new_axis):
            return UnivariateSpline(old_axis, spec, s=0, ext='zeros')(new_axis)
        self.fitted_spectra = np.apply_along_axis(interpolate, -1, specs, idl_axis, self.axis)
        # coldefs = idl_result.columns
        # coldefs.add_col(fits.Column(name='BESTFIT', array=bestfit.T, format='%dD'%bestfit.shape[0]))
        # self.idl_result = fits.FITS_rec.from_columns(coldefs)

    def extract_spectrum(self, binNumber):
        """
        Extract the fitted spectra in a given bin.

        Parameters
        ----------
        binNumber : int
            The number of the bin.

        Returns
        -------
        axis : 1D :class:`~numpy:numpy.ndarray`
            The corresponding axis, in Angstroms
        fit :  1D :class:`~numpy:numpy.ndarray`
            The fitted spectra
        """
        axis = self.idl_result['WAVE'][binNumber,:]
        fit = self.idl_result['FIT'][binNumber,:]
        return axis,fit

class NburstFitterList():
    """
    This class is a wrapper to fit different spectra, having different fiting inputs (e.g. metallicities or age guess, different parameters...).

    Because running a single :class:`NburstFitter` means opening and closing IDL, it would be too fastidious to just build a loop.
    Instead, this Fitter list concatenates all the Fitters in a single script to run them at once. It works the same way a single :class:`NburstFitter` does.

    Attributes
    ----------
    axis : 1D :class:`~numpy:numpy.ndarray`
        a regular wavelength axis in Angstroms
    spectra : 2 or 3D :class:`~numpy:numpy.ndarray`
        spectra of shape [x,z] or [x,y,z], interpolated on the ``axis``
    error : 2 or 3D :class:`~numpy:numpy.ndarray`
        an error array of the *same shape* as ``spectra``
    fwhm : 2 or 3D :class:`~numpy:numpy.ndarray`
        an fwhm array of the *same shape* as ``spectra``
    fitted_spectra :  1, 2 or 3D :class:`~numpy:numpy.ndarray`
        The Nburst-fitted spectra. Same dimension as input ``spectra``
    idl_result : :class:`~astropy:astropy.io.fits.BinTableHDU`
        The table in which the parameters are found (returned by ``NBurst``)

    nburst_working_dir : :class:`~path:path.Path`
        Class attribute. The working directory for nburst, where the library are stored.
        This is machine dependent and can be updated with the :func:`set_env` function.
    idl_binary_path : str
        Class attribute. Location of the idl binary. This is machine dependent and can be updated with the :func:`set_env` function.
    idl_startup_script : str
        Class attribute. Path of the script to be executed at startup of IDL, to compile the right libraries. This script should always be stored at ``~/.idl/start.pro``.
    """

    nburst_working_dir = Path('/Users/blaunet/Documents/M31/nburst/')
    idl_binary_path = "/usr/local/idl/idl/bin:/usr/local/idl/idl/bin/bin.darwin.x86_64:"
    idl_startup_script = "~/.idl/start.pro"

    @classmethod
    def set_env(cls, machine, user):
        '''
        Sets the right value for ``nburst_working_dir`` and ``idl_binary_path``, depending on the machine. This has been specifically implemented for B. Launet and should be modified when used by others.

        Parameters
        ----------
        machine : str
            hostname of the machine
        '''
        if machine == 'tycho':
            cls.nburst_working_dir = Path('/obs/%s/nburst/'%user)
            cls.idl_binary_path = "/usr/local/idl/bin:/usr/local/idl/bin/bin.linux.x86_64:"
        elif machine == 'barth':
            cls.nburst_working_dir = Path('/Volumes/TOSHIBA/M31/nburst/')
            cls.idl_binary_path = "/usr/local/idl/idl/bin:/usr/local/idl/idl/bin/bin.darwin.x86_64:"
        else:
            pass

    def __init__(self, fitters):
        """
        Initialize a NburstFitterList instance.

        Parameters
        ----------
        fitters : list of :class:`NburstFitter`
            The different fitters to concatenate.
        Returns
        -------
        :class:`NburstFitterList`
            A :class:`NburstFitterList` on which the fit can be performed.
        """
        self.fitters = fitters
        self.axis = self.fitters[0].axis
        self.spectra = np.concatenate([fitter.spectra for fitter in self.fitters], axis=0)
        self.error = np.concatenate([fitter.error for fitter in self.fitters], axis=0)
        self.fwhm = np.concatenate([fitter.fwhm for fitter in self.fitters], axis=0)

        self.fitted_spectra = None
    def configure_fit(self, **kwargs):
        """
        Configurator for the Nburst fit.
        The configuration will be applied to all the fitters present in the NburstFitterList.
        See :func:`NburstFitter.configure_fit` for details about the arguments.
        """
        for fitter in self.fitters:
            fitter.configure_fit(**kwargs)

    def run_each(self, silent = False):
        """
        Runs the :class:`NburstFitterList`, once properly configured.
        Same procedure as :func:`NburstFitter.run_fit` but avoid the trouble to open and close idl each time.

        Parameters
        ----------
        silent : bool
            (Optional) Default False. If True, the output of the subporocess is displayed.
        """
        env = os.environ
        env['PATH'] = self.idl_binary_path + env['PATH']
        env['IDL_STARTUP'] = self.idl_startup_script

        script_path = ' '.join([fitter.filedir / fitter.prefix+'_'+fitter.fit_name+'.pro' for fitter in self.fitters])
        p = subprocess.Popen('idl %s'%script_path, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
        if not silent:
            for line in p.stdout.readlines():
                print line,
                sys.stdout.flush()
        retval = p.wait()

        self.read_result()
    def read_result(self, force = False):
        """
        Read the raw NBurst result and convert it to a more natural format, by populating the ``fitted_spectra`` and ``idl_result`` attributes of the :class:`~sitelle.nburst.NburstFitterList`.

        Parameters
        ----------
        force : bool
            (Optional) If True, we recompute the ``fitted_spectra``, ``idl_result`` and ``bin_table`` attributes even if they are already populated. Default = False.
        """
        if self.fitted_spectra is not None and force is False:
            return None
        for fitter in self.fitters:
            fitter.read_result(force=force)
            if fitter.fitted_spectra is None:
                fitter.fitted_spectra = np.full_like(fitter.spectra, np.nan)
                fitter.idl_result = {}
                for col in self.fitters[0].idl_result.columns:
                    fitter.idl_result[col.name] = np.full_like(self.fitters[0].idl_result[col.name], np.nan)


        self.idl_result={}
        for col in self.fitters[0].idl_result.columns:
            if col.name == 'CHI2' or 'D' in col.format:
                self.idl_result[col.name] = np.concatenate([fitter.idl_result[col.name] for fitter in self.fitters], axis=0)
        self.fitted_spectra = np.concatenate([fitter.fitted_spectra for fitter in self.fitters], axis=0)


if 'johannes' in socket.gethostname() or 'tycho' in socket.gethostname():
    NburstFitter.set_env('tycho', os.getenv('USER'))
    NburstFitterList.set_env('tycho', os.getenv('USER'))
if 'MacBookAirdeBarthelemy' in socket.gethostname():
    NburstFitter.set_env('barth', os.getenv('USER'))
    NburstFitterList.set_env('barth', os.getenv('USER'))

