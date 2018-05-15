#Tweak for documentation building
import os
ON_RTD = os.environ.get('READTHEDOCS', None) == 'True'
ON_SPHINX = 'sphinx-build' in os.environ.get('_', '')
if not ON_RTD and not ON_SPHINX:
    from orcs.process import SpectralCube
else:
    class SpectralCube(object):
        pass

    
import logging
import numpy as np
import orb
import gvar
import scipy.interpolate
import warnings
from sitelle.parallel import available_cpu_count
from orb.utils.parallel import init_pp_server, close_pp_server
from orcs.core import Filter

__all__ = ['SpectralCubePatch']

class SpectralCubePatch(SpectralCube):
    """
    Overload of :class:`ORCS:orcs.process.SpectralCube`, to extend or correct some behaviors.

    """

    def get_filter_range(self, wavenumber=True):
        """
        Returns the filter range of a given SpectralCube.

        Parameters
        ----------
        wavenumber : bool, Default = True
            (Optional) If True, the range is in cm-1, else in Angstroms.

        See Also
        --------
        :func:`ORCS:orcs.process.SpectralCube.get_filter_range`
        """
        lims = super(SpectralCubePatch, self).get_filter_range()
        if wavenumber:
            return lims
        else:
            return np.flip(np.array([1e8/l for l in lims]), 0)


    def _extract_spectra_from_region(self, region, silent=False, return_theta=False):
        """
        Extracts **non integrated** spectra from the cube, given a region.
        The parallelization is automatically decided, given the number of pixel to treat.

        Parameters
        ----------
        region : tuple
            a region in pixel (obtained with :func:`numpy:numpy.where` for example), from which we want to get spectra.
        silent : bool, Default = False
            (Optional) If True, a log is displayed.
        return_theta : bool, Default = False
            (Optional) If True, returns the theta values in the ``region``

        Returns
        -------
        out : 3D :class:`~numpy:numpy.ndarray`
            Extracted spectra of the same shape as ``region``
        """

        def _interpolate_spectrum(spec, corr, wavenumber, step, order, base_axis):
            if wavenumber:
                corr_axis = orb.utils.spectrum.create_cm1_axis(
                    spec.shape[0], step, order, corr=corr)
                return orb.utils.vector.interpolate_axis(
                    spec, base_axis, 5, old_axis=corr_axis)
            else: raise NotImplementedError()

        def _extract_spectrum_in_column(data_col, calib_coeff_col, mask_col,
                                        wavenumber, base_axis, step, order,
                                        base_axis_corr):

            for icol in range(data_col.shape[0]):
                if mask_col[icol]:
                    corr = calib_coeff_col[icol]
                    if corr != base_axis_corr:
                        data_col[icol, :] = _interpolate_spectrum(
                            data_col[icol, :], corr, wavenumber, step, order, base_axis)
                else:
                    data_col[icol, :].fill(np.nan)
            return data_col

        calibration_coeff_map = self.get_calibration_coeff_map()

        mask = np.zeros((self.dimx, self.dimy), dtype=np.uint8)
        mask[region] = 1
        if not silent:
            logging.info('Number of extracted pixels: {}'.format(np.sum(mask)))

        if np.sum(mask) == 0: raise StandardError('A region must contain at least one valid pixel')

        elif np.sum(mask) == 1:
            ii = region[0][0] ; ij = region[1][0]
            data = _interpolate_spectrum(
                self.get_data(ii, ii+1, ij, ij+1, 0, self.dimz, silent=silent),
                calibration_coeff_map[ii, ij],
                self.params.wavenumber, self.params.step, self.params.order,
                self.params.base_axis)
            theta = self.get_theta_map()[ii,ij]
            data = data.reshape(1,1,self.dimz)

        else:
            mask_x_proj = np.nanmax(mask, axis=1).astype(float)
            mask_x_proj[np.nonzero(mask_x_proj == 0)] = np.nan
            mask_x_proj *= np.arange(self.dimx)
            x_min = int(np.nanmin(mask_x_proj))
            x_max = int(np.nanmax(mask_x_proj)) + 1

            mask_y_proj = np.nanmax(mask, axis=0).astype(float)
            mask_y_proj[np.nonzero(mask_y_proj == 0)] = np.nan
            mask_y_proj *= np.arange(self.dimy)
            y_min = int(np.nanmin(mask_y_proj))
            y_max = int(np.nanmax(mask_y_proj)) + 1

            #check if paralell extraction is necessary
            parallel_extraction = True
            #It takes roughly ncpus/4 s to initiate the parallel server
            #The non-parallel algo runs at ~400 pixel/s
            ncpus = self.params['ncpus']
            if ncpus/4. > np.sum(mask)/400.:
                parallel_extraction = False

            data = self.get_data(x_min, x_max, y_min, y_max,
                                    0, self.dimz, silent=silent).reshape(
                                        x_max-x_min, y_max-y_min, self.dimz
                                    )
            if parallel_extraction:
                logging.debug('Parallel extraction')
                # multi-processing server init

                job_server, ncpus = init_pp_server(available_cpu_count(), silent=silent)
                if not silent: progress = orb.core.ProgressBar(x_max - x_min)
                for ii in range(0, x_max - x_min, ncpus):
                    # no more jobs than columns
                    if (ii + ncpus >= x_max - x_min):
                        ncpus = x_max - x_min - ii

                    # jobs creation
                    jobs = [(ijob, job_server.submit(
                        _extract_spectrum_in_column,
                        args=(data[ii+ijob,:,:],
                              calibration_coeff_map[x_min + ii + ijob,
                                                    y_min:y_max],
                              mask[x_min + ii + ijob, y_min:y_max],
                              self.params.wavenumber,
                              self.params.base_axis, self.params.step,
                              self.params.order, self.params.axis_corr),
                        modules=("import logging",
                                 'import numpy as np',
                                 'import orb.utils.spectrum',
                                 'import orb.utils.vector'),
                        depfuncs=(_interpolate_spectrum,)))
                            for ijob in range(ncpus)]

                    for ijob, job in jobs:
                        data[ii+ijob,:,:] = job()

                    if not silent:
                        progress.update(ii)
                close_pp_server(job_server)
                if not silent: progress.end()

            else:
                logging.debug('Non Parallel extraction')
                local_mask = mask[x_min:x_max, y_min:y_max]
                local_calibration_coeff_map = calibration_coeff_map[x_min:x_max, y_min:y_max]
                if not silent:
                    progress = orb.core.ProgressBar(local_mask.size)
                    k = 0
                for i,j in np.ndindex(data.shape[:-1]):
                    if local_mask[i,j]:
                        corr = local_calibration_coeff_map[i,j]
                        if corr != self.params.axis_corr:
                            data[i,j] = _interpolate_spectrum(
                                data[i,j], corr, self.params.wavenumber,
                                self.params.step, self.params.order, self.params.base_axis)
                    else:
                        data[i,j].fill(np.nan)
                    if not silent:
                        k+=1
                        if k%100 == 0:
                            progress.update(k)
                if not silent: progress.end()

            data = data.reshape(x_max-x_min, y_max-y_min, self.dimz)
            theta = self.get_theta_map()[x_min:x_max, y_min:y_max]
        if return_theta:
            return data, theta
        else:
            return data

    def _extract_spectrum_from_region(self, region,
                                      subtract_spectrum=None,
                                      median=False,
                                      mean_flux=False,
                                      silent=False,
                                      return_spec_nb=False,
                                      return_mean_theta=False,
                                      return_gvar=False,
                                      output_axis=None):
        """
        Overloads :func:`ORCS:orcs.core._extract_spectra_from_region`,
        to make use of :func:`_extract_spectra_from_region`.
        """
        if median:
            warnings.warn('Median integration')
        mask = np.zeros((self.dimx, self.dimy), dtype=np.uint8)
        mask[region] = 1

        data = self._extract_spectra_from_region(region, silent=silent)

        all_nan_specs = np.apply_along_axis(lambda s: np.all(np.isnan(s)), 2, data)
        spec_nb = all_nan_specs.size - np.nansum(all_nan_specs.astype(int))
        if median:
            with np.warnings.catch_warnings():
                np.warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')
                spectrum = np.nanmedian(data, axis=(0,1)) * spec_nb
        else:
            spectrum = np.nansum(data, axis=(0,1))

        # add uncertainty on the spectrum
        if return_gvar:
            flux_uncertainty = self.get_flux_uncertainty()

            if flux_uncertainty is not None:
                uncertainty = np.nansum(flux_uncertainty[np.nonzero(mask)])
                logging.debug('computed mean flux uncertainty: {}'.format(uncertainty))
                spectrum = gvar.gvar(spectrum, np.ones_like(spectrum) * uncertainty)


        if subtract_spectrum is not None:
            spectrum -= subtract_spectrum * spec_nb

        if mean_flux:
            spectrum /= spec_nb

        returns = list()
        if output_axis is not None and np.all(output_axis == self.params.base_axis):
            spectrum[np.isnan(gvar.mean(spectrum))] = 0. # remove nans
            returns.append(spectrum)

        else:
            nonans = ~np.isnan(gvar.mean(spectrum))
            spectrum_function = scipy.interpolate.UnivariateSpline(
                self.params.base_axis[nonans], gvar.mean(spectrum)[nonans],
                s=0, k=1, ext=1)
            if return_gvar:
                spectrum_function_sdev = scipy.interpolate.UnivariateSpline(
                    self.params.base_axis[nonans], gvar.sdev(spectrum)[nonans],
                    s=0, k=1, ext=1)
                raise Exception('now a tuple is returned with both functions for mean and sdev, this will raise an error somewhere and must be checked before')
                spectrum_function = (spectrum_function, spectrum_function_sdev)

            if output_axis is None:
                returns.append(spectrum_function(gvar.mean(output_axis)))
            else:
                returns.append(spectrum_function)

        if return_spec_nb:
            returns.append(spec_nb)
        if return_mean_theta:
            theta_map = self.get_theta_map()
            mean_theta = np.nanmean(theta_map[np.nonzero(mask)])
            logging.debug('computed mean theta: {}'.format(mean_theta))
            returns.append(mean_theta)

        return returns

    def integrate(self, filter_function, xmin=None, xmax=None, ymin=None, ymax=None):
        """
        Integrate a cube under a filter function and generate an image

        :math:`I = \int F(\sigma)S(\sigma)d\sigma`

        with :math:`I`, the image, :math:`S` the spectral cube, :math:`F` the
        filter function.

        Contrary to :func:`ORCS:orcs.core.integrate`, it uses the correctly wavelength calibrated spectra.

        Parameters
        ----------
        filter_function : :class:`ORCS:orcs.core.Filter`
            The filter to use.
        xmin : int
            (Optional) lower boundary of the ROI along x axis in pixels (default None, i.e. 0)
        xmax : int
            (Optional) upper boundary of the ROI along y axis in pixels (default None, i.e. dimx)
        ymin : int
            (Optional) lower boundary of the ROI along y axis in pixels (default None, i.e. 0)
        ymax : int
            (Optional) upper boundary of the ROI along y axis in pixels (default None, i.e. dimy)

        Returns
        -------
        sframe : 2D :class:`~numpy:numpy.ndarray`
            The integrated frame
        """
        if not isinstance(filter_function, Filter):
            raise TypeError('filter_function must be an orcs.core.Filter instance')

        if (filter_function.start <= self.params.base_axis[0]
            or filter_function.end >= self.params.base_axis[-1]):
            raise ValueError('filter passband (>5%) between {} - {} out of cube band {} - {}'.format(
                filter_function.start,
                filter_function.end,
                self.params.base_axis[0],
                self.params.base_axis[-1]))

        if xmin is None: xmin = 0
        if ymin is None: ymin = 0
        if xmax is None: xmax = self.dimx
        if ymax is None: ymax = self.dimy

        xmin = int(np.clip(xmin, 0, self.dimx))
        xmax = int(np.clip(xmax, 0, self.dimx))
        ymin = int(np.clip(ymin, 0, self.dimy))
        ymax = int(np.clip(ymax, 0, self.dimy))

        start_pix, end_pix = orb.utils.spectrum.cm12pix(
            self.params.base_axis, [filter_function.start, filter_function.end])
        start_pix = int(round(start_pix))
        end_pix = int(round(end_pix))
        sframe = np.zeros((self.dimx, self.dimy), dtype=float)
        zsize = end_pix-start_pix+1
        # This splits the range in zsize//10 +1 chunks (not necessarily of same
        # size). The endpix is correctly handled in the extraction
        mask = np.zeros((self.dimx, self.dimy), dtype=float)
        mask[xmin:xmax, ymin:ymax] = 1
        data = self._extract_spectra_from_region(np.nonzero(mask))
        data = data[..., start_pix:end_pix+1]
        sframe[xmin:xmax, ymin:ymax] = np.sum(data
                                            * filter_function(
                                            self.params.base_axis[start_pix:end_pix+1].astype(float)),
                                            axis=2)
        sframe /= np.sum(filter_function(self.params.base_axis.astype(float)))
        return sframe
