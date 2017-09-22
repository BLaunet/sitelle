from __future__ import division
from orcs.process import SpectralCube
import numpy as np
import logging
from orb.utils import io
import argparse
import orb.core

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file",
                        help="file to process")
    parser.add_argument("-o", "--out_prefix",
                        help="Prefix for output path")
    parser.add_argument("-xmin", "--xmin",
                        help="Min index on the x axis, Default = 0",
                        default=0)
    parser.add_argument("-xmax", "--xmax",
                        help="Max index on the x axis, Default = 2048",
                        default=2048)
    parser.add_argument("-ymin", "--ymin",
                        help="Min index on the y axis, Default = 0",
                        default=0)
    parser.add_argument("-ymax", "--ymax",
                        help="Max index on the y axis, Default = 2064",
                        default=2064)

    return parser

def extract_unbinned_spectrum(cube, region,out_prefix ='.',
                                  subtract_spectrum=None,
                                  only_bandpass=False,
                                  silent=False,
                                  save=True,
                                  return_spectrum=False):
    """
    Extract spectra from given region of the cube.
    All spectra in the defined region are extracted pixel by pixel.

    This code adapts the code from the core function _extract_spectrum

    :param region: A list of the indices of the pixels integrated
      in the returned spectrum.

    :param subtract_spectrum: (Optional) Remove the given spectrum
      from the extracted spectrum before fitting
      parameters. Useful to remove sky spectrum. Both spectra must
      have the same size.

    :param only_bandpass: (Optional) Extract only the spectrum inside the
        bandpass of the filter (default False)

    :param silent: (Optional) If True, nothing is printed (default
      False).

    :return: an numpy.array with the same shape as 'region' containing the extracted spectra
    """
    def _interpolate_spectrum(spec, corr, wavenumber, step, order, base_axis):
        if wavenumber:
            corr_axis = orb.utils.spectrum.create_cm1_axis(
                spec.shape[0], step, order, corr=corr)
            return orb.utils.vector.interpolate_axis(
                spec, base_axis, 5, old_axis=corr_axis)
        else:
            corr_axis = orb.utils.spectrum.create_nm_axis(
                spec.shape[0], step, order, corr=corr)
            return orb.utils.vector.interpolate_axis(
                spec, base_axis, 5, old_axis=corr_axis)


    def _extract_spectrum_in_column(data_col, calib_coeff_col, mask_col,
                                    wavenumber, base_axis, step, order, new_axis):

        for icol in range(data_col.shape[0]):
            if mask_col[icol]:
                corr = calib_coeff_col[icol]
                data_col[icol, :] = _interpolate_spectrum(
                    data_col[icol, :], corr, wavenumber, step, order, base_axis)
            else:
                data_col[icol, :].fill(np.nan)

        data_col = np.apply_along_axis(_interpolate_on_new_axis, 1,
                                         data_col, base_axis=base_axis, new_axis=new_axis)
        return data_col

    def _interpolate_on_new_axis(spectrum, base_axis, new_axis):
        f = scipy.interpolate.UnivariateSpline(base_axis[~np.isnan(spectrum)],
                                               spectrum[~np.isnan(spectrum)],
                                               s=0, k=1, ext=1)
        #logging.info(str(new_axis.shape))
        return f(new_axis)

    calibration_coeff_map = cube.get_calibration_coeff_map()

    mask = np.zeros((cube.dimx, cube.dimy), dtype=np.uint8)
    mask[region] = 1

    base_axis = cube.params.base_axis.astype(float)
    irreg_axis = 1e8/base_axis
    reg_axis = np.linspace(irreg_axis[0], irreg_axis[-1], base_axis.shape[0] )
    #We flip it to make it wavelength ascending
    reg_axis = np.flip(reg_axis, 0)

    if only_bandpass:
        #We get extremas of the filter bandpass
        wl_min, wl_max = list(reversed([1e8/x for x in cube.params.filter_range]))
        #We conserve only spectrum inside this bandpass
        i_min = np.searchsorted(reg_axis, wl_min)
        i_max = np.searchsorted(reg_axis, wl_max)
        reg_axis = reg_axis[i_min:i_max]
    new_axis = 1e8/reg_axis #back in cm-1 to evaluate the spectrum function on it

    if not silent:
        logging.info('Number of integrated pixels: {}'.format(np.sum(mask)))

    if np.sum(mask) == 0: raise StandardError('A region must contain at least one valid pixel')

    elif np.sum(mask) == 1:
        ii = region[0][0] ; ij = region[1][0]
        spectrum = _interpolate_spectrum(
            cube.get_data(ii, ii+1, ij, ij+1, 0, cube.dimz, silent=silent),
            calibration_coeff_map[ii, ij],
            cube.params.wavenumber, cube.params.step, cube.params.order,
            cube.params.base_axis)
        counts = 1
        spec_arr = _interpolate_on_new_axis(spectrum, base_axis, new_axis.astype(float) )

    else:
        spectrum = np.zeros(cube.dimz, dtype=float)
        counts = 0

        # get range to check if a quadrants extraction is necessary
        mask_x_proj = np.nanmax(mask, axis=1).astype(float)
        mask_x_proj[np.nonzero(mask_x_proj == 0)] = np.nan
        mask_x_proj *= np.arange(cube.dimx)
        x_min = int(np.nanmin(mask_x_proj))
        x_max = int(np.nanmax(mask_x_proj)) + 1

        mask_y_proj = np.nanmax(mask, axis=0).astype(float)
        mask_y_proj[np.nonzero(mask_y_proj == 0)] = np.nan
        mask_y_proj *= np.arange(cube.dimy)
        y_min = int(np.nanmin(mask_y_proj))
        y_max = int(np.nanmax(mask_y_proj)) + 1
        y_0 = y_min

        spec_arr = np.zeros((x_max-x_min, y_max-y_min, len(new_axis)), dtype=float)

        if (x_max - x_min < cube.dimx / float(cube.config.DIV_NB)
            and y_max - y_min < cube.dimy / float(cube.config.DIV_NB)):
            quadrant_extraction = False
            QUAD_NB = 1
            DIV_NB = 1
        else:
            quadrant_extraction = True
            QUAD_NB = cube.config.QUAD_NB
            DIV_NB = cube.config.DIV_NB


        for iquad in range(0, QUAD_NB):

            if quadrant_extraction:
                # x_min, x_max, y_min, y_max are now used for quadrants boundaries
                x_min, x_max, y_min, y_max = cube.get_quadrant_dims(iquad)
            iquad_data = cube.get_data(x_min, x_max, y_min, y_max,
                                       0, cube.dimz, silent=silent)

            # multi-processing server init
            job_server, ncpus = cube._init_pp_server(silent=silent)
            if not silent: progress = orb.core.ProgressBar(x_max - x_min)
            for ii in range(0, x_max - x_min, ncpus):

                # no more jobs than columns
                if (ii + ncpus >= x_max - x_min):
                    ncpus = x_max - x_min - ii

                # jobs creation
                jobs = [(ijob, job_server.submit(
                    _extract_spectrum_in_column,
                    args=(iquad_data[ii+ijob,:,:],
                          calibration_coeff_map[x_min + ii + ijob,
                                                y_min:y_max],
                          mask[x_min + ii + ijob, y_min:y_max],
                          cube.params.wavenumber,
                          cube.params.base_axis, cube.params.step,
                          cube.params.order, new_axis.astype(float)),
                    modules=("import logging",
                             'import numpy as np',
                             'import orb.utils.spectrum',
                             'import orb.utils.vector',
                             'import scipy.interpolate'),
                    depfuncs=(_interpolate_spectrum,_interpolate_on_new_axis)))
                        for ijob in range(ncpus)]

                for ijob, job in jobs:
                    j = job()
                    #print(spec_arr[ii+ijob, (y_min-y_0):(y_max-y_0), :].shape)
                    spec_arr[ii+ijob,(y_min-y_0):(y_max-y_0), :] = j

                if not silent:
                    progress.update(ii, info="ext column : {}/{}".format(
                        ii, int(cube.dimx/float(DIV_NB))))
            cube._close_pp_server(job_server)
            if not silent: progress.end()

    h = gen_wavelength_header(cube, reg_axis)
    if save:
        ext=''
        if only_bandpass:
            ext='_bandpass'
        path = "./{}_{}_wl_grid{}".format(cube.params['object_name'],
                                    cube.params['filter_name'],
                                    ext)
        io.write_fits('%s.fits'%path, spec_arr, fits_header=h, overwrite=True)
    if return_spectrum:
        return reg_axis.astype(float), spec_arr

def gen_spectrum_header(cube, axis):
    h = cube.get_header()
    for k in cube.get_wcs_header():
        if k in h.keys() and k not in ['LONPOLE', 'LATPOLE', 'RADESYS', 'EQUINOX', 'MJD-OBS', 'DATE-OBS']:
            h.pop(k)
    h.pop('NAXIS2')

    h.pop('WAVTYPE')
    h.pop('CTYPE3')
    h.pop('CRVAL3')
    h.pop('CUNIT3')
    h.pop('CRPIX3')
    h.pop('CDELT3')
    h.pop('CROTA3')

    h['NAXIS'] = 1
    h['NAXIS1'] = len(axis)
    #h['NAXIS2'] = 2
    h['CRPIX1'] = 1
    h['CRVAL1'] = axis[0]
    h['CDELT1'] = axis[1]-axis[0]
    h['CUNIT1'] = 'Angstroms'
    h['CTYPE1'] = 'LINEAR'
    return h

def gen_wavelength_header(cube, axis):
    h = cube.get_header()
    h['NAXIS3'] = len(axis)
    h['CRPIX3'] = 1
    h['CRVAL3'] = axis[0]
    h['CDELT3'] = axis[1]-axis[0]
    h['CUNIT3'] = 'Angstroms'
    h['CTYPE3'] = 'LINEAR'
    return h

if __name__ == '__main__':

    parser = get_parser()
    args = parser.parse_args()
    path = args.file
    x_min, x_max, y_min, y_max = [args.xmin, args.xmax, args.ymin, args.ymax]
    out_prefix = args.out_prefix

    cube = SpectralCube(path)
    mask = np.zeros((cube.dimx, cube.dimy), dtype=bool)
    mask[int(x_min):int(x_max), int(y_min):int(y_max)] = True
    region = np.nonzero(mask)
    extract_unbinned_spectrum(cube, region, out_prefix=out_prefix, only_bandpass=True)
