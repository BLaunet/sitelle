import numpy as np
from sitelle.region import centered_square_region
from sitelle.plot import *
from photutils import CircularAperture, CircularAnnulus, EllipticalAperture, DAOStarFinder, IRAFStarFinder, find_peaks, detect_sources
from scipy.interpolate import UnivariateSpline
from astropy.stats import sigma_clipped_stats
from orb.astrometry import Astrometry
from astropy.convolution import Gaussian2DKernel
from astropy.table import Table

def filter_sources(sources, annulus):
    x,y = sources['ycentroid'], sources['xcentroid']
    return sources[annulus[np.round(x).astype(int), np.round(y).astype(int)].astype(bool)]

def filter_frame(frame, annulus, val=0):
    _frame = np.copy(frame)
    _frame[np.where(annulus == 0)] = val
    return _frame

def extract_max_frame(x,y, spectral_cube, detection_pos_frame):
    iframe = int(detection_pos_frame[x,y])
    data = spectral_cube.get_data(x-10, x+11, y-10,y+11, iframe-2,iframe+3)
    return np.sum(data, axis=2)

def extract_point_source(xy, cube):
    big_box = centered_square_region(*xy, b=30)
    small_box = centered_square_region(*xy, b=3)
    mask = np.zeros((cube.dimx, cube.dimy))
    mask[big_box]=1
    mask[small_box]=0
    _, bkg_spec = cube.extract_integrated_spectrum(np.nonzero(mask), median=True, mean_flux=True, silent=True)
    a,s, n = cube.extract_integrated_spectrum(small_box, silent=True, return_spec_nb = True)
    return a, s-n*bkg_spec

def check_source(x,y, spectral_cube, detection_pos_frame=None):
    a,s = extract_point_source((x,y), spectral_cube)
    f = plot_spectra(a,s)
    if detection_pos_frame is not None:
        f,ax = plot_map(extract_max_frame(x,y, spectral_cube, detection_pos_frame))
        ax.scatter(10,10, marker='+', color='red')
        wl = 1e8/spectral_cube.params.base_axis[int(detection_pos_frame[x,y])]
        ax.set_title('Frame at %.1f Angstroms'%wl)

    else:
        f,ax = plot_map(spectral_cube.get_deep_frame()[x-10:x+11, y-10:y+11])
        ax.scatter(10,10, marker='+', color='red')

def measure_coherence(source, detection_pos_frame):
    y,x = np.round(source[['xcentroid', 'ycentroid']]).astype(int)
    return 1/np.nanstd(detection_pos_frame[x-1:x+2, y-1:y+2])

def measure_source_fwhm(detection, data, rmax=10):
    x,y = np.array(detection[['xcentroid', 'ycentroid']])
    photo_flux = np.zeros(rmax)
    for r in range(rmax):
        if r == 0:
            aper = CircularAperture((x,y), 1.)
        else:
            aper = CircularAnnulus((x,y), r, r+1)
        photo_flux[r] = aper.do_photometry(data)[0]/aper.area()


    def get_fwhm(flux):
        #We assume max is on 0. If not, source is probably contaminated
        flux = flux - flux.min()
        spline = UnivariateSpline(np.arange(rmax), flux-flux[0]/2., s=0)
        if spline.roots().shape != (1,):
            #print spline.roots()
            return np.nan
        return spline.roots()[0]
    return (get_fwhm(photo_flux))


def get_sources(detection_frame, mask=False, sigma = 5.0, mode='DAO', fwhm = 2.5, threshold = None):
    if mask is False:
        mask = np.ones_like(detection_frame)
    mean, median, std = sigma_clipped_stats(detection_frame, sigma=3.0, iters=5,
                                        mask=~mask.astype(bool) )#On masque la region hors de l'anneau
    #On detecte sur toute la frame, mais on garde que ce qui est effectivement dans l'anneau
    if threshold is None:
        threshold = median+sigma*std

    if mode == 'DAO':
        daofind = DAOStarFinder(fwhm=fwhm, threshold=threshold)
        sources = daofind(detection_frame)
    elif mode == 'IRAF':
        irafind = IRAFStarFinder(threshold=threshold, fwhm=fwhm)
        sources = irafind(detection_frame)
    elif mode == 'PEAK':
        sources = find_peaks(detection_frame, threshold=threshold )
        sources.rename_column('x_peak', 'xcentroid')
        sources.rename_column('y_peak', 'ycentroid')
    elif mode == 'ORB':
        astro = Astrometry(detection_frame, instrument='sitelle')
        path, fwhm_arc = astro.detect_stars(min_star_number=5000, r_max_coeff=1. )
        star_list = astro.load_star_list(path)
        sources = Table([star_list[:,0], star_list[:,1]], names=('ycentroid', 'xcentroid'))
    elif mode == 'SEGM':

        s = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        kernel = Gaussian2DKernel(s, x_size=3, y_size=3)
        kernel.normalize()

        segm = detect_sources(detection_frame, threshold, npixels=7,
                              filter_kernel=kernel)
        def get_xy(bbox):
            xslice, yslice = bbox
            x = xslice.start + (xslice.stop - xslice.start -1)/2.
            y = yslice.start + (yslice.stop - yslice.start -1)/2.
            return x,y
        def get_positions(segm):
            pos = np.zeros((len(segm.slices), 2))
            for i, box in enumerate(segm.slices):
                pos[i] = np.array(get_xy(box))
            X = pos[:,0]
            Y = pos[:,1]
            return Table([Y, X], names=('xcentroid', 'ycentroid'))

        sources = get_positions(segm)
    sources = filter_sources(sources, mask) # On filtre
    df = sources.to_pandas()
    if 'id' in df:
        df.pop('id')
    return df
