import numpy as np
from sitelle.region import centered_square_region
from sitelle.plot import *
from photutils import CircularAperture, CircularAnnulus
from scipy.interpolate import UnivariateSpline
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

def check_source(x,y, spectral_cube, detection_pos_frame):
    a,s = extract_point_source((x,y), spectral_cube)
    f = plot_spectra(a,s)
    f,ax = plot_map(extract_max_frame(x,y, spectral_cube, detection_pos_frame))
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
