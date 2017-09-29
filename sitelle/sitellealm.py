#!/usr/bin/env python

import argparse
import numpy as np
from astropy.io import fits
from scipy.interpolate import interp1d
from glob import glob

import astropysics as ap
from astropysics.coords import ICRSCoordinates,GalacticCoordinates

__version__ = "sitelle processing tools, Ch. Morisset - IA-UNAM 2016, version 0.2"

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file",
                        help="file to process")
    parser.add_argument("-V", "--version", action="version", version=__version__,
                        help="Display version information and exit.")
    parser.add_argument("-U", "--usage",action="store_true",
                        help="Print out usage examples")
    parser.add_argument("-C", "--cut", action="store_true",
                        help="cut the file into Nx and Ny subfiles")
    parser.add_argument("-EC", "--extcube", action="store_true",
                        help="extract the useful an Nx x Ny data cube at the offset (Offx, Offy)")
    parser.add_argument("-Cr", "--cropcube", action="store_true",
                        help="extract the useful file from Nz1 to Nz2")
    parser.add_argument("-Nx", "--Nx", type=int,default=10,
                        help="Number of pieces in X, default=10")
    parser.add_argument("-Ny", "--Ny", type=int,default=10,
                        help="Number of pieces in Y, default=10")
    parser.add_argument("-Offx", "--Offx", type=float,default=-172.1,
                        help="Number of pieces in X, default=-172.1")
    parser.add_argument("-Offy", "--Offy", type=float,default=115.3,
                        help="Number of pieces in Y, default=115.3")
    parser.add_argument("-Nz1", "--Nz1", type=int,default=236,
                        help="Starting point in z, default=236")
    parser.add_argument("-Nz2", "--Nz2", type=int,default=644,
                        help="Ending point in z, default=644")
    parser.add_argument("-o", "--out_file",
                        help="output filename for the cut")
    parser.add_argument("-I", "--interpol_lambda", action="store_true",
                        help="Interpolate the cube in lambdas")
    parser.add_argument("-Npix", "--Npix", type=int,default=-1,
                        help="Number of pixels in lambda, default=-1 no rebin")
    return parser

#def crop_cube(cub_name, out_name, Nz1=236, Nz2=644):
def crop_cube(cub_name, out_name, Nz1=24, Nz2=75):
    dfits = fits.open(cub_name)
    d = dfits[0].data
    h = dfits[0].header
    dfits.close()

    nx = h['NAXIS1']
    ny = h['NAXIS2']
    nl = h['NAXIS3']

    x_inf = 0
    x_sup = nx-1
    #    x_inf = 500
    #    x_sup = 600

    y_inf = 0
    y_sup = ny-1
    #    y_inf = 500
    #    y_sup = 600
    #    h["CRPIX1"] = 1075
    #    h["CRPIX2"] = 1035
    #    h["CRVAL2"] = 0.4126898333333E+02

    print h["CRPIX1"],    h["CRPIX2"]
    print h["CRVAL1"],    h["CRVAL2"]
    print h["CDELT1"],    h["CDELT2"]
    ra  = h["CRVAL1"]
    dec = h["CRVAL2"]
    rao = ap.coords.AngularCoordinate(ra)
    deco = ap.coords.AngularCoordinate(dec)
    ras = str(rao.getHmsStr(sep=('h','m','s'),canonical = False))
    decs = str(deco.getDmsStr(sep=('d','m','s'),canonical = False))
    print  ras, decs, h["CRPIX1"],    h["CRPIX2"]

    #    myx = (x_sup - x_inf)/2.
    #    myy = (y_sup - y_inf)/2.
    #    ra = h["CRVAL1"]+h["CDELT1"]*((myx+1)-h["CRPIX1"])
    #    dec = h["CRVAL2"]+h["CDELT2"]*((myy+1)-h["CRPIX2"])
    #    h["CRPIX1"] = myx
    #    h["CRPIX2"] = myy
    #    h["CRVAL1"] = ra
    #    h["CRVAL2"] = dec
    #    print ra, dec
    #    rao = ap.coords.AngularCoordinate(ra)
    #    deco = ap.coords.AngularCoordinate(dec)
    #    ras = str(rao.getHmsStr(sep=('h','m','s'),canonical = False))
    #    decs = str(deco.getDmsStr(sep=('d','m','s'),canonical = False))
    #    print  ras, decs ,myx, myy

    nbo0  = h["CDELT3"]*(0-h["CRPIX3"])+h["CRVAL3"]
    nbo1  = h["CDELT3"]*(1-h["CRPIX3"])+h["CRVAL3"]
    nbo235  = h["CDELT3"]*(235-h["CRPIX3"])+h["CRVAL3"]
    nbo236  = h["CDELT3"]*(236-h["CRPIX3"])+h["CRVAL3"]
    nbo237  = h["CDELT3"]*(237-h["CRPIX3"])+h["CRVAL3"]
    print nbo0, nbo1, nbo235, nbo236, nbo237

    subd = d[Nz1:Nz2,y_inf:y_sup, x_inf:x_sup]
    dfits[0].data = subd
    h['NAXIS2'] = subd.shape[1]
    h['NAXIS1'] = subd.shape[2]
    h['NAXIS3'] = subd.shape[0]
    #    h['CRVAL3'] = nbo236
    hdulist = fits.HDUList(dfits)
    of = '{}.fits'.format(out_name)
    hdulist.writeto(of)
    print('Writing {}'.format(of))

def ext_cube(cub_name, out_name, Nx=100, Ny=100, Offx=-172.1, Offy=115.3):
    dfits = fits.open(cub_name)
    d = dfits[0].data
    h = dfits[0].header
    dfits.close()

    nx = h['NAXIS1']
    ny = h['NAXIS2']
    nl = h['NAXIS3']
    numx = (Offx)/3600./h["CDELT1"]+h["CRPIX1"]-1
    numy = (Offy)/3600./h["CDELT2"]+h["CRPIX2"]-1
    dec = h["CRVAL2"]+h["CDELT2"]*((numy+1)-h["CRPIX2"])
    ra = h["CRVAL1"]+(h["CDELT1"]*((numx+1)-h["CRPIX1"]))/np.cos(dec*np.pi/180.)
    rao = ap.coords.AngularCoordinate(ra)
    deco = ap.coords.AngularCoordinate(dec)
    ras = str(rao.getHmsStr(sep=('h','m','s'),canonical = False))
    decs = str(deco.getDmsStr(sep=('d','m','s'),canonical = False))
    print  ras, decs, Offx,Offy, numx, numy

    x_inf = 0
    x_sup = nx-1
    x_inf = int(numx-Nx/2.)
    x_sup = x_inf+Nx

    y_inf = 0
    y_sup = ny-1
    y_inf = int(numy-Ny/2)
    y_sup = y_inf+Ny
    print x_inf, x_sup, y_inf, y_sup, numx, numy


    # Center as new CRPIX1, CRPIX2
    myx = int((x_sup-x_inf)/2.)+x_inf
    myy = int((y_sup-y_inf)/2.)+y_inf
    dec = h["CRVAL2"]+h["CDELT2"]*((myy+1)-h["CRPIX2"])
    ra = h["CRVAL1"]+(h["CDELT1"]*((myx+1)-h["CRPIX1"]))/np.cos(dec*np.pi/180.)
    h["CRPIX1"] = Nx/2
    h["CRPIX2"] = Nx/2
    h["CRVAL1"] = ra
    h["CRVAL2"] = dec
    print ra, dec
    rao = ap.coords.AngularCoordinate(ra)
    deco = ap.coords.AngularCoordinate(dec)
    ras = str(rao.getHmsStr(sep=('h','m','s'),canonical = False))
    decs = str(deco.getDmsStr(sep=('d','m','s'),canonical = False))
    print  ras, decs ,myx, myy


    subd = d[0:nl,y_inf:y_sup, x_inf:x_sup]
    dfits[0].data = subd
    h['NAXIS2'] = subd.shape[1]
    h['NAXIS1'] = subd.shape[2]
    h['NAXIS3'] = subd.shape[0]
    hdulist = fits.HDUList(dfits)
    of = '{}.fits'.format(out_name)
    hdulist.writeto(of)
    print('Writing {}'.format(of))

def cut_cube(cub_name, out_name, Nx=10, Ny=10):
    dfits = fits.open(cub_name)
    d = dfits[0].data
    h = dfits[0].header
    dfits.close()

    nx = h['NAXIS1']
    ny = h['NAXIS2']
    nl = h['NAXIS3']
    crval1=h['CRVAL1']
    crval2=h['CRVAL2']
    crpix1=h['CRPIX1']
    crpix2=h['CRPIX2']
    cdelt1=h['CDELT1']
    cdelt2=h['CDELT2']

    sx = nx // Nx
    sy = ny // Ny

    x_inf = np.array([sx*i for i in  np.arange(Nx)])
    x_sup = np.array([sx*(i+1) for i in  np.arange(Nx)])
    x_sup[-1] = nx

    y_inf = np.array([sy*i for i in  np.arange(Ny)])
    y_sup = np.array([sy*(i+1) for i in  np.arange(Ny)])
    y_sup[-1] = ny

    for iy in np.arange(Ny):
        for ix in np.arange(Nx):
            subd = d[:,y_inf[iy]:y_sup[iy], x_inf[ix]:x_sup[ix]]
            dfits[0].data = subd
            h['NAXIS2'] = subd.shape[1]
            h['NAXIS1'] = subd.shape[2]
            # Center as new CRPIX1, CRPIX2
            myx = int((x_sup[ix]-x_inf[ix])/2.)+x_inf[ix]
            myy = int((y_sup[iy]-y_inf[iy])/2.)+y_inf[iy]
            dec = crval2+cdelt2*((myy+1)-crpix2)
            ra = crval1+(cdelt1*((myx+1)-crpix1))/np.cos(dec*np.pi/180.)
            print ra, dec, crval1, crval2,crpix1, crpix2, cdelt1, cdelt2
            rao = ap.coords.AngularCoordinate(ra)
            deco = ap.coords.AngularCoordinate(dec)
            ras = str(rao.getHmsStr(sep=('h','m','s'),canonical = False))
            decs = str(deco.getDmsStr(sep=('d','m','s'),canonical = False))
            print  ix, iy, ras, decs ,myx, myy
            h["CRPIX1"] = h['NAXIS1']/2.
            h["CRPIX2"] = h['NAXIS2']/2.
            h["CRVAL1"] = ra
            h["CRVAL2"] = dec

            hdulist = fits.HDUList(dfits)
            of = '{}_{}_{}.fits'.format(out_name, ix, iy)
            hdulist.writeto(of)
            print('Writing {}'.format(of))

def interpol_lambda(cub_name, Npix=-1):

    if "*" in cub_name:
        cub_names = glob(cub_name)
        for c_name in cub_names:
            interpol_lambda(c_name, Npix=Npix)
    else:
        dfits = fits.open(cub_name)
    d = dfits[0].data
    h = dfits[0].header
    dfits.close()
    wn_min = h['CRVAL3']
    wn_delt = h['CDELT3']
    nx = h['NAXIS1']
    ny = h['NAXIS2']
    nl = h['NAXIS3']
    if Npix == -1:
        Npix = nl
    wn_min = h['CRVAL3']
    wn_delt = h['CDELT3']
    wn_tab = wn_min + np.arange(nl)*wn_delt
    l_tab = 1e8 / wn_tab
    l_min = np.min(l_tab)
    l_delt = (np.max(l_tab) - np.min(l_tab)) / Npix
    l_tab2 = l_min + np.arange(Npix) * l_delt
    wn_tab2 = 1e8 / l_tab2
    new_data = np.zeros((Npix,ny, nx))
    for iy in np.arange(ny):
        for ix in np.arange(nx):
            interp = interp1d(wn_tab, d[:,iy, ix])
            new_data[:,iy, ix] = interp(wn_tab2)
    dfits[0].data = new_data.astype(np.float32)

    h['WAVTYPE'] = 'WAVELENGTH'
    h['CTYPE3'] = ('WAVE', 'wavelength in Angstrom')
    h['CRVAL3'] = (l_min, 'Minimum wavelength in Angstrom')
    h['CUNIT3'] = 'Angstrom'
    h['CDELT3'] = (l_delt, 'Angstrom per pixel')

    hdulist = fits.HDUList(dfits)
    of = 'li_' + cub_name
    hdulist.writeto(of, clobber=True)
    print('Writing {}'.format(of))
#log_file = open(cub_name+'.ok', 'w')
#log_file.write('Done')
#log_file.close()

if __name__ == '__main__':

    parser = get_parser()
    args = parser.parse_args()
    if args.usage:
        print("""
Usage examples:

To split the file M31SN3.fits into 10x10 subcubes, all with names starting with nM31:
sitellealm.py -C -f M31_SN3.fits -Nx=10 -Ny=10 -o cut/nM31 
sitellealm.py -C -f M31sn3.fits -Nx=30 -Ny=30 -o nM31 

To transform the file nM31_1_3.fits into li_nM31_1_3.fits where the 3rd axis is in wavelengths (Angstrom):
sitellealm.py -I -f nM31_0_1.fits
sitellealm.py -I -f M31extract1.fits
sitellealm.py -I -f M31extract2.fits
sitellealm.py -I -f M31extract3.fits
sitellealm.py -I -f M31extract4.fits
sitellealm.py -I -f M31extract5.fits

%run sitellealm.py -I -f M31extractNW.fits
%run sitellealm.py -I -f M31extractSE.fits
%run sitellealm.py -I -f M31extractSW.fits
%run sitellealm.py -I -f M31extractNE.fits

To run over a set of files:
sitellealm.py -I -f "cut/nM31_*_*.fits"

To crop the useful cube:
%run sitellealm.py -Cr -f M31_SN3.fits -Nz1=236 -Nz2=644 -o M31crop
%run sitellealm.py -Cr -f M31-Field1_SN2.merged.nm.1.0.fits -Nz1=480 -Nz2=514 -o M31SN2crop

To extract a piece of cube at a given offset:
%run sitellealm.py -EC -f M31crop.fits -Nx=100 -Ny=100 -Offx=-172.1 -Offy=115.3 -o M31extract1
%run sitellealm.py -EC -f M31crop.fits -Nx=100 -Ny=100 -Offx=-202.5 -Offy=109.7 -o M31extract2
%run sitellealm.py -EC -f M31crop.fits -Nx=100 -Ny=100 -Offx=133.4  -Offy=-90.3 -o M31extract3
%run sitellealm.py -EC -f M31crop.fits -Nx=100 -Ny=100 -Offx=145.8  -Offy=-90.3 -o M31extract4
%run sitellealm.py -EC -f M31crop.fits -Nx=100 -Ny=100 -Offx=-114.3 -Offy=79.0  -o M31extract5

%run sitellealm.py -EC -f M31crop.fits -Nx=800 -Ny=800 -Offx=-141. -Offy=110. -o M31extractNW
%run sitellealm.py -EC -f M31crop.fits -Nx=800 -Ny=800 -Offx=141.  -Offy=-110. -o M31extractSE
%run sitellealm.py -EC -f M31crop.fits -Nx=800 -Ny=800 -Offx=-141. -Offy=-110. -o M31extractSW
%run sitellealm.py -EC -f M31crop.fits -Nx=800 -Ny=800 -Offx=141.  -Offy=110. -o M31extractNE

""")

    elif args.cropcube:
        crop_cube(args.file, args.out_file, Nz1=args.Nz1, Nz2=args.Nz2)
    elif args.extcube:
        ext_cube(args.file, args.out_file, Nx=args.Nx, Ny=args.Ny,Offx=args.Offx,Offy=args.Offy)
    elif args.cut:
        cut_cube(args.file, args.out_file, Nx=args.Nx, Ny=args.Ny)
    elif args.interpol_lambda:
        interpol_lambda(args.file)
