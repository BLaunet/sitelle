#!/usr/bin/env python
# %run fitsview.py
# https://www.astro.rug.nl/software/kapteyn/maputilstutorial.html
#
import sys
import os
import numpy as np
import math   # This will import math module
import matplotlib
matplotlib.use('Qt4Agg')

import asciitable as ascii
import glob as glob
from PyPDF2 import PdfFileMerger, PdfFileReader


from astropy.table import Table
from astropy.cosmology import WMAP9 as cosmo
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatter
from matplotlib import pyplot as plt
import scipy.ndimage
import seaborn as sns
import pyfits
import pylab as plot
import lineid_plot

from astropy.utils.data import download_file
from astropy.io import fits
from kapteyn import wcsgrat, maputils, celestial
from matplotlib import pylab as plt

import pylab as plt
from astropy.table import Table
from astropy.cosmology import WMAP9 as cosmo
#from networkx import to_agraph,relabel_nodes,draw_networkx
from astropysics.coords import CoordinateSystem
import scipy.ndimage

import asciitable as ascii
import glob as glob
from PyPDF2 import PdfFileMerger, PdfFileReader

from astropy.coordinates import ICRS
from astropy import units as u
import astropy.coordinates as coord

#from __future__ import print_function, division
from PyAstronomy import pyasl
from astLib import astCalc
from astLib import astCoords
from astLib import astWCS
from PyWCSTools import wcs
from PyWCSTools import wcscon


import astropysics as ap
from astropysics.coords import ICRSCoordinates,GalacticCoordinates
from math import pi

from astropysics.coords import CoordinateSystem

import itertools


from kapteyn import maputils
from matplotlib import pyplot as plt
import matplotlib
#matplotlib.use('GTKAgg')
matplotlib.use('Agg')
#'GTKAgg', 'Cairo', 'GDK', 'GTK', 'Agg',

from astropysics.coords import CoordinateSystem
from astropy.coordinates import ICRS
from astropy import units as u
from astropy.io import fits
import astropysics as ap
from astropysics.coords import ICRSCoordinates,GalacticCoordinates
from math import pi

import sys
import os
import numpy as np

ooffx = np.zeros(9)
ooffy = np.zeros(9)

ooffx[1] = -172.1
ooffy[1] = 115.3

ooffx[2] = -202.5
ooffy[2] =  109.7

ooffx[3] =  133.4
ooffy[3] =  -90.3

ooffx[4] = 145.8
ooffy[4] = -90.3

ooffx[5] = 118.8
ooffy[5] = -114.3

for i in range(1, 6):
   name = 'li_M31extract'+str(i)
   print i, name
   dfits = fits.open(name+'.fits')
   d=dfits[0].data
   h=dfits[0].header
   dfits.close()
   wn_min = h['CRVAL3']
   wn_delt = h['CDELT3']
   nx = h['NAXIS1']   
   ny = h['NAXIS2']   
   nl = h['NAXIS3']
   cd = h['CDELT3']
   crv = h['CRVAL3']
   crp = h['CRPIX3']
   wn_tab = wn_min + np.arange(nl)*wn_delt
   new_data = np.zeros(nl)
   for iy in np.arange(ny):
      for ix in np.arange(nx):
         if np.sqrt((ix-h['CRPIX1'])*(ix-h['CRPIX1'])+(iy-h['CRPIX2'])*(iy-h['CRPIX2'])) * h['CDELT2'] *3600.  < 6.:
#         if np.sqrt((ix-h['CRPIX1'])*(ix-h['CRPIX1'])+(iy-h['CRPIX2'])*(iy-h['CRPIX2'])) * h['CDELT2'] * 3600.  < 1.:
#            print ix, iy, np.sqrt((ix-h['CRPIX1'])*(ix-h['CRPIX1'])+(iy-h['CRPIX2'])*(iy-h['CRPIX2'])) * h['CDELT2']
#            new_data[:] += d[:,iy,ix]*8.74/h['FLAMBDA']
            new_data[:] += d[:,iy,ix]/h['FLAMBDA']
   h['NAXIS'] = 1
   h['NAXIS1'] = nl
   h['CDELT1'] = cd
   h['CRPIX1'] = crp
   h['CRVAL1'] = crv
   h['CUNIT1'] = h['CUNIT3']
   h['CTYPE1'] = h['CTYPE3']
   del h['CRPIX2'] 
   del h['CRPIX3'] 
   del h['CRVAL2'] 
   del h['CRVAL3'] 
   del h['NAXIS2']
   del h['NAXIS3']
   dfits[0].data = new_data
   hdulist = fits.HDUList(dfits)
   of = 'spec_' + name + '.fits'
   hdulist.writeto(of, clobber=True)
   print('Writing {}'.format(of))

#plt.plot(xx,yy)
   plt.figure(figsize=(20,6))
#   fig.patch.set_facecolor('blue')

#f, axes = plt.subplots(2,2, sharex=True,figsize=(20,6))


#   line_wave = [4856.139, 4954.041, 5001.993]
#   line_label1 = ['Hb', 'OIII 4959A', 'OIII 5007A']
#
#   ak = lineid_plot.initial_annotate_kwargs()
#   ak
#   {'arrowprops': {'arrowstyle': '->', 'relpos': (0.5, 0.0)},
#    'horizontalalignment': 'center',
#    'rotation': 90,
#    'textcoords': 'data',
#    'verticalalignment': 'center',
#    'xycoords': 'data'}
#   ak['arrowprops']['arrowstyle'] = "->"
#   vmax = np.max(new_data)
#   vmin = np.min(new_data)
#   plt.ylim((vmin,vmax))
#   plt.xlim(6450.,6900.)
#   ax = fig.add_axes([0.1,0.06, 0.85, 0.35])
   plt.plot(wn_tab,new_data,drawstyle="steps")
   plt.xlim(6450.,6900.)
#   lineid_plot.plot_line_ids(wn_tab[0:408],new_data[0:408], line_wave, line_label1, ax=ax)
#  lineid_plot.plot_line_ids(wn_tab[0:408],new_data[0:408], line_wave, line_label1, max_iter=300, annotate_kwargs=ak)

#   Plt.ylim(0,12000.)
   plt.xlabel("Wavelength (A)")
   plt.ylabel("$\phi$ (8.74 x $ 10^{-17}$ erg/cm$^2$/s/A)")
   xmin,xmax = plt.xlim()
   ymin,ymax = plt.ylim()
   yval = ymax - (ymax-ymin)/20.
   val = np.sqrt(np.power(ooffx[i],2.)+np.power(ooffy[i],2.))
   Myval =  '%5.2f' % val
   MyText = "Offset: ("+str(ooffx[i])+"''"+","+str(ooffy[i])+"'') R="+str(Myval)+"''"
   plt.annotate(MyText, xy=(xmin, yval),  xycoords='data'  )
   MyText2 = "[SII] doublet"
   plt.annotate(MyText2, xy=(6710., 67.),  xycoords='data'  )
   MyText3 = "[NII] "
   plt.annotate(MyText3, xy=(6575., 67.),  xycoords='data'  )
   MyText3 = "Halpha"
   plt.annotate(MyText3, xy=(6550., 67.),  xycoords='data'  )
   os.system("rm -f "+name+'.pdf')
   toto = name+'.pdf'

   plt.savefig(toto, bbox_inches='tight')


