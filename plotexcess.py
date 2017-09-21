from kapteyn import wcsgrat, maputils, celestial
from matplotlib import pylab as plt
import matplotlib.lines as mlines

import sys
import os
import pylab as plt
import numpy as np
import math   # This will import math module
from astropy.table import Table
from astropy.cosmology import WMAP9 as cosmo
#from networkx import to_agraph,relabel_nodes,draw_networkx
from astropysics.coords import CoordinateSystem
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatter
import scipy.ndimage

from astropy.coordinates import ICRS
from astropy import units as u

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

Nmax = 275
nmax = 346
nmax = 464
nmax = 675 # 10 sigma
nmax = 352 # 12 sigma
nmax = 691 # 12 sigma

Basefits = maputils.FITSimage('/Users/melchior/M31/data_misc/Herschel/ana/img.fits')
Rotfits = Basefits.reproject_to(rotation=0.0, crota2=0.0)

Secondfits = maputils.FITSimage('/Users/melchior/M31//data_misc/Mosaic/mosaic_SDSSz.fits')
Reprojfits2 = Secondfits.reproject_to(Rotfits)

Thirdfits = maputils.FITSimage('/Users/melchior/M31/data_misc/Bogdan2008/fig7_left_chandra_ratio.fits')
Reprojfits3 = Thirdfits.reproject_to(Rotfits)

Rotfits.set_imageaxes(axnr1=1,axnr2=2)
Reprojfits2.set_imageaxes(axnr1=1,axnr2=2)
#Rotfits.set_limits(pxlim=(830,870), pylim=(745,785))
#Reprojfits2.set_limits(pxlim=(830,870), pylim=(745,785))
Rotfits.set_limits(pxlim=(833,860), pylim=(747,774))
Reprojfits2.set_limits(pxlim=(833,860), pylim=(747,774))
Rotfits.set_skyout(celestial.fk5)
Reprojfits2.set_skyout(celestial.fk5)

#fig = plt.figure()
fig = plt.figure(figsize=(4,4))
fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
frame = fig.add_subplot(1,1,1)

clipmin, clipmax = 0.005, 3.4
baseim = Rotfits.Annotatedimage(frame, clipmin=clipmin, clipmax=clipmax, blankcolor='w',cmap="gist_yarg")
#binary

baseim.Image()
baseim.set_histogrameq()
baseim.Graticule()

cbar1=baseim.Colorbar(fontsize=8)
cbar1.set_label(label='160 $\mu$m emission',fontsize=12)

# IRAM-PdB
pos='0h42m44.37s 41d16m8.34s'
pos = pos.strip()
pos = pos.split()
print pos[0], pos[1]
mc = ICRSCoordinates(pos)
rapos = mc.ra.degrees
decpos = mc.dec.degrees


mytelx = np.zeros(4)
mytely = np.zeros(4)
mycoordsx = np.zeros(6)
mycoordsy = np.zeros(6)

#vel = np.zeros((274),dtype=float, order='F')
#offx = np.zeros((274),dtype=float, order='F')
#offy = np.zeros((274),dtype=float, order='F')
#mytelx = np.zeros(4)
#mytely = np.zeros(4)
#mycoordsx = np.zeros(274)
#mycoordsy = np.zeros(274)
vel = np.zeros((nmax),dtype=float, order='F')
offx = np.zeros((nmax),dtype=float, order='F')
offy = np.zeros((nmax),dtype=float, order='F')
mytelx = np.zeros(4)
mytely = np.zeros(4)
mycoordsx = np.zeros(nmax)
mycoordsy = np.zeros(nmax)

myoffx = [-6.8, 20.2, 3.2, -16.8]
myoffy = [10.5, 14.5, -3.5, -21.5]

mycentx = [-6.78]
mycenty = [10.5]

myoffx[:] = [x - mycentx[0] for x in myoffx]
myoffy[:] = [x - mycenty[0] for x in myoffy]

mytelx[:] = [x/3600./np.cos(decpos*np.pi/180.) + rapos for x in myoffx]
mytely[:] = [x/3600. + decpos for x in myoffy]

mypos    = list("")
for i in range(4):
    Xtt = ap.coords.AngularCoordinate(mytelx[i])
    Ytt = ap.coords.AngularCoordinate(mytely[i])
    ras  = str(Xtt.getHmsStr(sep=('h','m','s'),canonical = False))
    decs = str(Ytt.getDmsStr(sep=('d','m','s'),canonical = False))
    ttl = list("")
    ttl.append(ras)
    ttl.append(decs)
    mypos.append(ttl)    

#f = open('../outputs/excess.data','r')
#f = open('../scripts/excess.data','r')
f = open('../work_test1/excess.data','r')
os.system('wc ../work_test1/excess.data')
j = 0
for line in f.readlines():
    line    = line.strip()
    columns = line.split()
    offx[j] =  float(columns[3]) # the offsets are not inverted.
    offy[j] =  float(columns[4])
    vel[j]  =  float(columns[5])
    j       = j+1
f.close()

mycoordsx[:] = [x/3600./np.cos(decpos*np.pi/180.) + rapos for x in offx]
mycoordsy[:] = [x/3600. + decpos for x in offy]

#mycoordsx[:] = offx[:]/3600./np.cos(decpos*np.pi/180.) + rapos
#mycoordsy[:] = offy[:]/3600. + decpos

mypos2    = list("")
#for i in range(274):
for i in range(nmax):
    Xtt = ap.coords.AngularCoordinate(mycoordsx[i])
    Ytt = ap.coords.AngularCoordinate(mycoordsy[i])
    ras  = str(Xtt.getHmsStr(sep=('h','m','s'),canonical = False))
    decs = str(Ytt.getDmsStr(sep=('d','m','s'),canonical = False))
    ttl = list("")
    ttl.append(ras)
    ttl.append(decs)
    mypos2.append(ttl)    


mypos    = list("")
for i in range(4):
    Xtt = ap.coords.AngularCoordinate(mytelx[i])
    Ytt = ap.coords.AngularCoordinate(mytely[i])
    ras  = str(Xtt.getHmsStr(sep=('h','m','s'),canonical = False))
    decs = str(Ytt.getDmsStr(sep=('d','m','s'),canonical = False))
    ttl = list("")
    ttl.append(ras)
    ttl.append(decs)
    mypos.append(ttl)    



#ttl = pos
pos='0h42m44.37s 41d16m8.34s'
print pos,type(pos),len(pos)
overlayim2 = Rotfits.Annotatedimage(frame, clipmin=0, clipmax=4.e-6,boxdat=Reprojfits3.boxdat)
levels = np.arange(2.e-6,3.e-6,0.5e-6)
overlayim2.Marker(pos=pos, marker='+', markersize=40, color='b')


pos = '0h42m48.15s 41d15m25.s'
beam = overlayim2.Beam(0.0009361, 0.000677, pa=69.84, pos=pos, fc='black', fill=True, alpha=0.4)
mj = 0.0009361*3600.
mn = 0.000677*3600.

#overlayim2.Marker(pos=pos, marker='+', markersize=40, color='b')
#for i in range(274):
for i in range(nmax):
    xval = mypos2[i]
    val=str(xval).strip('[]')
    overlayim2.Marker(pos=val, marker='+', markersize=2, color='b')
    overlayim2.Skypolygon(prescription="ellipse", xc=mycoordsx[i], yc=mycoordsy[i], cpos=None, major=mj, minor=mn,  pa=69.84, units='arcsec',fill=True,color='pink')
#    overlayim2.Skypolygon(prescription="ellipse", xc=mytelx[i], yc=mytely[i], cpos=None, major=35.8, minor=35.8,  pa=0.0, units='arcsec',fill=False,color='r')

for i in range(4):
    xval = mypos[i]
    val=str(xval).strip('[]')
    overlayim2.Marker(pos=val, marker='+', markersize=20, color='r')
    overlayim2.Skypolygon(prescription="ellipse", xc=mytelx[i], yc=mytely[i], cpos=None, major=43.7, minor=43.7,  pa=0.0, units='arcsec',fill=False,color='r')


overlayim2.Skypolygon(prescription="ellipse", xc=rapos, yc=decpos, cpos=None, major=7.2, minor=7.2,  pa=0.0, units='arcsec',fill=False,color='b',linestyle="solid")
overlayim2.Skypolygon(prescription="ellipse", xc=rapos, yc=decpos, cpos=None, major=13.4, minor=13.4,  pa=0.0, units='arcsec',fill=False,color='b',linestyle="dashed")


overlayim = Rotfits.Annotatedimage(frame, clipmin=0, clipmax=4.e-6,boxdat=Reprojfits2.boxdat)
levels = np.logspace(1.,2.,6)

Line_colours = ('BlueViolet', 'Crimson', 'ForestGreen', 'Indigo', 'Tomato', 'Maroon')
overlayim.Contours(levels=levels,linestyles=('solid', 'solid','solid','solid','solid','solid'),   linewidths=(1,1,1,1,1,1),colors='black')



baseim.plot()
overlayim.plot()
overlayim2.plot()


baseim.interact_toolbarinfo()
baseim.interact_imagecolors()
baseim.interact_writepos()

print 'OK'

#plt.axis("equal")
#plt.xlim(-50,50)
#plt.ylim(-60,30)

#plt.gca().invert_xaxis()
#plt.grid()

#os.system('rm -f posexcess.pdf')
plt.savefig('../outputs/posexcess.pdf', bbox_inches='tight')
#plt.show()
plt.close()


