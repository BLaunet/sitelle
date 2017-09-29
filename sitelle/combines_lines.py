from scipy.interpolate import interp1d
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import sys
import os


Nx = 30
Ny = 30

MyOff=np.ndarray(shape=(7),dtype='a17')
MyOff[1] = "-172.1'', 115.3''"
MyOff[2] = "-202.5'', 109.7''"
MyOff[3] = " 133.4'', -90.3''"
MyOff[4] = " 145.8'', -90.3''"
MyOff[5] = " 118.8'',-114.3''"

vstack=np.arange(-600.,0.,20)
yall = np.zeros(shape=(6,30))
ystack = np.zeros(30)
ystack2 = np.zeros(30)
for num in np.arange(1,6):
    f, axes = plt.subplots(7,1, sharex=True,figsize=(15,20))
    name = 'spec_li_M31extract'+str(num)
    dfits = fits.open(name+'.fits')
    d = dfits[0].data    
    h = dfits[0].header
    nl = h['NAXIS1']   
    dfits.close()
    lam = h['CRVAL1'] + h['CDELT1']*np.arange(d.shape[0])
    ld = np.zeros(6)
    mask=np.ndarray(shape=(6,nl),dtype='bool')
    MyText=np.ndarray(shape=(7),dtype='a17')
#[NII]
    mask[1] = (lam < 6548.03) & (lam > 6534.93)
    mask[1] = (lam < 6550.) & (lam > 6532.)
    ld[1] = 6548.03 
#Halpha
    mask[2] = (lam < 6562.8) & (lam > 6549.67)
    mask[2] = (lam < 6564.) & (lam > 6547.)
    ld[2] = 6562.8
#[NII]
    mask[3] = (lam < 6583.51) & (lam > 6570.24)
    mask[3] = (lam < 6585.) & (lam > 6568.)
    ld[3] = 6583.41
#[SII]
    mask[4] = (lam < 6716.47) & (lam > 6703.04)
    mask[4] = (lam < 6718.) & (lam > 6700)
    ld[4] = 6716.47
#[SII]
    mask[5] = (lam < 6730.85) & (lam > 6717.39)
    mask[5] = (lam < 6732.) & (lam > 6715)
    ld[5] =  6730.85
#
    MyText[1] = "[NII] 6548.03A"
    MyText[2] = "Halpha 6562.80A"
    MyText[3] = "[NII] 6583.41A"
    MyText[4] = "[SII] 6716.47A"
    MyText[5] = "[SII] 6730.85A"
    MyText[6] = "CO(2-1) 1.3mm"
#
    for ix in np.arange(1,6):
        ax = axes[ix-1]
        vel = (lam-ld[ix])/ld[ix]*3.e5
    #    ax.plot(lam[mask[ix]]-ld[ix], (d)[mask[ix]],drawstyle="steps")
        ax.plot(vel[mask[ix]], (d)[mask[ix]],drawstyle="steps",color='k')
        interp = interp1d(vel[mask[ix]], d[mask[ix]])
        yall[ix,:] = interp(vstack)
        ax.set_xlim(-600.,0.)
        ax.relim()
        ymin,ymax = ax.get_ylim()
        vmin = np.zeros(len(d[mask[ix]]))
        vmin.fill(ymin)
        vmax = np.zeros(len(d[mask[ix]]))
        vmax.fill(ymax)
        x  = np.arange(-600.,0,10)
        yi = np.zeros(60)
        ya = np.zeros(60)
        yi.fill(ymin)
        ya.fill(ymax*1.05)
        ax.fill_between(x, yi, ya, where=(x <= -300.), facecolor='blue', alpha=0.5)
        ax.fill_between(x, yi, ya, where=(x >= -300.), facecolor='red', alpha=0.5,interpolate=True)
        ax.yaxis.labelpad = 3
        ax.tick_params(axis='x', which='major', pad=8)
#        start, end = ax.get_ylim()
#        stepsize = (end-start)/4.
#        ax.yaxis.set_ticks(np.arange(start, end, stepsize))
        ax.locator_params(axis='y',nbins=4)
        
#for ax, where in zip(ax_list, ['pre', 'post', 'mid']):
#    ax.step(x, y, where=where, color='r', zorder=5, lw=5)
#    fill_between_steps(ax, x, y, 0, step_where=where)

#    ax.xaxis.set_ticks_position("none")
#    ax.yaxis.set_ticks_position("none")
#    ax.xaxis.set_ticklabels([])
#    ax.yaxis.set_ticklabels([])
        print('{}'.format(ix))
        plt.xlabel("Velocity (km/s)")
    #    plt.ylabel("$\phi$ (8.74 $\times 10^{-17}$ erg/cm$^2$/s/A)",'right')
        if ix==3:
            ax.set_ylabel(r'$\phi$ (8.74 $\times 10^{-17}$ erg/cm$^2$/s/A)', size =16)
            ax.get_yaxis().set_label_coords(-0.1,0.5)
            
        ax.set_ylim(ymin,1.05*ymax)
        xmin,xmax = ax.get_xlim()
        yval = ymax - (ymax-ymin)/5.
        ax.annotate(MyText[ix], xy=(-580, yval),  xycoords='data'  )

        if ix==1: 
            ax.annotate(MyOff[num], xy=(-150, yval),  xycoords='data'  )
        
        f.subplots_adjust(hspace=0.2)
        f.subplots_adjust(wspace=0.)
 
# Stacking
    for ix in np.arange(1,6):
        ystack[:] += yall[ix,:]
        if ix !=1:
            ystack2[:] += yall[ix,:]
    ystack = ystack/np.mean(ystack)
    ystack2 = ystack2/np.mean(ystack2)
    ax = axes[5]
    #    ax.plot(lam[mask[ix]]-ld[ix], (d)[mask[ix]],drawstyle="steps")
    ax.plot(vstack, ystack2,drawstyle="steps",color='k')
    ax.plot(vstack, ystack,drawstyle="steps",color='b')

    ax.relim()
    ymin,ymax = ax.get_ylim()
    vmin = np.zeros(len(d[mask[ix]]))
    vmin.fill(ymin)
    vmax = np.zeros(len(d[mask[ix]]))
    vmax.fill(ymax)
    x  = np.arange(-600.,0,10)
    yi = np.zeros(60)
    ya = np.zeros(60)
    yi.fill(ymin)
    ya.fill(ymax*1.05)
    ax.fill_between(x, yi, ya, where=(x <= -300.), facecolor='blue', alpha=0.5)
    ax.fill_between(x, yi, ya, where=(x >= -300.), facecolor='red', alpha=0.5,interpolate=True)
    ax.yaxis.labelpad = 3
    ax.tick_params(axis='x', which='major', pad=8)
    ax.locator_params(axis='y',nbins=4)
    ax.set_ylim(ymin,1.05*ymax)
    xmin,xmax = ax.get_xlim()
    yval = ymax - (ymax-ymin)/3.
    yval2 = ymax - (ymax-ymin)/6.
    ax.annotate('Stack 5 lines', xy=(-580, yval),  xycoords='data'  )
    ax.annotate('Stack 4 lines', xy=(-580, yval2),  xycoords='data',color='blue' )
    ax.set_ylabel(r'Arbitrary norm', size =16)
    ax.get_yaxis().set_label_coords(-0.1,0.5)
    f.subplots_adjust(hspace=0.2)
    f.subplots_adjust(wspace=0.)
    

# CO(2-1) data
    g = open('/users/melchior/M31/211-11/spec'+str(num)+'_co21.dat')
    i= 0
    if num==1:
        atab = np.zeros(512)
        btab = np.zeros(512)
    else:
        atab = np.zeros(465)
        btab = np.zeros(465)
        
    for line in g.readlines(): 
        line = line.strip()
        columns = line.split()
        yy = float(columns[1])
        xx = float(columns[0])
        atab[i] = xx
        btab[i] = yy
        i = i +1      
    ax = axes[6]
    ax.plot(atab, btab,drawstyle="steps",color='k')
    ax.set_xlim(-600.,0.)
    if num == 1:
        ax.set_ylim(-0.15,0.4)
    if num == 2:
        ax.set_ylim(-0.1,0.45)
    if num == 3:
        ax.set_ylim(-0.075,0.15)
    if num == 4:
        ax.set_ylim(-0.05,0.2)
    if num == 5:
        ax.set_ylim(-0.05,0.12)
    start, end = ax.get_ylim()
    stepsize = (end-start)/4.
#   ax.yaxis.set_ticks(np.arange(start, end, stepsize))
    ax.locator_params(axis='y',nbins=4)
    ax.set_ylabel(r'T$_{mb}$ (K)', size =16)
    ax.get_yaxis().set_label_coords(-0.1,0.5)
    ymin,ymax = ax.get_ylim()
    ax.set_ylim(ymin,1.05*ymax)
    yval = ymax - (ymax-ymin)/5.
    ax.annotate(MyText[6], xy=(-580, yval),  xycoords='data'  )
     
    f.subplots_adjust(hspace=0.2)
    f.subplots_adjust(wspace=0.)

#    yy, locs = plt.yticks()
#    ll = ['%.1f' % a for a in yy]
#    plt.yticks(yy, ll)


    os.system("rm -f "+name+'vel.png')
    toto = name+'vel.png'
    plt.savefig(toto, bbox_inches='tight')

