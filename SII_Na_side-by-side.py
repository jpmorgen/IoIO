#!/usr/bin/python3
# https://matplotlib.org/examples/pylab_examples/contourf_demo.html
import os

import numpy as np
from matplotlib import gridspec
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LogNorm
from skimage.measure import block_reduce

from astropy.io import fits
# Eventually I want to get propert C* WCS keywords into headers
from ReduceCorObs import plate_scale

origin = 'lower'
vmin = (300, 20)
vmax = 3000
# Figure is way too big when saved by pyplot  as eps.  Try to make it smaller
# --> Best way to do this is to save as png and then let 'convert' do
# the conversion to eps.  Note that block_size=2 causes lines in pdf output
block_size = 1
binning = 1
linewidth = 2

#rdir = '/data/io/IoIO/reduced/2018-05-05/'
#fnums = range(3,47,5)
#fig, axes = plt.subplots(nrows=len(fnums), ncols=1, figsize=(5,12))
#
#rdir = '/data/io/IoIO/reduced/2018-05-14/'
#fnums = range(1,21,3)

rdir = '/data/io/IoIO/reduced/2018-06-06/'
#nfs = 8
#SII_fnums = range(2,40,5)
#Na_fnums = range(2,10)
#fig, axes = plt.subplots(nrows=nfs, ncols=2, figsize=(11,10))
nfs = 6
SII_fnums = range(2,30,5)
Na_fnums = range(2,8)
fig, axes = plt.subplots(nrows=nfs, ncols=2, figsize=(10,6))
#nfs = 4
#SII_fnums = range(2,20,5)
#Na_fnums = range(2,6)
#fig, axes = plt.subplots(nrows=nfs, ncols=2, figsize=(11,5))
fig.subplots_adjust(hspace=0, wspace=0)

# https://scipy-cookbook.readthedocs.io/items/Rebinning.html
def rebin( a, newshape ):
    '''Rebin an array to a new shape.
    '''
    assert len(a.shape) == len(newshape)

    slices = [ slice(0,old, float(old)/new) for old,new in zip(a.shape,newshape) ]
    coordinates = np.mgrid[slices]
    indices = coordinates.astype('i')   #choose the biggest smaller integer index
    return a[tuple(indices)]

def rebin_factor( a, newshape ):
    '''Rebin an array to a new shape.
    newshape must be a factor of a.shape.
    '''
    assert len(a.shape) == len(newshape)
    assert not np.sometrue(np.mod( a.shape, newshape ))

    slices = [ slice(None,None, old/new) for old,new in zip(a.shape,newshape) ]
    return a[slices]

SII_fnames = [os.path.join(rdir,
                           f'SII_on-band_{i:03d}r.fits') for i in SII_fnums]
Na_fnames = [os.path.join(rdir,
                           f'Na_on-band_{i:03d}r.fits') for i in Na_fnums]

# https://stackoverflow.com/questions/6963035/pyplot-axes-labels-for-subplots
ax = fig.add_subplot(111, frameon=False)
# Turn off axis lines and ticks of the big subplot
ax.spines['top'].set_color('none')
ax.spines['bottom'].set_color('none')
ax.spines['left'].set_color('none')
ax.spines['right'].set_color('none')
ax.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
ax.grid(False)
# Set common label
#ax.set_xlabel('Rj')
ax.set_ylabel('Rj')

# https://stackoverflow.com/questions/13784201/matplotlib-2-subplots-1-colorbar
for icol, fnames in enumerate((Na_fnames,  SII_fnames)):
    for fname, ax in zip(fnames, axes[:,icol]):
        with fits.open(fname) as HDUList:
            # https://stackoverflow.com/questions/7066121/how-to-set-a-single-main-title-above-all-the-subplots-with-pyplot
            header = HDUList[0].header
            fig.suptitle(header['DATE-OBS'].split('T')[0])
            im = HDUList[0].data
            center = (np.asarray(im.shape)/2).astype(int)
            #im = im[center[0]-80:center[0]+80, center[1]-300:center[1]+300]
            im = im[center[0]-70:center[0]+70, center[1]-300:center[1]+300]
            im = im
            im = block_reduce(im,
                              block_size=(block_size, block_size),
                              func=np.median)
            im = rebin(im, np.asarray(im.shape)/binning)
            badc = np.where(im < 0)
            im[badc] = 1
            Rjpix = (header['ANGDIAM']/2/plate_scale
                     / (block_size*binning)) # arcsec / (arcsec/pix) / (pix/bin)
            nr, nc = im.shape
            x = (np.arange(nc) - nc/2) / Rjpix
            y = (np.arange(nr) - nr/2) / Rjpix
            X, Y = np.meshgrid(x, y)
            #plt.pcolormesh(X, Y, im, norm=LogNorm(vmin=vmin, vmax=vmax), cmap='YlOrRd')
            plotted = ax.pcolormesh(X, Y, im, norm=LogNorm(vmin=vmin[icol], vmax=vmax), cmap='gist_heat')
            #plt.pcolormesh(X, Y, im, vmin=vmin, vmax=vmax, cmap='gist_heat')
            # https://stackoverflow.com/questions/2934878/matplotlib-pyplot-preserve-aspect-ratio-of-the-plot
            ax.axis('scaled')
            ax.xaxis.set_visible(False)
            if icol == 0:
                ax.yaxis.set_visible(True)
                Na_plotted = plotted
            else:
                ax.yaxis.set_visible(False)
                SII_plotted = plotted
    ax.xaxis.set_visible(True)
    ax.set_xlabel('Rj')
fig.subplots_adjust(left=0.15, right=0.85)
cbar_ax = fig.add_axes([0.08, 0.10, 0.01, 0.8])
cbar = fig.colorbar(Na_plotted, cax=cbar_ax)
cbar.ax.set_ylabel('Na Surface brightness (R)')
cbar.ax.yaxis.set_ticks_position('left')
cbar.ax.yaxis.set_label_position('left')
cbar_ax = fig.add_axes([0.85, 0.10, 0.01, 0.8])
cbar = fig.colorbar(SII_plotted, cax=cbar_ax)
cbar.ax.set_ylabel('[SII] Surface brightness (R)')
plt.savefig('SII_Na_seq_transparent.png', transparent=True)    
plt.show()
