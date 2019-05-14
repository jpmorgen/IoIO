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
vmin = 10
vmax = 3000
block_size = 2
binning = 2
linewidth = 2

#rdir = '/data/io/IoIO/reduced/2018-05-05/'
#fnums = range(3,47,5)
#fig, axes = plt.subplots(nrows=len(fnums), ncols=1, figsize=(5,12))
#
#rdir = '/data/io/IoIO/reduced/2018-05-14/'
#fnums = range(1,21,3)

rdir = '/data/io/IoIO/reduced/2018-06-06/'
fnums = range(2,40,5)
fig, axes = plt.subplots(nrows=len(fnums), ncols=1, figsize=(6,10))

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

fnames = [os.path.join(rdir,
                       f'SII_on-band_{i:03d}r.fits') for i in fnums]

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
ax.set_xlabel('Rj')
ax.set_ylabel('Rj')

# https://stackoverflow.com/questions/13784201/matplotlib-2-subplots-1-colorbar
for fname, ax in zip(fnames, axes):
    with fits.open(fname) as HDUList:
        # https://stackoverflow.com/questions/7066121/how-to-set-a-single-main-title-above-all-the-subplots-with-pyplot
        header = HDUList[0].header
        fig.suptitle(header['DATE-OBS'].split('T')[0])
        im = HDUList[0].data
        center = (np.asarray(im.shape)/2).astype(int)
        im = im[center[0]-80:center[0]+80, center[1]-300:center[1]+300]
        im = im
        im = block_reduce(im, block_size=(block_size, block_size), func=np.median)
        im = rebin(im, np.asarray(im.shape)/binning)
        badc = np.where(im < 0)
        im[badc] = 1
        Rjpix = header['ANGDIAM']/2/plate_scale / (block_size*binning) # arcsec / (arcsec/pix) / (pix/bin)
        nr, nc = im.shape
        x = (np.arange(nc) - nc/2) / Rjpix
        y = (np.arange(nr) - nr/2) / Rjpix
        X, Y = np.meshgrid(x, y)
        #plt.pcolormesh(X, Y, im, norm=LogNorm(vmin=vmin, vmax=vmax), cmap='YlOrRd')
        plotted = ax.pcolormesh(X, Y, im, norm=LogNorm(vmin=vmin, vmax=vmax), cmap='gist_heat')
        #plt.pcolormesh(X, Y, im, vmin=vmin, vmax=vmax, cmap='gist_heat')
        # https://stackoverflow.com/questions/2934878/matplotlib-pyplot-preserve-aspect-ratio-of-the-plot
        ax.axis('scaled')
        ax.xaxis.set_visible(False)
ax.xaxis.set_visible(True)
#plt.xlabel('Rj')
fig.subplots_adjust(right=0.85)
cbar_ax = fig.add_axes([0.8, 0.10, 0.05, 0.8])
cbar = fig.colorbar(plotted, cax=cbar_ax)
cbar.ax.set_ylabel('Surface brightness (R)')
plt.savefig('SII_seq_transparent.png', transparent=True)    
plt.show()
