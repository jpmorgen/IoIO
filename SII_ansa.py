#!/usr/bin/python3
# https://matplotlib.org/examples/pylab_examples/contourf_demo.html
import os

import numpy as np
from matplotlib import gridspec
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LogNorm
from scipy import optimize
from skimage.measure import block_reduce

from astropy.io import fits
# Eventually I want to get propert C* WCS keywords into headers
from ReduceCorObs import plate_scale

# --> 2D Gaussian is probably what I want
# https://scipy-cookbook.readthedocs.io/items/FittingData.html

def gaussian(height, center_x, center_y, width_x, width_y):
    """Returns a gaussian function with the given parameters"""
    width_x = float(width_x)
    width_y = float(width_y)
    return lambda x,y: height*np.exp(
                -(((center_x-x)/width_x)**2+((center_y-y)/width_y)**2)/2)

def moments(data, X, Y):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution by calculating its
    moments """
    total = data.sum()
    #X, Y = np.indices(data.shape)
    x = (X*data).sum()/total
    y = (Y*data).sum()/total
    col = data[:, int(x)]
    width_y = np.sqrt(np.abs((X[:,0]-x)**2*col).sum()/col.sum())
    row = data[int(y), :]
    width_x = np.sqrt(np.abs((Y[0,:]-y)**2*row).sum()/row.sum())
    height = data.max()
    return height, x, y, width_x, width_y

def fitgaussian(data, X, Y):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution found by a fit"""
    params = moments(data, X, Y)
    errorfunction = lambda p: np.ravel(gaussian(*p, X, Y)(*np.indices(data.shape)) -
                                 data)
    p, success = optimize.leastsq(errorfunction, params)
    return p

origin = 'lower'
vmin = 20
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
# nfs = 6
# SII_fnums = range(2,30,5)
# Na_fnums = range(2,8)
# --> fig, axes = plt.subplots(nrows=nfs, ncols=2, figsize=(10,6))
#nfs = 4
#SII_fnums = range(2,20,5)
#Na_fnums = range(2,6)
#fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(11,5))
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(11,8))
# --> fig.subplots_adjust(hspace=0, wspace=0)


SII_fnum = 30
SII_fnum = 7
SII_fnum = 8
# 13 -- 16 have bad calibration or something
#SII_fnum = 13
#SII_fnum = 16
#SII_fnum = 17
SII_fname = os.path.join(rdir, f'SII_on-band_{SII_fnum:03d}r.fits')
print(SII_fname)
with fits.open(SII_fname) as HDUList:
    # https://stackoverflow.com/questions/7066121/how-to-set-a-single-main-title-above-all-the-subplots-with-pyplot
    header = HDUList[0].header
    #fig.suptitle(header['DATE-OBS'].split('T')[0])
    im = HDUList[0].data
#im = im[center[0]-70:center[0]+70, center[1]-300:center[1]+300]
badc = np.where(im < 0)
im[badc] = 1
im[badc] = 1

center = (np.asarray(im.shape)/2).astype(int)
Rjpix = (header['ANGDIAM']/2/plate_scale
                      / (block_size*binning)) # arcsec / (arcsec/pix) / (pix/bi
nr, nc = im.shape
x = (np.arange(nc) - nc/2) / Rjpix
y = (np.arange(nr) - nr/2) / Rjpix
wansa_xidx = np.flatnonzero(np.logical_and(4.9 < x, x < 6.9))
wansa_yidx = np.flatnonzero(np.logical_and(-2 < y, y < 2))
eansa_xidx = np.flatnonzero(np.logical_and(-6.9 < x, x < -4.9))
eansa_yidx = wansa_yidx
wX, wY = np.meshgrid(x[wansa_xidx], y[wansa_yidx])
eX, eY = np.meshgrid(x[eansa_xidx], y[eansa_yidx])

# Tricky!  Index by slice, not index
wim = im[wansa_yidx[0]:wansa_yidx[-1]+1, wansa_xidx[0]:wansa_xidx[-1]+1]
eim = im[eansa_yidx[0]:eansa_yidx[-1]+1, eansa_xidx[0]:eansa_xidx[-1]+1]

#gparams = fitgaussian(im X, Y)
wgparams = moments(wim, wX, wY)
wbot = wgparams[2] - wgparams[4]
wtop = wgparams[2] + wgparams[4]
wbot_pix = np.int((wbot - (- 2))* Rjpix)
wtop_pix = np.int((wtop - (- 2))* Rjpix)
egparams = moments(eim, eX, eY)
ebot = egparams[2] - egparams[4]
etop = egparams[2] + egparams[4]
ebot_pix = np.int((ebot - (- 2))* Rjpix)
etop_pix = np.int((etop - (- 2))* Rjpix)


print(wbot, wtop, wbot_pix, wtop_pix)
print(ebot, etop, ebot_pix, etop_pix)
wyav = [np.average(wim[wbot_pix:wtop_pix, i]) for i in wansa_xidx-wansa_xidx[0]]
eyav = [np.average(eim[ebot_pix:etop_pix, i]) for i in eansa_xidx-eansa_xidx[0]]

#ax = fig.add_subplot(111, frameon=False)
#plotted = ax.pcolormesh(wX, wY,
#                        wim,
#                        norm=LogNorm(vmin=vmin, vmax=vmax),
#                        cmap='gist_heat')
#ax.axis('scaled')
#plt.show()

#ax = fig.add_subplot(111, frameon=False)
#plotted = ax.pcolormesh(wX[wbot_pix:wtop_pix,:], wY[wbot_pix:wtop_pix,:],
#                        im[wbot_pix:wtop_pix,:],
#                        norm=LogNorm(vmin=vmin, vmax=vmax),
#                        cmap='gist_heat')
#ax.axis('scaled')
#plt.show()

# ax = fig.add_subplot(111, frameon=False)
# plotted = ax.pcolormesh(eX, eY,
#                         eim,
#                         norm=LogNorm(vmin=vmin, vmax=vmax),
#                         cmap='gist_heat')
# ax.axis('scaled')
# plt.show()

# ax = fig.add_subplot(111, frameon=False)
# plotted = ax.pcolormesh(eX[ebot_pix:etop_pix,:], eY[ebot_pix:etop_pix,:],
#                         eim[ebot_pix:etop_pix,:],
#                         norm=LogNorm(vmin=vmin, vmax=vmax),
#                         cmap='gist_heat')
# ax.axis('scaled')
# plt.show()



# --> 2D Gaussian is probably what I want
# https://scipy-cookbook.readthedocs.io/items/FittingData.html


# --> Eventually I want to change the im Y idx to follow the ansa
# based on the xav profile
#wyav = [np.average(im[0:-1, i]) for i in wansa_xidx-wansa_xidx[0]]
plt.scatter(x[wansa_xidx], wyav, 100, 'b')
plt.scatter(-x[eansa_xidx], eyav, 100, 'r')
axes.tick_params(axis='both', which='major', labelsize=24)
axes.tick_params(axis='both', which='minor', labelsize=24)
plt.xlabel('Rj', fontsize=24)
plt.ylabel('Approx surface brightness (R)', fontsize=24)
plt.legend(['West ansa', 'East ansa'], loc='upper right', fontsize=24)
plt.axvline(x=5.52, ymin=0, ymax=1, linewidth=5, color='b')
plt.axvline(x=5.93, ymin=0, ymax=1, linewidth=5, color='r')
plt.show()

# 
# 
# 
# # https://scipy-cookbook.readthedocs.io/items/Rebinning.html
# def rebin( a, newshape ):
#     '''Rebin an array to a new shape.
#     '''
#     assert len(a.shape) == len(newshape)
# 
#     slices = [ slice(0,old, float(old)/new) for old,new in zip(a.shape,newshape) ]
#     coordinates = np.mgrid[slices]
#     indices = coordinates.astype('i')   #choose the biggest smaller integer index
#     return a[tuple(indices)]
# 
# def rebin_factor( a, newshape ):
#     '''Rebin an array to a new shape.
#     newshape must be a factor of a.shape.
#     '''
#     assert len(a.shape) == len(newshape)
#     assert not np.sometrue(np.mod( a.shape, newshape ))
# 
#     slices = [ slice(None,None, old/new) for old,new in zip(a.shape,newshape) ]
#     return a[slices]
# 
# SII_fnames = [os.path.join(rdir,
#                            f'SII_on-band_{i:03d}r.fits') for i in SII_fnums]
# Na_fnames = [os.path.join(rdir,
#                            f'Na_on-band_{i:03d}r.fits') for i in Na_fnums]
# 
# # https://stackoverflow.com/questions/6963035/pyplot-axes-labels-for-subplots
# ax = fig.add_subplot(111, frameon=False)
# # Turn off axis lines and ticks of the big subplot
# ax.spines['top'].set_color('none')
# ax.spines['bottom'].set_color('none')
# ax.spines['left'].set_color('none')
# ax.spines['right'].set_color('none')
# ax.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
# ax.grid(False)
# # Set common label
# #ax.set_xlabel('Rj')
# ax.set_ylabel('Rj')
# 
# #
# https://stackoverflow.com/questions/13784201/matplotlib-2-subplots-1-colorbar
# for icol, fnames in enumerate((Na_fnames, SII_fnames)): for fname,
# ax in zip(fnames, axes[:,icol]): with fits.open(fname) as HDUList: #
# https://stackoverflow.com/questions/7066121/how-to-set-a-single-main-title-above-all-the-subplots-with-pyplot
# header = HDUList[0].header
# fig.suptitle(header['DATE-OBS'].split('T')[0]) im = HDUList[0].data
# center = (np.asarray(im.shape)/2).astype(int) #im =
# im[center[0]-80:center[0]+80, center[1]-300:center[1]+300] im =
# im[center[0]-70:center[0]+70, center[1]-300:center[1]+300] im = im
# im = block_reduce(im, block_size=(block_size, block_size),
# func=np.median) im = rebin(im, np.asarray(im.shape)/binning) badc =
# np.where(im < 0) im[badc] = 1 Rjpix =
# (header['ANGDIAM']/2/plate_scale / (block_size*binning)) # arcsec /
# (arcsec/pix) / (pix/bin) nr, nc = im.shape x = (np.arange(nc) -
# nc/2) / Rjpix y = (np.arange(nr) - nr/2) / Rjpix X, Y =
# np.meshgrid(x, y) #plt.pcolormesh(X, Y, im, norm=LogNorm(vmin=vmin,
# vmax=vmax), cmap='YlOrRd') plotted = ax.pcolormesh(X, Y, im,
# norm=LogNorm(vmin=vmin[icol], vmax=vmax), cmap='gist_heat')
# #plt.pcolormesh(X, Y, im, vmin=vmin, vmax=vmax, cmap='gist_heat') #
# https://stackoverflow.com/questions/2934878/matplotlib-pyplot-preserve-aspect-ratio-of-the-plot
# ax.axis('scaled') ax.xaxis.set_visible(False) if icol == 0:
# ax.yaxis.set_visible(True) Na_plotted = plotted else:
# ax.yaxis.set_visible(False) SII_plotted = plotted
# ax.xaxis.set_visible(True) ax.set_xlabel('Rj')
# fig.subplots_adjust(left=0.15, right=0.85) cbar_ax =
# fig.add_axes([0.08, 0.10, 0.01, 0.8]) cbar =
# fig.colorbar(Na_plotted, cax=cbar_ax) cbar.ax.set_ylabel('Na Surface
# brightness (R)') cbar.ax.yaxis.set_ticks_position('left')
# cbar.ax.yaxis.set_label_position('left') cbar_ax =
# fig.add_axes([0.85, 0.10, 0.01, 0.8]) cbar =
# fig.colorbar(SII_plotted, cax=cbar_ax) cbar.ax.set_ylabel('[SII]
# Surface brightness (R)') plt.savefig('SII_Na_seq_transparent.png',
# transparent=True) plt.show()
