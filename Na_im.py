#!/usr/bin/python3
# https://matplotlib.org/examples/pylab_examples/contourf_demo.html
import numpy as np
from matplotlib import gridspec
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LogNorm
from skimage.measure import block_reduce

from precisionguide import get_HDUList
# Eventually I want to get propert C* WCS keywords into headers
from ReduceCorObs import plate_scale

origin = 'lower'
vmin = 10
vmax = 3000
binning = 4

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

fig = plt.figure(figsize=(6, 2.6))
gs = gridspec.GridSpec(1, 2, width_ratios=[1,1.24])

ax = plt.subplot(gs[0])
rect15 = patches.Rectangle((-7.5,-7.5),15,15,linewidth=1,edgecolor='C0',facecolor='none')
rect30 = patches.Rectangle((-15,-15),30,30,linewidth=1,edgecolor='C1',facecolor='none')
rect40 = patches.Rectangle((-20,-20),40,40,linewidth=1,edgecolor='C2',facecolor='none')
rect50 = patches.Rectangle((-25,-25),50,50,linewidth=1,edgecolor='C2',facecolor='none')

# These are less bright in background than 2018-03-02
#fname = '/data/io/IoIO/reduced/2018-02-27/IPT_Na_R_043r.fits'

fname = '/data/io/IoIO/reduced/2018-02-27/IPT_Na_R_003r.fits'


# Like this but maybe too much uniform background
#fname = '/data/io/IoIO/reduced/2018-03-02/IPT_Na_R_040r.fits'
#fname = '/data/io/IoIO/noCrashPlan/reduced.previous_versions/sent_to_coauthors/2018-02-27/IPT_Na_R_060r.fits'

# Go through more images on this date
# Way too bright
#fname = '/data/io/IoIO/reduced/2018-03-02/IPT_Na_R_003r.fits'
# Wow!  Beautiful extended structure, but brighht background
#fname = '/data/io/IoIO/reduced/2018-03-02/IPT_Na_R_020r.fits'
#fname = '/data/io/IoIO/reduced/2018-03-02/IPT_Na_R_023r.fits'
#fname = '/data/io/IoIO/reduced/2018-03-02/IPT_Na_R_020r.fits'
#fname = '/data/io/IoIO/reduced/2018-03-02/IPT_Na_R_040r.fits'
#fname = '/data/io/IoIO/reduced/2018-03-02/IPT_Na_R_056r.fits'
# Background getting brighter
#fname = '/data/io/IoIO/reduced/2018-03-02/IPT_Na_R_072r.fits'
#fname = '/data/io/IoIO/reduced/2018-03-02/IPT_Na_R_089r.fits'

plt.title('2018-03-02')
HDUList = get_HDUList(fname)
header = HDUList[0].header
im = HDUList[0].data
im = block_reduce(im, block_size=(binning, binning), func=np.median)
im = rebin(im, np.asarray(im.shape)/binning)
badc = np.where(im < 0)
im[badc] = 1
Rjpix = header['ANGDIAM']/2/plate_scale / binning**2 # arcsec / (arcsec/pix) / (pix/bin)
nr, nc = im.shape
x = (np.arange(nc) - nc/2) / Rjpix
y = (np.arange(nr) - nr/2) / Rjpix
X, Y = np.meshgrid(x, y)
#plt.pcolormesh(X, Y, im, norm=LogNorm(vmin=vmin, vmax=vmax), cmap='YlOrRd')
plt.pcolormesh(X, Y, im, norm=LogNorm(vmin=vmin, vmax=vmax), cmap='gist_heat')
ax.add_patch(rect15)
ax.add_patch(rect30)
ax.add_patch(rect40)
ax.add_patch(rect50)
plt.ylabel('Rj')
plt.xlabel('Rj')
# https://stackoverflow.com/questions/2934878/matplotlib-pyplot-preserve-aspect-ratio-of-the-plot
plt.axis('scaled')
#cbar = plt.colorbar()
#cbar.ax.set_ylabel('Surface brightness (approx. R)')


ax = plt.subplot(gs[1])
rect15 = patches.Rectangle((-7.5,-7.5),15,15,linewidth=1,edgecolor='C0',facecolor='none')
rect30 = patches.Rectangle((-15,-15),30,30,linewidth=1,edgecolor='C1',facecolor='none')
rect40 = patches.Rectangle((-20,-20),40,40,linewidth=1,edgecolor='C2',facecolor='none')
rect50 = patches.Rectangle((-25,-25),50,50,linewidth=1,edgecolor='C2',facecolor='none')

# Was using this one
fname = '/data/io/IoIO/reduced/2018-06-12/Na_on-band_005r.fits'

# Too bright
#fname = '/data/io/IoIO/reduced/2018-06-12/Na_on-band_001r.fits'
# Better
#fname = '/data/io/IoIO/reduced/2018-06-12/Na_on-band_002r.fits'

# This looks pretty good
fname = '/data/io/IoIO/reduced/2018-06-12/Na_on-band_003r.fits'

#fname = '/data/io/IoIO/reduced/2018-06-12/Na_on-band_004r.fits'
#fname = '/data/io/IoIO/reduced/2018-06-12/Na_on-band_005r.fits'
# Oversubtracted
#fname = '/data/io/IoIO/reduced/2018-06-12/Na_on-band_006r.fits'
#fname = '/data/io/IoIO/reduced/2018-06-12/Na_on-band_007r.fits'
# OK
#fname = '/data/io/IoIO/reduced/2018-06-12/Na_on-band_008r.fits'
# Ugly
#fname = '/data/io/IoIO/reduced/2018-06-12/Na_on-band_010r.fits'

#fname = '/data/io/IoIO/noCrashPlan/reduced.previous_versions/sent_to_coauthors/2018-06-12/Na_on-band_005r.fits'
plt.title('2018-06-12')
HDUList = get_HDUList(fname)
header = HDUList[0].header
im = HDUList[0].data
im = block_reduce(im, block_size=(binning, binning), func=np.median)
im = rebin(im, np.asarray(im.shape)/binning)
badc = np.where(im < 0)
im[badc] = 1
Rjpix = header['ANGDIAM']/2/plate_scale / binning**2 # arcsec / (arcsec/pix) / (pix/bin)
nr, nc = im.shape
x = (np.arange(nc) - nc/2) / Rjpix
y = (np.arange(nr) - nr/2) / Rjpix
X, Y = np.meshgrid(x, y)
#plt.pcolormesh(X, Y, im, norm=LogNorm(vmin=vmin, vmax=vmax), cmap='YlOrRd')
plt.pcolormesh(X, Y, im, norm=LogNorm(vmin=vmin, vmax=vmax), cmap='gist_heat')
ax.add_patch(rect15)
ax.add_patch(rect30)
ax.add_patch(rect40)
ax.add_patch(rect50)
plt.xlabel('Rj')
plt.axis('scaled')
cbar = plt.colorbar()
cbar.ax.set_ylabel('Surface brightness (R)')


#badc = np.where(np.logical_or(im < 10, im > chop))
#im[badc] = 0

# https://matplotlib.org/examples/pylab_examples/pcolor_log.html

#plt.pcolor(X, Y, im, norm=LogNorm(vmin=im.min(), vmax=im.max()), cmap='PuBu_r')
#plt.subplot(2, 1, 1)
#plt.pcolormesh(X, Y, im, norm=LogNorm(vmin=1, vmax=5000), cmap='hsv')
#plt.pcolormesh(X, Y, im, norm=LogNorm(vmin=1, vmax=5000), cmap='autumn')

#plt.subplot(2, 1, 2)
#plt.pcolor(X, Y, im, norm=LogNorm(vmin=0, vmax=8000), cmap='PuBu_r')
#plt.colorbar()

plt.show()


# # https://matplotlib.org/gallery/images_contours_and_fields/contourf_log.html#sphx-glr-gallery-images-contours-and-fields-contourf-log-py
# 
# # Automatic selection of levels works; setting the
# # log locator tells contourf to use a log scale:
# fig, ax = plt.subplots()
# 
# CS = ax.contourf(X, Y, im,
#                  locator=ticker.LogLocator(),
#                  cmap=plt.cm.viridis,
#                  origin=origin)
# 
# #CS = plt.contourf(X, Y, im, 10,
# #                  #[-1, -0.1, 0, 0.1],
# #                  #alpha=0.5,
# #                  cmap=plt.cm.viridis,
# #                  origin=origin)
# 
# cbar = plt.colorbar(CS)
# cbar.ax.set_ylabel('Surface brightness (approx. R)')

#plt.figure()
#plt.show()

#import numpy as np
#import matplotlib.pyplot as plt
#
#
#delta = 0.025
#
#x = y = np.arange(-3.0, 3.01, delta)
#X, Y = np.meshgrid(x, y)
#Z1 = plt.mlab.bivariate_normal(X, Y, 1.0, 1.0, 0.0, 0.0)
#Z2 = plt.mlab.bivariate_normal(X, Y, 1.5, 0.5, 1, 1)
#Z = 10 * (Z1 - Z2)
#
#nr, nc = Z.shape
#
## put NaNs in one corner:
#Z[-nr//6:, -nc//6:] = np.nan
## contourf will convert these to masked
#
#
#Z = np.ma.array(Z)
## mask another corner:
#Z[:nr//6, :nc//6] = np.ma.masked
#
## mask a circle in the middle:
#interior = np.sqrt((X**2) + (Y**2)) < 0.5
#Z[interior] = np.ma.masked
#
## We are using automatic selection of contour levels;
## this is usually not such a good idea, because they don't
## occur on nice boundaries, but we do it here for purposes
## of illustration.
#CS = plt.contourf(X, Y, Z, 10,
#                  #[-1, -0.1, 0, 0.1],
#                  #alpha=0.5,
#                  cmap=plt.cm.bone,
#                  origin=origin)
#
## Note that in the following, we explicitly pass in a subset of
## the contour levels used for the filled contours.  Alternatively,
## We could pass in additional levels to provide extra resolution,
## or leave out the levels kwarg to use all of the original levels.
#
#CS2 = plt.contour(CS, levels=CS.levels[::2],
#                  colors='r',
#                  origin=origin)
#
#plt.title('Nonsense (3 masked regions)')
#plt.xlabel('word length anomaly')
#plt.ylabel('sentence length anomaly')
#
## Make a colorbar for the ContourSet returned by the contourf call.
#cbar = plt.colorbar(CS)
#cbar.ax.set_ylabel('verbosity coefficient')
## Add the contour line levels to the colorbar
#cbar.add_lines(CS2)
#
#plt.figure()
#
## Now make a contour plot with the levels specified,
## and with the colormap generated automatically from a list
## of colors.
#levels = [-1.5, -1, -0.5, 0, 0.5, 1]
#CS3 = plt.contourf(X, Y, Z, levels,
#                   colors=('r', 'g', 'b'),
#                   origin=origin,
#                   extend='both')
## Our data range extends outside the range of levels; make
## data below the lowest contour level yellow, and above the
## highest level cyan:
#CS3.cmap.set_under('yellow')
#CS3.cmap.set_over('cyan')
#
#CS4 = plt.contour(X, Y, Z, levels,
#                  colors=('k',),
#                  linewidths=(3,),
#                  origin=origin)
#plt.title('Listed colors (3 masked regions)')
#plt.clabel(CS4, fmt='%2.1f', colors='w', fontsize=14)
#
## Notice that the colorbar command gets all the information it
## needs from the ContourSet object, CS3.
#plt.colorbar(CS3)
#
## Illustrate all 4 possible "extend" settings:
#extends = ["neither", "both", "min", "max"]
#cmap = plt.cm.get_cmap("winter")
#cmap.set_under("magenta")
#cmap.set_over("yellow")
## Note: contouring simply excludes masked or nan regions, so
## instead of using the "bad" colormap value for them, it draws
## nothing at all in them.  Therefore the following would have
## no effect:
## cmap.set_bad("red")
#
#fig, axs = plt.subplots(2, 2)
#fig.subplots_adjust(hspace=0.3)
#
#for ax, extend in zip(axs.ravel(), extends):
#    cs = ax.contourf(X, Y, Z, levels, cmap=cmap, extend=extend, origin=origin)
#    fig.colorbar(cs, ax=ax, shrink=0.9)
#    ax.set_title("extend = %s" % extend)
#    ax.locator_params(nbins=4)
#
#plt.show()
