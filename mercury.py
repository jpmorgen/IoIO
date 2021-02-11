import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import MultipleLocator

from astropy import units as u
from astropy.convolution import Gaussian2DKernel, convolve
from astropy.stats import gaussian_fwhm_to_sigma, sigma_clipped_stats

from photutils import make_source_mask
#from photutils import detect_threshold
from photutils import detect_sources
from photutils import deblend_sources
from photutils import source_properties
from photutils.background import Background2D

from ccdmultipipe import ccddata_read
from cormultipipe import (CorMultiPipe, Calibration, 
                          nd_filter_mask, mask_nonlin_sat, detflux)

from IoIO import CorObsData
from ReduceCorObs import plate_scale

c = Calibration(start_date='2020-07-07', stop_date='2020-08-22', reduce=True)
cmp = CorMultiPipe(auto=True, calibration=c,
                   post_process_list=[detflux])
fname1 = '/data/Mercury/raw/2020-05-27/Mercury-0005_Na-on.fit'
fname2 = '/data/Mercury/raw/2020-05-27/Mercury-0005_Na_off.fit'
pout = cmp.pipeline([fname1, fname2], outdir='/data/Mercury/analysis/2020-05-27/', overwrite=True)

####
on = ccddata_read('/data/Mercury/analysis/2020-05-27/Mercury-0005_Na-on_r.fit')
off = ccddata_read('/data/Mercury/analysis/2020-05-27/Mercury-0005_Na_off_r.fit')

#nd_edge_expand = 40
#obs_data = CorObsData(off.to_hdu(), edge_mask=-nd_edge_expand)
# Put cleaner-looking off-band Mercury ND filter image into on-band image
obs_data = CorObsData(fname2)
on.data[obs_data.ND_coords] = off.data[obs_data.ND_coords]
off.data[obs_data.ND_coords] = 0
#mcenter = obs_data.obj_center
# Not working automatically. 
mcenter = np.asarray((1100, 1294))

# from IoIO_reduction.notebk Mon Jan 04 08:49:42 2021 EST  jpmorgen@snipe
off_on_ratio = 4.941459261792191
#pff_on_ratio = 4.16666
#off_on_ratio = 3
#off_on_ratio = 6
off_scaled = off.divide(off_on_ratio, handle_meta='first_found')
off_sub = on.subtract(off_scaled, handle_meta='first_found')

sigma = 10.0 * gaussian_fwhm_to_sigma # FWHM = 10
kernel = Gaussian2DKernel(sigma)
kernel.normalize()
# Make a source mask to enable optimal background estimation
mask = make_source_mask(off_sub.data, nsigma=2, npixels=5,
                        filter_kernel=kernel, mask=off_sub.mask,
                        dilate_size=11)
#impl = plt.imshow(mask, origin='lower',
#                  cmap=plt.cm.gray,
#                  filternorm=0, interpolation='none')
#plt.show()

##mean, median, std = sigma_clipped_stats(data, sigma=3.0, mask=mask)
#

box_size = int(np.mean(off_sub.shape) / 10)
back = Background2D(off_sub, box_size, mask=mask, coverage_mask=off_sub.mask)
threshold = back.background + (2.0* back.background_rms)

print(f'background_median = {back.background_median}, background_rms_median = {back.background_rms_median}')

#impl = plt.imshow(back.background, origin='lower',
#                  cmap=plt.cm.gray,
#                  filternorm=0, interpolation='none')
#back.plot_meshes()
#plt.show()

###
bsub_no_rot = off_sub.subtract(back.background*u.electron/u.s,
                               handle_meta='first_found')

# Rayleigh calibration as per Mercury.notebk
# Fri Jan 29 12:37:23 2021 EST  jpmorgen@snipe
F = 74.14 * 2.512*10**(3.01-6.473) *u.photon*u.cm**-2*u.A**-1*u.s**-1
FWHM = 12*u.A

Ff = F*FWHM
theta = (plate_scale * u.arcsec)
I = 4*np.pi*Ff / (theta**2)
I = I.to(u.R)
S = 1786 * u.electron/u.s

eps_to_R = I / S

bsub_no_rot = bsub_no_rot.multiply(eps_to_R, handle_meta='first_found')

bsub_no_rot.write('/data/Mercury/analysis/2020-05-27/Mercury-0005_Na-bsub.fit', overwrite=True)

bsub = bsub_no_rot.copy()

f = plt.figure(figsize=[8.5, 3.5])
date_obs = bsub.meta['DATE-OBS']
just_date, _ = date_obs.split('T')

plt.title(f'IoIO observatory NaD lines {just_date}')

s = np.asarray(bsub.shape)
center = s/2
to_shift = center - mcenter
bsub.data = ndimage.interpolation.shift(bsub.data, to_shift)
bsub.mask = ndimage.interpolation.shift(bsub.mask, to_shift)
bsub.uncertainty.array = ndimage.interpolation.shift(bsub.uncertainty.array, to_shift)
psang = 83
bsub.data = ndimage.interpolation.rotate(bsub.data, 90 - psang)
bsub.mask = ndimage.interpolation.rotate(bsub.mask, 90 - psang)
bsub.uncertainty.array = ndimage.interpolation.rotate(bsub.uncertainty.array, 90 - psang)

block_size = 1
binning = 1
#vmin = 0.01
#vmax = 1.3
vmin = 100
vmax = 10000
ang_dia = 6.8 # arcsec
Rmpix = ang_dia/plate_scale / (block_size*binning) # arcsec / (arcsec/pix) / (pix/bin)

# Keep negatives from going white on log
bsub.data[np.where(bsub.data < 0)] = vmin
s = np.asarray(bsub.shape)
center = s/2
center = center.astype(int)
subim_corners = np.asarray(((round(center[0]-20*Rmpix), round(center[1]-20*Rmpix)),
                            (round(center[0]+20*Rmpix), s[1]-100)))
#subim_corners = np.asarray(((round(center[0]-2*Rmpix), round(center[1]-20*Rmpix)),
#                            (round(center[0]+2*Rmpix), s[1]-100)))
bsub_full_im = bsub.copy()
bsub = bsub[subim_corners[0,0]:subim_corners[1,0],
            subim_corners[0,1]:subim_corners[1,1]]
nr, nc = bsub.shape
#x = (np.arange(nc) - nc/2) / Rmpix
#y = (np.arange(nr) - nr/2) / Rmpix
x = (np.arange(nc) - (center[1] - subim_corners[0,1])) / Rmpix
y = (np.arange(nr) - (center[0] - subim_corners[0,0])) / Rmpix
X, Y = np.meshgrid(x, y)
#plt.pcolormesh(X, Y, bsub, norm=LogNorm(vmin=vmin, vmax=vmax), cmap='YlOrRd')
plt.pcolormesh(X, Y, bsub, norm=LogNorm(vmin=vmin, vmax=vmax), cmap='gist_heat')
#plt.pcolormesh(X, Y, bsub, vmin=vmin, vmax=vmax, cmap='gist_heat')
plt.ylabel('Rm')
plt.xlabel('Rm')
plt.axis('scaled')
#cbar = plt.colorbar(orientation='horizontal', fraction=0.08)
cbar = plt.colorbar(shrink=0.6)
cbar.ax.set_xlabel(bsub.unit.to_string())
plt.savefig(f'/data/Mercury/analysis/2020-05-27/IoIO_{just_date}.png', transparent=True)
plt.show()

yaxis = (np.arange(nr) - 20*Rmpix) / Rmpix
ax1 = plt.subplot(1, 4, 1)
p1 = np.sum(bsub[:, round((7+20)*Rmpix):round((25+20)*Rmpix)], 1)
ax1.plot(p1 / ((25-7)*Rmpix), yaxis)
ax1.set_ylim(-17, 17)

ax2 = plt.subplot(1, 4, 2)
p2 = np.sum(bsub[:, round((50+20)*Rmpix):round((80+20)*Rmpix)], 1)
ax2.plot(p2/((80-50)*Rmpix), yaxis)
ax2.set_ylim(-17, 17)

ax3 = plt.subplot(1, 4, 3)
p3 = np.sum(bsub[:, round((110+20)*Rmpix):round((150+20)*Rmpix)], 1)
ax3.plot(p3/((150-110)*Rmpix), yaxis)
ax3.set_ylim(-17, 17)
ax3.set_xlim(100, 250)
plt.savefig(f'/data/Mercury/analysis/2020-05-27/IoIO_{just_date}_cross-tail_profiles.png', transparent=True)
plt.show()

subim_corners = np.asarray(((round(center[0]-2*Rmpix), round(center[1]-20*Rmpix)),
                            (round(center[0]+2*Rmpix), s[1]-100)))
prof_im = bsub_full_im[subim_corners[0,0]:subim_corners[1,0],
               subim_corners[0,1]:subim_corners[1,1]]

#impl = plt.imshow(prof_im, origin='lower',
#                  cmap=plt.cm.gray,
#                  filternorm=0, interpolation='none')
#plt.show()

prof = np.sum(prof_im, 0)

xaxis = (np.arange(nc) - 20*Rmpix) / Rmpix
plt.plot(xaxis, prof/(4*Rmpix))
plt.title(f'IoIO observatory NaD lines {just_date}')
axes = plt.gca()
axes.set_xlim(3, 155)
m1 = MultipleLocator(5)
m2 = MultipleLocator(500)
plt.axes().xaxis.set_minor_locator(m1)
plt.axes().yaxis.set_minor_locator(m2)
plt.xlabel('Rm')
plt.ylabel('R')
plt.xscale("log")
plt.yscale("log")
plt.savefig(f'/data/Mercury/analysis/2020-05-27/IoIO_{just_date}_profile.png', transparent=True)
plt.show()



### s = np.asarray(bsub.shape)
### subim_corners = np.asarray(((mcenter[0]-200, mcenter[1]-20),
###                             (mcenter[0]+200, s[1]-100)))
### #bsub = bsub[mcenter[0]-20:mcenter[0]+20, mcenter[1]-20:-100]
### bsub = bsub[subim_corners[0,0]:subim_corners[1,0],
###             subim_corners[0,1]:subim_corners[1,1]]
### block_size = 1
### binning = 1
### vmin = 0.01
### vmax = 1.3
### ang_dia = 6.8 # arcsec
### Rmpix = ang_dia/plate_scale / (block_size*binning) # arcsec / (arcsec/pix) / (pix/bin)
### nr, nc = bsub.shape
### #x = (np.arange(nc) - nc/2) / Rmpix
### #y = (np.arange(nr) - nr/2) / Rmpix
### x = (np.arange(nc) - (mcenter[1] - subim_corners[0,1])) / Rmpix
### y = (np.arange(nr) - (mcenter[0] - subim_corners[0,0])) / Rmpix
### X, Y = np.meshgrid(x, y)
### #plt.pcolormesh(X, Y, bsub, norm=LogNorm(vmin=vmin, vmax=vmax), cmap='YlOrRd')
### plt.pcolormesh(X, Y, bsub, norm=LogNorm(vmin=vmin, vmax=vmax), cmap='gist_heat')
### #plt.pcolormesh(X, Y, bsub, vmin=vmin, vmax=vmax, cmap='gist_heat')
### plt.ylabel('Rm')
### plt.xlabel('Rm')
### plt.axis('scaled')
### cbar = plt.colorbar(orientation='horizontal')
### cbar.ax.set_xlabel('Electron/s')
### plt.show()

#impl = plt.imshow(bsub, origin='lower',
#                  norm=LogNorm(), cmap='gist_heat',
#                  filternorm=0, interpolation='none')
#plt.show()
#impl = plt.imshow(bsub, origin='lower', cmap='gist_heat',
#                  filternorm=0, interpolation='none')
#plt.show()


### block_size = 1
### binning = 1
### vmin = 0.01
### vmax = 1.3
### ang_dia = 6.8 # arcsec
### Rmpix = ang_dia/plate_scale / (block_size*binning) # arcsec / (arcsec/pix) / (pix/bin)
### nr, nc = bsub.shape
### #x = (np.arange(nc) - nc/2) / Rmpix
### #y = (np.arange(nr) - nr/2) / Rmpix
### x = (np.arange(nc) - mcenter[1]) / Rmpix
### y = (np.arange(nr) - mcenter[0]) / Rmpix
### X, Y = np.meshgrid(x, y)
### #plt.pcolormesh(X, Y, bsub, norm=LogNorm(vmin=vmin, vmax=vmax), cmap='YlOrRd')
### plt.pcolormesh(X, Y, bsub, norm=LogNorm(vmin=vmin, vmax=vmax), cmap='gist_heat')
### #plt.pcolormesh(X, Y, bsub, vmin=vmin, vmax=vmax, cmap='gist_heat')
### plt.ylabel('Rm')
### plt.xlabel('Rm')
### plt.axis('scaled')
### #cbar = plt.colorbar()
### cbar = plt.colorbar(orientation='horizontal')
### cbar.ax.set_xlabel('Electron/s')
### plt.show()
### 
### #impl = plt.imshow(bsub, origin='lower',
### #                  norm=LogNorm(), cmap='gist_heat',
### #                  filternorm=0, interpolation='none')
### #plt.show()
### #impl = plt.imshow(bsub, origin='lower', cmap='gist_heat',
### #                  filternorm=0, interpolation='none')
### #plt.show()

