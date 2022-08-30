#!/usr/bin/python3

import os
import warnings

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import MultipleLocator

import astropy.units as u
from astropy.io.fits import getheader, PrimaryHDU
from astropy.wcs import WCS, FITSFixedWarning
from astropy.time import Time
from astropy.coordinates import SkyCoord

import ccdproc as ccdp

#from IoIO.cormultipipe import ASTROMETRY_ROOT
from IoIO.utils import savefig_overwrite
from IoIO.simple_show import simple_show
from IoIO.photometry import create_rmat, rot_wcs, rot_angle_by_wcs, rot_to
from IoIO.cor_photometry import CorPhotometry, read_wcs
from IoIO.cordata import CorData
from astropy.wcs.utils import pixel_to_skycoord, skycoord_to_pixel

vmin = 4E3
vmax = 80E3

x0, y0 = (50, 50)
vlen = 50

ccd = CorData.read('/data/Mercury/2021-10-28/Mercury-0001_Na_on-back-sub.fits')

f = plt.figure(figsize=[6, 2])
Rm = ccd.meta['Mercury_ang_width'] / 2 * u.arcsec
pixscale = ccd.wcs.proj_plane_pixel_scales() * u.deg / u.pixel
Rmscale = Rm/pixscale
Rmscale = Rmscale.decompose()
Rmscale = Rmscale.value
Rmscale = np.mean(Rmscale)
center = ccd.wcs.world_to_pixel(ccd.sky_coord)
l = round(center[0]-50*Rmscale)
r = ccd.shape[1]-0
b = round(center[1]-40*Rmscale)
t = round(center[1]+40*Rmscale)
subim = ccd[b:t, l:r]
nr, nc = subim.shape
x = (np.arange(nc) - (center[0] - l)) / Rmscale
y = (np.arange(nr) - (center[1] - b)) / Rmscale
X, Y = np.meshgrid(x, y)

north = ccd.meta['Mercury_NPole_ang']*u.deg
antisun = ccd.meta['Mercury_sunTargetPA']*u.deg
print(f'original north {north}')
print(f'original antisun {antisun}')
north = rot_angle_by_wcs(north, ccd)
antisun = rot_angle_by_wcs(antisun, ccd)
print(north)
print(antisun)


subim.data[subim.data < vmin] = vmin

# Rough stellar calibration
# Tue May 24 16:23:20 2022 EDT  jpmorgen@snipe
#subim = subim.multiply(0.4, handle_meta='first_found')

plt.pcolormesh(X, Y, subim,
#               norm=LogNorm(vmin=vmin, vmax=vmax), cmap='gist_heat')
               norm=LogNorm(vmin=vmin, vmax=vmax), cmap='gist_ncar')
plt.ylabel('Sky Plane Rm')
plt.xlabel('Sky Plane Rm')
plt.axis('scaled')
plt.title(ccd.meta['DATE-OBS'])
plt.tight_layout()

#plt.arrow(x0, y0, north_dxy[0], north_dxy[1])
#plt.arrow(x0, y0, antisun_dxy[0], antisun_dxy[1])

cbar = plt.colorbar(shrink=0.5)
#cbar = plt.colorbar()
cbar.ax.set_xlabel(ccd.unit.to_string())

#savefig_overwrite('/data/Mercury/analysis/IoIO_2021-10-28.png')
savefig_overwrite('/home/jpmorgen/Talks/PSI/2022_retreat/IoIO_2021-10-28.png')
plt.show()
plt.close()
