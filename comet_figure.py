import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

from astropy import log

from ccdmultipipe import as_single

from cormultipipe import reduced_dir
from cormultipipe import Calibration, OffCorMultiPipe, nd_filter_mask

log.setLevel('DEBUG')

#calibration = Calibration(reduce=True)
#directory = '/data/io/IoIO/raw/20200514/'
#rd = reduced_dir(directory, create=True)
#
#flist = [directory + 'CK17T020-S001-R001-C001-R_dupe-4.fts']
#cmp = OffCorMultiPipe(auto=True, calibration=calibration,
#                      fits_fixed_ignore=True, outname_ext='.fits', 
##                      post_process_list=[nd_filter_mask, as_single])
#                      post_process_list=[as_single])
#pout = cmp.pipeline(flist, outdir=rd, overwrite=True, outname_ext='.fits')

from astropy.nddata import CCDData
plt.rcParams.update({'font.size': 18})

ccd = CCDData.read('/data/io/IoIO/reduced/20200514/CK17T020-S001-R001-C001-R_dupe-4_p.fits')

comet_center = np.asarray((1013, 1663))
cropx = 250
cropy = cropx
plate_scale = 0.7

exptime = ccd.meta['EXPTIME']

# Code from RedCorData
patch_half_width = cropx / 2
patch_half_width = int(patch_half_width)
icom_center = comet_center.astype(int)
print('icom_center:', icom_center)
ll = icom_center - patch_half_width
ur = icom_center + patch_half_width
print('ll', ll)
print('ur', ur)
patch = ccd[ll[0]:ur[0], ll[1]:ur[1]]
patch.data /= exptime
vmin = 200/exptime
vmax = 2000/exptime

# Code from Na_im
nr, nc = patch.shape
x = (np.arange(nc) - nc/2) / plate_scale
y = (np.arange(nr) - nr/2) / plate_scale
X, Y = np.meshgrid(x, y)

fig = plt.figure(figsize=(7.7, 6))
plt.pcolormesh(X, Y, patch, cmap=plt.cm.viridis,
               norm=colors.LogNorm(), vmin=vmin, vmax=vmax)
# Preserve asepct ratio with extra space in plot
#plt.axis('equal')
# Preserve asepct ratio with extra space outside of plot
plt.axis('scaled')
plt.xlabel('arcsec')
plt.ylabel('arcsec')
fig.set_tight_layout(True)

cbar = plt.colorbar()
cbar.ax.set_ylabel('electron/s')

#plt.imshow(patch, origin='lower', cmap=plt.cm.viridis, #plt.cm.rainbow,
#           norm=colors.LogNorm(), vmin=200, vmax=1800)
plt.show()

