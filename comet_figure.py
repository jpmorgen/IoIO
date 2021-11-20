import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib import colors

from astropy import log
import astropy.units as u

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
delta = 1.67376532448597 * u.au
# Scale by 1000 so reads in 1000 km
delta = delta.to(u.km) / 1000

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

fig, ax = plt.subplots(figsize=(7.7, 6))
#fig = plt.figure(figsize=(7.7, 6))
plt.pcolormesh(X, Y, patch, cmap=plt.cm.viridis,
               norm=colors.LogNorm(), vmin=vmin, vmax=vmax)
# Preserve asepct ratio with extra space in plot
#ax.axis('equal')
# Preserve asepct ratio with extra space outside of plot
ax.axis('scaled')
ax.set_xlabel('arcsec')
ax.set_ylabel('arcsec')
ax.xaxis.set_major_locator(mtick.MultipleLocator(100))
ax.yaxis.set_major_locator(mtick.MultipleLocator(100))
ax.xaxis.set_minor_locator(mtick.AutoMinorLocator())
ax.yaxis.set_minor_locator(mtick.AutoMinorLocator())

#formatter = mtick.LogFormatter(10, labelOnlyBase=False)
#formatter.format_data('%2.2f')
#formatter = mtick.LogFormatterMathtext(10, labelOnlyBase=False)
formatter = '%2.f'
#formatter = mtick.EngFormatter(places=2)
cbar = plt.colorbar(format=formatter)
#cbar = plt.colorbar()
cbar.ax.set_ylabel('electron/s')


# [tan](cometocentric / delta) = radians
def pix_to_cometocentric(npix):
    return np.deg2rad(npix*plate_scale/3600) * delta.value

def cometocentric_to_pix(cometocentric):
    return np.rad2deg(cometocentric/delta.value)/3600/plate_scale

def arcsec_to_cometocentric(arcsec):
    return np.deg2rad(arcsec/3600) * delta.value

def cometocentric_to_arcsec(cometocentric):
    return np.rad2deg(cometocentric/delta.value)/3600

#secax = ax.secondary_xaxis('top', functions=(pix_to_cometocentric,
#                                             cometocentric_to_pix))
secax = ax.secondary_xaxis('top', functions=(arcsec_to_cometocentric,
                                             cometocentric_to_arcsec))
secax.set_xlabel('cometocentric distance x 1000 km')
secax.minorticks_on()
secay = ax.secondary_yaxis('right', functions=(arcsec_to_cometocentric,
                                               cometocentric_to_arcsec))
secay.minorticks_on()
secay.set_yticklabels('')

#secax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.0e'))

#secax.ticklabel_format(axis='x', style='sci', scilimits=(0,3), useOffset=False)

fig.set_tight_layout(True)


#plt.imshow(patch, origin='lower', cmap=plt.cm.viridis, #plt.cm.rainbow,
#           norm=colors.LogNorm(), vmin=200, vmax=1800)

plt.savefig('/home/jpmorgen/Proposals/NSF/Landslides_2021/figures/C2017T2_PANNSTARRS.png')
plt.show()


