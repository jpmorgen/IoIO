"""Reduce files for Henry Hsieh"""

import os
import glob

from ccdmultipipe import as_single

from cormultipipe import reduced_dir
from cormultipipe import Calibration, OffCorMultiPipe, nd_filter_mask

#directory = '/data/io/IoIO/raw/20210909'
directory = '/data/io/IoIO/raw/20210914'
rd = reduced_dir(directory, create=True)
glob_include = ['00003*', '17857*']

calibration = Calibration(reduce=True)

flist = []
for gi in glob_include:
    flist += glob.glob(os.path.join(directory, gi))

cmp = OffCorMultiPipe(auto=True, calibration=calibration,
                   outname_ext='.fits', 
                   post_process_list=[nd_filter_mask, as_single])
pout = cmp.pipeline(flist, outdir=rd, overwrite=True, outname_ext='.fits')
#pout = cmp.pipeline([flist[18]], outdir=rd, overwrite=True, outname_ext='.fits')

#cmp = OffCorMultiPipe(auto=True, calibration=calibration,
#                   outname_ext='.fits', 
#                   post_process_list=[nd_filter_mask])
#pout = cmp.pipeline(flist[0:3], outdir='/tmp',
#                    as_scaled_int=True, overwrite=True, outname_ext='.fits')

#cmp = OffCorMultiPipe(auto=True, calibration=calibration,
#                   outname_ext='.fits', 
#                   post_process_list=[nd_filter_mask])
#pout = cmp.pipeline(flist[0:3], outdir='/tmp/double', overwrite=True, outname_ext='.fits')

# IMPORTANT: Use the DATE-AVG keyword as your midpoint of the
# observations, don't calculated it yourself!  As detailed below, there
# are a lot of instrumental details to get there.  The DATE-AVG keyword
# name is recommended in the WCS standard, which is why I use it.
# 
# DATE-AVG details: DATE-AVG is carefully constructed using calibration
# measurements detailed in the nearby FITS cards.  The basic issue: the
# shutter is commanded via the USB bus on a Windows 10 machine, so there
# is a variable latency in the real shutter open time.  The shutter close
# time, which is needed for EXPTIME, is an even bigger mess, at least for
# exposures >0.7s, involving the aformentioned USB latency and an extra
# ~2s round-trip.  DATE-AVG and EXPTIME are therefore best estimates and
# their respective uncertainties, DATE-AVG-UNCERTAINTY and
# EXPTIME-UNCERTAINTY, are noted.  If you want to be really thorough, feel
# free to use the uncertainties to put error bars on the time axis of your
# analyses.
# 
# Plate scale in arcsec can be calculated using the following keywords
# which (unless you buy me a fancier camera) are unlikely to change.  
# 
# XPIXSZ = 4.5390600000000001 # micron
# FOCALLEN = 1200.0000000000000 # mm
# PIXSCALE = 206265 * XPIXSZ/1000 / FOCALLEN
# 0.78020767575 # arcsec
# 
# or astropythonically:
# 
# import astropy.units as u
# from astropy.nddata import CCDData
# 
# # Example file
# ccd = CCDData.read('/data/io/IoIO/reduced/20210820/TOI-2109-S001-R001-C003-R_p.fts')
# # Be completely general -- not all CCDs have square pixels
# pixel_size = (ccd.meta['XPIXSZ']**2 + ccd.meta['YPIXSZ'])**0.5
# focal_length = ccd.meta['FOCALLEN']
# # Proper FITS header units would make these obsolete
# pixel_size *= u.micron 
# focal_length *= u.mm
# plate_scale = pixel_size / focal_length
# plate_scale *= u.radian
# plate_scale = plate_scale.to(u.arcsec)
# print(f'Plate scale: {plate_scale}')
# # Proper FITS header units would make this easier
# ccd.meta['PIXSCALE'] = (plate_scale.value, '[arcsec]')
# 
# 
# Let me know if you have any questions or recommendations to make
