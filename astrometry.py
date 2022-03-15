"""IoIO astrometry experiments"""

import os
import subprocess

import numpy as np

from matplotlib.colors import LogNorm

from astropy import log
import astropy.units as u
from astropy.coordinates import Angle
from astropy.table import QTable

from IoIO.utils import simple_show
from IoIO.cordata import CorData
from IoIO.calibration import Calibration
from IoIO.cor_process import cor_process
from IoIO.photometry import Photometry

c = Calibration(reduce=True)

# Taking too long to solve
#fname = '/data/IoIO/raw/2018-05-08/Na_on-band_011.fits'
# Bad focus
#fname = '/data/IoIO/raw/20220312/HAT-P-36b-S002-R010-C002-R.fts'
fname = '/data/IoIO/raw/20220308/HAT-P-36b-S002-R010-C002-R.fts'
rccd = CorData.read(fname)
ccd = cor_process(rccd, calibration=c, auto=True, gain=False)
#ccd = ccd.divide(1*u.s)

#simple_show(ccd, norm=LogNorm())
p = Photometry(ccd=ccd)
cat = p.source_catalog
source_qtable = cat.to_table(['label',
	                      'xcentroid',
	                      'ycentroid',
                              'min_value',
	                      'max_value',
                              'local_background',
                              'segment_flux',
                              'segment_fluxerr'])

# I am going to want to extract this from the outname, which is going
# to make this a little hard in the bigmultipipe formalism.  Or I
# cheat and have outfname and/or write continue the processing
#bname = os.path.basename(fname)
outname = '/tmp/test_astrometry.fits'
bname = os.path.basename(outname)


#source_qtable.show_in_browser()

# Ack.  Astropy is not making this easy.  Writing the QTable fails
# entirely, possibly because it has mixed Quantity and non-Quantity
# columns.  Error has something about not being able to represent an
# object
if source_qtable.has_mixin_columns:
    # Move collection of Quantity columns to regular columns with
    # quantities, which does write properly, but
    source_table = Table(source_qtable)
# I need to work harder to rewrite all columns as Quantities, but 
# that would just be for testing purposes
#for col in source_qtable.colnames:
#    try:
#        source_qtable[col].unit = u.Angstrom
#    except:
#        pass

# This didn't help
#with u.add_enabled_units('electron'):
#    source_qtable.write('/tmp/test_photometry_qtable.xyls',
#                        format='fits', overwrite=True)

# This, unfortunately does not save the units like I would expect
# because they do not get written
with u.add_enabled_units('electron'):
    source_table.write('/tmp/test_photometry_table.xyls',
                       format='fits', overwrite=True)

    test_table = QTable.read('/tmp/test_photometry_table.xyls')

#t = QTable([1, 2] * u.angstrom)
t = QTable([1, 2] * u.electron)
t.write('/tmp/my_table.fits', overwrite=True)
qt = QTable.read('/tmp/my_table.fits')
qt

# This is the RA and DEC table extened by catalog info (--tag-all)
rdls = QTable.read('/tmp/test_photometry_table.rdls')

ra = Angle(ccd.meta['objctra'])
ra = ra.to_string(sep=':')
dec = Angle(ccd.meta['objctdec'])
dec = dec.to_string(alwayssign=True, sep=':')
naxis1 = ccd.meta['naxis1']
naxis2 = ccd.meta['naxis2']
pixscale = ccd.meta['PIXSCALE']
radius = np.linalg.norm((naxis1, naxis2)) * u.pixel
radius = radius * pixscale *u.arcsec/u.pixel
radius = radius.to(u.deg)

print(f'solve-field --x-column xcentroid --y-column ycentroid '
      f'--ra {ra} --dec {dec} -radius {2*radius.value:.2f} '
      f'--width {naxis1} --height {naxis2} --scale-low pixscale*-0.8 '
      f'--scale-low pixscale*-1.2 --scale-units arcsecperpix '
      f'--tag-all --overwrite ')
