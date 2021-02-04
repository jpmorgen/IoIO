"""Play with photometry for calibration sources"""

import glob

import numpy as np
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import pandas as pd

from astropy import log
from astropy import units as u
from astropy.time import Time
from astropy.convolution import Gaussian2DKernel, convolve
from astropy.stats import gaussian_fwhm_to_sigma, sigma_clipped_stats
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize

from photutils import make_source_mask
#from photutils import detect_threshold
from photutils import detect_sources
from photutils import deblend_sources
from photutils import source_properties
from photutils.background import Background2D

from bigmultipipe import multi_logging, prune_pout

from ccdmultipipe import ccddata_read
from cormultipipe import (CorMultiPipe, Calibration, 
                          nd_filter_mask, mask_nonlin_sat)

def photometry_process(ccd, pipe_meta, in_name='', **kwargs):
    # This is going to expand by a factor of 15
    sigma = 10.0 * gaussian_fwhm_to_sigma # FWHM = 10
    kernel = Gaussian2DKernel(sigma)
    kernel.normalize()
    # Make a source mask to enable optimal background estimation
    mask = make_source_mask(ccd.data, nsigma=2, npixels=5,
                            filter_kernel=kernel, mask=ccd.mask,
                            dilate_size=11)
    #impl = plt.imshow(mask, origin='lower',
    #                  cmap=plt.cm.gray,
    #                  filternorm=0, interpolation='none')
    #plt.show()
    
    ##mean, median, std = sigma_clipped_stats(data, sigma=3.0, mask=mask)
    #
    
    box_size = int(np.mean(ccd.shape) / 10)
    back = Background2D(ccd, box_size, mask=mask, coverage_mask=ccd.mask)
    threshold = back.background + (2.0* back.background_rms)

    print(f'background_median = {back.background_median}, background_rms_median = {back.background_rms_median}')
    
    #impl = plt.imshow(back.background, origin='lower',
    #                  cmap=plt.cm.gray,
    #                  filternorm=0, interpolation='none')
    #back.plot_meshes()
    #plt.show()
    
    npixels = 5
    segm = detect_sources(ccd.data, threshold, npixels=npixels,
                          filter_kernel=kernel, mask=ccd.mask)
    
    if segm is None:
        # detect_sources logs a WARNING and returns None if no sources
        # are found.  We need to return at this point our else
        # deblend_sources throws an inscrutable error
        return (None, {})

    # It does save a little time and a factor ~1.3 in memory if we
    # don't deblend
    segm_deblend = deblend_sources(ccd.data, segm, npixels=npixels,
                                   filter_kernel=kernel, nlevels=32,
                                   contrast=0.001)
    
    #impl = plt.imshow(segm, origin='lower',
    #                  cmap=plt.cm.gray,
    #                  filternorm=0, interpolation='none')
    #plt.show()
    
    cat = source_properties(ccd.data, segm_deblend, error=ccd.uncertainty.array,
                            mask=ccd.mask)
    tbl = cat.to_table()
    tbl.sort('source_sum', reverse=True)

    # Reject source with saturated pixels
    nonlin = ccd.meta['NONLIN']
    if tbl['max_value'][0] >= nonlin:
        return(None, {})
    
    # tbl['xcentroid'].info.format = '.2f'  # optional format
    # tbl['ycentroid'].info.format = '.2f'
    # tbl['cxx'].info.format = '.2f'
    # tbl['cxy'].info.format = '.2f'
    # tbl['cyy'].info.format = '.2f'
    # tbl['gini'].info.format = '.2f'
    # print(tbl)
    # print(tbl['source_sum'])
    # 
    # print(ccd.meta['AIRMASS'])
    
    # What I really want is source sum and airmass as metadata put into header and
    # http://classic.sdss.org/dr2/algorithms/fluxcal.html
    # Has some nice formulae:
    #     aa = zeropoint
    #     kk = extinction coefficient
    #     airmass
    # f/f0 = counts/exptime * 10**(0.4*(aa + kk * airmass))
    
    airmass = ccd.meta['AIRMASS']
    exptime = ccd.meta['EXPTIME']
    detflux = tbl['source_sum'][0] / exptime
    detflux_err = tbl['source_sum_err'][0] / exptime
    xcentrd = tbl['xcentroid'][0].value
    ycentrd = tbl['ycentroid'][0].value
    center = np.asarray(ccd.shape)/2
    radius = ((xcentrd-center[1])**2 + (ycentrd-center[0])**2)**0.5
    # --> could check to see if too close to edge
    
    ccd.meta['DETFLUX'] = (detflux, '(electron/s)')
    ccd.meta['HIERARCH DETFLUX_ERR'] = (detflux_err, '(electron/s)')
    ccd.meta['xcentrd'] = xcentrd
    ccd.meta['ycentrd'] = ycentrd
    ccd.meta['radius'] = radius    

    date_obs = ccd.meta['DATE-OBS']
    just_date, _ = date_obs.split('T')
    tm = Time(date_obs, format='fits')
    
    return (ccd,
            {'object': ccd.meta['OBJECT'],
             'filter': ccd.meta['FILTER'],
             'date': just_date,
             'time_obs': tm,
             'jd': tm.jd,
             'airmass': airmass,
             'objalt': ccd.meta['OBJCTALT'],
             'exptime': exptime,
             'detflux': detflux,
             'detflux_err': detflux_err,
             'xcentroid': xcentrd,
             'ycentroid': ycentrd,
             'radius': radius,
             'background_median': back.background_median,
             'background_rms_median': back.background_rms_median})
    
#c = Calibration(start_date='2020-07-07', stop_date='2020-08-22', reduce=True)
##fname = '/data/io/IoIO/raw/20200708/HD 118648-S001-R001-C001-Na_on.fts'
#fname = '/data/io/IoIO/raw/2020-07-15/HD87696-0016_Na_off.fit'
#cmp = CorMultiPipe(auto=True, calibration=c,
#                   post_process_list=[nd_filter_mask,
#                                      photometry_process],
#                   process_expand_factor=15)
#pout = cmp.pipeline([fname], outdir='/tmp', overwrite=True)
#out_fnames, pipe_meta = zip(*pout)
#
#print(pipe_meta)

#directory = '/data/io/IoIO/raw/2020-07-15'
directory = '/data/io/IoIO/raw/20200708/'
glob_include = ['HD*']
flist = []
for gi in glob_include:
    flist += glob.glob(os.path.join(directory, gi))

c = Calibration(start_date='2020-07-07', stop_date='2020-08-22', reduce=True)
cmp = CorMultiPipe(auto=True, calibration=c,
                   post_process_list=[nd_filter_mask,
                                      photometry_process],
                   process_expand_factor=15)
pout = cmp.pipeline(flist, outdir='/tmp', overwrite=True)
pout, flist = prune_pout(pout, flist)
out_fnames, pipe_meta = zip(*pout)

print(pipe_meta)

df = pd.DataFrame(pipe_meta)

ax = plt.subplot(1, 1, 1)
plt.title(directory)
ax.tick_params(which='both', bottom=True, top=True, left=True, right=True)
plt.plot(df['airmass'], df['detflux'], 'k.')
plt.xlabel('airmass')
plt.ylabel('detflux')
plt.show()

show = True
objects = list(set(df['object']))
filters = list(set(df['filter']))
for object in objects:
    objdf = df[df['object'] == object]
    min_airmass = np.amin(objdf['airmass'])
    max_airmass = np.amax(objdf['airmass'])
    f = plt.figure(figsize=[8.5, 11])
    plt.suptitle(f'{directory}      {object}')
    for ifilt, filt in enumerate(filters):
        filtdf = objdf[objdf['filter'] == filt]
        ax = plt.subplot(9, 1, ifilt+1)
        ax.tick_params(which='both', direction='inout',
                       bottom=True, top=True, left=True, right=True)
        ax.set_xlim([min_airmass, max_airmass])
        plt.plot(filtdf['airmass'], filtdf['detflux'], 'k.')
        plt.ylabel(filt)
        #ax.xaxis.set_ticklabels([])
    ax.tick_params(reset = True, which='both', direction='inout',
                   bottom=True, top=True, left=True, right=True)
    plt.xlabel('airmass')
    #plt.savefig((outbase + '_.png'), transparent=True)
    if show:
        plt.show()
    plt.close()
