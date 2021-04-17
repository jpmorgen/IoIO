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

from cormultipipe import (CorMultiPipe, Calibration, 
                          nd_filter_mask, mask_nonlin_sat)
import sx694

def photometry_process(ccd,
                       bmp_meta=None,
                       in_name=None,
                       seeing=5,
                       min_ND_multiple=3,
                       **kwargs):
    """Parameters
    ----------
    seeing : number
        Seeing FWHM (pixel)

    min_ND_multiple : number

        Number of ND filter *widths* away from the *center* of the ND
        filter that the source needs to be to be considered good.
        Note that expansion by `cormultipipe.ND_EDGE_EXPAND` is not
        considered in this calculation, but the ND filter is likely to
        be masked using that quantity if `cormultipipe.nd_filter_mask`
        has been called in the `cormultipipe.post_process_list`

    """
    # This is going to expand by a factor of 15
    sigma = seeing * gaussian_fwhm_to_sigma
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
        return None

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
        log.warning(f'Too bright: {in_name}')
        return None
    
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
    
    xcentrd = tbl['xcentroid'][0].value
    ycentrd = tbl['ycentroid'][0].value
    ccd.obj_center = (ycentrd, xcentrd)
    ccd.quality = 10
    ND_edges = ccd.ND_edges(ycentrd)
    min_ND_dist = min_ND_multiple * (ND_edges[1] - ND_edges[0])
    print(ND_edges, min_ND_dist)
    if ccd.obj_to_ND < min_ND_dist:
        log.warning(f'Too close: ccd.obj_to_ND = {ccd.obj_to_ND} {in_name}')
        return None
    center = np.asarray(ccd.shape)/2
    radius = ((xcentrd-center[1])**2 + (ycentrd-center[0])**2)**0.5

    airmass = ccd.meta['AIRMASS']
    exptime = ccd.meta['EXPTIME']
    oexptime = ccd.meta.get('oexptime')
    detflux = tbl['source_sum'][0] / exptime
    detflux_err = tbl['source_sum_err'][0] / exptime
    if oexptime is None:
        odetflux = None
        odetflux_err = None
    else:
        odetflux = tbl['source_sum'][0] / oexptime
        odetflux_err = tbl['source_sum_err'][0] / oexptime
    
    ccd.meta['DETFLUX'] = (detflux, '(electron/s)')
    ccd.meta['HIERARCH DETFLUX_ERR'] = (detflux_err, '(electron/s)')
    ccd.meta['xcentrd'] = xcentrd
    ccd.meta['ycentrd'] = ycentrd
    ccd.meta['radius'] = radius    

    date_obs = ccd.meta['DATE-OBS']
    just_date, _ = date_obs.split('T')
    tm = Time(date_obs, format='fits')
    
    tmeta = {'object': ccd.meta['OBJECT'],
             'filter': ccd.meta['FILTER'],
             'date': just_date,
             'time_obs': tm,
             'jd': tm.jd,
             'airmass': airmass,
             'objalt': ccd.meta['OBJCTALT'],
             'exptime': exptime,
             'oexptime': oexptime,
             'detflux': detflux,
             'detflux_err': detflux_err,
             'odetflux': odetflux,
             'odetflux_err': odetflux_err,
             'xcentroid': xcentrd,
             'ycentroid': ycentrd,
             'obj_to_ND': ccd.obj_to_ND,
             'radius': radius,
             'background_median': back.background_median,
             'background_rms_median': back.background_rms_median}
    bmp_meta.update(tmeta)

    return ccd
    
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
#directory = '/data/io/IoIO/raw/20200708/' # 2 -- 3.4, spotty data
#directory = '/data/io/IoIO/raw/20210315/' # 2.3 -- 3.0
#directory = '/data/io/IoIO/raw/20210311/' # 2.3 -- 2.7
#directory = '/data/io/IoIO/raw/20210310/' # 2.3 lots of measurements
#directory = '/data/io/IoIO/raw/20210307/' # 2.3 -- 2.6
#directory = '/data/io/IoIO/raw/20201011/' # Crumy
directory = '/data/io/IoIO/raw/20201010/'  # 1.85 -- 1.90 ~very good

#glob_include = ['HD*']
#glob_include = ['HD*118648*SII_on*']
glob_include = ['HD*SII_on*']

# Pythonic way of checking for a non-assigned variable
try:
    pout
except NameError:
    # Run our pipeline code
    flist = []
    for gi in glob_include:
        flist += glob.glob(os.path.join(directory, gi))

    c = Calibration(start_date='2020-01-01', stop_date='2021-12-31',
                    reduce=True)
    cmp = CorMultiPipe(auto=True, calibration=c,
                       post_process_list=[nd_filter_mask,
                                          photometry_process],
                       process_expand_factor=15)

    pout = cmp.pipeline(flist, outdir='/tmp', overwrite=True)
    pout, flist = prune_pout(pout, flist)
    out_fnames, pipe_meta = zip(*pout)

df = pd.DataFrame(pipe_meta)

#ax = plt.subplot(1, 1, 1)
#plt.title(directory)
#ax.tick_params(which='both', bottom=True, top=True, left=True, right=True)
#plt.plot(df['airmass'], df['detflux'], 'k.')
#plt.xlabel('airmass')
#plt.ylabel('detflux')
#plt.show()

show = True
#show = False
objects = list(set(df['object']))
filters = list(set(df['filter']))
exposure_corrects = []
exposure_correct_jds = []
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

        # Work with exposure time correction
        filtdf.sort_values(by=['jd'], inplace=True)
        exptimes = list(set(filtdf['exptime']))
        if (len(exptimes) > 1
            and np.min(exptimes) <= sx694.max_accurate_exposure
            and np.max(exptimes) > sx694.max_accurate_exposure):
            # Code from cormultipipe.bias_dark_fdict_creator to find
            # jumps in time
            jds1 = filtdf['jd'].iloc[1:].to_numpy()
            jds0 = filtdf['jd'].iloc[0:-1].to_numpy()
            deltas = jds1 - jds0
            # This didn't work for some annoying reason
            #deltas = filtdf['jd'].iloc[1:] - filtdf['jd'].iloc[0:-1]#)#*u.day
            #deltas = filtdf['jd'] - filtdf['jd']
            print('deltas', deltas)
            #print(filtdf['time_obs'])
            #deltats = filtdf['time_obs'][1:] - filtdf['time_obs'][0:-1]#)#*u.day
            # Group measurements in chunks recorded over < 10 minutes
            jump = np.flatnonzero(deltas*u.day > 10*u.min)
            tslices = np.append(0, jump+1)
            tslices = np.append(tslices, None)
            print('tslices', tslices)
            for it in range(len(tslices)-1):
                # For each distinct measurement time
                tdf = filtdf[slice(tslices[it], tslices[it+1])]
                oexptimes = tdf['oexptime'].to_numpy()
                exptimes = tdf['exptime'].to_numpy()
                oexptimes = np.where(np.isnan(oexptimes), exptimes, oexptimes)
                uoes = np.unique(oexptimes)
                if (len(uoes) < 2
                    or np.min(oexptimes) > sx694.max_accurate_exposure
                    or np.max(oexptimes) <= sx694.max_accurate_exposure):
                    # Make sure we have some times that straddle the
                    # accurate exposure time boundary
                    continue
                detfluxes = []
                counts = []
                print('uoes, exptimes', uoes, exptimes)
                for uoe in uoes:
                    # For each unique oexptime
                    exp_idx = np.flatnonzero(oexptimes == uoe)
                    print('exp_idx', exp_idx)
                    # Have many nights with 3 measurements per exposure time
                    mdetflux_err = np.median(tdf['detflux_err'].iloc[exp_idx])
                    detflux_std = np.std(tdf['detflux'].iloc[exp_idx])
                    mdetflux = np.mean(tdf['detflux'].iloc[exp_idx])
                    count = mdetflux * np.median(tdf['exptime'].iloc[exp_idx])
                    print('mdetflux, count', mdetflux, count)
                    print('detflux_std, mdetflux_err',
                          detflux_std, mdetflux_err)
                    #if detflux_std > 10*mdetflux_err:
                    #    print('unstable')
                    #    mdetflux = np.NAN
                    #    count = np.NAN
                    detfluxes.append(mdetflux)
                    counts.append(count)
                print('detfluxes, counts', detfluxes, counts)
                uoes = np.asarray(uoes)
                detfluxes = np.asarray(detfluxes)
                counts =  np.asarray(counts)
                flux_idx = np.flatnonzero(uoes <= sx694.max_accurate_exposure)
                flux = np.mean(detfluxes[flux_idx])
                exposure_correct = counts[1:]/flux - uoes[1:]
                exposure_correct = exposure_correct.tolist()
                print('exposure_correct, exposure_corrects', exposure_correct, exposure_corrects)
                #print('tolist', exposure_correct.tolist())
                exposure_corrects.extend(exposure_correct)
                exposure_correct_jds.append(np.mean(tdf['jd']))
        
                #tdf = tdf.where(np.isnan(tdf['oexptime']), tdf['exptime'])
                #print('tdf', tdf)
                ## Retrieve original exposure time
                #oexptimes = tdf['oexptime'].to_numpy()
                #print(oexptimes)
                ##oexptime_idx = np.flatnonzero(oexptimes is not None)
                ##print(oexptime_idx)
                #np.isnan(oexptimes)
    print('jds', exposure_correct_jds)
    print('exposure_corrects', exposure_corrects)

    ax.tick_params(reset = True, which='both', direction='inout',
                   bottom=True, top=True, left=True, right=True)
    plt.xlabel('airmass')
    #plt.savefig((outbase + '_.png'), transparent=True)
    if show:
        plt.show()
    plt.close()


plt.plot(exposure_correct_jds, exposure_corrects, 'k.')
plt.ylabel('Exposure correction (s)')
plt.show()
