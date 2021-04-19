"""Play with photometry for calibration sources"""

import glob

import numpy as np
from numpy.polynomial import Polynomial

import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import pandas as pd

from astropy import log
from astropy import units as u
from astropy.time import Time
from astropy.convolution import Gaussian2DKernel, convolve
from astropy.stats import gaussian_fwhm_to_sigma, sigma_clipped_stats
from astropy.stats import mad_std, biweight_location

from photutils import make_source_mask
#from photutils import detect_threshold
from photutils import detect_sources
from photutils import deblend_sources
from photutils import source_properties
from photutils.background import Background2D

from bigmultipipe import no_outfile, prune_pout

from cormultipipe import (CorMultiPipe, Calibration, 
                          nd_filter_mask, mask_nonlin_sat)
import sx694

def photometry_process(ccd,
                       bmp_meta=None,
                       in_name=None,
                       seeing=5,
                       min_ND_multiple=1,
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

    #print(f'background_median = {back.background_median}, background_rms_median = {back.background_rms_median}')
    
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
        log.warning(f'No sources found: {in_name}')
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
             'date_obs': tm,
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
directory = '/data/io/IoIO/raw/20210310/' # 2.34 lots of measurements
#directory = '/data/io/IoIO/raw/20210307/' # 2.3 -- 2.6
#directory = '/data/io/IoIO/raw/20201011/' # Crumy
#directory = '/data/io/IoIO/raw/20201010/'  # 1.85 -- 1.90 ~very good

glob_include = ['HD*']
#glob_include = ['HD*118648*SII_on*']
#glob_include = ['HD*SII_on*']
#glob_include = ['HD*-R.fts', 'HD*-R_*.fts']

# Pythonic way of checking for a non-assigned variable
try:
    pout
except NameError:
    # Avoid confusing exception
    pout = None
    
if pout is None:
    # Run our pipeline code
    flist = []
    for gi in glob_include:
        flist += glob.glob(os.path.join(directory, gi))

    c = Calibration(start_date='2020-01-01', stop_date='2021-12-31',
                    reduce=True)
    cmp = CorMultiPipe(auto=True, calibration=c, num_processes=12,
                       post_process_list=[nd_filter_mask,
                                          photometry_process,
                                          no_outfile],
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
# Reasonable maximum length of time between individual exposures in a
# photometry measurement
photometry_time_window = 1*u.min
# Factor to multiply as shown in code
#stability_factor = 10
stability_factor = np.inf
min_airmass_points = 3


objects = list(set(df['object']))
filters = list(set(df['filter']))
# Collect extinction data into an array of dicts
extinction_data = []
# Collect exposure correction measurements for the whole night in one
# array, since this is independent of object and filter
exposure_corrects = []
exposure_correct_plot_dates = []
for object in objects:
    # Each object has its own flux which we measure at different
    # airmasses and through different filters
    objdf = df[df['object'] == object]
    # --> might want to move these in case there are some bad
    # --> measurements skewing the endpoints
    min_airmass = np.amin(objdf['airmass'])
    max_airmass = np.amax(objdf['airmass'])
    f = plt.figure(figsize=[8.5, 11])
    plt.suptitle(f'{directory}      {object}')
    for ifilt, filt in enumerate(filters):
        # Collect fluxes and airmasses measured throughout the night
        # for this object and filter
        filtdf = objdf[objdf['filter'] == filt]
        n_meas = len(filtdf.index) 
        if n_meas < 3:
            log.warning(f'Not enough measurements ({n_meas}) in filter {filt} for object {object} in {directory}')
            continue
        true_fluxes = []
        instr_mags = []
        airmasses = []

        filtdf = filtdf.sort_values(by=['jd'])
        # Code from cormultipipe.bias_dark_fdict_creator to find
        # jumps in time
        jds1 = filtdf['jd'].iloc[1:].to_numpy()
        jds0 = filtdf['jd'].iloc[0:-1].to_numpy()
        deltas = jds1 - jds0
        # This didn't work for some annoying reason
        #deltas = filtdf['jd'].iloc[1:] - filtdf['jd'].iloc[0:-1]#)#*u.day
        #deltas = filtdf['jd'] - filtdf['jd']
        #print('deltas', deltas)
        #print(filtdf['date_obs'])
        #deltats = filtdf['date_obs'][1:] - filtdf['date_obs'][0:-1]#)#*u.day
        # Group measurements in chunks recorded over < 10 minutes
        jump = np.flatnonzero(deltas*u.day > photometry_time_window)
        tslices = np.append(0, jump+1)
        tslices = np.append(tslices, None)
        #print('tslices', tslices)
        for it in range(len(tslices)-1):
            # For each distinct measurement time
            tdf = filtdf[slice(tslices[it], tslices[it+1])]
            n_meas = len(tdf.index) 
            if n_meas < 3:
                log.warning(f"Not enough measurements ({n_meas}) in filter {filt} for object {object} in {directory} from {tdf['date_obs'].iloc[0]} to {tdf['date_obs'].iloc[-1]}")
                continue
            # Collect oexptimes to monitor exposure_correct
            oexptimes = tdf['oexptime'].to_numpy()
            exptimes = tdf['exptime'].to_numpy()
            if np.all(oexptimes == None):
                oexptimes = exptimes
            else:
                # pands.dataframe.to_numpy() outputs NAN for None when
                # some of the elements are None and some aren't
                oexptimes = np.where(np.isnan(oexptimes), exptimes, oexptimes)
            uoes = np.unique(oexptimes)
        
            valid_uoes = []
            detfluxes = []
            counts = []
            for uoe in uoes:
                # Have many nights with 3 measurements per exposure time

                # For each unique oexptime
                exp_idx = np.flatnonzero(oexptimes == uoe)
                n_meas = len(exp_idx)
                if n_meas < 3:
                    log.warning(f"Not enough measurements ({n_meas}) in filter {filt} for object {object} in {directory} for exposure time {uoe} from {tdf['date_obs'].iloc[0]} to {tdf['date_obs'].iloc[-1]}")
                    continue
                # Thow out the first exposure
                exp_idx = exp_idx[1:]
                #print(f"{tdf[['date_obs','detflux', 'background_median', 'obj_to_ND']].iloc[exp_idx]}")# {tdf['detflux'].iloc[exp_idx]}")
                mdetflux_err = np.mean(tdf['detflux_err'].iloc[exp_idx])
                detflux_std = np.std(tdf['detflux'].iloc[exp_idx])
                mdetflux = np.mean(tdf['detflux'].iloc[exp_idx])
                    
                #mdetflux_err = np.median(tdf['detflux_err'].iloc[exp_idx])
                #detflux_std = np.std(tdf['detflux'].iloc[exp_idx])
                #mdetflux = np.mean(tdf['detflux'].iloc[exp_idx])

                count = mdetflux * uoe


                if detflux_std > stability_factor*mdetflux_err:
                    log.warning(f"Unstable atmosphere detflux_std = {detflux_std} > {stability_factor} * {mdetflux_err} for filter {filt} object {object} in {directory} from {tdf['date_obs'].iloc[0]} to {tdf['date_obs'].iloc[-1]}")
                    mdetflux = np.NAN
                    count = np.NAN

                valid_uoes.append(uoe) 
                detfluxes.append(mdetflux)
                counts.append(count)

            valid_uoes = np.asarray(valid_uoes)
            detfluxes = np.asarray(detfluxes)
            counts =  np.asarray(counts)
            # Before precise calibration of exposure_correct, flux is
            # only reliable for exposure times <=
            # sx694.max_accurate_exposure
            true_flux_idx = np.flatnonzero(valid_uoes
                                           <= sx694.max_accurate_exposure)
            if len(true_flux_idx) == 0:
                log.warning(f"No good measurements at exposure times <= sx694.max_accurate_exposure for filter {filt} object {object} in {directory} from {tdf['date_obs'].iloc[0]} to {tdf['date_obs'].iloc[-1]}")
                continue
            true_flux = np.nanmean(detfluxes[true_flux_idx])
            if np.isnan(true_flux):
                log.warning(f"No good flux measurements left for filter {filt} object {object} in {directory} from {tdf['date_obs'].iloc[0]} to {tdf['date_obs'].iloc[-1]}")
                continue

            airmasses.append(np.mean(tdf['airmass']))
            true_fluxes.append(true_flux)
            instr_mag = u.Magnitude(true_flux*u.electron/u.s)
            instr_mags.append(instr_mag.value)
            

            if (len(valid_uoes) > 1
                and np.min(valid_uoes) <= sx694.max_accurate_exposure
                and np.max(valid_uoes) > sx694.max_accurate_exposure):
                # We have some exposures in this time interval
                # straddling the max_accurate_exposure time, so we can
                # calculate exposure_correct.  We may have more than
                # one, so calculate for each individually
                ec_idx = np.flatnonzero(valid_uoes
                                        > sx694.max_accurate_exposure)
                exposure_correct = counts[ec_idx]/true_flux - valid_uoes[ec_idx]
                exposure_correct = exposure_correct.tolist()
                exposure_corrects.extend(exposure_correct)
                date_obs = tdf['date_obs'].iloc[ec_idx].tolist()
                plot_dates = [d.plot_date for d in date_obs]
                exposure_correct_plot_dates.extend(plot_dates)

        # Having collected fluxes for this object and filter over the
        # night, fit mag vs. airmass to get top-of-the-atmosphere
        # magnitude and extinction coef
        if true_fluxes is None:
            log.warning(f"No good flux measurements for filter {filt} object {object} in {directory}")

        #print('airmasses, true_fluxes, instr_mags', airmasses, true_fluxes, instr_mags)

        # Create strip-chart style plot for each filter
        ax = plt.subplot(9, 1, ifilt+1)
        ax.tick_params(which='both', direction='inout',
                       bottom=True, top=True, left=True, right=True)
        ax.set_xlim([min_airmass, max_airmass])
        #plt.plot(airmasses, true_fluxes, 'k.')
        plt.plot(airmasses, instr_mags, 'k.')
        plt.ylabel(filt)

        num_airmass_points = len(airmasses)
        if num_airmass_points < min_airmass_points:
            log.warning(f"not enough points ({num_airmass_points}) for extinction measurement {filtdf['date'].iloc[0]} filter {filt}")
            continue
        am_range = [min(airmasses), max(airmasses)]
        # Fit a line to airmass vs. instr_mags
        poly = Polynomial.fit(airmasses, instr_mags, deg=1)
        xfit, yfit = poly.linspace()
        plt.plot(xfit, yfit)
        instr_mag_am0 = poly(0)
        extinction = poly.deriv()
        extinction = extinction(0)
        airmasses = np.asarray(airmasses)
        dof = num_airmass_points - 2
        red_chisq = np.sum((poly(airmasses) - instr_mags)**2) / dof
        plt.text(0.5, 0.8, 
                 f'Top of atm. instr mag = {instr_mag_am0:.2f} (electron/s)',
                 ha='center', va='bottom', transform=ax.transAxes)

        plt.text(0.5, 0.1, 
                 f'Extinction = {extinction:.3f} (mag/airmass)',
                 ha='center', transform=ax.transAxes)
        plt.text(0.13, 0.8, 
                 f'Red. Chisq = {red_chisq:.4f}',
                     ha='center', va='bottom', transform=ax.transAxes)

        extinction_dict = {'date': filtdf['date'].iloc[0],
                           'object': object,
                           'filter': filt,
                           'instr_mag_am0': instr_mag_am0,
                           'extinction': extinction,
                           'am_range': am_range,
                           'num_airmass_points ': num_airmass_points,
                           'red_chisq': red_chisq}

        extinction_data.append(extinction_dict)

    # Finish our strip-chart plot for this object
    # --> We will want to store the data to a CSV or something
    ax.tick_params(reset = True, which='both', direction='inout',
                   bottom=True, top=True, left=True, right=True)
    plt.xlabel('airmass')
    #plt.savefig((outbase + '_.png'), transparent=True)
    if show:
        plt.show()
    plt.close()

# Plot our 
# --> figure out a way to store these data
biweight_exposure_correct = biweight_location(exposure_corrects,
                                              ignore_nan=True)
mad_std_exposure_correct = mad_std(exposure_corrects,
                                   ignore_nan=True)
print(biweight_exposure_correct, mad_std_exposure_correct)
ax = plt.subplot()
plt.plot_date(exposure_correct_plot_dates, exposure_corrects, 'k.')
plt.axhline(biweight_exposure_correct, color='red')
plt.axhline(biweight_exposure_correct-mad_std_exposure_correct,
            linestyle='--', color='k', linewidth=1)
plt.axhline(biweight_exposure_correct+mad_std_exposure_correct,
            linestyle='--', color='k', linewidth=1)
plt.text(0.5, biweight_exposure_correct + 0.1*mad_std_exposure_correct, 
         f'{biweight_exposure_correct:.2f} +/- {mad_std_exposure_correct:.2f}',
         ha='center', transform=ax.get_yaxis_transform())

plt.ylabel('Exposure correction (s)')
plt.gcf().autofmt_xdate()  # orient date labels at a slant
if show:
    plt.show()
plt.close()
