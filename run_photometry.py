#!/usr/bin/python3

"""Run standard star photometry pipeline for the IoIO coronagraph"""

# Avoid circular reference of [Red]CorData and photometry pipeline by separating out the Photometry class

import gc
import os
import glob
import pickle
import csv
import argparse
from pathlib import Path

import numpy as np
from numpy.polynomial import Polynomial

import matplotlib.pyplot as plt
from matplotlib.dates import datestr2num

import pandas as pd

from astropy import log
from astropy import units as u
from astropy.time import Time
from astropy.stats import mad_std, biweight_location

# bmp_cleanup can go away now that Photometry is an object
from bigmultipipe import bmp_cleanup, no_outfile, prune_pout

import sx694
from cormultipipe import IoIO_ROOT, RAW_DATA_ROOT
from cormultipipe import assure_list, reduced_dir, get_dirs_dates
from cormultipipe import RedCorData, CorMultiPipe, Calibration
from cormultipipe import nd_filter_mask, mask_nonlin_sat

from photometry import Photometry, is_flux

PHOTOMETRY_ROOT = os.path.join(IoIO_ROOT, 'StandardStar')

#def kernel_process(ccd,
#                   seeing=5,
#                   bmp_meta=None,
#                   **kwargs):
#    sigma = seeing * gaussian_fwhm_to_sigma
#    kernel = Gaussian2DKernel(sigma)
#    kernel.normalize()
#    bmp_meta['kernel'] = kernel
#    ccd = bmp_cleanup(ccd, bmp_meta=bmp_meta, add='kernel')
#    return ccd
#    
#def source_mask_process(ccd,
#                        bmp_meta=None,
#                        show_mask=False,
#                        source_mask_dilate=11,
#                        **kwargs):
#    kernel = bmp_meta.get('kernel')
#    if kernel is None:
#        ccd = kernel_process(ccd, bmp_meta=bmp_meta, **kwargs)
#        kernel = bmp_meta.get('kernel')
#    # Make a source mask to enable optimal background estimation
#    mask = make_source_mask(ccd.data, nsigma=2, npixels=5,
#                            filter_kernel=kernel, mask=ccd.mask,
#                            dilate_size=source_mask_dilate)
#    if show_mask:
#        impl = plt.imshow(mask, origin='lower',
#                          cmap=plt.cm.gray,
#                          filternorm=0, interpolation='none')
#        plt.show()
#    bmp_meta['source_mask'] = mask
#    ccd = bmp_cleanup(ccd, bmp_meta=bmp_meta, add='source_mask')
#    return ccd
#
#def background_process(ccd,
#                       bmp_meta=None,
#                       in_name=None,
#                       **kwargs):
#    # Make sure our dependencies have run.  kernel comes along for
#    # free with source_mask
#    source_mask = bmp_meta.get('source_mask')
#    if source_mask is None:
#        ccd = source_mask_process(ccd, bmp_meta=bmp_meta, **kwargs)
#        source_mask = bmp_meta.get('source_mask')
#    kernel = bmp_meta['kernel']
#    
#    box_size = int(np.mean(ccd.shape) / 10)
#    back = Background2D(ccd, box_size, mask=source_mask, coverage_mask=ccd.mask)
#
#    return ccd
#    
#def source_catalog_process(ccd,
#                           bmp_meta=None,
#                           in_name=None,
#                           **kwargs):
#    """CorMultiPipe post-processing routine to create a `~astropy.photutils.SourceCatalog`
#
#    If no sources are found, a warning is logged, the return value is
#    `None` and bmp_meta is set to {}.
#
#    NOTE: the `~astropy.photutils.SourceCatalog` is constructed to
#    (hopefully) have the correct units for all Quantities, in
#    particular `segment_flux.`  That means this and related Quantities
#    will track the units of the `ccd` that was passed to this object
#
#    This should be used in concert with another function that extracts
#    the desired information from the
#    `~astropy.photutils.SourceCatalog`.  Then
#    `source_catalog_cleanup` should be used for the reasons stated
#    in its docstring
#
#    Parameters
#    ----------
#    --> consider making (one or more) plot keyword(s)
#
#    """
#    # Make sure our dependencies have run.  kernel comes along for
#    # free with source_mask
#    source_mask = bmp_meta.get('source_mask')
#    if source_mask is None:
#        ccd = source_mask_process(ccd, bmp_meta=bmp_meta, **kwargs)
#        source_mask = bmp_meta.get('source_mask')
#    kernel = bmp_meta['kernel']
#
#    # This is going to expand by a factor of 15 for default kernel
#    # with seeing=5
#    
#    ##mean, median, std = sigma_clipped_stats(data, sigma=3.0, mask=mask)
#    #
#    
#    box_size = int(np.mean(ccd.shape) / 10)
#    back = Background2D(ccd, box_size, mask=source_mask, coverage_mask=ccd.mask)
#
#
#
#
#    threshold = back.background + (2.0* back.background_rms)
#
#    #print(f'background_median = {back.background_median}, background_rms_median = {back.background_rms_median}')
#    
#    #impl = plt.imshow(back.background, origin='lower',
#    #                  cmap=plt.cm.gray,
#    #                  filternorm=0, interpolation='none')
#    #back.plot_meshes()
#    #plt.show()
#    
#    npixels = 5
#    segm = detect_sources(ccd.data*ccd.unit, threshold, npixels=npixels,
#                          filter_kernel=kernel, mask=ccd.mask)
#    
#    if segm is None:
#        # detect_sources logs a WARNING and returns None if no sources
#        # are found.  Treat this as a fatal error
#        log.warning(f'No sources found: {in_name}')
#        bmp_meta.clear()
#        return None
#
#    # It does save a little time and a factor ~1.3 in memory if we
#    # don't deblend
#    segm_deblend = deblend_sources(ccd.data, segm, 
#                                   npixels=npixels,
#                                   filter_kernel=kernel, nlevels=32,
#                                   contrast=0.001)
#    
#    #impl = plt.imshow(segm, origin='lower',
#    #                  cmap=plt.cm.gray,
#    #                  filternorm=0, interpolation='none')
#    #plt.show()
#
#    # https://photutils.readthedocs.io/en/stable/segmentation.html
#
#
#    # Very strange this is not working!
#    #print(ccd.unit, back.background_rms.unit, effective_gain.unit)
#    #inputs = [ccd, back.background_rms, effective_gain]
#    #has_unit = [hasattr(x, 'unit') for x in inputs]
#    #use_units = all(has_unit)
#    #print(has_unit)
#    #print(use_units)
#    #use_units = np.all(has_unit)
#    #print(use_units)
#
#    # As per calc_total_error documentation, effective_gain converts
#    # ccd.data into count-based units, so it is exptime when we have
#    # flux units
#    # --> Eventually get units into this properly with the
#    # CardQuantity stuff I have been working on, or however that turns
#    # out, for now assume everything is in seconds
#    if is_flux(ccd.unit):
#        effective_gain = ccd.meta['EXPTIME']*u.s
#    else:
#        effective_gain = 1*u.s
#
#    ###try:
#    ###    #t = ccd.data*ccd.unit
#    ###    #print(f'ccd.data*ccd.unit unit {t.unit} {back.background_rms} {effective_gain.unit}')
#    ###    error = calc_total_error(ccd.data,
#    ###                             back.background_rms,
#    ###                             effective_gain)
#    ###    print(f'success: {in_name}')
#    ###except Exception as e:
#    ###    print(f'fail {e} for {in_name}')
#    ###    error = calc_total_error(ccd.data,
#    ###                             back.background_rms,
#    ###                             effective_gain) 
#
#    # Use the advertised error, which is supposed to be a bit fluffier
#    #sc = SourceCatalog(ccd.data, segm_deblend, 
#    #                   error=ccd.uncertainty.array,
#    #                   mask=ccd.mask)
#
#    #sc = SourceCatalog(ccd.data, segm_deblend, 
#    #                   error=error,
#    #                   mask=ccd.mask)
#
#
#    #total_error = np.sqrt(back.background_rms.value**2 + ccd.uncertainty.array)
#    #compare = total_error - error
#    #print(f'max val {np.max(ccd.uncertainty.array):.2f} max difference {np.max(np.abs(compare)):.2f} {in_name}')
#    #sc = SourceCatalog(ccd.data, segm_deblend, 
#    #                   error=total_error,
#    #                   mask=ccd.mask)
#
#    if ccd.uncertainty is None:
#        log.warning(f'photometry being conducted on ccd data with no uncertainty.  Is this file being properly reduced? {in_name}')
#        if ccd.unit == u.adu:
#            log.warning(f'File is still in adu, cannot calculate proper '
#                        f'Poisson error for sources {in_name}')
#            total_error = np.zeros_like(ccd)*u.adu
#        else:
#            total_error = calc_total_error(ccd,
#                                           back.background_rms.value,
#                                           effective_gain) 
#    else:
#        uncert = ccd.uncertainty.array*ccd.unit
#        if ccd.uncertainty.uncertainty_type == 'std':
#            total_error = np.sqrt(back.background_rms**2 + uncert**2)
#        elif ccd.uncertainty.uncertainty_type == 'var':
#            # --> if I work in units make var units ccd.unit**2
#            var  = uncert*ccd.unit
#            total_error = np.sqrt(back.background_rms**2 + var)
#        else:
#            raise ValueError(f'Unsupported uncertainty type {ccd.uncertainty.uncertainty_type} for {in_name}')
#        
#    ###compare = total_error - error
#    ###print(f'max val {np.max(ccd.uncertainty.array):.2f} max difference {np.max(np.abs(compare)):.2f} {in_name}')
#    # Call to SourceCatalog is a little awkward because it expects a Quantity
#    sc = SourceCatalog(ccd.data*ccd.unit, segm_deblend, 
#                       error=total_error,
#                       mask=ccd.mask)
#    bmp_meta['source_catalog'] = sc
#    ccd = bmp_cleanup(ccd, bmp_meta=bmp_meta, add='source_catalog')
#    return ccd

def standard_star_process(ccd,
                          bmp_meta=None,
                          in_name=None,
                          min_ND_multiple=1,
                          photometry=None,
                          **kwargs):
    """CorMultiPipe post_processing routine to collect standard star photometry.  
    NOTE: Star is assumed to be the brightest in the field.  

    NOTE: If this is adapted for other photometric processing,
    particularly those involving ephemerides, DATE-AVG should be
    recorded in the bmp_meta.  It was not recorded here because the
    information needed to generate it is derived from the results of
    this routine.

    min_ND_multiple : number

        Number of ND filter *widths* away from the *center* of the ND
        filter that the source needs to be to be considered good.
        Note that expansion by `cormultipipe.ND_EDGE_EXPAND` is not
        considered in this calculation, but the ND filter is likely to
        be masked using that quantity if `cormultipipe.nd_filter_mask`
        has been called in the `cormultipipe.post_process_list`


    photometry : Photometry object
        Used to calculate SourceCatalog
    """

    # Fundamentally, we are going to work from a source catalog table,
    # but we need to start from a SourceCatalog object.  The 
    # Photometry object passed into our pipeline will help us do that
    # in a way that naturally disposes of the large intermediate
    # images when we are done with the pipeline.

    if bmp_meta is None:
        bmp_meta = {}
    if photometry is None:
        photometry = Photometry(**kwargs)
    photometry.ccd = ccd
    sc = photometry.source_catalog
    if sc is None:
        log.error(f'No source catalog: {in_name}')
        bmp_meta.clear()
        return None
    
    tbl = sc.to_table()
    
    tbl.sort('segment_flux', reverse=True)

    # Reject sources with saturated pixels
    # --> This unit will get better if Card units are improved
    nonlin = ccd.meta['NONLIN']*ccd.unit
    max_val = tbl['max_value'][0]
    if max_val >= nonlin:
        log.warning(f'Too bright: {in_name}')
        # Don't forget to clear out the meta
        bmp_meta.clear()
        return None
    
    # tbl['xcentroid'].info.format = '.2f'  # optional format
    # tbl['ycentroid'].info.format = '.2f'
    # tbl['cxx'].info.format = '.2f'
    # tbl['cxy'].info.format = '.2f'
    # tbl['cyy'].info.format = '.2f'
    # tbl['gini'].info.format = '.2f'
    # print(tbl)
    # print(tbl['segment_flux'])
    # 
    # print(ccd.meta['AIRMASS'])
    
    # What I really want is source sum and airmass as metadata put into header and
    # http://classic.sdss.org/dr2/algorithms/fluxcal.html
    # Has some nice formulae:
    #     aa = zeropoint
    #     kk = extinction coefficient
    #     airmass
    # f/f0 = counts/exptime * 10**(0.4*(aa + kk * airmass))
    
    # These are in pixels, but are not Quantities
    xcentrd = tbl['xcentroid'][0]
    ycentrd = tbl['ycentroid'][0]
    ccd.obj_center = (ycentrd, xcentrd)
    ccd.quality = 10
    ND_edges = ccd.ND_edges(ycentrd)
    min_ND_dist = min_ND_multiple * (ND_edges[1] - ND_edges[0])
    if ccd.obj_to_ND < min_ND_dist:
        log.warning(f'Too close: ccd.obj_to_ND = {ccd.obj_to_ND} {in_name}')
        bmp_meta.clear()
        return None
    center = np.asarray(ccd.shape)/2
    radius = ((xcentrd-center[1])**2 + (ycentrd-center[0])**2)**0.5

    airmass = ccd.meta['AIRMASS']
    # --> This might get better with Card units, but is implicitly in
    # s for now, preparing for any weird exptime units in the future
    exptime = ccd.meta['EXPTIME']*u.s
    # Guarantee exposure time always reads in s
    exptime = exptime.decompose()
    # Prepare to measure our exposure time gap problem by recording
    # the old exptime and calculating the resulting (incorrect)
    # detector flux.  exptime and oexptime will help work back and
    # forth as needed from flux to counts in standard_star_directory
    # Note that if we have prepared the source catalog with detflux
    # processed images, the exposure time uncertainty will be
    # incorporated.  Otherwise we need to compute it ourselves
    oexptime = ccd.meta.get('oexptime') # Might be None
    detflux = tbl['segment_flux'][0]
    assert isinstance(detflux, u.Quantity), ('SourceCatalog was not prepared properly')
    detflux_err = tbl['segment_fluxerr'][0]
    if not is_flux(detflux.unit):
        # Work in flux units
        detflux /= exptime
        detflux_err /= exptime
        exptime_uncertainty = ccd.meta.get('EXPTIME-UNCERTAINTY')
        if exptime_uncertainty is not None:
            detflux_err *= np.sqrt((detflux_err/detflux)**2
                                   + (exptime_uncertainty/exptime.value)**2)
    if oexptime is None:
        odetflux = None
        odetflux_err = None
    else:
        # Exptime might actually be close to right, so convert our
        # detflux, which is in proper flux units calculated above,
        # back into incorrect odetflux.  The error is unchanged
        # because all we have to work with is the error we found in
        # our ccd.meta.  Prepare here for dataframe by working as
        # scalars, not Quantities, since odetflux may be None and
        # can't be converted spontaneously like other Quantities
        odetflux = detflux.value * exptime.value / oexptime
        odetflux_err = detflux_err * exptime.value / oexptime
    
    # --> These will get better when Card units are implemented
    ccd.meta['DETFLUX'] = (detflux.value, f'({detflux.unit.to_string()})')
    ccd.meta['HIERARCH DETFLUX-ERR'] = (detflux_err.value,
                                        f'({detflux.unit.to_string()})')
    ccd.meta['xcentrd'] = xcentrd
    ccd.meta['ycentrd'] = ycentrd
    ccd.meta['radius'] = radius    

    date_obs = ccd.meta['DATE-OBS']
    just_date, _ = date_obs.split('T')
    tm = Time(date_obs, format='fits')

    max_frac_nonlin = max_val/nonlin

    # We are going to turn this into a Pandas dataframe, which does
    # not do well with units, so just return everything
    tmeta = {'object': ccd.meta['OBJECT'],
             'filter': ccd.meta['FILTER'],
             'date': just_date,
             'date_obs': tm,
             'jd': tm.jd,
             'airmass': airmass,
             'objalt': ccd.meta['OBJCTALT'],
             'exptime': exptime.value,
             'oexptime': oexptime, # Already value or None
             'detflux': detflux.value,
             'detflux_err': detflux_err.value,
             'odetflux': odetflux,
             'odetflux_err': odetflux_err,
             # exptime will always be in s, but detflux may be in
             # photons or electrons or whatever
             'detflux_units': detflux.unit.to_string(),
             'xcentroid': xcentrd,
             'ycentroid': ycentrd,
             'max_frac_nonlin': max_frac_nonlin.value,
             'obj_to_ND': ccd.obj_to_ND,
             'radius': radius}
             # --> background stuff will take more work:
             # https://photutils.readthedocs.io/en/stable/segmentation.html
             #'background_mean': tbl['background_mean'][0]}

    bmp_meta['standard_star'] = tmeta
    return ccd
    
def standard_star_pipeline(directory,
                           glob_include=None,
                           calibration=None,
                           photometry=None,
                           num_processes=None,
                           read_pout=False,
                           write_pout=False,
                           outdir_root=PHOTOMETRY_ROOT,
                           fits_fixed_ignore=False,
                           **kwargs): 
    """
    Parameters
    ----------
    directory : str
        Directory to process

    glob_include : list of 

    read_pout : str or bool
        See write_pout.  If file read, simply return that without
        running pipeline.  Default is ``False``

    write_pout : str or bool
        If str, full filename to write pickled pout to.  If True,
        write to 'standard_star.pout' in `directory`.  Default is ``False``

    **kwargs passed on to Photometry and CorMultiPipe

    """

    rd = reduced_dir(directory, outdir_root, create=False)
    default_outname = os.path.join(rd, 'standard_star.pout')
    if read_pout is True:
        read_pout = default_outname
    if isinstance(read_pout, str):
        try:
            pout = pickle.load(open(read_pout, "rb"))
            return (directory, pout)
        except:
            log.debug(f'running code because file not found: {read_pout}')
        
    if glob_include is None:
        glob_include = ['HD*']
    glob_include = assure_list(glob_include)

    if calibration is None:
        calibration = Calibration(reduce=True)
    if photometry is None:
        photometry = Photometry(precalc=True, **kwargs)

    flist = []
    for gi in glob_include:
        flist += glob.glob(os.path.join(directory, gi))

    if len(flist) == 0:
        return (directory, [])

    cmp = CorMultiPipe(auto=True,
                       calibration=calibration,
                       photometry=photometry,
                       post_process_list=[nd_filter_mask,
                                          standard_star_process,
                                          no_outfile],
                       fits_fixed_ignore=fits_fixed_ignore, 
                       num_processes=num_processes,
                       process_expand_factor=15,
                       **kwargs)
    # Pipeline is set with no_outfile so it won't produce any files,
    # but get ready to write to reduced directory if necessary
    pout = cmp.pipeline(flist, outdir=rd, overwrite=True)
    pout, _ = prune_pout(pout, flist)
    if write_pout is True:
        write_pout = default_outname
    if isinstance(write_pout, str):
        os.makedirs(os.path.dirname(write_pout), exist_ok=True)
        pickle.dump(pout, open(write_pout, "wb"))
    return (directory, pout)

def exposure_correct_plot(exposure_correct_data,
                          show=False,
                          outname=None,
                          latency_change_dates=None):
    if len(exposure_correct_data) == 0:
        return
    latency_change_dates = assure_list(latency_change_dates)
    latency_change_dates =  ['1000-01-01'] + latency_change_dates
    latency_changes = [datestr2num(d) for d in latency_change_dates]
    latency_changes += [np.inf] # optimistic about end date of IoIO observations
    plot_dates = [ecd['plot_date'] for ecd in exposure_correct_data]
    exposure_corrects = [ecd['exposure_correct']
                         for ecd in exposure_correct_data]
    plot_dates = np.asarray(plot_dates)
    exposure_corrects = np.asarray(exposure_corrects)

    # Plot our exposure correction data
    f = plt.figure()
    ax = plt.subplot()
    plt.plot_date(plot_dates, exposure_corrects, 'k.')

    # Fit a constant offset to each segment in which we have a
    # constant latency
    for iseg in range(len(latency_changes)-1):
        sidx = np.flatnonzero(
            np.logical_and(latency_changes[iseg] <= plot_dates,
                           plot_dates < latency_changes[iseg+1]))
        if len(sidx) == 0:
            continue
        mindate = min(plot_dates[sidx])
        maxdate = max(plot_dates[sidx])
        biweight_exposure_correct = biweight_location(exposure_corrects[sidx],
                                                      ignore_nan=True)
        mad_std_exposure_correct = mad_std(exposure_corrects[sidx],
                                           ignore_nan=True)
        plt.plot_date([mindate, maxdate],
                      [biweight_exposure_correct]*2, 'r-')

        plt.plot_date([mindate, maxdate],
                      [biweight_exposure_correct-mad_std_exposure_correct]*2,
                      'k--')
        plt.plot_date([mindate, maxdate],
                      [biweight_exposure_correct+mad_std_exposure_correct]*2,
                      'k--')

        plt.text((mindate + maxdate)/2,
                 0.5*biweight_exposure_correct, 
                 f'{biweight_exposure_correct:.2f} +/- {mad_std_exposure_correct:.2f}',
                 ha='center')#, transform=ax.get_yaxis_transform())

    plt.ylabel('Exposure correction (s)')
    ax.set_ylim([0, 5])
    plt.gcf().autofmt_xdate()  # orient date labels at a slant
    if outname is not None:
        plt.savefig(outname, transparent=True)
    if show:
        plt.show()
    plt.close()
    

def standard_star_directory(directory,
                            read_csvs=False,
                            write_csvs=True,
                            pout=None,
                            read_pout=True,
                            write_pout=True,
                            show=False,
                            photometry_time_window=1*u.min,
                            stability_factor=np.inf,
                            min_airmass_points=3,
                            outdir_root=PHOTOMETRY_ROOT,
                            **kwargs):
    """
    Parameters
    ----------
    directory : str
        Directory to process

    pout : list 
        Output of previous pipeline run if available as a variable.
        Use `read_pout` keyword of underlying
        :func:`standard_star_pipeline` to read previous run off of disk

    read_csvs : bool
        If True and output CSVs exist, don't do calculations.  If
        tuple, the two input CSV filenames, in the order
        (extinction.csv, exposure_correction.csv)

    read_pout, write_pout, see standard_star_process
        Defaults here are ``True`` -- just reuse previous pipeline output

    photometry_time_window: number 

        Reasonable maximum length of time between individual exposures
        in a photometry measurement

    stability_factor : number
        Higher number accepts more data (see code)

    """
    rd = reduced_dir(directory, outdir_root, create=False)
    default_extinction_outname = os.path.join(rd, 'extinction.csv')
    default_exposure_correction_outname = \
        os.path.join(rd, 'exposure_correction.csv')
    if read_csvs is True:
        read_csvs = (default_extinction_outname,
                     default_exposure_correction_outname)
    if isinstance(read_csvs, tuple):
        try:
            extinction_outname = read_csvs[0]
            exposure_correct_outname = read_csvs[1]
            extinction_data = []
            exposure_correct_data = []
            # The getsize calls allows 0-size files to just pass
            # through empty lists, so we don't have to re-run code to
            # find we have no results
            if os.path.getsize(extinction_outname) > 0:
                with open(extinction_outname, newline='') as csvfile:
                    csvr = csv.DictReader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
                    for row in csvr:
                        extinction_data.append(row)
            if os.path.getsize(exposure_correct_outname) > 0:
                with open(exposure_correct_outname, newline='') as csvfile:
                    csvr = csv.DictReader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
                    for row in csvr:
                        exposure_correct_data.append(row)
                if show:
                    # extinction plot is too complicated to extract at
                    # the moment
                    exposure_correct_plot(exposure_correct_data,
                                          show=show)
            log.debug(f'returning csvs from {directory}')
            return (directory, extinction_data, exposure_correct_data)
        except Exception as e:
            log.debug(f'running code because of exception {e}')

    if pout is None:
        directory, pout = standard_star_pipeline(directory,
                                                 read_pout=read_pout,
                                                 write_pout=write_pout,
                                                 outdir_root=outdir_root,
                                                 **kwargs)

    if len(pout) == 0:
        log.debug(f'no photometry measurements found in {directory}')
        return (directory, [], [])

    _ , pipe_meta = zip(*pout)
    standard_star_list = [pm['standard_star'] for pm in pipe_meta]
    df = pd.DataFrame(standard_star_list)
    just_date = df['date'].iloc[0]

    if write_csvs is True:
        write_csvs = (default_extinction_outname,
                      default_exposure_correction_outname)
    if isinstance(write_csvs, tuple):
        extinction_outname = write_csvs[0]
        exposure_correct_outname = write_csvs[1]
        os.makedirs(os.path.dirname(extinction_outname), exist_ok=True)        

    objects = list(set(df['object']))
    filters = list(set(df['filter']))

    # Collect extinction and exposure correction data  into arrays of dicts
    extinction_data = []
    exposure_correct_data = []
    for object in objects:
        # Each object has its own flux which we measure at different
        # airmasses and through different filters
        objdf = df[df['object'] == object]
        # --> might want to move these in case there are some bad
        # --> measurements skewing the endpoints
        min_airmass = np.amin(objdf['airmass'])
        max_airmass = np.amax(objdf['airmass'])
        min_tm = np.amin(objdf['date_obs'])
        max_tm = np.amax(objdf['date_obs'])
        min_mjd = min_tm.mjd
        max_mjd = max_tm.mjd
        min_date = min_tm.value
        max_date = max_tm.value
        _, min_time = min_date.split('T')
        _, max_time = max_date.split('T')
        min_time, _ = min_time.split('.')
        max_time, _ = max_time.split('.')
        f = plt.figure(figsize=[8.5, 11])
        plt.suptitle(f"{object} {just_date} {min_time} -- {max_time} instrument mags")
        valid_extinction_data = False
        for ifilt, filt in enumerate(filters):
            # Collect fluxes and airmasses measured throughout the night
            # for this object and filter
            filtdf = objdf[objdf['filter'] == filt]
            n_meas = len(filtdf.index) 
            if n_meas < 3:
                log.warning(f'Not enough measurements ({n_meas}) in filter {filt} for object {object} in {directory}')
                continue
            max_frac_nonlin = np.max(filtdf['max_frac_nonlin'])
            true_fluxes = []
            best_fluxes = []
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
                # Collect oexptimes to monitor exposure_correct.
                # To make bookkeeping easy, just put everything in
                # oexptimes and determine if there are multiple groups 
                oexptimes = tdf['oexptime'].to_numpy()
                exptimes = tdf['exptime'].to_numpy()
                # pands.dataframe.to_numpy() outputs NAN for None when
                # some of the elements are None and some aren't
                if np.all(oexptimes == None):
                    oexptimes = exptimes
                else:
                    oexptimes = np.where(np.isnan(oexptimes),
                                         exptimes, oexptimes)
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
                    #print(f"{tdf[['date_obs','detflux', 'background_median', 'obj_to_ND']].iloc[exp_idx]}")# {tdf['detflux'].iloc[exp_idx]}")
                    # Thow out the first exposure
                    exp_idx = exp_idx[1:]
                    mdetflux_err = np.mean(tdf['detflux_err'].iloc[exp_idx])
                    detflux_std = np.std(tdf['detflux'].iloc[exp_idx])
                    mdetflux = np.mean(tdf['detflux'].iloc[exp_idx])

                    #mdetflux_err = np.median(tdf['detflux_err'].iloc[exp_idx])
                    #detflux_std = np.std(tdf['detflux'].iloc[exp_idx])
                    #mdetflux = np.mean(tdf['detflux'].iloc[exp_idx])

                    # Multiply by the exposure time we divided by when
                    # calculating detflux
                    count = mdetflux * exptimes[exp_idx[0]]

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
                # Before precise calibration of exposure_correct, flux
                # is only reliable for exposure times <=
                # sx694.max_accurate_exposure.  After calibration, we
                # can do a reasonable job, but bookkeep it separately
                # as best_flux, since we use true_flux to check the
                # exposure_correction
                true_flux_idx = np.flatnonzero(valid_uoes
                                               <= sx694.max_accurate_exposure)
                if len(true_flux_idx) > 0:
                    best_flux_idx = true_flux_idx
                    true_flux = np.nanmean(detfluxes[true_flux_idx])
                else:
                    best_flux_idx = np.arange(len(detfluxes))
                    true_flux = None
                best_flux = np.nanmean(detfluxes[best_flux_idx])
                if np.isnan(best_flux):
                    log.warning(f"No good flux measurements left for filter {filt} object {object} in {directory} from {tdf['date_obs'].iloc[0]} to {tdf['date_obs'].iloc[-1]}")
                    continue

                #if len(true_flux_idx) == 0:
                #    log.warning(f"No good measurements at exposure times <= sx694.max_accurate_exposure for filter {filt} object {object} in {directory} from {tdf['date_obs'].iloc[0]} to {tdf['date_obs'].iloc[-1]}")
                #    continue
                #true_flux = np.nanmean(detfluxes[true_flux_idx])
                #if np.isnan(true_flux):
                #    log.warning(f"No good flux measurements left for filter {filt} object {object} in {directory} from {tdf['date_obs'].iloc[0]} to {tdf['date_obs'].iloc[-1]}")
                #    continue

                airmasses.append(np.mean(tdf['airmass']))
                best_fluxes.append(best_flux)
                instr_mag = u.Magnitude(best_flux*u.electron/u.s)
                instr_mags.append(instr_mag.value)


                if (true_flux is not None
                    and len(valid_uoes) > 1
                    and np.min(valid_uoes) <= sx694.max_accurate_exposure
                    and np.max(valid_uoes) > sx694.max_accurate_exposure):
                    # We have some exposures in this time interval
                    # straddling the max_accurate_exposure time, so we can
                    # calculate exposure_correct.  We may have more than
                    # one, so calculate for each individually
                    ec_idx = np.flatnonzero(valid_uoes
                                            > sx694.max_accurate_exposure)
                    exposure_corrects = (counts[ec_idx]/true_flux
                                        - valid_uoes[ec_idx])
                    exposure_corrects = exposure_corrects.tolist()
                    # --> This may be more efficiently done using julian2num
                    date_obs = tdf['date_obs'].iloc[ec_idx].tolist()
                    texposure_correct_data = \
                        [{'plot_date': d.plot_date,
                          'exposure_correct': exposure_correct}
                         for d, exposure_correct in
                         zip(date_obs, exposure_corrects)]
                    exposure_correct_data.extend(texposure_correct_data)

                    # for i in range(len(ec_idx)):
                    #     date_obs = tdf['date_obs'].iloc[ec_idx[i]]
                    #     exposure_correct_dict = \
                    #         {'plot_date': d.plot_date,
                    #          'exposure_correct': exposure_corrects[i]}
                             
                    # plot_dates = [d.plot_date for d in date_obs]
                    # exposure_correct_dicts = [{
                    # exposure_correct_data.extend(exposure_correct_dicts)
                    # exposure_corrects.extend(exposure_correct)
                    # exposure_correct_plot_dates.extend(plot_dates)

            # Having collected fluxes for this object and filter over the
            # night, fit mag vs. airmass to get top-of-the-atmosphere
            # magnitude and extinction coef
            valid_extinction_data = True
            if best_fluxes is None:
                log.warning(f"No good flux measurements for filter {filt} object {object} in {directory}")

            # Create strip-chart style plot for each filter
            ax = plt.subplot(9, 1, ifilt+1)
            ax.tick_params(which='both', direction='inout',
                           bottom=True, top=True, left=True, right=True)
            ax.set_xlim([min_airmass, max_airmass])
            plt.gca().invert_yaxis()
            plt.plot(airmasses, instr_mags, 'k.')
            plt.ylabel(filt)
            plt.text(0.87, 0.1, 
                     f'max_frac_nonlin = {max_frac_nonlin:.2f}',
                         ha='center', va='bottom', transform=ax.transAxes)

            num_airmass_points = len(airmasses)
            if num_airmass_points < min_airmass_points:
                log.warning(f"not enough points ({num_airmass_points}) for object {object} for extinction measurement {just_date} filter {filt}")
                continue

            # Fit a line to airmass vs. instr_mags
            # --> This might be a good place for iter_linfit?
            # In any case, throwing out bad points would be good
            # --> I will eventually want to extract uncertainties for
            # the fit quantities
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

            # --> Min and Max JD or MJD would probably be better
            # --> Needs uncertainties
            extinction_dict = {'date': just_date,
                               'object': object,
                               'filter': filt,
                               'instr_mag_am0': instr_mag_am0,
                               'extinction_coef': extinction,
                               'min_am': min(airmasses),
                               'max_am': max(airmasses),
                               'num_airmass_points ': num_airmass_points,
                               'red_chisq': red_chisq,
                               'max_frac_nonlin': max_frac_nonlin,
                               'min_mjd': min_mjd,
                               'max_mjd': max_mjd}

            extinction_data.append(extinction_dict)

        if not valid_extinction_data:
            log.warning(f"No good extinction measurements for object {object} in {directory}")
            # Close plot without writing
            plt.close()
            continue
            
        # Finish our strip-chart plot for this object and write the PNG file
        ax.tick_params(reset = True, which='both', direction='inout',
                       bottom=True, top=True, left=True, right=True)
        plt.xlabel('airmass')
        fname = f"{just_date}_{object}_extinction.png"
        outname = os.path.join(rd, fname)
        os.makedirs(rd, exist_ok=True)
        plt.savefig(outname, transparent=True)
        if show:
            plt.show()
        plt.close()

    if len(extinction_data) == 0:
        log.warning(f'No good extinction measurements in {directory}')
        # Signal that we were here and found nothing
        Path(extinction_outname).touch()
    else:
        # Write our extinction CSV
        fieldnames = list(extinction_dict.keys())
        with open(extinction_outname, 'w', newline='') as csvfile:
            csvdw = csv.DictWriter(csvfile, fieldnames=fieldnames,
                                   quoting=csv.QUOTE_NONNUMERIC)
            csvdw.writeheader()
            for ed in extinction_data:
                csvdw.writerow(ed)

    if len(exposure_correct_data) == 0:
        log.warning(f'No exposure correction measurements in {directory}')
        # Signal that we were here and found nothing
        Path(exposure_correct_outname).touch()
    else:
        outname = os.path.join(rd, "exposure_correction.png")
        exposure_correct_plot(exposure_correct_data,
                              show=show,
                              outname=outname)
        # Write our exposure correction CSV file
        with open(exposure_correct_outname, 'w', newline='') as csvfile:
            fieldnames = list(exposure_correct_data[0].keys())
            csvdw = csv.DictWriter(csvfile, fieldnames=fieldnames,
                                   quoting=csv.QUOTE_NONNUMERIC)
            csvdw.writeheader()
            for ecd in exposure_correct_data:
                csvdw.writerow(ecd)

    # Running into problems after several directories have run, but it
    # is not a specific file, since they all run OK when re-run.
    # Error:
    #
    # Exception ignored in: <function Image.__del__ at 0x7f711dab0dc0>
    # Traceback (most recent call last):
    #   File "/usr/lib/python3.9/tkinter/__init__.py", line 4017, in __del__
    #     self.tk.call('image', 'delete', self.name)
    # RuntimeError: main thread is not in main loop
    # Exception ignored in: <function Image.__del__ at 0x7f711dab0dc0>
    #
    # https://mail.python.org/pipermail/tkinter-discuss/2019-December/004153.html
    # suggests might be solved by just GCing occationally
    # This might be because I didn't close plots properly when I
    # didn't find any excintion data.  Well, that didn't take long to fail!
    gc.collect()

    return (directory, extinction_data, exposure_correct_data)

def standard_star_tree(data_root=RAW_DATA_ROOT,
                       start=None,
                       stop=None,
                       calibration=None,
                       photometry=None,
                       read_csvs=True,
                       show=False,
                       ccddata_cls=RedCorData,
                       outdir_root=PHOTOMETRY_ROOT,                       
                       **kwargs):
    dirs_dates = get_dirs_dates(data_root, start=start, stop=stop)
    dirs, _ = zip(*dirs_dates)
    if len(dirs) == 0:
        log.warning('No directories found')
        return
    if calibration is None:
        calibration = Calibration(reduce=True)
    if photometry is None:
        photometry = Photometry(precalc=True, **kwargs)

    extinction_data = []
    exposure_correct_data = []
    for d in dirs:
        _, extinct, expo = \
            standard_star_directory(d,
                                    calibration=calibration,
                                    photometry=photometry,
                                    read_csvs=read_csvs,
                                    ccddata_cls=ccddata_cls,
                                    outdir_root=outdir_root,
                                    **kwargs)
        extinction_data.extend(extinct)
        exposure_correct_data.extend(expo)

    rd = reduced_dir(data_root, outdir_root, create=True)
    # This is fast enough so that I don't need to save the plots.
    # Easier to just run with show=True and manipilate the plot by hand
    if start is None:
        start_str = ''
    else:
        start_str = start + '--'
    if stop is None:
        stop_str = ''
    else:
        stop_str = stop + '_'
    outname = os.path.join(rd,
                           f"{start_str}{stop_str}exposure_correction.png")
    exposure_correct_plot(exposure_correct_data,
                          outname=outname,
                          latency_change_dates=sx694.latency_change_dates,
                          show=show)
    return dirs, extinction_data, exposure_correct_data

def standard_star_cmd(args):
    calibration_start = args.calibration_start
    calibration_stop = args.calibration_stop
    c = Calibration(start_date=calibration_start,
                    stop_date=calibration_stop,
                    reduce=True,
                    num_processes=args.num_processes)
    standard_star_tree(data_root=args.data_root,
                       outdir_root=args.outdir_root,
                       start=args.start,
                       stop=args.stop,
                       calibration=c,
                       read_csvs=args.read_csvs,
                       read_pout=args.read_pout,
                       show=args.show,
                       num_processes=args.num_processes,
                       fits_fixed_ignore=args.fits_fixed_ignore)
    
if __name__ == '__main__':
    log.setLevel('DEBUG')
    parser = argparse.ArgumentParser(
        description='IoIO standard star photometric reduction')
    parser.add_argument(
        '--data_root', help='raw data root',
        default=RAW_DATA_ROOT)
    parser.add_argument(
        '--outdir_root', help='photometry output root',
        default=PHOTOMETRY_ROOT)
    parser.add_argument(
        '--calibration_start', help='calibration start date')
    parser.add_argument(
        '--calibration_stop', help='calibration stop date')
    parser.add_argument(
        '--start', help='start directory/date (default: earliest)')
    parser.add_argument(
        '--stop', help='stop directory/date (default: latest)')
    parser.add_argument(
        '--num_processes', type=float, default=0,
        help='number of subprocesses for parallelization; 0=all cores, <1 = fraction of total cores')
    parser.add_argument(
        '--read_csvs', action=argparse.BooleanOptionalAction, default=True,
        help='re-read previous results from CSV files in each subdirectory')
    parser.add_argument(
        '--read_pout', action=argparse.BooleanOptionalAction, default=True,
        help='re-read previous pipeline output in each subdirectory')
    parser.add_argument(
        '--show', action=argparse.BooleanOptionalAction,
        help='show PyPlot of top-level results (pauses terminal)',
        default=False)
    parser.add_argument(
        '--fits_fixed_ignore', action=argparse.BooleanOptionalAction,
        help='turn off WCS warning messages',
        default=False)
    
    args = parser.parse_args()
    standard_star_cmd(args)


## Pythonic way of checking for a non-assigned variable
#try:
#    pout
#except NameError:
#    # Avoid confusing double-exception if code to generate variable fails
#    pout = None
#    
#if pout is None:
#    # Run our pipeline code
#    c = Calibration(start_date='2020-01-01', stop_date='2021-12-31', reduce=True)
#    directory, pout = photometry_pipeline('/data/io/IoIO/raw/20210310/',
#                                          glob_include=['HD*SII_on*'],
#                                          calibration=c)


#photometry_tree(start='2021-03-10', stop='2021-03-11')

#c = Calibration(start_date='2020-01-01', stop_date='2021-12-31', reduce=True)
#photometry_tree(start='2017-03-07', stop='2017-03-07',
#                calibration=c)

#standard_star_tree(start='2021-03-10', stop='2021-03-11',
#                   glob_include=['HD*SII_on*'])

#c = Calibration(start_date='2020-01-01', stop_date='2021-12-31', reduce=True)
#test = '/data/io/IoIO/raw/20210310/HD 132052-S001-R001-C002-R.fts'
#cmp = CorMultiPipe(auto=True, calibration=c,
#                   post_process_list=[nd_filter_mask,
#                                      standard_star_process],
#                   process_expand_factor=15)
#pout = cmp.pipeline([test], outdir='/tmp', overwrite=True)

#standard_star_tree(start='2021-03-10', stop='2021-03-10',
#                   glob_include=['HD*SII_on*'])

#c = Calibration(start_date='2020-01-01', stop_date='2021-12-31', reduce=True)
#standard_star_tree(start='2021-05-07', stop='2021-05-18', calibration=c)

#from astropy.nddata import CCDData
#from cormultipipe import cor_process, RedCorData
#test = '/data/io/IoIO/raw/20210310/HD 132052-S001-R001-C002-R.fts'
##ccd = CCDData.read(test, unit=u.adu)
##bmp_meta = {}
##ccd = source_catalog_process(ccd,
##                         bmp_meta=bmp_meta,
##                         in_name=test)
#
#c = Calibration(start_date='2020-01-01', stop_date='2021-12-31',
#                reduce=True)
#
#ccd = RedCorData.read(test)
#ccd = cor_process(ccd, calibration=c, auto=True)
#bmp_meta = {}
#ccd = source_catalog_process(ccd,
#                     bmp_meta=bmp_meta,
#                     in_name=test)
#

#c = Calibration(start_date='2020-01-01', stop_date='2021-12-31', reduce=True)
##standard_star_directory('/data/io/IoIO/raw/20210508/', calibration=c)
#standard_star_directory('/data/io/IoIO/raw/20210510/', calibration=c)
#directory, extinct, expo = \
#    standard_star_directory('/data/io/IoIO/raw/20210510/',
#                            calibration=c,
#                            read_csvs=True, show=True)

#c = Calibration(start_date='2020-01-01', stop_date='2021-12-31', reduce=True)
#dirs, extinction_data, exposure_correct_data = \
#    standard_star_tree(start='2021-05-10', stop='2021-05-11',
#                       show=True,
#                       calibration=c)

#c = Calibration(start_date='2020-01-01', stop_date='2021-12-31', reduce=True)
##standard_star_directory('/data/io/IoIO/raw/20210508/', calibration=c)
#standard_star_directory('/data/io/IoIO/raw/20210517/', calibration=c)
#directory, extinct, expo = \
#    standard_star_directory('/data/io/IoIO/raw/20210510/',
#                            calibration=c,
#                            read_csvs=False, show=True)

#standard_star_directory('/data/io/IoIO/raw/20210517/',
#                        read_pout=False, read_csvs=False)

#test = '/data/io/IoIO/raw/20210310/HD 132052-S001-R001-C002-R.fts'
#cmp = CorMultiPipe(auto=True, post_process_list=[nd_filter_mask,
#                                                 standard_star_process],
#                   process_expand_factor=15)
#pout = cmp.pipeline([test], outdir='/tmp', overwrite=True)

#standard_star_pipeline('/data/io/IoIO/raw/20200720/',
#                       read_pout=False, write_pout=True)

#standard_star_directory('/data/io/IoIO/raw/20200720/',
#                        read_pout=False, read_csvs=False)


#test = '/data/io/IoIO/raw/20210310/HD 132052-S001-R001-C002-R.fts'
#ccd = FwRedCorData.read(test, unit=u.adu)
#p = Photometry(ccd=ccd)

#dirs, extinction_data, exposure_correct_data = \
#    standard_star_tree(start='2021-05-10', stop='2021-05-11',
#                       show=True)

#dirs, extinction_data, exposure_correct_data = \
#    standard_star_tree(start='2020-07-20', stop='2020-07-20',
#                       read_pout=False, read_csvs=False, show=True)


#dirs, extinction_data, exposure_correct_data = \
#    standard_star_tree(start='2021-05-10', stop='2021-05-10',
#                       read_pout=False, read_csvs=False, show=True)

#log.setLevel('DEBUG')
#standard_star_directory('/data/io/IoIO/raw/20210510/',
#                        read_pout=False, read_csvs=False)
#standard_star_directory('/data/io/IoIO/raw/20200720/',
#                        read_pout=False, read_csvs=False)

