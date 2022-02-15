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

#from scipy.interpolate import interp1d

import matplotlib.pyplot as plt
from matplotlib.dates import datestr2num

import pandas as pd

from astropy import log
from astropy import units as u
from astropy.time import Time
from astropy.stats import mad_std, biweight_location
from astropy.coordinates import SkyCoord
from astroquery.simbad import Simbad
from astropy.modeling import models, fitting
from astropy.visualization import quantity_support

from specutils import Spectrum1D, SpectralRegion
from specutils.manipulation import (extract_region,
                                    LinearInterpolatedResampler)


# bmp_cleanup can go away now that Photometry is an object
from bigmultipipe import no_outfile, prune_pout

from burnashev import Burnashev

import IoIO.sx694 as sx694
from IoIO.utils import assure_list, reduced_dir, get_dirs_dates
from IoIO.cordata_base import CorDataBase
from IoIO.cormultipipe import (IoIO_ROOT, RAW_DATA_ROOT,
                               CorMultiPipeBase,
                               nd_filter_mask, mask_nonlin_sat)
from IoIO.calibration import Calibration
from IoIO.photometry import Photometry, is_flux

PHOTOMETRY_ROOT = os.path.join(IoIO_ROOT, 'StandardStar')
FILT_ROOT = '/data/IoIO/observing'
# Small filters are 15 arcmin.  Could even fluf this out if I think I
# haven't synced the telescope on some old observations
POINTING_TOLERANCE = 15*u.arcmin
AVOID_HBETA = 5000*u.AA

class CorBurnashev(Burnashev):
    def calc_spec(self, entry, plot=False, title=''):
        spec = super().calc_spec(entry)
        spec_med_dlambda = np.median(spec.spectral_axis[1:]
                                     - spec.spectral_axis[0:-1])
        spec_med_dlambda = spec_med_dlambda.to(spec.spectral_axis.unit)
        last_spec_lambda = spec.spectral_axis[-1]
        lstart = last_spec_lambda.to(spec.spectral_axis.unit) + spec_med_dlambda
        lstop = 1*u.micron
        lstop = lstop.to(spec.spectral_axis.unit)
        num_extra = int(np.round((lstop - lstart) / spec_med_dlambda))
        lstop = lstart + num_extra * spec_med_dlambda
        extra_lambda =  np.linspace(lstart.value, lstop.value, num_extra+1)
        full_lambda = np.concatenate((spec.spectral_axis.value, extra_lambda))

        # HACK Total hacky decision on where to start fitting
        model_bandpass = SpectralRegion(AVOID_HBETA,
                                        spec.spectral_axis[-1])
        to_model = extract_region(spec, model_bandpass)


        # HACK ALERT!  Round off exponential with sloping line
        tail_model = models.Linear1D() + models.Exponential1D()
        # This works for Vega but always pegs at upper limit.  -1000
        # might be the best 
        #tail_model.tau_1.bounds = (-1500, -500)
        tail_model.tau_1.bounds = (-1500, -600)

        ## Not so bad for Vega, wrong way on other random star
        #tail_model = models.Const1D() + models.Logarithmic1D()

        ## Too steep
        #tail_model = models.Const1D() + models.Exponential1D()
        #tail_model.amplitude_0.bounds = (0, to_model.flux[-1])
        #tail_model.tau_1.bounds = (-1500, -500)

        ## Problems with undershoot or blowup
        #tail_model = models.Linear1D()
        #tail_model = models.Polynomial1D(2)

        #tail_model = models.Shift() | models.Exponential1D() + models.Const1D()
        #tail_model.offset_0 = model_bandpass.lower.value
        #tail_model.offset_0.fixed = True
        #tail_model.tau_1.max = -400

        ## Doesn't work, possibly need constrains
        #tail_model = BlackBody()
        ## Doesn't work, possibly need constrains
        #tail_model = models.Const1D() + models.PowerLaw1D()

        fiter = fitting.LevMarLSQFitter()
        tail_fit = fiter(tail_model, to_model.spectral_axis.value, to_model.flux.value)

        yfit = tail_fit(to_model.spectral_axis.value)

        extrap_flux = tail_fit(extra_lambda)
        extrap_flux[extrap_flux < 0] = 0
        full_flux = np.concatenate((spec.flux.value, extrap_flux))
        full_lambda *= spec.spectral_axis.unit
        full_spec = Spectrum1D(spectral_axis=full_lambda,
                               flux=full_flux*spec.flux.unit)
        

        if plot:
            quantity_support()
            f, ax = plt.subplots()
            ax.step(full_spec.spectral_axis, full_spec.flux)
            ax.step(spec.spectral_axis, spec.flux)
            ax.step(to_model.spectral_axis, yfit)
            ax.set_title(title)
            plt.show()

        return full_spec
        

    def get_filt(self, filt_name, **kwargs):
        fname = os.path.join(FILT_ROOT, filt_name+'.txt')
        try:
            filt_arr = np.loadtxt(fname, **kwargs)
        except:
            filt_arr = np.loadtxt(fname, delimiter=',')
        wave = filt_arr[:,0]*u.nm
        trans = filt_arr[:,1]
        # Clean up datatheif reading
        #trans = np.asarray(trans)
        trans[trans < 0] = 0
        trans *= u.percent
        filt = Spectrum1D(spectral_axis=wave, flux=trans)
        return filt

    def flux_in_filt(self, spec, filt,
                     resampler=None, energy=False,
                     plot=False, title=''):
        # Make filter spectral axis unit consistent with star spectrum.
        # Need to make a new filter spectrum as a result
        filt_spectral_axis = filt.spectral_axis.to(spec.spectral_axis.unit)
        filt = Spectrum1D(spectral_axis=filt_spectral_axis,
                          flux=filt.flux)
        filter_bandpass = SpectralRegion(np.min(filt.spectral_axis),
                                         np.max(filt.spectral_axis))
        # Work only with our filter bandpass
        spec = extract_region(spec, filter_bandpass)

        if (len(spec.spectral_axis) == len(filt.spectral_axis)
            and np.all((spec.spectral_axis == filt.spectral_axis))):
            # No need to resample
            pass
        else:
            if resampler is None:
                resampler = LinearInterpolatedResampler()
            # Resample lower resolution to higher
            spec_med_dlambda = np.median(spec.spectral_axis[1:]
                                         - spec.spectral_axis[0:-1])
            filt_med_dlambda = np.median(filt.spectral_axis[1:]
                                     - filt.spectral_axis[0:-1])
            if spec_med_dlambda > filt_med_dlambda:
                spec = resampler(spec, filt.spectral_axis)
            else:
                filt = resampler(filt, spec.spectral_axis)

        filt = resampler(filt, spec.spectral_axis)

        orig_spec = spec
        spec = spec * filt
        if plot:
            quantity_support()
            star_color='tab:blue'
            filter_color='tab:red'
            filtered_color='tab:green'
            f, ax = plt.subplots()
            ax.step(orig_spec.spectral_axis, orig_spec.flux,
                    color=star_color,
                    label='Star')
            ax.step(spec.spectral_axis, spec.flux,
                    color=filtered_color,
                    label='Filter*Star')
            ax.legend(loc='upper left')
            ax2 = ax.twinx()
            ax2.tick_params(axis='y', labelcolor=filter_color)
            ax2.step(filt.spectral_axis, filt.flux,
                     color=filter_color,
                     label='Filter')
            ax.set_title(title)
            ax2.legend(loc='upper right')
            plt.show()

        if energy:
            filt_flux = line_flux(spec)
            filt_flux = filt_flux.to(u.erg * u.cm**-2 * u.s**-1)
        else:
            # Photon.  Integrate by hand with simple trapezoidal,
            # non-interpolated technique.  THis is OK, since our filter
            # curves are nicely sampled
            spec_dlambda = spec.spectral_axis[1:] - spec.spectral_axis[0:-1]
            av_bin_flux = (spec.photon_flux[1:] + spec.photon_flux[0:-1])/2
            filt_flux = np.nansum(spec_dlambda*av_bin_flux)
        return filt_flux

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
    ND_width = ccd.ND_params[1, 1] - ccd.ND_params[1, 0]
    min_ND_dist = min_ND_multiple * ND_width
    if ccd.obj_to_ND < min_ND_dist:
        log.warning(f'Too close: ccd.obj_to_ND = {ccd.obj_to_ND} {in_name}')
        bmp_meta.clear()
        return None
    center = np.asarray(ccd.shape)/2
    radius = ((xcentrd-center[1])**2 + (ycentrd-center[0])**2)**0.5

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

    # After binning pixel size
    pixsz = np.asarray((ccd.meta['XPIXSZ'], ccd.meta['YPIXSZ']))
    pix_area = np.prod(pixsz)
    pix_area *= u.micron**2
    focal_length = ccd.meta['FOCALLEN']*u.mm
    pix_solid_angle = pix_area / focal_length**2
    pix_solid_angle *= u.rad**2
    pix_solid_angle = pix_solid_angle.to(u.arcsec**2)    

    # We are going to turn this into a Pandas dataframe, which does
    # not do well with units, so just return everything
    tmeta = {'object': ccd.meta['OBJECT'],
             'filter': ccd.meta['FILTER'],
             'date': just_date,
             'date_obs': tm,
             'jd': tm.jd,
             'objctra': ccd.meta['OBJCTRA'],
             'objctdec': ccd.meta['OBJCTDEC'],
             'objalt': ccd.meta['OBJCTALT'],
             'airmass': ccd.meta['AIRMASS'],
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
             'radius': radius,
             'pix_solid_angle': pix_solid_angle.value}
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

    cmp = CorMultiPipeBase(auto=True,
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

    # Add Vega onto our objects.  Will take it off, below
    objects = list(set(df['object']))
    objects.append('Vega')
    filters = list(set(df['filter']))

    s = Simbad()
    burnashev = CorBurnashev()
    s.add_votable_fields('flux(U)', 'flux(B)', 'flux(V)', 'flux(R)',
                         'flux(I)')
    simbad_results = s.query_objects(objects)
    vega_entry = simbad_results[-1]
    simbad_results = simbad_results[0:-2]
    vega_coords = SkyCoord(vega_entry['RA'],
                           vega_entry['DEC'],
                           unit=(u.hour, u.deg))
    bname, bsep, _ = burnashev.closest_name_to(vega_coords)
    assert bsep < POINTING_TOLERANCE, 'Vega must be found in Burnashevc catalog!'
    vega_spec = burnashev.get_spec(bname)
    vega_standard_list = []
    for filt_name in filters:
        filt = burnashev.get_filt(filt_name)
        filt_flux = burnashev.flux_in_filt(vega_spec, filt,
                                           plot=False,
                                           title=f'Vega {filt_name}')
        flux_col = 'FLUX_' + filt_name
        if flux_col in vega_entry.colnames:
            filt_mag = vega_entry[flux_col]
        else:
            # All our narrow-band filters are currently in R-band, so make
            # that the logarithmic reference
            filt_mag = vega_entry['FLUX_R']
        vega_standard_list.append({'filt_name': filt_name,
                                   'filt_flux': filt_flux.value,
                                   'filt_mag': filt_mag})
    pd_vega = pd.DataFrame(vega_standard_list, index=filters)

    
    # Collect extinction and exposure correction data  into arrays of dicts
    extinction_data = []
    exposure_correct_data = []
    # --> This assumes our OBJECT assignments are correct and the
    # telescope was really pointed at the named object
    for obj, simbad_entry in zip(objects, simbad_results):
        # Each object has its own flux which we measure at different
        # airmasses and through different filters.  We also want to
        # compare with standard stars
        objdf = df[df['object'] == obj]

        # Get object coordinates from our FITS header.  Assume
        # cor_process with correct_obj_offset has run, which puts
        # object's original RA and DEC in these keywords in
        # HHhMMmSS.Ss, etc. format.  NOTE: for objects recorded by
        # hand with MaxIm, the actual telescope RA and DEC will be
        # used, which will be off by a few arcmin, since there is no
        # RAOFF or DECOFF record
        row1 = objdf.index[0]
        ra = objdf['objctra'][row1]
        dec = objdf['objctdec'][row1]
        obj_coords = SkyCoord(ra, dec)
        simbad_coords = SkyCoord(simbad_entry['RA'],
                                 simbad_entry['DEC'],
                                 unit=(u.hour, u.deg))
        pix_solid_angle = objdf['pix_solid_angle'][row1]
        pix_solid_angle *= u.arcsec**2

        print(f"{obj}, simbad {simbad_coords.to_string(style='hmsdms')}, "
              f"commanded {obj_coords.to_string(style='hmsdms')}")
        #print(simbad_coords.to_string(style='hmsdms'))
        #print(obj_coords.to_string(style='hmsdms'))
        sep = obj_coords.separation(simbad_coords)
        print(sep)
        simbad_match = sep < POINTING_TOLERANCE
        if simbad_match:
            log.debug(f'Pointed to within {POINTING_TOLERANCE} of {obj}')
            # Use authoritative coordinate of object for Burnashev match
            bname, bsep, bcoords = burnashev.closest_name_to(simbad_coords)
        else:
            log.debug('Not finding object close enough in Simbad, checking Burnashev anyway')
            bname, bsep, bcoords = burnashev.closest_name_to(obj_coords)
        burnashev_match = bsep < POINTING_TOLERANCE
        if burnashev_match:
            log.debug(f'Found {obj} in Burnashev catalog')
            burnashev_spec = burnashev.get_spec(bname)
                                    
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
        plt.suptitle(f"{obj} {just_date} {min_time} -- {max_time} instrument mags")
        valid_extinction_data = False
        for ifilt, filt in enumerate(filters):
            # Collect fluxes and airmasses measured throughout the night
            # for this object and filter
            filtdf = objdf[objdf['filter'] == filt]
            n_meas = len(filtdf.index) 
            if n_meas < 3:
                log.warning(f'Not enough measurements ({n_meas}) in filter {filt} for object {obj} in {directory}')
                continue
            max_frac_nonlin = np.max(filtdf['max_frac_nonlin'])
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
                    log.warning(f"Not enough measurements ({n_meas}) in filter {filt} for object {obj} in {directory} from {tdf['date_obs'].iloc[0]} to {tdf['date_obs'].iloc[-1]}")
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
                        log.warning(f"Not enough measurements ({n_meas}) in filter {filt} for object {obj} in {directory} for exposure time {uoe} from {tdf['date_obs'].iloc[0]} to {tdf['date_obs'].iloc[-1]}")
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
                        log.warning(f"Unstable atmosphere detflux_std = {detflux_std} > {stability_factor} * {mdetflux_err} for filter {filt} object {obj} in {directory} from {tdf['date_obs'].iloc[0]} to {tdf['date_obs'].iloc[-1]}")
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
                    log.warning(f"No good flux measurements left for filter {filt} object {obj} in {directory} from {tdf['date_obs'].iloc[0]} to {tdf['date_obs'].iloc[-1]}")
                    continue

                #if len(true_flux_idx) == 0:
                #    log.warning(f"No good measurements at exposure times <= sx694.max_accurate_exposure for filter {filt} object {obj} in {directory} from {tdf['date_obs'].iloc[0]} to {tdf['date_obs'].iloc[-1]}")
                #    continue
                #true_flux = np.nanmean(detfluxes[true_flux_idx])
                #if np.isnan(true_flux):
                #    log.warning(f"No good flux measurements left for filter {filt} object {obj} in {directory} from {tdf['date_obs'].iloc[0]} to {tdf['date_obs'].iloc[-1]}")
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

            # Having collected fluxes for this object and filter over
            # the night, fit mag vs. airmass to get airless magnitude
            # and extinction coef
            if best_fluxes is None:
                log.warning(f"No good flux measurements for filter {filt} object {obj} in {directory}")

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
                log.warning(f"not enough points ({num_airmass_points}) for object {obj} for extinction measurement {just_date} filter {filt}")
                continue

            valid_extinction_data = True
            # Fit a line to airmass vs. instr_mags
            # --> This might be a good place for iter_linfit?
            # In any case, throwing out bad points would be good
            # --> I will eventually want to extract uncertainties for
            # the fit quantities
            poly = Polynomial.fit(airmasses, instr_mags, deg=1)
            xfit, yfit = poly.linspace()
            plt.plot(xfit, yfit)
            instr_mag_am0 = poly(0)*u.mag(u.ph/u.cm**2/u.s)
            #print(simbad_entry['FLUX_'+filt])
            flux_col = 'FLUX_'+filt
            zero_point = ''
            calibration_str = np.NAN*u.dimensionless_unscaled
            rayleigh_conversion = np.NAN*u.dimensionless_unscaled

            # Get our Vega flux and mag reference
            vega_flux = pd_vega.loc[filt:filt,'filt_flux']
            vega_flux = vega_flux[0]
            vega_mag0 = pd_vega.loc[filt_name:filt_name,'filt_mag']
            vega_mag0 = vega_mag0.values[0]
            vega_mag0 = vega_mag0*u.mag(u.ph/u.cm**2/u.s)
            vega_flux_mag = u.Magnitude(vega_flux*u.ph/u.cm**2/u.s)
            if (simbad_match
                and flux_col in simbad_entry.colnames
                and not np.ma.is_masked(simbad_entry[flux_col])):
                # Prefer Simbad-listed mags for UBVRI
                star_mag = simbad_entry[flux_col]
                star_mag *= u.mag(u.ph/u.cm**2/u.s)
                # Convert to physical flux using algebra and the
                # forward derivation below
                star_flux_mag = star_mag + (vega_flux_mag - vega_mag0)
                star_flux = star_flux_mag.physical
            elif burnashev_match:
                # Integrating the Burnashev catalog flux over our
                # filters gives pretty comparable results (see code in __main__)
                filt_prof = burnashev.get_filt(filt)
                star_flux = burnashev.flux_in_filt(burnashev_spec, filt_prof,
                                                   plot=False,
                                                   title=f'{obj} {filt_name}')
                star_flux_mag = u.Magnitude(star_flux)
                star_mag = star_flux_mag - (vega_flux_mag - vega_mag0)
                # http://sirius.bu.edu/planetary/obstools/starflux/starcalib/starcalib.htm                
            star_sb = 4*np.pi * star_flux / pix_solid_angle
            star_sb = star_sb.to(u.R)
            # Convert our measurement back to flux units for
            # comparison to integral
            flux_am0 = u.Magnitude(instr_mag_am0).physical    
            flux_am0 *= u.electron/u.s
            rayleigh_conversion = star_sb / flux_am0
            zero_point = star_mag - instr_mag_am0
            extinction = poly.deriv()
            extinction = extinction(0)
            airmasses = np.asarray(airmasses)
            dof = num_airmass_points - 2
            red_chisq = np.sum((poly(airmasses) - instr_mags)**2) / dof
            plt.text(0.5, 0.75, 
                     f'$M_0$ = {zero_point:.2f};   {rayleigh_conversion:0.2e}',
                     ha='center', va='bottom', transform=ax.transAxes)

            plt.text(0.5, 0.5, 
                     f'Airless instr mag = {instr_mag_am0:.2f} (electron/s)',
                     ha='center', va='bottom', transform=ax.transAxes)

            plt.text(0.5, 0.1, 
                     f'Extinction = {extinction:.3f} (mag/airmass)',
                     ha='center', va='bottom', transform=ax.transAxes)
            plt.text(0.13, 0.1, 
                     f'Red. Chisq = {red_chisq:.4f}',
                         ha='center', va='bottom', transform=ax.transAxes)

            # --> Min and Max JD or MJD would probably be better
            # --> Needs uncertainties
            extinction_dict = {'date': just_date,
                               'object': obj,
                               'filter': filt,
                               'instr_mag_am0': instr_mag_am0,
                               'zero_point': zero_point,
                               'rayleigh_conversion': rayleigh_conversion.value,
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
            log.warning(f"No good extinction measurements for object {obj} in {directory}")
            # Close plot without writing
            plt.close()
            continue
            
        # Finish our strip-chart plot for this object and write the PNG file
        ax.tick_params(reset = True, which='both', direction='inout',
                       bottom=True, top=True, left=True, right=True)
        plt.xlabel('airmass')
        fname = f"{just_date}_{obj}_extinction.png"
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
    # didn't find any excintqion data.  Well, that didn't take long to fail!
    gc.collect()

    return (directory, extinction_data, exposure_correct_data)

def standard_star_tree(data_root=RAW_DATA_ROOT,
                       start=None,
                       stop=None,
                       calibration=None,
                       photometry=None,
                       read_csvs=True,
                       show=False,
                       ccddata_cls=CorDataBase,
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

#filt_name = 'B'
#c = CorBurnashev()
#filt = c.get_filt(filt_name)
#f, ax = plt.subplots()
#ax.step(filt.spectral_axis, filt.flux)
#ax.set_title(f"{filt_name}")
#plt.show()
#

## Not helpful without knowing catalog to use
##v = Vizier(columns=['U', 'B', 'V', 'R', 'I'])
#v = Vizier(columns=['n_Vmag', 'u_Vmag', 'B-V', 'u_B-V', 'U-B',
#                    'u_U-B', 'R-I', 'u_R-I'])
#result_table = v.query_object("HD 64648", catalog="V/50")
#result_table[0]
#

# So Simbad seems to be more useful

## --> this sort of worked, but the catalog doesn't have all the fluxes!
#s = Simbad()
#s.add_votable_fields('flux(U)', 'flux(B)', 'flux(V)', 'flux(R)',
#                     'flux(I)')
##, "flux(u')" "flux(g')", "flux(r')",
##                     "flux(i')", "flux(z')")
#objects = ["HD 64648", "HD 4647"]
#simbad_results = s.query_objects(objects)
##simbad_results.show_in_browser()
#row = simbad_results[0]
#my_star = SkyCoord(row['RA'], row['DEC'], unit=(u.hour, u.deg))
#
#b = Burnashev()
#my_star = SkyCoord(f'07h 55m 39.9s', f'+19d 53m 02s')
#
#ra = '00 45 34.61'
#ra = ra.split()
#ra = f'{ra[0]}h {ra[1]}m {ra[2]}s'


#fname = '/data/IoIO/raw/20190930/0029P-S001-R001-C001-H2O+_dupe-1.fts'
#nccd = RedCorData.read(fname)
#
#objctra = nccd.meta.get('OBJCTRA')
#objctdec = nccd.meta.get('OBJCTDEC')
## These are in files written by MaxIm and ACP *when the telescope is connected*
#if objctra is not None and objctdec is not None:
#    raoff = nccd.meta.get('RAOFF') or 0
#    decoff = nccd.meta.get('DECOFF') or 0
#    raoff = Angle(raoff*u.arcmin)
#    decoff = Angle(decoff*u.arcmin)
#    cent = SkyCoord(objctra, objctdec, unit=(u.hour, u.deg))
#    target = SkyCoord(cent.ra - raoff, cent.dec - decoff)
#    ra_dec = target.to_string(style='hmsdms').split()
#    nccd.meta['OBJCTRA'] = (ra_dec[0],
#                            '[hms J2000] Target right assention')
#    nccd.meta['OBJCTDEC'] = (ra_dec[1],
#                             '[dms J2000] Target declination')
#    nccd.meta.comments['RA'] = '[hms J2000] Center nominal right assention'
#    nccd.meta.comments['DEC'] = '[dms J2000] Center nominal declination'
#print(nccd.meta)

      
#my_star = SkyCoord(f'07 55 39.9', f'+19 53 02', unit=(u.hour, u.deg))
#print(my_star.to_string(style='hmsdms'))
#raoff = Angle(3*u.arcmin)
#decoff = Angle(3*u.arcmin)
#print(cent.to_string(style='hmsdms'))

#from cormultipipe import cor_process
#fname = '/data/IoIO/raw/2020-07-15/HD85235-0005_Na_off.fit'
#ccd = RedCorData.read(fname)
#nccd = cor_process(ccd, auto=True, calibration=True)
#bmp_meta = {}
#standard_star_process(nccd, bmp_meta=bmp_meta)

                    
#log.setLevel('DEBUG')
#directory = '/data/IoIO/raw/2020-07-15'
##standard_star_pipeline(directory, read_pout=False, write_pout=True)
#standard_star_directory(directory, read_pout=True, read_csvs=False)

#pixsz = np.asarray((4.539, 4.539))
#pix_area = np.prod(pixsz)
#pix_area *= u.micron**2
#focal_length = 1200*u.mm
#pix_solid_angle = pix_area / focal_length**2
#pix_solid_angle *= u.rad**2
#pix_solid_angle.to(u.arcsec**2)    

## Check the standard Alpha Lyrae
#my_star = SkyCoord('18h 36m 56.33635s', '+38d 47m 01.2802s')
#my_star = SkyCoord(f'12h23m42.2s', f'-26d18m22.2s')

# b = CorBurnashev()
# name, dist, coord = b.closest_name_to(my_star)
# spec = b.get_spec(name)
# spec_med_dlambda = np.median(spec.spectral_axis[1:]
#                              - spec.spectral_axis[0:-1])
# spec_med_dlambda = spec_med_dlambda.to(spec.spectral_axis.unit)
# last_spec_lambda = spec.spectral_axis[-1]
# lstart = last_spec_lambda.to(spec.spectral_axis.unit) + spec_med_dlambda
# lstop = 1*u.micron
# lstop = lstop.to(spec.spectral_axis.unit)
# num_extra = int(np.round((lstop - lstart) / spec_med_dlambda))
# lstop = lstart + num_extra * spec_med_dlambda
# extra_lambda =  np.linspace(lstart.value, lstop.value, num_extra+1)
# full_lambda = np.concatenate((spec.spectral_axis.value, extra_lambda))
# 
# # HACK Total hacky decision on where to start fitting
# model_bandpass = SpectralRegion(last_spec_lambda-1500*u.AA,
#                                 spec.spectral_axis[-1])
# to_model = extract_region(spec, model_bandpass)
# 
# 
# # HACK ALERT!  Round off exponential with sloping line
# tail_model = models.Linear1D() + models.Exponential1D()
# # This works for Vega but always pegs at upper limit
# #tail_model.tau_1.bounds = (-1500, -500)
# tail_model.tau_1.bounds = (-1500, -600)
# 
# ## Not so bad for Vega, wrong way on other random star
# #tail_model = models.Const1D() + models.Logarithmic1D()
# 
# ## Too steep
# #tail_model = models.Const1D() + models.Exponential1D()
# #tail_model.amplitude_0.bounds = (0, to_model.flux[-1])
# #tail_model.tau_1.bounds = (-1500, -500)
# 
# ## Problems with undershoot or blowup
# #tail_model = models.Linear1D()
# #tail_model = models.Polynomial1D(2)
# 
# #tail_model = models.Shift() | models.Exponential1D() + models.Const1D()
# #tail_model.offset_0 = model_bandpass.lower.value
# #tail_model.offset_0.fixed = True
# #tail_model.tau_1.max = -400
# 
# ## Doesn't work, possibly need constrains
# #tail_model = BlackBody()
# ## Doesn't work, possibly need constrains
# #tail_model = models.Const1D() + models.PowerLaw1D()
# 
# fiter = fitting.LevMarLSQFitter()
# tail_fit = fiter(tail_model, to_model.spectral_axis.value, to_model.flux.value)
# 
# yfit = tail_fit(to_model.spectral_axis.value)
# 
# #m = models.Exponential1D()
# # --> decent parameters
# #yfit = m.evaluate(to_model.spectral_axis.value, 3E1, -1000.)
# 
# extrap_flux = tail_fit(extra_lambda)
# full_flux = np.concatenate((spec.flux.value, extrap_flux))
# full_spec = Spectrum1D(spectral_axis=full_lambda*spec.spectral_axis.unit,
#                        flux=full_flux*spec.flux.unit)
# f, ax = plt.subplots()
# ax.step(full_spec.spectral_axis, full_spec.flux)
# ax.step(spec.spectral_axis, spec.flux)
# ax.step(to_model.spectral_axis, yfit)
# plt.show()

#blew up
#extrapolator = interp1d(spec.spectral_axis, spec.flux,
#                        #kind='quadratic', fill_value='extrapolate')
#                        kind='cubic', fill_value='extrapolate')
#new_flux = extrapolator(full_lambda)
#spec = Spectrum1D(spectral_axis=full_lambda*spec.spectral_axis.unit,
#                  flux=new_flux*spec.flux.unit)

# Wrong shape -- no reddening, I guess
#num_pts = len(spec.spectral_axis) + num_extra
#full_lambda = np.linspace(lstart.value, lstop.value, spec_med_dlambda.value)
#bb = BlackBody(temperature=9602*u.K)
#bbspec = Spectrum1D(spectral_axis=spec.spectral_axis,
#                    flux=bb(spec.spectral_axis))
#bbspec *=1E3


#f, ax = plt.subplots()
##ax.step(bbspec.spectral_axis, bbspec.flux)
#ax.step(spec.spectral_axis, spec.flux)
#plt.show()

# spec = orig_spec
#filt_root = '/data/IoIO/observing'
#filt_names = ['U', 'B', 'V', 'R', 'I',
#              'SII_on', 'SII_off', 'Na_on', 'Na_off']
#
#for filt_name in filt_names:
#    filt = b.get_filt(filt_name)
#    f, ax = plt.subplots()
#    ax.step(filt.spectral_axis, filt.flux)
#    ax.set_title(f"{filt_name}")
#    plt.show()
#    #ax.set_xlabel("Wavelenth")  
#    #ax.set_ylabel("Transmission")
#    #ax.set_title(f"{filt_name}")
#    #plt.show()
#
#    spec = orig_spec
#    filt_flux = b.flux_in_filt(orig_spec, filt)
#    print(f'{filt_name} flux = {filt_flux}')


####################################################################
### Test showing integrating Burnashev flux over our filters gives
### comparable results to cataloged filter mags
####################################################################
##burnashev = CorBurnashev()
##
##filt_names = ['U', 'B', 'V', 'R', 'I',
##              'SII_on', 'SII_off', 'Na_on', 'Na_off']
##star_list = ['HD 1280  ',
##             'HD 6695  ',
##             'HD 12869 ',
##             'HD 18411 ',
##             'HD 25867 ',
##             'HD 32301 ',
##             'HD 39357 ',
##             'HD 50635 ',
##             'HD 64648 ',
##             'HD 77350 ',
##             'HD 85376 ',
##             'HD 95310 ',
##             'HD 103578',
##             'HD 112412',
##             'HD 118232',
##             'HD 120136',
##             'HD 128167',
##             'HD 133640',
##             'HD 143894',
##             'HD 154029',
##             'HD 166230',
##             'HD 177196',
##             'HD 187013',
##             'HD 191747',
##             'HD 198639',
##             'HD 204414',
##             'HD 210459',
##             'HD 217782']
##
##star_list.append('Vega')
##s = Simbad()
##s.add_votable_fields('flux(U)', 'flux(B)', 'flux(V)', 'flux(R)',
##                     'flux(I)')
##simbad_results = s.query_objects(star_list)
##vega_entry = simbad_results[-1]
##simbad_results = simbad_results[0:-2]
##vega_coords = SkyCoord(vega_entry['RA'],
##                       vega_entry['DEC'],
##                       unit=(u.hour, u.deg))
##bname, bsep, _ = burnashev.closest_name_to(vega_coords)
##assert bsep < POINTING_TOLERANCE, 'Vega must be found in Burnashev catalog!'
##vega_spec = burnashev.get_spec(bname)
##vega_standard_list = []
##for filt_name in filt_names:
##    filt = burnashev.get_filt(filt_name)
##    filt_flux = burnashev.flux_in_filt(vega_spec, filt,
##                                       plot=False,
##                                       title=f'Vega {filt_name}')
##    flux_col = 'FLUX_' + filt_name
##    if flux_col in vega_entry.colnames:
##        filt_mag = vega_entry[flux_col]
##    else:
##        # All our narrow-band filters are currently in R-band, so make
##        # that the logarithmic reference
##        filt_mag = vega_entry['FLUX_R']
##    
##    print(f'{filt_name} Vega {filt_flux} {filt_mag}')
##    vega_standard_list.append({'filt_name': filt_name,
##                               'filt_flux': filt_flux.value,
##                               'filt_mag': filt_mag})
##pd_vega = pd.DataFrame(vega_standard_list, index=filt_names)
##    
##for obj, simbad_entry in zip(star_list, simbad_results):
##    simbad_coords = SkyCoord(simbad_entry['RA'],
##                             simbad_entry['DEC'],
##                             unit=(u.hour, u.deg))
##    name, _, _ = burnashev.closest_name_to(simbad_coords)
##    burnashev_spec = burnashev.get_spec(name)#, plot=True, title=obj)
##    print(f'{obj}')
##    for filt_name in filt_names:
##        flux_col = 'FLUX_' + filt_name
##        filt_prof = burnashev.get_filt(filt_name)
##        star_flux = burnashev.flux_in_filt(burnashev_spec, filt_prof,
##                                   plot=False,
##                                   title=f'{obj} {filt_name}')
##        #print(f'{filt_name} {star_flux}')
##        #print(f'Vega {pd_vega.loc[filt_name:filt_name,"filt_flux"]}')
##
##        vega_flux = pd_vega.loc[filt_name:filt_name,'filt_flux']
##        vega_flux = vega_flux[0]
##        vega_mag0 = pd_vega.loc[filt_name:filt_name,'filt_mag']
##        vega_mag0 = vega_mag0.values[0]
##        vega_mag0 = vega_mag0*u.mag(u.ph/u.cm**2/u.s)
##        vega_flux_mag = u.Magnitude(vega_flux*u.ph/u.cm**2/u.s)
##        star_flux_mag = u.Magnitude(star_flux)
##        filt_prof_star_mag = star_flux_mag - (vega_flux_mag - vega_mag0)
##        #print(f'star_flux_mag = {star_flux_mag}')
##        #print(f'vega_flux_mag = {vega_flux_mag}')
##        #print(f'star relative mag = {star_flux_mag - vega_flux_mag}')
##        #print(f'corrected for vega_mag0 = {filt_prof_star_mag}')
##
##        if (flux_col in simbad_entry.colnames
##            and not np.ma.is_masked(simbad_entry[flux_col])):
##            star_mag = simbad_entry[flux_col]
##            star_mag *= u.mag(u.ph/u.cm**2/u.s)
##            delta = filt_prof_star_mag-star_mag
##            print(f'{filt_name} {star_mag:.2f}; delta = {delta:.2f}')
##        
##        #f, ax = plt.subplots()
##        #ax.step(filt.spectral_axis, filt.flux)
##        #ax.set_title(f"{filt_name}")
##        #plt.show()
##        #ax.set_xlabel("Wavelenth")  
##        #ax.set_ylabel("Transmission")
##        #ax.set_title(f"{filt_name}")
##        #plt.show()
##        #break
##    break

