#!/usr/bin/python3

"""Run standard star photometry pipeline for the IoIO coronagraph"""

# Avoid circular reference of [Red]CorData and photometry pipeline by separating out the Photometry class

import gc
import os
import glob
import argparse

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.dates import datestr2num

import pandas as pd

from astropy import log
from astropy import units as u
from astropy.time import Time
from astropy.stats import mad_std, biweight_location
from astropy.coordinates import Angle, SkyCoord
# Rainy day project to get rid of pandas
#from astropy.table import QTable
from astroquery.simbad import Simbad
from astropy.modeling import models, fitting
from astropy.visualization import quantity_support
from astropy.nddata import CCDData
from astropy.nddata.nduncertainty import StdDevUncertainty

from specutils import Spectrum1D, SpectralRegion
from specutils.analysis import line_flux
from specutils.manipulation import (extract_region,
                                    LinearInterpolatedResampler)

from bigmultipipe import assure_list, no_outfile, cached_pout, prune_pout

from precisionguide import pgproperty

from burnashev import Burnashev

import IoIO.sx694 as sx694
from IoIO.utils import (Lockfile, reduced_dir, get_dirs_dates,
                        cached_csv, iter_polyfit, savefig_overwrite)
from IoIO.cordata_base import CorDataBase
from IoIO.cormultipipe import (IoIO_ROOT, RAW_DATA_ROOT,
                               CorMultiPipeBase,
                               nd_filter_mask)
from IoIO.calibration import Calibration, CalArgparseHandler
from IoIO.photometry import Photometry, is_flux

STANDARD_STAR_ROOT = os.path.join(IoIO_ROOT, 'StandardStar')
FILT_ROOT = os.path.join(IoIO_ROOT, 'observing')

LOCKFILE = '/tmp/standard_star_reduce.lock'

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
        """Returns star flux passing through filt

        Parameters
        ----------
        spec : specutils.Spectrum1D
            Star spectrum read from Burnashev catalog

        filt : specutils.Spectrum1D
            Filter response curve

        resampler : specutils.manipulation or None
            Resampler to translate between filter and star spectrum.
           If ``None``
           `~specutils.manipulation.LinearInterpolatedResampler()` is used
            Default is ``None``

        energy : bool
            Return flux in energy units.  Otherwise photon flux is returned
            Default is ``False``
        
"""
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

def object_to_objctradec(ccd_in, **kwargs):
    """cormultipipe post-processing routine to query Simbad for RA and DEC

    """    
    ccd = ccd_in.copy()
    s = Simbad()
    simbad_results = s.query_object(ccd.meta['OBJECT'])
    obj_entry = simbad_results[0]
    ra = Angle(obj_entry['RA'], unit=u.hour)
    dec = Angle(obj_entry['DEC'], unit=u.deg)
    ccd.meta['OBJCTRA'] = (ra.to_string(),
                      '[hms J2000] Target right assention')
    ccd.meta['OBJCTDEC'] = (dec.to_string(),
                       '[dms J2000] Target declination')
    ccd.meta.insert('OBJCTDEC',
                    ('HIERARCH OBJECT_TO_OBJCTRADEC', True,
                     'OBJCT* point to OBJECT'),
                    after=True)
    return ccd

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
        # --> Consider returning these anyway so I can do stripchart
        # of brightness values
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
    ccd.center_quality = 10
    ND_width = ccd.ND_params[1, 1] - ccd.ND_params[1, 0]
    min_ND_dist = min_ND_multiple * ND_width
    if ccd.obj_to_ND < min_ND_dist:
        # --> Consider returning these stars anyway so I can play with
        # limits at the directory level
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
             'detflux_unit': detflux.unit.to_string(),
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
    #log.debug(f'returning {detflux:.2e} +/- {detflux_err:.2e}')
    return ccd
    
def standard_star_pipeline(directory,
                           glob_include=None,
                           calibration=None,
                           photometry=None,
                           num_processes=None,
                           fits_fixed_ignore=False,
                           **kwargs): 
    """
    Parameters
    ----------
    directory : str
        Directory to process

    glob_include : list of 

    **kwargs passed on to Photometry and CorMultiPipe

    """
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
        return []

    # Pipeline is set with no_outfile so it won't produce any files,
    # but outdir is set in calling code and passed as **kwarg just in case
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
    pout = cmp.pipeline(flist, overwrite=True)
    pout, _ = prune_pout(pout, flist)
    return pout

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
    f.autofmt_xdate()  # orient date labels at a slant
    if outname is not None:
        savefig_overwrite(outname, transparent=True)
    if show:
        plt.show()
    plt.close()
    
def standard_star_directory(directory,
                            pout=None,
                            read_pout=True,
                            write_pout=True,
                            outdir=None,
                            create_outdir=True,
                            show=False,
                            photometry_time_window=1*u.min,
                            stability_factor=np.inf,
                            min_airmass_points=3,
                            **kwargs):
    """
    Parameters
    ----------
    directory : str
        Directory to process

    outdir : str
        Directory in which to write output files

    create_outdir : bool
        Default is ``True``

    pout : list 
        Output of previous pipeline run if available as a variable.
        Use `read_pout` keyword of underlying
        :func:`standard_star_pipeline` to read previous run off of disk

    read_pout : str or bool
        See write_pout.  If file read, simply return that without
        running pipeline.  Default is ``True``

    write_pout : str or bool
        If str, full filename to write pickled pout to.  If True,
        write to 'standard_star.pout' in `directory`.  Default is ``True``

    photometry_time_window: number 

        Reasonable maximum length of time between individual exposures
        in a photometry measurement

    stability_factor : number
        Higher number accepts more data (see code)

    returns: [extinction_data, exposure_correct_data]
        Where extinction_data and exposure_correct_data are lists of type dict
    """
    poutname = os.path.join(outdir, 'standard_star.pout')
    pout = pout or cached_pout(standard_star_pipeline,
                               poutname=poutname,
                               read_pout=read_pout,
                               write_pout=write_pout,
                               directory=directory,
                               outdir=outdir,
                               create_outdir=create_outdir,
                               **kwargs)
        
    if len(pout) == 0:
        log.debug(f'no photometry measurements found in {directory}')
        return [[], []]

    _ , pipe_meta = zip(*pout)
    standard_star_list = [pm['standard_star'] for pm in pipe_meta]
    #df = pd.DataFrame(standard_star_list)
    df = pd.DataFrame(standard_star_list)
    just_date = df['date'].iloc[0]

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
        detflux_unit = objdf['detflux_unit'][row1]
        detflux_unit = u.Unit(detflux_unit)
        obj_coords = SkyCoord(ra, dec)
        simbad_coords = SkyCoord(simbad_entry['RA'],
                                 simbad_entry['DEC'],
                                 unit=(u.hour, u.deg))
        pix_solid_angle = objdf['pix_solid_angle'][row1]
        pix_solid_angle *= u.arcsec**2

        #print(f"{obj}, simbad {simbad_coords.to_string(style='hmsdms')}, "
        #      f"commanded {obj_coords.to_string(style='hmsdms')}")
        #print(simbad_coords.to_string(style='hmsdms'))
        #print(obj_coords.to_string(style='hmsdms'))
        sep = obj_coords.separation(simbad_coords)
        #print(sep)
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
        else:
            log.warning(f'Did not find {obj} in Burnashev catalog')
                                    
        # --> might want to move these in case there are some bad
        # --> measurements skewing the endpoints
        min_airmass = np.amin(objdf['airmass'])
        max_airmass = np.amax(objdf['airmass'])
        min_tm = np.amin(objdf['date_obs'])
        max_tm = np.amax(objdf['date_obs'])
        min_plot_date = min_tm.plot_date
        max_plot_date = max_tm.plot_date
        min_date = min_tm.value
        max_date = max_tm.value
        _, min_time = min_date.split('T')
        _, max_time = max_date.split('T')
        min_time, _ = min_time.split('.')
        max_time, _ = max_time.split('.')
        f = plt.figure(figsize=[8.5, 11])
        plt.suptitle(f"{obj} {just_date} {min_time} -- {max_time} Extinction")
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
                instr_mag = u.Magnitude(best_flux*detflux_unit)
                instr_mag_unit = instr_mag.unit
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
            # --> I will eventually want to extract uncertainties for
            # the fit quantities
            poly = iter_polyfit(airmasses, instr_mags, deg=1, max_resid=1)
            xfit, yfit = poly.linspace()
            plt.plot(xfit, yfit)

            instr_mag_am0 = poly(0)*instr_mag_unit
            flux_col = 'FLUX_'+filt

            # Get our Vega flux in photons and mag reference
            vega_flux = pd_vega.loc[filt:filt,'filt_flux']
            vega_flux = vega_flux[0]
            vega_mag0 = pd_vega.loc[filt_name:filt_name,'filt_mag']
            vega_mag0 = vega_mag0.values[0]
            vega_mag0 = vega_mag0*u.mag(filt_flux.unit)
            vega_flux_mag = u.Magnitude(vega_flux*filt_flux.unit)
            if (simbad_match
                and flux_col in simbad_entry.colnames
                and not np.ma.is_masked(simbad_entry[flux_col])):
                # Prefer Simbad-listed mags for UBVRI
                star_mag = simbad_entry[flux_col]
                star_mag *= u.mag(filt_flux.unit)
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
            else:
                star_mag = np.NAN*u.mag(filt_flux.unit)
                star_flux = np.NAN*filt_flux.unit

            # http://sirius.bu.edu/planetary/obstools/starflux/starcalib/starcalib.htm                
            star_sb = 4*np.pi * star_flux / pix_solid_angle
            star_sb = star_sb.to(u.R)
            # Convert our measurement back to flux units for
            # comparison to integral
            flux_am0 = u.Magnitude(instr_mag_am0).physical    
            rayleigh_conversion = star_sb / flux_am0
            zero_point = star_mag - instr_mag_am0
            extinction_poly = poly.deriv()
            extinction = extinction_poly(0)*instr_mag_am0.unit
            airmasses = np.asarray(airmasses)
            dof = num_airmass_points - 2
            red_chisq = np.sum((poly(airmasses) - instr_mags)**2) / dof
            plt.text(0.5, 0.75, 
                     f'$M_0$ = {zero_point:.2f};   {rayleigh_conversion:0.2e}',
                     ha='center', va='bottom', transform=ax.transAxes)

            plt.text(0.5, 0.5, 
                     f'Airless instr mag = {instr_mag_am0:.2f}',
                     ha='center', va='bottom', transform=ax.transAxes)

            plt.text(0.5, 0.1, 
                     f'Extinction = {extinction:.3f} / airmass',
                     ha='center', va='bottom', transform=ax.transAxes)
            plt.text(0.13, 0.1, 
                     f'Red. Chisq = {red_chisq:.4f}',
                         ha='center', va='bottom', transform=ax.transAxes)

            # --> Min and Max JD or MJD would probably be better
            # --> Needs uncertainties
            extinction_dict = \
                {'date': just_date,
                 'object': obj,
                 'filter': filt,
                 'instr_mag_am0': instr_mag_am0.value,
                 'instr_mag_unit': instr_mag_am0.unit.to_string(),
                 'zero_point': zero_point.value,
                 'zero_point_unit': zero_point.unit.to_string(),
                 'rayleigh_conversion': rayleigh_conversion.value,
                 'rayleigh_conversion_unit': 
                 rayleigh_conversion.unit.to_string(), 
                 'extinction_coef': extinction.value,
                 'extinction_coef_unit': extinction.unit.to_string(),
                 'min_am': min(airmasses),
                 'max_am': max(airmasses),
                 'num_airmass_points ': num_airmass_points,
                 'red_chisq': red_chisq,
                 'max_frac_nonlin': max_frac_nonlin,
                 'min_plot_date': min_plot_date,
                 'max_plot_date': max_plot_date}

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
        outname = os.path.join(outdir, fname)
        if create_outdir:
            os.makedirs(outdir, exist_ok=True)
        savefig_overwrite(outname, transparent=True)
        if show:
            plt.show()
        plt.close()

    if len(exposure_correct_data) > 0:
        outname = os.path.join(outdir, "exposure_correction.png")
        exposure_correct_plot(exposure_correct_data,
                              show=show,
                              outname=outname)

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

    return [extinction_data, exposure_correct_data]

# --> This is not necessarily the best name
def filter_stripchart(df=None,
                      title=None,
                      column=None,
                      outname=None,
                      show=False):
    filters = list(set(df['filter']))
    filters.sort()
    filters.reverse()
    nfilt = len(filters)
    if outname or show:
        f = plt.figure(figsize=[8.5, 11])
        plt.suptitle(f"{title}")
        plot_date_range = [np.min(df['min_plot_date']),
                           np.max(df['max_plot_date'])]
    summary_list = []
    for ifilt, filt in enumerate(filters):
        filtdf = df[df['filter'] == filt]
        to_plot = filtdf[column]
        biweight_loc = biweight_location(to_plot, ignore_nan=True)
        mads = mad_std(to_plot, ignore_nan=True)
        if f'{column}_unit' in df.columns:
            col_unit = df.iloc[0][f'{column}_unit']
        else:
            col_unit = ''
        summary_dict = {'filt': filt,
                        'biweight_loc': biweight_loc,
                        'mad_std': mads,
                        'unit': col_unit}
        summary_list.append(summary_dict)
        if not(outname or show):
            continue

        # If we made it here, we want to include a plot as output
        ax = plt.subplot(nfilt, 1, ifilt+1)
        ax.tick_params(which='both', direction='inout',
                       bottom=True, top=True, left=True, right=True)
        plt.plot_date(filtdf['min_plot_date'], filtdf[column], 'k.')
        plt.plot_date(plot_date_range, [biweight_loc + mads]*2, 'k--')
        plt.plot_date(plot_date_range, [biweight_loc]*2, 'r-')
        plt.plot_date(plot_date_range, [biweight_loc - mads]*2, 'k--')
        
        ax.set_xlim(plot_date_range)
        ax.set_ylim([biweight_loc - 5*mads,
                     biweight_loc + 5*mads])
        plt.ylabel(filt)
        plt.text(0.5, 0.8, 
                 f' {biweight_loc:.4g} +/- {mads:.2g}',
                 ha='center', va='bottom', transform=ax.transAxes)

    if outname or show:
        # Put slanted date axis on bottom plot
        f.autofmt_xdate()
    if outname:
        savefig_overwrite(outname, transparent=True)
    if show:
        plt.show()

    return summary_list

def filt_quantity(filt, data):
    """Returns tuple of Quantity: biweight_loc and mad_std for a particular filter in a list of dictionaries containing the appropriate tags"""
    filt_data = [r for r in data if r['filt'] == filt]
    filt_data = filt_data[0]
    unit = u.Unit(filt_data['unit'])
    return (filt_data['biweight_loc']*unit,
            filt_data['mad_std']*unit)

def extinction_correct(flex_input, airmass=None, ext_coef=None,
                       inverse=False, standard_star_obj=None,
                       **kwargs):
    """Returns atmospheric extinction corrected instr_mag

    Parameters
    ----------
    flex_input : list, `~astropy.nddata.CCDData`, `~astropy.unit.Quantity` or float 
        (1) list: routine will be run on each member, with the output
        formed into a list (useful for list of `~astropy.nddata.CCDData`)

        (2) `~astropy.nddata.CCDData`: this input cannot be expressed
        in a functional `~astropy.unit.Quantity` (e.g.,
        `~astropy.units.Magnitude`), so it will be kept in its current
        units and scaled by the extinction correction in physical
        units

        (3) `~astropy.unit.Quantity` or `float`:  Measured instrument
        magnitude(s) 

    airmass : float, numpy.array, or None
        Airmass(es) at which correction is to be applied.  See 
        standard_star_obj

    ext_coef : `~astropy.unit.Quantity`, float, or None
        Extinction coef in units of instr_mag (per airmass).  See
        standard_star_obj

    inverse : bool
        Invert the extinction correction, where instr_mag is the
        extinction-corrected value and the raw instr_mag is returned
        Default is `False`

    standard_star_obj : ~IoIO.standard_star.StandardStar
        `~IoIO.standard_star.StandardStar` object used in the case
        that flex_input is a `~astropy.nddata.CCDData` to produce
        ext_coef (from ccd.meta['FILTER'])

    **kwargs : dict, optional
        Allows routine to be used as cormultipipe post-processing routine

    Returns
    -------
    Extinction-corrected instrument magnitude as `list`,
    `~astropy.unit.Quantity` or `float` depending on input.  If
    ``inverse`` keyword ``True`` conversion run backward

    """
    if isinstance(flex_input, list):
        return [extinction_correct(fi, airmass=airmass,
                                   ext_coef=ext_coef, inverse=inverse,
                                   standard_star_obj=None,
                                   **kwargs)
                for fi in flex_input]
    if (isinstance(flex_input, CCDData)):
        if airmass is not None or ext_coef is not None:
            raise ValueError('ERROR: airmass and/or ext_coef supplied.  '
                             'CCDData metadata are used for these quantities')
        ccd = flex_input.copy()
        old_ext_corr_val = ccd.meta.get('EXT_CORR_VAL')
        if inverse and (old_ext_corr_val is None
                        or old_ext_corr_val == 1):
            # 1 would be inverse already applied
            raise ValueError('Inversion requested, but data are not extinction corrected')
        filt = ccd.meta['FILTER']
        airmass = ccd.meta['AIRMASS']
        ext_coef, ext_coef_err = standard_star_obj.extinction_coef(filt)
        mag = u.Magnitude(1*ccd.unit)
        ecmag = extinction_correct(mag, airmass, ext_coef,
                                    inverse=inverse)
        ecphys = ecmag.physical
        ccd = ccd.multiply(ecphys.value, handle_meta='first_found')
        ccd.meta['EXT_COEF'] = (
            ext_coef.value,
            f'extinction coefficient ({ext_coef.unit.to_string()})')
        ccd.meta['HIERARCH EXT_COEF_ERR'] = (
            ext_coef_err.value, f'[{ext_coef.unit.to_string()}]')
        ccd.meta['HIERARCH EXT_CORR_VAL'] = (
            ecphys.value, 'extinction cor. factor applied')
        return ccd
    instr_mag = flex_input
    if (isinstance(instr_mag, u.Quantity)
        and isinstance(ext_coef, u.Quantity)):
        assert instr_mag.unit == ext_coef.unit
        # Units get a little sticky when extinction correcting.  In
        # log space, we have a simple linear relationship where the
        # extinction coef is the slope in units of instr_mag/airmass,
        # where airmass is dimensionless.  Were we to convert to
        # non-logarithmic units, we would be doing an exponentiation
        # of the airmass by the extinction coefficient and the result
        # multiplies the measured instrument magnitude in a
        # non-dimensional way.  This is what is confusing to astropy,
        # since, if we were to work with exponentiating mag(electron /
        # s)[/airmass], the units would be weird.  So astropy just
        # fails.  Thus, just work in value space and multiply by
        # proper unit after the fact
        unit = instr_mag.unit
        instr_mag = instr_mag.value
        ext_coef = ext_coef.value
    else:
        unit = 1
    if inverse:
        ext_coef *= -1
    ac = instr_mag - ext_coef*airmass
    return ac*unit
        
def rayleigh_convert(ccd_in, standard_star_obj=None, inverse=False, **kwargs):
    """cormultipipe post-processing routine to convert to rayleighs"""
    if isinstance(ccd_in, list):
        return [to_rayleigh(ccd, inverse=inverse,
                            standard_star_obj=None,
                            **kwargs)
                for ccd in ccd_in]
    ext_corr_val = ccd_in.meta.get('ext_corr_val')
    if not inverse and (ext_corr_val is None
                        or ext_corr_val == 1):
        raise ValueError('Extinction correction needs to be done first')
    if inverse and ext_corr_val is not None and ext_corr_val > 1:
        raise ValueError('Extinction correction still applied.  Invert that first')
    ccd = ccd_in.copy()
    rc, rc_err = standard_star_obj.rayleigh_conversion(ccd.meta['FILTER'])
    if inverse:
        ccd = ccd.divide(rc, handle_meta='first_found')
    else:
        ccd = ccd.multiply(rc, handle_meta='first_found')
    ccd.meta['HIERARCH RAYLEIGH_CONVERSION'] = (
        rc.value, f'[{rc.unit.to_string()}]')
    ccd.meta['HIERARCH RAYLEIGH_CONVERSION_ERR'] = (
        rc_err.value, f'[{rc_err.unit.to_string()}]')
    return ccd
    
def standard_star_tree(raw_data_root=RAW_DATA_ROOT,
                       start=None,
                       stop=None,
                       calibration=None,
                       photometry=None,
                       read_csvs=True,
                       write_csvs=True,
                       create_outdir=True,                       
                       show=False,
                       ccddata_cls=CorDataBase,
                       outdir_root=STANDARD_STAR_ROOT,                       
                       **kwargs):
    dirs_dates = get_dirs_dates(raw_data_root, start=start, stop=stop)
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
    for directory in dirs:
        rd = reduced_dir(directory, outdir_root, create=False)
        extinct_csv = os.path.join(rd, 'extinction.csv')
        expo_cor_csv = os.path.join(rd, 'exposure_correction.csv')
        csvnames = (extinct_csv, expo_cor_csv)
        extinct, expo = cached_csv(standard_star_directory,
                                   csvnames=csvnames,
                                   read_csvs=read_csvs,
                                   write_csvs=write_csvs,
                                   outdir=rd,
                                   create_outdir=create_outdir,
                                   directory=directory,
                                   calibration=calibration,
                                   photometry=photometry,
                                   ccddata_cls=ccddata_cls,
                                   **kwargs)
        extinction_data.extend(extinct)
        exposure_correct_data.extend(expo)

    return extinction_data, exposure_correct_data

# --> Eventually add avability to get extinction for individual night
# I am sensing it would be nice to have a base class that handles the
# start and stop, etc. keyword transfers to property.  Maybe IoIOReduction
class StandardStar():
    def __init__(self,
                 reduce=False,
                 raw_data_root=RAW_DATA_ROOT,
                 start=None,
                 stop=None,
                 calibration=None,
                 photometry=None,
                 read_csvs=True,
                 write_csvs=True,
                 read_pout=True,
                 write_pout=True,
                 create_outdir=True,                       
                 show=False,
                 ccddata_cls=CorDataBase,
                 outdir_root=STANDARD_STAR_ROOT,
                 write_summary_plots=False,
                 lockfile=LOCKFILE,
                 **kwargs):
        self.raw_data_root = raw_data_root
        self.start = start
        self.stop = stop
        self.calibration = calibration
        self.photometry = photometry
        self.read_csvs = read_csvs
        self.write_csvs = write_csvs
        self.read_pout = read_pout
        self.write_pout = write_pout
        self.create_outdir = create_outdir
        self.show = show
        self.ccddata_cls = ccddata_cls
        self.outdir_root = outdir_root
        self.write_summary_plots = write_summary_plots
        self._lockfile = lockfile
        self._kwargs = kwargs
        # These can't be handled by pgproperty becaues they are
        # entangled with the plots via filter_stripchart
        self._zero_points = None
        self._rayleigh_conversions = None
        self._extinction_coefs = None
        if reduce:
            # This will reduce all the data with standard_star_tree
            # and write the following three plots, if desired
            self.zero_points
            self.rayleigh_conversions
            self.extinction_coefs
            # This plot is separate
            if self.write_summary_plots:
                self.exposure_correct_plot()

    @pgproperty
    def calibration(self):
        # If the user has opinions about the time range over which
        # calibration should be done, they should be expressed by
        # creating the calibration object externally and passing it in
        # at instantiation time
        return Calibration(reduce=True)

    # The standard_star_tree code uses IoIO.utils.cached_csv, which
    # returns a list of dictionary lists, one per CSV/file reduction
    # product.  Unfortunately, you have to know based on the code,
    # rather than a dictionary, which reduction product is which.
    # Thus, all of the property, below and standard_star_tree are
    # closely linked
    @pgproperty
    def reduction_products(self):
        lock = Lockfile(self._lockfile)
        lock.create()
        rp = standard_star_tree(raw_data_root=self.raw_data_root,
                                start=self.start,
                                stop=self.stop,
                                calibration=self.calibration,
                                photometry=self.photometry,
                                read_csvs=self.read_csvs,
                                write_csvs=self.write_csvs,
                                read_pout=self.read_pout,
                                write_pout=self.write_pout,
                                create_outdir=self.create_outdir,
                                show=self.show,
                                ccddata_cls=self.ccddata_cls,
                                outdir_root=self.outdir_root,
                                **self._kwargs)
        lock.clear()
        return rp

    @pgproperty
    def extinction_data(self):
        return self.reduction_products[0]

    @pgproperty
    def exposure_correct_data(self):
        return self.reduction_products[1]

    # This no longer depends on order-specific output of standard_star_tree
    @pgproperty
    def extinction_data_frame(self):
        # --> I tried filtering like this and it didn't really change
        # the result
        # -->     df = df[df['extinction_coef'] > 0]
        return pd.DataFrame(self.extinction_data)

    @property
    def start_str(self):
        if self.start is None:
            return ''
        return self.start + '--'

    @property
    def stop_str(self):
        if self.stop is None:
            return ''
        return self.stop + '_'

    @pgproperty
    def created_outdir_root(self):
        return reduced_dir(self.raw_data_root, self.outdir_root,
                           create=self.write_summary_plots)

    def exposure_correct_plot(self):
        if self.write_summary_plots:
            outname = os.path.join(self.created_outdir_root,
                                   f'{self.start_str}{self.stop_str}'
                                   f'exposure_correction.png')
        else:
            outname = None
        exposure_correct_plot(self.exposure_correct_data,
                              outname=outname,
                              latency_change_dates=sx694.latency_change_dates,
                              show=self.show)

    @property
    def zero_points(self):
        if self._zero_points is not None and not self.write_summary_plots:
            return self._zero_points

        zero_point_unit = self.extinction_data_frame.iloc[0]['zero_point_unit']
        outbase = os.path.join(self.created_outdir_root,
                               f'{self.start_str}{self.stop_str}zero_point')
        if self.write_summary_plots:
            outname = outbase+'.png'
        else:
            outname = None
        self._zero_points = cached_csv(filter_stripchart, outbase+'.csv',
                                       read_csvs=False, write_csvs=True,
                                       show=self.show,
                                       df=self.extinction_data_frame,
                                       title=(f'Vega zero point magnitiudes '
                                              f'{zero_point_unit}'),
                                       column='zero_point',
                                       outname=outname)
        return self._zero_points

    @property
    def rayleigh_conversions(self):
        if (self._rayleigh_conversions is not None
            and not self.write_summary_plots):
            return self._rayleigh_conversions

        outbase = os.path.join(self.created_outdir_root,
                               f'{self.start_str}{self.stop_str}'
                               f'rayleigh_conversion')
        if self.write_summary_plots:
            outname = outbase+'.png'
        else:
            outname = None
        rayleigh_conversion_unit = \
            self.extinction_data_frame.iloc[0]['rayleigh_conversion_unit']
        self._rayleigh_conversions = \
            cached_csv(filter_stripchart, outbase+'.csv',
                       read_csvs=False,
                       write_csvs=True,
                       show=self.show,
                       df=self.extinction_data_frame,
                       title=(f'Rayleigh Conversion '
                              f'{rayleigh_conversion_unit}'),
                       column='rayleigh_conversion',
                       outname=outname)
        return self._rayleigh_conversions

    @property
    def extinction_coefs(self):
        if (self._extinction_coefs is not None
            and not self.write_summary_plots):
            return self._extinction_coefs

        outbase = os.path.join(self.created_outdir_root,
                               f'{self.start_str}{self.stop_str}'
                               f'extinction_coefs')
        if self.write_summary_plots:
            outname = outbase+'.png'
        else:
            outname = None
        self._extinction_coefs = \
            cached_csv(filter_stripchart, outbase+'.csv',
                       read_csvs=False,
                       write_csvs=True,
                       show=self.show,
                       df=self.extinction_data_frame,
                       title=f'Extinction coefficients',
                       column='extinction_coef',
                       outname=outname)
        return self._extinction_coefs

    def zero_point(self, filt):
        return filt_quantity(filt, self.zero_points)

    def rayleigh_conversion(self, filt):
        return filt_quantity(filt, self.rayleigh_conversions)

    def extinction_coef(self, filt):
        return filt_quantity(filt, self.extinction_coefs)

    def extinction_correct(self, instr_mag, airmass, filt, **kwargs):
        """Returns atmospheric extinction corrected instr_mag

        Parameters
        ----------
        instr_mag : `~astropy.unit.Quantity`
            Measured instrument magnitude(s) to be corrected, one per 
            airmass

        airmass : float or numpy.array
            Airmass(es) at which correction is to be applied, one per 
            instr_mag

        filt : str
            Filter for which extinction correction has been derived
            by `~standard_star.StandarStar` object

        Returns
        -------
        atmospheric extinction corrected instrument magnitude as
        `~astropy.unit.Quantity` 
        
        """
        ext_coef, _ = self.extinction_coef(filt)
        return extinction_correct(instr_mag, airmass, ext_coef, **kwargs)

class SSArgparseHandler(CalArgparseHandler):
    def add_standard_star_root(self, 
                               default=STANDARD_STAR_ROOT,
                               help=None,
                               **kwargs):
        option = 'standard_star_root'
        if help is None:
            help = f'standard star root (default: {default})'
        self.parser.add_argument('--' + option, 
                                 default=default, help=help, **kwargs)

    def add_standard_star_start(self, 
                                default=None,
                                help=None,
                                **kwargs):
        option = 'standard_star_start'
        if help is None:
            help = 'start directory/date (default: earliest)'
        self.parser.add_argument('--' + option, 
                                 default=default, help=help, **kwargs)

    def add_standard_star_stop(self, 
                               default=None,
                               help=None,
                               **kwargs):
        option = 'standard_star_stop'
        if help is None:
            help = 'stop directory/date (default: latest)'
        self.parser.add_argument('--' + option, 
                                 default=default, help=help, **kwargs)

    def add_show(self, 
                 default=False,
                 help=None,
                 **kwargs):
        option = 'show'
        if help is None:
            help = (f'Show plots interactively')
        self.parser.add_argument('--' + option,
                                 action=argparse.BooleanOptionalAction,
                                 default=default,
                                 help=help, **kwargs)

    def add_write_summary_plots(self, 
                                default=False,
                                help=None,
                                **kwargs):
        option = 'write_summary_plots'
        if help is None:
            help = (f'Write summary plots to standard_star_root')
        self.parser.add_argument('--' + option,
                                 action=argparse.BooleanOptionalAction,
                                 default=default,
                                 help=help, **kwargs)

    def add_all(self):
        """Add options used in cmd"""
        super().add_all()
        self.add_standard_star_root()
        self.add_standard_star_start()
        self.add_standard_star_stop()
        self.add_read_pout(default=True)
        self.add_write_pout(default=True)        
        self.add_read_csvs(default=True)
        self.add_write_csvs(default=True)
        self.add_show()
        self.add_write_summary_plots()

    def cmd(self, args):
        c = CalArgparseHandler.cmd(self, args)
        ss = StandardStar(reduce=True,
                          raw_data_root=args.raw_data_root,
                          outdir_root=args.standard_star_root,
                          start=args.standard_star_start,
                          stop=args.standard_star_stop,
                          calibration=c,
                          read_csvs=args.read_csvs,
                          write_csvs=args.write_csvs,
                          read_pout=args.read_pout,
                          write_pout=args.write_pout,
                          show=args.show,
                          write_summary_plots=args.write_summary_plots,
                          num_processes=args.num_processes,
                          fits_fixed_ignore=args.fits_fixed_ignore)
        return c, ss

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run standard star reduction')
    aph = SSArgparseHandler(parser)
    aph.add_all()
    args = parser.parse_args()
    aph.cmd(args)


    #log.setLevel('DEBUG')
    #parser = argparse.ArgumentParser(
    #    description='IoIO standard star photometric reduction')
    #parser.add_argument(
    #    '--outdir_root', help='root of output file directory tree.  '
    #    'Subdirectories organized by date',
    #    default=STANDARD_STAR_ROOT)
    #parser.add_argument(
    #    '--raw_data_root', help='raw data root (default: {RAW_DATA_ROOT})',
    #    default=RAW_DATA_ROOT)
    #parser.add_argument(
    #    '--calibration_root',
    #    help=f'calibration root (default: {CALIBRATION_ROOT})',
    #    default=CALIBRATION_ROOT)
    #parser.add_argument(
    #    '--calibration_start',
    #    help='calibration start date (default: earliest)')
    #parser.add_argument(
    #    '--calibration_stop', help='calibration stop date (default: latest)')
    #parser.add_argument(
    #    '--start', help='start directory/date (default: earliest)')
    #parser.add_argument(
    #    '--stop', help='stop directory/date (default: latest)')
    #parser.add_argument(
    #    '--num_processes', type=float, default=0,
    #    help='number of subprocesses for parallelization; 0=all cores, <1 = fraction of total cores')
    #parser.add_argument(
    #    '--read_csvs', action=argparse.BooleanOptionalAction, default=True,
    #    help='re-read previous results from CSV files in each subdirectory')
    #parser.add_argument(
    #    '--read_pout', action=argparse.BooleanOptionalAction, default=True,
    #    help='re-read previous pipeline output in each subdirectory')
    #parser.add_argument(
    #    '--show', action=argparse.BooleanOptionalAction,
    #    help='show PyPlot of top-level results (pauses terminal)',
    #    default=False)
    #parser.add_argument(
    #    '--fits_fixed_ignore', action=argparse.BooleanOptionalAction,
    #    help='turn off WCS warning messages',
    #    default=False)
    #
    #args = parser.parse_args()
    #standard_star_cmd(args)


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

### log.setLevel('DEBUG')
### 
### #extinction_data, exposure_correct_data = standard_star_tree(read_csvs=False,
### #                                                            start='2019-06-02')
### #extinction_data, exposure_correct_data = standard_star_tree(read_csvs=False,
### #                                                            start='2019-07-03',
### #                                                            stop='2019-07-03')
### #extinction_data, exposure_correct_data = standard_star_tree(read_csvs=False)
### extinction_data, exposure_correct_data = standard_star_tree()
### df = pd.DataFrame(extinction_data)
### plot_date_range = [np.min(df['min_plot_date']),
###                    np.max(df['max_plot_date'])]
### filters = list(set(df['filter']))
### filters.sort()
### filters.reverse()
### nfilt = len(filters)
### 
### f = plt.figure(figsize=[8.5, 11])
### plt.suptitle(f"Vega zero point magnitiudes {df.iloc[0]['zero_point_unit']}")
### for ifilt, filt in enumerate(filters):
###     filtdf = df[df['filter'] == filt]
###     zero_points = filtdf['zero_point']
###     zp_biweight_loc = biweight_location(zero_points, ignore_nan=True)
###     zp_mad_std = mad_std(zero_points, ignore_nan=True)
###     ax = plt.subplot(nfilt, 1, ifilt+1)
###     ax.tick_params(which='both', direction='inout',
###                    bottom=True, top=True, left=True, right=True)
###     plt.plot_date(filtdf['min_plot_date'], filtdf['zero_point'], 'k.')
###     plt.plot_date(plot_date_range, [zp_biweight_loc + zp_mad_std]*2, 'k--')
###     plt.plot_date(plot_date_range, [zp_biweight_loc]*2, 'r-')
###     plt.plot_date(plot_date_range, [zp_biweight_loc - zp_mad_std]*2, 'k--')
###     
###     ax.set_xlim(plot_date_range)
###     ax.set_ylim([zp_biweight_loc - 5*zp_mad_std,
###                  zp_biweight_loc + 5*zp_mad_std])
###     plt.ylabel(filt)
###     plt.text(0.5, 0.8, 
###              f' {zp_biweight_loc:.2f} +/- {zp_mad_std:.2f}',
###              ha='center', va='bottom', transform=ax.transAxes)
### plt.gcf().autofmt_xdate()
### plt.show()
### 
### f = plt.figure(figsize=[8.5, 11])
### plt.suptitle(f"Rayleigh Conversion {df.iloc[0]['zero_point_unit']}")
### for ifilt, filt in enumerate(filters):
###     filtdf = df[df['filter'] == filt]
###     zero_points = filtdf['rayleigh_conversion']
###     zp_biweight_loc = biweight_location(zero_points, ignore_nan=True)
###     zp_mad_std = mad_std(zero_points, ignore_nan=True)
###     ax = plt.subplot(nfilt, 1, ifilt+1)
###     ax.tick_params(which='both', direction='inout',
###                    bottom=True, top=True, left=True, right=True)
###     plt.plot_date(filtdf['min_plot_date'], filtdf['rayleigh_conversion'], 'k.')
###     plt.plot_date(plot_date_range, [zp_biweight_loc + zp_mad_std]*2, 'k--')
###     plt.plot_date(plot_date_range, [zp_biweight_loc]*2, 'r-')
###     plt.plot_date(plot_date_range, [zp_biweight_loc - zp_mad_std]*2, 'k--')
###     
###     ax.set_xlim(plot_date_range)
###     ax.set_ylim([zp_biweight_loc - 5*zp_mad_std,
###                  zp_biweight_loc + 5*zp_mad_std])
###     plt.ylabel(filt)
###     plt.text(0.5, 0.8, 
###              f' {zp_biweight_loc:.2e} +/- {zp_mad_std:.2e}',
###              ha='center', va='bottom', transform=ax.transAxes)
### plt.gcf().autofmt_xdate()
### plt.show()
### 

#ss = StandardStar()
#print(ss.extinction_coefs)
#
#print(ss.extinction_coef('Na_on'))

