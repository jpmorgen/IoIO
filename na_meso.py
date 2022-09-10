#!/usr/bin/python3

"""Construct model of telluric Na emission. """

import os
import warnings

import numpy as np

import matplotlib.pyplot as plt

from astropy import log
from astropy import units as u
from astropy.table import QTable
from astropy.time import Time
from astropy.stats import mad_std, biweight_location
from astropy.coordinates import (Angle, SkyCoord,
                                 solar_system_ephemeris, get_body,
                                 AltAz)
from astropy.wcs import FITSFixedWarning
from astropy.modeling import models, fitting

from ccdproc import ImageFileCollection

from bigmultipipe import no_outfile, cached_pout, prune_pout

from precisionguide import pgproperty

from IoIO.utils import (Lockfile, reduced_dir, get_dirs_dates,
                        multi_glob, closest_in_coord,
                        valid_long_exposure, im_med_min_max,
                        dict_to_ccd_meta, add_history,
                        csvname_creator, cached_csv, iter_polyfit,
                        savefig_overwrite)
from IoIO.simple_show import simple_show
from IoIO.cordata_base import CorDataBase
from IoIO.cormultipipe import (IoIO_ROOT, RAW_DATA_ROOT,
                               CorMultiPipeBase, angle_to_major_body,
                               nd_filter_mask, combine_masks,
                               mask_nonlin_sat, parallel_cached_csvs)
from IoIO.calibration import Calibration, CalArgparseHandler
from IoIO.photometry import (SOLVE_TIMEOUT, JOIN_TOLERANCE,
                             JOIN_TOLERANCE_UNIT)
from IoIO.cor_photometry import CorPhotometry, add_astrometry
from IoIO.standard_star import (StandardStar, extinction_correct,
                                rayleigh_convert)

BASE = 'Na_meso'  
OUTDIR_ROOT = os.path.join(IoIO_ROOT, BASE)
LOCKFILE = '/tmp/na_meso_reduce.lock'
AWAY_FROM_JUPITER = 5*u.deg # nominally 10 is what I used

# For Photometry -- number of boxes for background calculation
N_BACK_BOXES = 20

# sin wave amplitude for mesospheric emission 
MESO_AMPLITUDE = 15*u.R
MESO_BASELINE = 5*u.R
MESO_AV = MESO_BASELINE + MESO_AMPLITUDE

def shadow_height(alt, sun_alt):
    """This is not the most robust shadow height calculation, it uses thin
    slab approximation and doesn't account for sun on other side of
    earth very well
    
    Parameters
    ----------
    alt : Angle
        Altitude angle of object

    sun_alt : Angle
        Altitude of sun (must be negative)

    """
    assert np.all(sun_alt < 0)
    re = 1*u.R_earth
    re = re.to(u.km)
    tan_alpha_s = -np.tan(sun_alt) # sundown angle
    tan_alpha = np.tan(alt) # observing alt angle
    sh = re * tan_alpha_s**2 * tan_alpha / (tan_alpha + tan_alpha_s)
    return sh

def sun_angles(ccd_in,
               bmp_meta=None,
               **kwargs):
    """cormultipipe post-processing routine that calculates sun alt-az and
    angle between pointing direction and sun.  Adds shadow height via
    simple thin-slab calculation

    """
    ccd = ccd_in.copy()
    with solar_system_ephemeris.set('builtin'):
        sun_coord = get_body('sun', ccd.tavg, ccd.obs_location)
    # https://docs.astropy.org/en/stable/coordinates/common_errors.html
    # notes that separation is order dependent, with the calling
    # object's frame (e.g., sun_coord) used as the reference frame
    # into which the argument (e.g., ccd.sky_coord) is transformed
    sun_angle =  sun_coord.separation(ccd.sky_coord)
    sun_altaz = sun_coord.transform_to(
        AltAz(obstime=ccd.tavg, location=ccd.obs_location))
    objctalt = ccd.meta['OBJCTALT']
    sh = shadow_height(objctalt, sun_altaz.alt)
    adict = {'sun_angle': sun_angle,
             'sun_alt': sun_altaz.alt,
             'sun_az': sun_altaz.az,
             'shadow_height': sh}
    ccd = dict_to_ccd_meta(ccd, adict)
    bmp_meta.update(adict)
    return ccd

def na_meso_process(data,
                    in_name=None,
                    bmp_meta=None,
                    calibration=None,
                    photometry=None,
                    standard_star_obj=None,
                    n_back_boxes=N_BACK_BOXES,
                    show=False,
                    off_on_ratio=None,
                    **kwargs):
    """post-processing routine that processes a *pair* of ccd images
    in the order on-band, off-band"""
    if bmp_meta is None:
        bmp_meta = {}
    if off_on_ratio is None and calibration is None:
        calibration = Calibration(reduce=True)
    if photometry is None:
        photometry = CorPhotometry(precalc=True,
                                   n_back_boxes=n_back_boxes,
                                   **kwargs)
    standard_star_obj = standard_star_obj or StandardStar(reduce=True)
    if off_on_ratio is None:
        off_on_ratio, _ = calibration.flat_ratio('Na')
    jup_dist = angle_to_major_body(data[0], 'jupiter')
    flux_ims = []
    for ccd in data:
        photometry.ccd = ccd
        exptime = ccd.meta['EXPTIME']*u.s
        flux = photometry.background / exptime
        flux_ims.append(flux)

    # Mesosphere is above the stratosphere, where the density of the
    # atmosphere diminishes to very small values.  So all attenuation
    # has already happened by the time we get up to the mesospheric
    # sodium layer.  So do our extinction correction and rayleigh
    # conversion now in hopes that later extinction_correct will get 
    # time-dependent from real measurements
    # https://en.wikipedia.org/wiki/Atmosphere_of_Earth#/media/File:Comparison_US_standard_atmosphere_1962.svg
    background = flux_ims[0] - flux_ims[1]/off_on_ratio
    background = CorDataBase(background, meta=data[0].meta, mask=ccd.mask)
    background = extinction_correct(background,
                                    standard_star_obj=standard_star_obj,
                                    bmp_meta=bmp_meta)
    background = rayleigh_convert(background,
                                  standard_star_obj=standard_star_obj,
                                  bmp_meta=bmp_meta)
    if show:
        simple_show(background)

    # Unfortunately, a lot of data were taken with the filter wheel
    # moving.  This uses the existing bias light/dark patch routine to
    # get uncontaminated part --> consider making this smarter
    best_back, _ = im_med_min_max(background)*background.unit
    best_back_std = np.std(background)*background.unit

    ccd = background
    objctalt = ccd.meta['OBJCTALT']
    objctaz = ccd.meta['OBJCTAZ']
    airmass = ccd.meta.get('AIRMASS')

    tmeta = {'tavg': ccd.tavg,
             'ra': ccd.sky_coord.ra,
             'dec': ccd.sky_coord.dec,
             'jup_dist': jup_dist,
             'best_back': best_back,
             'best_back_std': best_back_std,
             'alt': objctalt*u.deg,
             'az': objctaz*u.deg,
             'airmass': airmass}
    ccd = sun_angles(ccd, bmp_meta=tmeta, **kwargs)
    
    bmp_meta.update(tmeta)

    # In production, we don't plan to write the file, but prepare the
    # name just in case
    in_base = os.path.basename(in_name[0])
    in_base, _ = os.path.splitext(in_base)
    bmp_meta['outname'] = f'Na_meso_{in_base}_airmass_{airmass:.2}.fits'
    # Return one image
    ccd.meta['OFFBAND'] = in_name[1]
    ccd.meta['HIERARCH N_BACK_BOXES'] = (n_back_boxes, 'Background grid for photutils.Background2D')
    ccd.meta['BESTBACK'] = (best_back.value,
                             f'Best background value ({best_back.unit})')
    add_history(ccd.meta,
                'Subtracted OFFBAND, smoothed over N_BACK_BOXES')

    return ccd

def na_meso_collection(directory,
                       glob_include='*-Na_*',
                       warning_ignore_list=[],
                       fits_fixed_ignore=False,
                       **kwargs):

    # I think the best thing to do is here make sure we are a valid
    # long exposure, offset from Jupiter, and that the on- and off
    # centers are within some tolerance of each other, since they
    # should nominally be identical.

    # --> I could consider limiting to 5-min on-band exposures and 60s
    # off-band, but I seem to be able to spot bad outliers pretty
    # easily in the summary plots

    flist = multi_glob(directory, glob_include)
    if len(flist) == 0:
        # Empty collection
        return ImageFileCollection(directory, glob_exclude='*')
    # Create a collection of valid long Na exposures that are
    # pointed away from Jupiter
    collection = ImageFileCollection(directory, filenames=flist)
    st = collection.summary
    valid = ['Na' in f for f in st['filter']]
    valid = np.logical_and(valid, valid_long_exposure(st))
    if np.all(~valid):
        return ImageFileCollection(directory, glob_exclude='*')
    if np.any(~valid):
        # There are generally lots of short exposures, so shorten the
        # list for lengthy calculations below
        fbases = st['file'][valid]
        flist = [os.path.join(directory, f) for f in fbases]
        collection = ImageFileCollection(directory, filenames=flist)
        st = collection.summary
        valid = np.full(len(flist), True)

    # angle_to_major_body(ccd, 'jupiter') would do some good here, but
    # it is designed to work on the ccd CorData level.  So use the
    # innards of that code
    # --> Might be nice to restructure those innards to be available
    # to collections

    # Alternately I could assume Jupiter doesn't move much in one
    # night.  But this isn't too nasty to reproduce
    ras = st['objctra']
    decs = st['objctdec']
    # This is not the exact tavg, but we are just getting close enough
    # to make sure we are not pointed at Juptier
    dateobs_strs = st['date-obs']
    scs = SkyCoord(ras, decs, unit=(u.hourangle, u.deg))
    times = Time(dateobs_strs, format='fits')
    # What I can assume is that a directory has only one observatory
    sample_fname = st[valid]['file'][0]
    sample_fname = os.path.join(directory, sample_fname)
    if fits_fixed_ignore:
        warning_ignore_list.append(FITSFixedWarning)
    with warnings.catch_warnings():
        for w in warning_ignore_list:
            warnings.filterwarnings("ignore", category=w)
        ccd = CorDataBase.read(sample_fname)
    with solar_system_ephemeris.set('builtin'):
        body_coords = get_body('jupiter', times, ccd.obs_location)
    # It is very nice that the code is automatically vectorized
    seps = scs.separation(body_coords)
    valid = np.logical_and(valid, seps > AWAY_FROM_JUPITER)
    if np.all(~valid):
        return ImageFileCollection(directory, glob_exclude='*')
    if np.any(~valid):
        fbases = st['file'][valid]
        flist = [os.path.join(directory, f) for f in fbases]
    collection = ImageFileCollection(directory, filenames=flist)
    return collection
    
def na_meso_pipeline(directory_or_collection=None,
                     calibration=None,
                     photometry=None,
                     n_back_boxes=N_BACK_BOXES,
                     num_processes=None,
                     outdir=None,
                     outdir_root=OUTDIR_ROOT,
                     create_outdir=True,
                     **kwargs):

    if isinstance(directory_or_collection, ImageFileCollection):
        # We are passed a collection when running multiple directories
        # in parallel
        directory = directory_or_collection.location
        collection = directory_or_collection
    else:
        directory = directory_or_collection
        collection = na_meso_collection(directory, **kwargs)

    outdir = outdir or reduced_dir(directory, outdir_root, create=False)

    # At this point, our collection is composed of Na on and off
    # exposures.  Create pairs that have minimum angular separation
    f_pairs = closest_in_coord(collection, ('Na_on', 'Na_off'),
                               valid_long_exposure,
                               directory=directory)

    if len(f_pairs) == 0:
        #log.warning(f'No matching set of Na background files found '
        #            f'in {directory}')
        return []

    calibration = calibration or Calibration(reduce=True)
    photometry = photometry or CorPhotometry(
        precalc=True,
        n_back_boxes=n_back_boxes,
        **kwargs)
        
    cmp = CorMultiPipeBase(
        auto=True,
        calibration=calibration,
        photometry=photometry,
        fail_if_no_wcs=False,
        create_outdir=create_outdir,
        post_process_list=[nd_filter_mask,
                           combine_masks,
                           mask_nonlin_sat,
                           add_astrometry,
                           na_meso_process,
                           no_outfile],                           
        num_processes=num_processes,
        **kwargs)

    # but get ready to write to reduced directory if necessary
    #pout = cmp.pipeline([f_pairs[0]], outdir=outdir, overwrite=True)
    pout = cmp.pipeline(f_pairs, outdir=outdir, overwrite=True)
    pout, _ = prune_pout(pout, f_pairs)
    return pout

def na_meso_directory(directory_or_collection,
                      read_pout=True,
                      write_pout=True,
                      write_plot=True,
                      outdir=None,
                      outdir_root=OUTDIR_ROOT,
                      create_outdir=True,
                      show=False,
                      **kwargs):

    if isinstance(directory_or_collection, ImageFileCollection):
        # We are passed a collection when running multiple directories
        # in parallel
        directory = directory_or_collection.location
        collection = directory_or_collection
    else:
        directory = directory_or_collection
        collection = na_meso_collection(directory, **kwargs)

    outdir = outdir or reduced_dir(directory, outdir_root, create=False)
    poutname = os.path.join(outdir, BASE + '.pout')
    pout = cached_pout(na_meso_pipeline,
                       poutname=poutname,
                       read_pout=read_pout,
                       write_pout=write_pout,
                       directory_or_collection=collection,
                       outdir=outdir,
                       create_outdir=create_outdir,
                       **kwargs)
    if pout is None or len(pout) == 0:
        return QTable()

    _ , pipe_meta = zip(*pout)
    t = QTable(rows=pipe_meta)
    return t

def na_meso_tree(data_root=RAW_DATA_ROOT,
                 start=None,
                 stop=None,
                 calibration=None,
                 photometry=None,
                 standard_star_obj=None,
                 keep_intermediate=None,
                 solve_timeout=SOLVE_TIMEOUT,
                 join_tolerance=JOIN_TOLERANCE*JOIN_TOLERANCE_UNIT,
                 read_csvs=True,
                 write_csvs=True,
                 create_outdir=True,                       
                 show=False,
                 ccddata_cls=None,
                 outdir_root=OUTDIR_ROOT,                 
                 **kwargs):
    dirs_dates = get_dirs_dates(data_root, start=start, stop=stop)
    if len(dirs_dates) == 0:
        log.warning(f'No data in time range {start} {stop}')
        return QTable()
    dirs, _ = zip(*dirs_dates)
    if len(dirs) == 0:
        log.warning('No directories found')
        return
    calibration = calibration or Calibration()
    if photometry is None:
        photometry = CorPhotometry(
            precalc=True,
            solve_timeout=solve_timeout,
            join_tolerance=join_tolerance)
    standard_star_obj = standard_star_obj or StandardStar(reduce=True)

    cached_csv_args = {
        'csvnames': csvname_creator,
        'csv_base': BASE + '.ecsv',
        'write_csvs': write_csvs,
        'calibration': calibration,
        'photometry': photometry,
        'standard_star_obj': standard_star_obj,
        'outdir_root': outdir_root,
        'create_outdir': create_outdir}
    cached_csv_args.update(**kwargs)
    summary_table = parallel_cached_csvs(dirs,
                                         code=na_meso_directory,
                                         collector=na_meso_collection,
                                         files_per_process=3,
                                         read_csvs=read_csvs,
                                         **cached_csv_args)
    if summary_table is not None:
        summary_table.write(os.path.join(outdir_root, BASE + '.ecsv'),
                            overwrite=True)
    return summary_table

class NaMeso:
    def __init__(self,
                 qtable_fname=None):
        if qtable_fname is None:
            qtable_fname = os.path.join(OUTDIR_ROOT, BASE + '.ecsv')
        self.qtable_fname = qtable_fname
        self._qtable = None

    @property
    def qtable(self):
        """Basic storage of data.  --> may add reduction step if not found"""
        if self._qtable is None:
            self._qtable = QTable.read(self.qtable_fname)
        return self._qtable

    @property
    def meso_mag(self):
        """Mesospheric emission expressed in mags, after correction for
        tropospheric extinction"""
        if 'meso_mag' not in self.qtable.colnames:
            self.qtable['meso_mag'] = u.Magnitude(self.qtable['best_back'])
        return self.qtable['meso_mag']


    @pgproperty
    def meso_airmass_poly(self):
        """Returns order 1 polynomial fit to observed mesophereic emission in
        magnitudes vs Kasten & Young (1989) airmass.  All datapoints
        are used.  This could be refinded geometrically to get a
        better path length through the mesopheric Na layer.  Could
        also use astropy's native iterative fitting.

        """
        return iter_polyfit(self.qtable['airmass'], self.meso_mag,
                            deg=1, max_resid=1)

    @pgproperty
    def meso_col_per_airmass(self):
        """Uses the dataset as a whole to calculate mesospheric column
        density per Kasten & Young (1989) airmass

        """
        meso_col_poly = self.meso_airmass_poly.deriv()
        return meso_col_poly(0)*self.meso_mag.unit

    def meso_mag_to_vcol(self, meso_mag, airmass, inverse=False):
        """Returns predicted mesospheric vertical column emission in magnitudes

        This could be improved geometrically so pathlength in
        mesosphere is used rather than airmass in troposphere.  In
        principle, this function has a seasonal dependence as the
        parameters of the sodium layer, like thickness and centroid,
        change.  See Dunker et al 2015

        Parameters
        ----------
        meso_mag : Quantity
            Mesospheric emission in magnitudes (observed instr_mag
            corrected for tropospheric airmass)

        airmass : float-like
            Airmass (ideally in troposphere)

        inverse : bool
            Invert the model

        """
        return extinction_correct(meso_mag,
                                  airmass=airmass,
                                  ext_coef=self.meso_col_per_airmass,
                                  inverse=inverse)
        
    @property
    def model_meso_vcol(self):
        """Returns predicted mesospheric vertical column emission in
        magnitudes for all measured points by correcting for fitted
        meso_vcol_per_airmass

        """
        if 'model_meso_vcol' not in self.qtable.colnames:
            self.qtable['model_meso_vcol'] = self.meso_mag_to_vcol(
                self.meso_mag, self.qtable['airmass'])
        return self.qtable['model_meso_vcol']

    # --> This is going to become obsolete
    @property
    def shadow_height(self):
        if 'shadow_height' not in self.qtable.colnames:
            self.qtable['shadow_height'] = shadow_height(
                self.qtable['alt'],
                self.qtable['sun_alt'])
        return self.qtable['shadow_height']        

    @property
    def dex_shadow_height(self):
        if 'dex_shadow_height' not in self.qtable.colnames:
            self.qtable['dex_shadow_height'] = u.Dex(
                self.qtable['shadow_height'])
        return self.qtable['dex_shadow_height']


    @pgproperty
    def vcol_dex_shadow_height_poly(self):
        """Returns order 3 polynomial fit to observed mesospheric vertical
        column in magnitudes vs. log shadow height.  All datapoints
        are used.  

        """
        return iter_polyfit(self.dex_shadow_height,
                            self.model_meso_vcol,
                            deg=4, max_resid=1)

    def vcol_dex_shadow_height_poly_quantity(self, dex_shadow_height):
        v = self.vcol_dex_shadow_height_poly(dex_shadow_height.value)
        return v*self.model_meso_vcol.unit
    
    @pgproperty
    def vcol_dex_shadow_height_poly_av(self):
        return np.mean(
            self.vcol_dex_shadow_height_poly_quantity(
                self.dex_shadow_height))

    def vcol_to_shadow_corrected(self, vcol, shadow_height, inverse=False):
        model = self.vcol_dex_shadow_height_poly_quantity(
            u.Dex(shadow_height))
        av = self.vcol_dex_shadow_height_poly_av
        if inverse:
            return vcol + model - av
        return vcol - model + av

    @property
    def model_vcol_shadow_corrected(self):
        if 'model_vcol_shadow_corrected' not in self.qtable.colnames:
            self.qtable['model_vcol_shadow_corrected'] = \
                self.vcol_to_shadow_corrected(
                    self.model_meso_vcol, self.shadow_height)
        return self.qtable['model_vcol_shadow_corrected']

    @pgproperty
    def meso_vcol_corrected_sin_physical(self):
        """Returns hand-fit sin function of shadow height-corrected meso
        vertical column vs. JD.  Don in physical units, since that
        makes more sense than logarithmic, though that sort of works too.

        """
        # This doesn't do any fit.  I have just tweaked the parameters
        # by hand
        # --! Aim for the low side to see if that helps make sure
        # background is not over-estimated

        # --> Might want to add in a Gaussian around the 1st of the year
        
        fit = fitting.LevMarLSQFitter()
        sin_init = (models.Sine1D(amplitude=MESO_AMPLITUDE.value,
                                  frequency=2/u.year,
                                  phase=(-20*u.deg).to(u.rad))
                    + models.Const1D(amplitude=MESO_AV.value))
        #sin_init = (models.Sine1D(amplitude=MESO_AMPLITUDE.value,
        #                          frequency=2/u.year,
        #                          phase=(-10*u.deg).to(u.rad))
        #            + models.Sine1D(amplitude=MESO_AV.value,
        #                            frequency=2/u.year,
        #                            phase=(-30*u.deg).to(u.rad))
        #            + models.Const1D(amplitude=MESO_AV.value))
        #sin_init = (models.Sine1D(amplitude=MESO_AMPLITUDE.value,
        #                          frequency=1/u.year,
        #                          phase=(+5*u.deg).to(u.rad))
        #            + models.Const1D(amplitude=MESO_AV.value))
        return fit(sin_init,
                   self.qtable['tavg'].jd*u.day,
                   self.model_vcol_shadow_corrected.physical)

    def meso_vcol_corrected_sin_quantity(self, tavg):
        v = self.meso_vcol_corrected_sin_physical(tavg.jd*u.day)
        return v

    def shadow_corrected_to_no_time(self, meso_shadow_corrected, tavg,
                                    inverse=False):
        """Transforms shadow height-corrected mesospheric vertical column
        density to time-corrected and inverse

        """
        model = self.meso_vcol_corrected_sin_quantity(tavg)
        if inverse:
            return meso_shadow_corrected + model - MESO_AV
        return meso_shadow_corrected - model + MESO_AV

    def meso_model_inverse(self, tavg, airmass, shadow_height):
        shadow_corrected = self.shadow_corrected_to_no_time(
            MESO_AV, tavg, inverse=True)
        vcol = self.vcol_to_shadow_corrected(
            u.Magnitude(shadow_corrected), shadow_height, inverse=True)
        meso_mag = self.meso_mag_to_vcol(vcol, airmass, inverse=True)
        meso = meso_mag.physical
        return meso        

    @property
    def ijd(self):
        if 'ijd' not in self.qtable.colnames:
            ijd = self.qtable['tavg'].jd.astype(int)
            self.qtable['ijd'] = self.qtable['tavg'].jd.astype(int)
        return self.qtable['ijd']

    @property
    def itavg(self):
        if 'itavg' not in self.qtable.colnames:
            self.qtable['itavg'] = Time(self.ijd, format='jd')
        return self.qtable['itavg']

    @property
    def daily_shadow_corrected(self):
        """Returns daily biweight location of shadow height-corrected Na meso
        emission.  Also calculates mad_std and groups qtable by ijd

        """
        if 'daily_shadow_corrected' not in self.qtable.colnames:
            unit  =self.model_vcol_shadow_corrected.physical.unit
            self.qtable['daily_shadow_corrected'] = np.NAN*unit
            self.qtable['daily_shadow_corrected_std'] = np.NAN*unit
            for i, ijd in enumerate(self.ijd):
                mask = self.ijd == ijd
                t_shadow_corrected = self.model_vcol_shadow_corrected[mask]
                t_shadow_corrected = t_shadow_corrected.physical
                tbiweight = biweight_location(t_shadow_corrected)
                tstd = mad_std(t_shadow_corrected)
                self.qtable['daily_shadow_corrected'][mask] = tbiweight
                self.qtable['daily_shadow_corrected_std'][mask] = tstd

        self._qtable = self.qtable.group_by('ijd')
        return self.qtable['daily_shadow_corrected']

    @property
    def daily_shadow_corrected_std(self):
        """Daily mad std of shadow height-corrected Na meso emission

        """
        if 'daily_shadow_corrected_std' not in self.qtable.colnames:
            self.daily_shadow_corrected
        return self.qtable['daily_shadow_corrected_std']

    @pgproperty
    def shadow_corrected_av_err(self):
        """Avervage error for all shadow height-corrected data.  Uses biweight
        location of daily shadow height-corrected mad_stds

        """
        return biweight_location(self.daily_shadow_corrected_std,
                                 ignore_nan=True)

    def best_na_meso(self, tavg, airmass, shadow_height): 
        """Returns the best-estimate Na mesospheric emission 

        Uses all Na background meausrements to construct an empirical,
        time-dependent model of mesospheric Na emission.  This routine
        first looks to see if background observations were recorded on
        the date of observation and uses the biweight distribution of
        those

        Parameters
        ----------
        tavg : `~astropy.time.Time`
            Time of observation

        airmass : float
            Airmass of observation look direction

        shadow_height : Quantity
            Height above earth surface where line of sight intersects
            earth's shadow 

        Returns
        -------
        meso, meso_err : tuple of Quantity
            best-estimate Na mesospheric emission

        """
        ijd = tavg.jd.astype(int)
        self.daily_shadow_corrected
        mask = self.qtable.groups.keys['ijd'] == ijd
        # Shadow-corrected vertical column
        ijdt = self.qtable.groups[mask]
        shadow_corrected = ijdt['daily_shadow_corrected']
        shadow_corrected_err = ijdt['daily_shadow_corrected_std']
        #if (len(shadow_corrected.value) == 0
        #    or not np.isfinite(shadow_corrected)):
        if len(shadow_corrected.value) == 0:
            meso = self.meso_model_inverse(tavg, airmass, shadow_height)
            meso_err = self.shadow_corrected_av_err
            return meso, meso_err

        shadow_corrected = u.Magnitude(shadow_corrected[0])
        shadow_corrected_err = shadow_corrected_err[0]
        vcol = self.vcol_to_shadow_corrected(u.Magnitude(shadow_corrected),
                                             shadow_height,
                                             inverse=True)
        meso_mag = self.meso_mag_to_vcol(vcol, airmass, inverse=True)
        meso = meso_mag.physical
        return meso, shadow_corrected_err

    def plots(self):
        f = plt.figure(figsize=[11, 8.5])
        ax = plt.subplot(3, 3, 1)
        plt.plot(self.qtable['airmass'], self.qtable['best_back'], 'k.')
        plt.xlabel(f'airmass')
        plt.ylabel(f'meso Na emission ({self.qtable["best_back"].unit})')

        ax = plt.subplot(3, 3, 2)
        plt.plot(self.qtable['best_back_std'],
                 self.qtable['best_back'], 'k.')
        ax.set_xscale('linear')
        plt.xlabel(f'meso Na emission ({self.qtable["best_back"].unit})')
        plt.ylabel(f'meso Na emission std ({self.qtable["best_back_std"].unit})')
        ax = plt.subplot(3, 3, 3)
        plt.plot(self.qtable['airmass'], self.meso_mag, 'k.')
        plt.plot(self.qtable['airmass'],
                 self.meso_airmass_poly(self.qtable['airmass']), 'r-')
        ax.invert_yaxis()        
        plt.xlabel(f'airmass')
        plt.ylabel(f'meso Na emission ({self.model_meso_vcol.unit})')

        #ax = plt.subplot(3, 3, 3)
        #plt.plot(self.qtable['airmass'], self.model_meso_vcol, 'k.')
        #plt.xlabel(f'airmass')
        #ax.invert_yaxis()        
        #plt.ylabel(f'Vert. col. meso Na ({self.model_meso_vcol.unit})')

        #ax = plt.subplot(3, 3, 4)
        #plt.plot(self.shadow_height, self.model_meso_vcol, 'k.')
        #ax.set_xscale('log')
        #ax.invert_yaxis()        
        #plt.xlabel(f'Shadow height ({self.shadow_height.unit})')
        #plt.ylabel(f'Vert. col. Na ({self.model_meso_vcol.unit})')

        ax = plt.subplot(3, 3, 4)
        plt.plot(self.shadow_height, self.model_meso_vcol, 'k.')
        plt.plot(self.shadow_height,
                 self.vcol_dex_shadow_height_poly_quantity(
                     self.dex_shadow_height), 'r.')
        ax.set_xscale('log')
        ax.invert_yaxis()        
        plt.xlabel(f'Shadow height ({self.shadow_height.unit})')
        plt.ylabel(f'Vert. col. Na ({self.model_meso_vcol.unit})')

        #ax = plt.subplot(3, 3, 6)
        #plt.plot(self.shadow_height, self.model_vcol_shadow_corrected, 'k.')
        #ax.set_xscale('log')
        #ax.invert_yaxis()        
        #plt.xlabel(f'Shadow height (SH) ({self.shadow_height.unit})')
        #plt.ylabel(f'SH corr. vert. col. Na ({self.model_meso_vcol.unit})')

        #ax = plt.subplot(3, 3, 7)
        #plt.plot(self.qtable['jup_dist'],
        #         self.model_vcol_shadow_corrected, 'k.')
        #ax.set_xscale('linear')
        #ax.invert_yaxis()        
        #plt.xlabel(f'Jupiter distance ({self.qtable["jup_dist"].unit})')
        #plt.ylabel(f'SH corr. vert. col. Na ({self.model_meso_vcol.unit})')

        #ax = plt.subplot(3, 3, 7)
        #plt.plot(self.qtable['jup_dist'],
        #         self.meso_mag, 'k.')
        #ax.set_xscale('linear')
        #ax.invert_yaxis()        
        #plt.xlabel(f'Jupiter distance ({self.qtable["jup_dist"].unit})')
        #plt.ylabel(f'SH corr. vert. col. Na ({self.model_meso_vcol.unit})')

        #ax = plt.subplot(3, 3, 7)
        #plt.plot(self.qtable['tavg'].datetime,
        #         self.model_vcol_shadow_corrected,
        #         'k.')
        #dt = self.qtable['tavg'].datetime
        ##days = np.arange(np.min(dt), np.max(dt))
        #plt.xlabel(f'Date')
        #plt.ylabel(f'SH corr. vert. col. Na ({self.model_meso_vcol.unit})')
        #ax.invert_yaxis()        
        ##plt.plot(dt, self.meso_vcol_corrected_sin_physical(days), 'r.')
        ##ax.set_ylim([0, 0.02])
        #ax.tick_params(axis='x', labelrotation = 45)

        ax = plt.subplot(3, 3, 5)
        plt.plot(self.qtable['tavg'].datetime,
                 self.model_vcol_shadow_corrected.physical, 'k.')
        plt.xlabel(f'Date')
        plt.ylabel(f'SH corr. vert. col. Na ({self.model_meso_vcol.physical.unit})')
        plt.plot(self.qtable['tavg'].datetime,
                 self.meso_vcol_corrected_sin_physical(
                     self.qtable['tavg'].jd*u.day), 'r.')
        ax.tick_params(axis='x', labelrotation = 45)

        ax = plt.subplot(3, 3, 6)
        plt.plot(self.qtable['tavg'].datetime,
                 self.shadow_corrected_to_no_time(
                     self.model_vcol_shadow_corrected.physical,
                     self.qtable['tavg']),
                 'k.')
        plt.errorbar(self.qtable['tavg'].datetime,
                     self.daily_shadow_corrected.value,
                     self.daily_shadow_corrected_std.value,
                     fmt='r.')
        plt.xlabel(f'Date')
        plt.ylabel(f'Model residual ({self.model_meso_vcol.physical.unit})')
        ax.tick_params(axis='x', labelrotation = 45)

        plt.tight_layout()
        plt.show()


#m = NaMeso()
#m.plots()
#t = Time('2020-01-01T00:01:00', format='fits')
#print(m.best_na_meso(t, 2, 1E5*u.km))
#t = Time('2019-10-23T00:01:00', format='fits')
#print(m.best_na_meso(t, 2, 1E5*u.km))

    
##on_fname = '/data/IoIO/raw/20210617/Jupiter-S007-R001-C001-Na_on.fts'
##off_fname = '/data/IoIO/raw/20210617/Jupiter-S007-R001-C001-Na_off.fts'
##on = CorDataBase.read(on_fname)
##off = CorDataBase.read(off_fname)
##bmp_meta = {'ads': 2}
##ccd = na_meso_process([on, off], in_name=[on_fname, off_fname],
##                      bmp_meta=bmp_meta, show=True)
#
#
#directory = '/data/IoIO/raw/20210617'
#directory = '/data/IoIO/raw/2017-07-01/'
#
##pout = na_meso_pipeline(directory, fits_fixed_ignore=True)
#
##t = na_meso_directory(directory, fits_fixed_ignore=True)
#
##t = na_back_tree(start='2021-06-17', stop='2021-06-17', fits_fixed_ignore=True)
#
##collection = na_meso_collection(directory)
#
#t = na_meso_tree(start='2021-06-17', stop='2021-06-17', fits_fixed_ignore=True)
#
##t = na_meso_tree(start='2021-06-01', stop='2021-06-17', fits_fixed_ignore=True)
#
##t = na_meso_tree(start='2021-12-01', stop='2021-12-17', fits_fixed_ignore=True)
#
##t = na_meso_tree(start='2017-07-01', stop='2017-07-01', fits_fixed_ignore=True)
#
##fp = closest_in_time(collection, ('Na_on', 'Na_off'),
##                     valid_long_exposure,
##                     directory=directory)
##cp = closest_in_coord(collection, ('Na_on', 'Na_off'),
##                     valid_long_exposure,
##                     directory=directory)
#
##t = na_meso_tree(fits_fixed_ignore=True)
#
#from IoIO.standard_star import extinction_correct
#t = QTable.read(os.path.join(OUTDIR_ROOT, BASE + '.ecsv'))
#t['bbeps'] = t['best_back'] / (3610 * u.R * u.s/u.electron)
#t['bbmag'] = u.Magnitude(t['bbeps'])
#t['instr_mag'] = extinction_correct(t['bbmag'], t['airmass'],
#                                    t['extinction_coef'], 
#                                    inverse=True)
#t['nbmag'] =  extinction_correct(t['instr_mag'], t['airmass'],
#                                 0.366*t['extinction_coef'].unit)
#t['new_back'] = t['nbmag'].physical * 3610 * u.R * u.s/u.electron
#
## Shadow height above observatory
#t['shadow_height'] = 1*u.R_earth * (1/np.cos(t['sun_alt']) - 1)
## Refine to shadow height along line of sight
#re = 1*u.R_earth
#re = re.to(u.m)
#sign = np.ones(len(t))
#dang = Angle(t['sun_az'] - t['az'])
#dang = dang.wrap_at(180*u.deg)
#dang = np.abs(dang)
##sign[dang > 90*u.deg] = -1
#tan_alpha_s = -np.tan(t['sun_alt']) # sundown angle
#tan_alpha = np.tan(t['alt']) # observing alt angle
#t['shadow_height'] = (re * tan_alpha_s**2 * tan_alpha
#                      / (tan_alpha + sign*tan_alpha_s))
##plt.plot(t['tavg'].datetime, t['new_back'] - t['best_back'], 'k.')
##plt.plot(t['tavg'].datetime, t['best_back'], 'k.')
##plt.plot(t['tavg'].datetime, t['new_back'], 'k.')
##plt.plot(t['tavg'].datetime, t['sun_az'] - t['az'], 'k.')
##plt.plot(t['tavg'].datetime,
##         np.abs(Angle(t['sun_az'] - t['az']).wrap_at(180*u.deg)), 'k.')
##plt.gcf().autofmt_xdate()  # orient date labels at a slant
##plt.plot(t['airmass'], t['best_back'], 'k.')
#plt.plot(t['airmass'], t['new_back'], 'k.')
#axes = plt.gca()
#axes.set_yscale('log')
#
##plt.plot(t['shadow_height'], t['new_back'], 'k.')
##axes = plt.gca()
##axes.set_xscale('log')
###axes.set_yscale('log')
#
##plt.plot(t['shadow_height'], t['sun_angle'], 'k.')
##plt.plot(t['shadow_height'], t['airmass'], 'k.')
##axes = plt.gca()
##axes.set_xscale('log')
#
##plt.plot(t['airmass'], t['sun_angle'], 'k.')
##plt.plot(t['sun_angle'], t['best_back'], 'k.')
##mask = t['sun_angle'] < 80*u.deg # linear
##mask = np.logical_and(80*u.deg < t['sun_angle'], t['sun_angle'] < 120*u.deg) # some swoops
##mask = np.logical_and(120*u.deg < t['sun_angle'], t['sun_angle'] < 180*u.deg) #most swoops
##mask = np.logical_and(120*u.deg < t['sun_angle'], t['sun_angle'] < 140*u.deg) # lower swoop
##mask = np.logical_and(140*u.deg < t['sun_angle'], t['sun_angle'] < 180*u.deg) # upper swoop + no swoop
##mask = np.logical_and(160*u.deg < t['sun_angle'], t['sun_angle'] < 180*u.deg) # less upper swoop & no swoop
##plt.plot(t['airmass'][mask], t['best_back'][mask], 'k.')
##plt.plot(t['sun_angle'] - t['alt'], t['best_back'], 'k.')
##plt.plot(t['shadow_height'], t['airmass'], 'k.')
#
#
## Shows nice swoops.  But these are away from the earth.  Ah HA!  This
## might be the earth's sodium tail.  No.  Meso Na layer is 80 -- 105
## km.  This might be a shadow height effect, where part of that layer
## is illuminated
##plt.plot(t['sun_alt'], t['best_back'], 'k.')
##plt.plot(np.abs(t['az'] - t['sun_az']), t['best_back'], 'k.')
##plt.plot(np.abs(t['alt'] - t['sun_alt']), t['best_back'], 'k.')
##plt.plot(1/np.cos(-t['sun_alt']), t['best_back'], 'k.')
#
##mask = -40*u.deg < -t['sun_alt']
##plt.plot(t['airmass'][mask], t['best_back'][mask], 'k.')
#
#plt.show()
