#!/usr/bin/python3

"""Construct model of telluric Na emission. """

import os
import argparse
import warnings

import numpy as np

import matplotlib.pyplot as plt

from astropy import log
from astropy import units as u
from astropy.table import QTable, unique, hstack
from astropy.time import Time
from astropy.stats import mad_std, biweight_location
from astropy.convolution import Box1DKernel
from astropy.coordinates import (Angle, SkyCoord,
                                 solar_system_ephemeris, get_body,
                                 AltAz)
from astropy.wcs import FITSFixedWarning

from ccdproc import ImageFileCollection

from bigmultipipe import no_outfile, cached_pout, prune_pout

from precisionguide import pgproperty

from IoIO.utils import (Lockfile, reduced_dir, get_dirs_dates,
                        multi_glob, closest_in_coord,
                        valid_long_exposure, im_med_min_max,
                        dict_to_ccd_meta, add_history,
                        csvname_creator, cached_csv, iter_polyfit,
                        daily_biweight, daily_convolve,
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
from IoIO.cor_photometry import (CorPhotometry,
                                 CorPhotometryArgparseMixin, add_astrometry)
from IoIO.standard_star import (StandardStar, SSArgparseHandler,
                                extinction_correct, rayleigh_convert)

BASE = 'Na_meso'  
OUTDIR_ROOT = os.path.join(IoIO_ROOT, BASE)
LOCKFILE = '/tmp/na_meso_reduce.lock'
AWAY_FROM_JUPITER = 5*u.deg # nominally 10 is what I used

# For Photometry -- number of boxes for background calculation
N_BACK_BOXES = 20

# sin wave amplitude for mesospheric emission 
MESO_AMPLITUDE = 20*u.R
MESO_BASELINE = 8*u.R
MESO_AV = MESO_BASELINE + MESO_AMPLITUDE
MEDFILT_WIDTH = 31

def shadow_height(alt, sun_alt):
    """This is not the most robust shadow height calculation, it uses thin
    slab approximation and doesn't account for sun on other side of
    earth very well
    
    Parameters
    ----------
    alt : Angle or Quantity
        Altitude angle of object

    sun_alt : Angle or Quantity
        Altitude of sun (must be negative)

    """
    if not ((isinstance(alt, u.Quantity) or isinstance(alt, Angle))
            and (isinstance(sun_alt, u.Quantity)
                 or isinstance(sun_alt, Angle))):
        raise ValueError(f'Not Quantities alt = {alt} {type(alt)}, sun_alt = {sun_alt} {type(alt)}')
    if np.any(sun_alt > 0):
        raise ValueError(f'Sun above horizon {sun_alt}')
    re = 1*u.R_earth
    re = re.to(u.km)
    tan_alpha_s = -np.tan(sun_alt) # sundown angle
    tan_alpha = np.tan(alt)
    sh = re * tan_alpha_s**2 * tan_alpha / (tan_alpha + tan_alpha_s)
    return sh

def sun_angles(ccd,
               bmp_meta=None,
               **kwargs):
    """cormultipipe post-processing routine that calculates sun alt-az and
    angle between pointing direction and sun.  Adds shadow height via
    simple thin-slab calculation

    """
    with solar_system_ephemeris.set('builtin'):
        sun_coord = get_body('sun', ccd.tavg, ccd.obs_location)
    # https://docs.astropy.org/en/stable/coordinates/common_errors.html
    # notes that separation is order dependent, with the calling
    # object's frame (e.g., sun_coord) used as the reference frame
    # into which the argument (e.g., ccd.sky_coord) is transformed
    sun_angle =  sun_coord.separation(ccd.sky_coord)
    sun_altaz = sun_coord.transform_to(
        AltAz(obstime=ccd.tavg, location=ccd.obs_location))
    sh = shadow_height(ccd.alt_az.alt, sun_altaz.alt)
    adict = {'sun_angle': sun_angle,
             'sun_alt': sun_altaz.alt,
             'sun_az': sun_altaz.az,
             'shadow_height': sh}
    ccd_out = dict_to_ccd_meta(ccd, adict)
    bmp_meta.update(adict)
    return ccd_out

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

    # Copy some of our CCD meta into bmp_meta so we can play with it
    # in our QTable
    ccd = background
    objctalt = ccd.meta['OBJCTALT']*u.deg
    objctaz = ccd.meta['OBJCTAZ']*u.deg
    airmass = ccd.meta.get('AIRMASS')

    tmeta = {'tavg': ccd.tavg,
             'ra': ccd.sky_coord.ra,
             'dec': ccd.sky_coord.dec,
             'jup_dist': jup_dist,
             'best_back': best_back,
             'best_back_std': best_back_std,
             'alt': objctalt,
             'az': objctaz,
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
                 qtable=None,
                 outdir_root=OUTDIR_ROOT,
                 base=BASE,
                 qtable_fname=None,
                 plot_fname=None,
                 show=False):
        self.outdir_root = outdir_root
        self.base = base
        self.qtable_fname = (qtable_fname
                             or os.path.join(self.outdir_root,
                                             self.base + '.ecsv'))
        self.plot_fname = (plot_fname
                           or os.path.join(self.outdir_root,
                                           self.base + '.png'))
        self.show = show

    @pgproperty
    def qtable(self):
        """Basic storage of data"""
        return QTable.read(self.qtable_fname)

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

    @property
    def dex_shadow_height(self):
        if 'dex_shadow_height' not in self.qtable.colnames:
            self.qtable['dex_shadow_height'] = u.Dex(
                self.qtable['shadow_height'])
        return self.qtable['dex_shadow_height']

    @pgproperty
    def vcol_dex_shadow_height_poly(self):
        """Returns polynomial fit to observed mesospheric vertical
        column in magnitudes vs. log shadow height.  All datapoints
        are used.  

        """
        return iter_polyfit(self.dex_shadow_height,
                            self.model_meso_vcol,
                            deg=5, max_resid=1)

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
                    self.model_meso_vcol, self.qtable['shadow_height'])
        return self.qtable['model_vcol_shadow_corrected']

    @property
    def ijd(self):
        if 'ijd' not in self.qtable.colnames:
            self.qtable['ijd'] = self.qtable['tavg'].jd.astype(int)
        return self.qtable['ijd']

    @property
    def model_vcol_shadow_corrected_physical(self):
        if 'model_vcol_shadow_corrected_physical' not in self.qtable.colnames:
            self.qtable['model_vcol_shadow_corrected_physical'] = \
                self.model_vcol_shadow_corrected.physical
            return self.qtable['model_vcol_shadow_corrected_physical']

    @property
    def daily_shadow_corrected(self):
        self.ijd
        self.model_vcol_shadow_corrected_physical
        if 'daily_shadow_corrected' not in self.qtable.colnames:
            daily_biweight(
                self.qtable,
                day_col='ijd',
                data_col='model_vcol_shadow_corrected_physical',
                biweight_col='daily_shadow_corrected',
                std_col='daily_shadow_corrected_std')
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

    def calc_phased_doy(self, tavg):
        return np.mod(tavg.jd, 365)

    @property
    def phased_doy(self):
        if 'phased_doy' not in self.qtable.colnames:
            yphase = self.calc_phased_doy(self.qtable['tavg'])
            self.qtable['phased_doy'] = yphase
        return self.qtable['phased_doy']

    @property
    def phased_idoy(self):
        if 'phased_idoy' not in self.qtable.colnames:
            self.qtable['phased_idoy'] = self.phased_doy.astype(int)
        return self.qtable['phased_idoy']

    @property
    def shadow_corrected_by_doy(self):
        if 'shadow_corrected_by_doy' not in self.qtable.colnames:
            self.phased_idoy
            unit = self.model_vcol_shadow_corrected_physical.unit
            daily_biweight(
                self.qtable,
                day_col='phased_idoy',
                data_col='model_vcol_shadow_corrected_physical',
                biweight_col='shadow_corrected_by_doy',
                std_col='shadow_corrected_by_doy_std')

    @property
    def shadow_corrected_by_doy_std(self):
        """Daily mad std of shadow height-corrected Na meso emission

        """
        if 'shadow_corrected_by_doy_std' not in self.qtable.colnames:
            self.shadow_corrected_by_doy
        return self.qtable['shadow_corrected_by_doy_std']

    @pgproperty
    def doy_table(self):
        self.shadow_corrected_by_doy
        dt = unique(self.qtable, keys='phased_idoy')
        dt = QTable([dt['phased_idoy'],
                     dt['shadow_corrected_by_doy'],
                     dt['shadow_corrected_by_doy_std']],
                    names=('doy',
                           'shadow_corrected',
                           'shadow_corrected_std'))
        dt = daily_convolve(dt,
                            'doy',
                            'shadow_corrected',
                            'boxfilt_shadow_corrected',
                            Box1DKernel(20),
                            all_days=np.arange(365))
        return dt

    #@pgproperty
    #def doy_table(self):
    #    """Returns table with one row per phased DOY.  Columns are
    #    phased_idoy, doy_shadow_corrected and
    #    doy_shadow_corrected_std.  Note the columns are calculated
    #    from the phased data, unlike daily_shadow_corrected in self.qtable
    #
    #    """
    #    t = QTable()
    #    unit = self.model_vcol_shadow_corrected_physical.unit
    #    t['doy'] = np.arange(365)
    #    #t['doy'] = np.unique(self.phased_idoy)
    #
    #    t['shadow_corrected'] = np.NAN*unit
    #    t['shadow_corrected_std'] = np.NAN*unit
    #    for i, phased_idoy in enumerate(t['doy']):
    #        mask = self.phased_idoy == phased_idoy
    #        t_shadow_corrected = self.model_vcol_shadow_corrected[mask]
    #        t_shadow_corrected = t_shadow_corrected.physical
    #        tbiweight = biweight_location(t_shadow_corrected,
    #                                      ignore_nan=True)
    #        tstd = mad_std(t_shadow_corrected, ignore_nan=True)
    #        t['shadow_corrected'][i] = tbiweight
    #        t['shadow_corrected_std'][i] = tstd
    #    box_kernel = Box1DKernel(20)
    #    med = medfilt(t['shadow_corrected'], MEDFILT_WIDTH)
    #    t['medfilt_shadow_corrected'] = med
    #    box = convolve(t['shadow_corrected'], box_kernel)
    #    t['boxfilt_shadow_corrected'] = box
    #    return t        

    def tavg_to_shadow_corrected(self, tavg):
        doy = self.calc_phased_doy(tavg)
        doy = doy.astype(int)
        return self.doy_table['boxfilt_shadow_corrected'][doy]

    def meso_model_inverse(self, tavg, airmass, shadow_height):
        shadow_corrected = self.tavg_to_shadow_corrected(tavg)
        vcol = self.vcol_to_shadow_corrected(
            u.Magnitude(shadow_corrected), shadow_height, inverse=True)
        meso_mag = self.meso_mag_to_vcol(vcol, airmass, inverse=True)
        meso = meso_mag.physical
        return meso        

    def best_na_meso(self, tavg, airmass, shadow_height): 
        """Returns dict of best-estimate Na mesospheric emission

        Uses all Na background measurements to construct an empirical,
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
        assert isinstance(shadow_height, u.Quantity)
        model = self.meso_model_inverse(tavg, airmass, shadow_height)
        model_err = self.shadow_corrected_av_err
        # Calculate measured meso on this day
        ijd = tavg.jd.astype(int)
        self.daily_shadow_corrected
        self.qtable = self.qtable.group_by('ijd')
        mask = self.qtable.groups.keys['ijd'] == ijd
        # Shadow-corrected vertical column
        ijdt = self.qtable.groups[mask]
        shadow_corrected = ijdt['daily_shadow_corrected']
        shadow_corrected_err = ijdt['daily_shadow_corrected_std']
        if (len(shadow_corrected.value) == 0
            or not np.all(np.isfinite(shadow_corrected))):
            meso = np.NAN*model.unit
            meso_err = np.NAN*model.unit
            meso_or_model = model
            meso_or_model_err = model_err
        else:
            shadow_corrected = u.Magnitude(shadow_corrected[0])
            shadow_corrected_err = shadow_corrected_err[0]
            vcol = self.vcol_to_shadow_corrected(
                u.Magnitude(shadow_corrected),
                shadow_height,
                inverse=True)
            meso_mag = self.meso_mag_to_vcol(vcol, airmass, inverse=True)
            meso = meso_mag.physical
            meso_err = shadow_corrected_err
            meso_or_model = meso
            meso_or_model_err = meso_err
        return {'measured_meso': meso,
                'measured_meso_err': meso_err,
                'model_meso': model,
                'model_meso_err': model_err,
                'meso_or_model': meso_or_model,
                'meso_or_model_err': meso_or_model_err}

    def add_best_na_meso(self):
        bnm = [self.best_na_meso(tavg, airmass, shadow_height)
               for tavg, airmass, shadow_height
               in zip(self.qtable['tavg'],
                      self.qtable['airmass'],
                      self.qtable['shadow_height'])]
        t = QTable(rows=bnm)
        self.qtable = hstack([self.qtable, t])

    def plots(self, show=False, plot_fname=None):
        show = show or self.show
        plot_fname = plot_fname or self.plot_fname

        f = plt.figure(figsize=[11, 8.5])
        ax = plt.subplot(3, 2, 1)
        plt.plot(self.qtable['airmass'], self.qtable['best_back'], 'k.')
        plt.xlabel(f'airmass')
        plt.ylabel(f'meso Na emission ({self.qtable["best_back"].unit})')

        ax = plt.subplot(3, 2, 2)
        plt.plot(self.qtable['best_back_std'],
                 self.qtable['best_back'], 'k.')
        ax.set_xscale('linear')
        plt.xlabel(f'meso Na emission std ({self.qtable["best_back_std"].unit})')
        plt.ylabel(f'meso Na emission ({self.qtable["best_back"].unit})')
        ax = plt.subplot(3, 2, 3)
        plt.plot(self.qtable['airmass'], self.meso_mag, 'k.')
        plt.plot(self.qtable['airmass'],
                 self.meso_airmass_poly(self.qtable['airmass']), 'r-')
        ax.invert_yaxis()        
        plt.xlabel(f'airmass')
        plt.ylabel(f'meso Na emission ({self.model_meso_vcol.unit})')

        #ax = plt.subplot(3, 2, 3)
        #plt.plot(self.qtable['airmass'], self.model_meso_vcol, 'k.')
        #plt.xlabel(f'airmass')
        #ax.invert_yaxis()        
        #plt.ylabel(f'Vert. col. meso Na ({self.model_meso_vcol.unit})')

        #ax = plt.subplot(3, 2, 4)
        #plt.plot(self.shadow_height, self.model_meso_vcol, 'k.')
        #ax.set_xscale('log')
        #ax.invert_yaxis()        
        #plt.xlabel(f'Shadow height ({self.shadow_height.unit})')
        #plt.ylabel(f'Vert. col. Na ({self.model_meso_vcol.unit})')

        ax = plt.subplot(3, 2, 4)
        plt.plot(self.qtable['shadow_height'], self.model_meso_vcol, 'k.')
        plt.plot(self.qtable['shadow_height'],
                 self.vcol_dex_shadow_height_poly_quantity(
                     self.dex_shadow_height), 'r.')
        ax.set_xscale('log')
        ax.invert_yaxis()        
        plt.xlabel(f'Shadow height ({self.qtable["shadow_height"].unit})')
        plt.ylabel(f'Vert. col. Na ({self.model_meso_vcol.unit})')

        #ax = plt.subplot(3, 2, 6)
        #plt.plot(self.shadow_height, self.model_vcol_shadow_corrected, 'k.')
        #ax.set_xscale('log')
        #ax.invert_yaxis()        
        #plt.xlabel(f'Shadow height (SH) ({self.shadow_height.unit})')
        #plt.ylabel(f'SH corr. vert. col. Na ({self.model_meso_vcol.unit})')

        #ax = plt.subplot(3, 2, 7)
        #plt.plot(self.qtable['jup_dist'],
        #         self.model_vcol_shadow_corrected, 'k.')
        #ax.set_xscale('linear')
        #ax.invert_yaxis()        
        #plt.xlabel(f'Jupiter distance ({self.qtable["jup_dist"].unit})')
        #plt.ylabel(f'SH corr. vert. col. Na ({self.model_meso_vcol.unit})')

        #ax = plt.subplot(3, 2, 7)
        #plt.plot(self.qtable['jup_dist'],
        #         self.meso_mag, 'k.')
        #ax.set_xscale('linear')
        #ax.invert_yaxis()        
        #plt.xlabel(f'Jupiter distance ({self.qtable["jup_dist"].unit})')
        #plt.ylabel(f'SH corr. vert. col. Na ({self.model_meso_vcol.unit})')

        #ax = plt.subplot(3, 2, 7)
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

        #ax = plt.subplot(3, 2, 5)
        #plt.plot(self.qtable['tavg'].datetime,
        #         self.model_vcol_shadow_corrected.physical, 'k.')
        #plt.xlabel(f'Date')
        #plt.ylabel(f'SH corr. vert. col. Na ({self.model_meso_vcol.physical.unit})')
        #plt.plot(self.qtable['tavg'].datetime,
        #         self.meso_vcol_corrected_sin_physical(
        #             self.qtable['tavg'].jd*u.day), 'r.')
        #ax.tick_params(axis='x', labelrotation = 45)

        ax = plt.subplot(3, 2, 5)
        plt.plot(self.phased_doy,
                 self.model_vcol_shadow_corrected.physical,
                 'k.')
        plt.errorbar(self.doy_table['doy'],
                     self.doy_table['shadow_corrected'].value,
                     self.doy_table['shadow_corrected_std'].value,
                     fmt='r.')
        #plt.plot(self.doy_table['doy'],
        #         self.doy_table['medfilt_shadow_corrected'], 'y-')
        plt.plot(self.doy_table['doy'],
                 self.doy_table['boxfilt_shadow_corrected'], 'b-')
        plt.xlabel(f'Phased DOY')
        plt.ylabel(f'SH corr. vert. col. Na ({self.model_meso_vcol.physical.unit})')

        self.add_best_na_meso()
        ax = plt.subplot(3, 2, 6)
        plt.errorbar(self.qtable['tavg'].datetime,
                     self.qtable['meso_or_model'].value,
                     self.qtable['meso_or_model_err'].value,
                     fmt=('k.'))
        plt.xlabel(f'Date')
        unit = self.qtable['meso_or_model'].unit
        plt.ylabel(f'Best meso Na ({unit})')
        ax.tick_params(axis='x', labelrotation = 45)

#        ax = plt.subplot(3, 2, 6)
#        plt.plot(self.qtable['tavg'].datetime,
#                 self.shadow_corrected_to_no_time(
#                     self.model_vcol_shadow_corrected.physical,
#                     self.qtable['tavg']),
#                 'k.')
#        #plt.errorbar(self.qtable['tavg'].datetime,
#        #             self.daily_shadow_corrected.value,
#        #             self.daily_shadow_corrected_std.value,
#        #             fmt='r.')
#        plt.errorbar(self.qtable['tavg'].datetime,
#                     self.shadow_corrected_to_no_time(
#                         self.daily_shadow_corrected,
#                         self.qtable['tavg']).value,
#                         self.daily_shadow_corrected_std.value,
#                     fmt='r.')
#        plt.xlabel(f'Date')
#        plt.ylabel(f'Model residual ({self.model_meso_vcol.physical.unit})')
#        ax.tick_params(axis='x', labelrotation = 45)

        plt.tight_layout()
        savefig_overwrite(plot_fname)

        if show:
            plt.show()
        plt.close()

    def highest_product(self):
        self.add_best_na_meso

    def qtable_write(self, qtable_fname=None):
        qtable_fname = qtable_fname or self.qtable_fname
        self.qtable.write(qtable_fname, overwrite=True)
        self.plots()
    
def na_meso_meta(ccd,
                 na_meso_obj=None,
                 bmp_meta=None,
                 **kwargs):
    """Calculates Na mesospheric emission & inserts it into CCD and BMP
    metadata

    """
    best_meso = na_meso_obj.best_na_meso(ccd.tavg,
                                         ccd.meta['airmass'],
                                         ccd.meta['shadow_height']*u.km)
    ccd_out = dict_to_ccd_meta(ccd, best_meso)
    bmp_meta.update(best_meso)
    return ccd_out

class NaMesoArgparseHandler(SSArgparseHandler,
                            CorPhotometryArgparseMixin,
                            CalArgparseHandler):
    def add_all(self):
        """Add options used in cmd"""
        self.add_reduced_root(option='na_meso_root',
                              default=OUTDIR_ROOT)
        self.add_base(default=BASE)
        self.add_start(option='na_meso_start')
        self.add_stop(option='na_meso_stop')
        self.add_show()
        self.add_read_pout(default=True)
        self.add_write_pout(default=True)        
        self.add_read_csvs(default=True)
        self.add_write_csvs(default=True)
        self.calc_highest_product()
        self.add_solve_timeout()
        self.add_join_tolerance()
        self.add_join_tolerance_unit()
        self.add_keep_intermediate()
        super().add_all()

    def cmd(self, args):
        c, ss = super().cmd(args)
        # Eventually, I am going to want to have this be a proper
        # command-line thing
        t = na_meso_tree(raw_data_root=args.raw_data_root,
                         start=args.na_meso_start,
                         stop=args.na_meso_stop,
                         calibration=c,
                         standard_star_obj=ss,
                         keep_intermediate=args.keep_intermediate,
                         solve_timeout=args.solve_timeout,
                         join_tolerance=(
                             args.join_tolerance
                             *u.Unit(args.join_tolerance_unit)),
                         read_pout=args.read_pout,
                         write_pout=args.write_pout,
                         read_csvs=args.read_csvs,
                         write_csvs=args.write_csvs,
                         show=args.show,
                         create_outdir=args.create_outdir,
                         outdir_root=args.na_meso_root,
                         num_processes=args.num_processes,
                         mem_frac=args.mem_frac,
                         fits_fixed_ignore=args.fits_fixed_ignore)
        m = NaMeso(qtable=t)
        if args.calc_highest_product:
            m.highest_product()
            m.qtable_write()
        return c, ss, m

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run Na mesospheric emission reduction')
    aph = NaMesoArgparseHandler(parser)
    aph.add_all()
    args = parser.parse_args()
    aph.cmd(args)

#m = NaMeso()
#dt = m.test_doy_table
#t = m.add_best_na_meso()
#print(np.mod(m.qtable['tavg'].jd, (1*u.year).value) * (1*u.year).to(u.day))
#m.plots(show=True)
#t = Time('2020-01-01T00:01:00', format='fits')
#print(m.best_na_meso(t, 2, 1E5*u.km))
#t = Time('2020-04-30T00:01:00', format='fits')
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
