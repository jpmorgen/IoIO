"""
The cormultipipe module implements the IoIO coronagraph data reduction
pipeline using ccdmultipipe/bigmultipipe as its base
"""

import inspect
import os
import time
import datetime
import glob
import psutil
from pathlib import Path

import numpy as np
import numpy.ma as ma
from scipy import signal, stats, interpolate

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import pandas as pd

from astropy import log
from astropy import units as u
from astropy.io import fits
from astropy.nddata import CCDData, StdDevUncertainty
from astropy.table import QTable
from astropy.time import Time, TimeDelta
from astropy.stats import mad_std, biweight_location

from photutils import Background2D, MedianBackground

import ccdproc as ccdp

from bigmultipipe import num_can_process, WorkerWithKwargs, NoDaemonPool
from bigmultipipe import multi_logging, prune_pout
from ccdmultipipe import CCDMultiPipe, ccddata_read

import sx694
from IoIO import CorObsData

# Processing global variables.  Since I avoid use of the global
# statement and don't reassign these at global scope, they stick to
# these values and provide handy defaults for routines and object
# inits.  It is also a way to be lazy about documenting all of the
# code :-o


# Tests with first iteration of pipeline showed that the real gain in
# speed is from the physical processors, not the logical processes
# (threads).  Threads automatically make the physical processes
# faster.  Going to num_processes greater than the number of physical
# processes does go faster, but only asymptotically, probably because
# wait times are minimized.  Rather than try to milk the asymptote for
# speed, just max out on physical processors to get the steepest gains
# and leave the asymptote for other jobs
max_num_processes = psutil.cpu_count(logical=False)
max_mem_frac = 0.8

# Calculate the maximum CCDdata size based on 64bit primary & uncert +
# 8 bit mask / 8 bits per byte.  It will be compared to
# psutil.virtual_memory() at runtime to optimize computational tasks
# is my do-it-yourself multiprocessing routines
max_ccddata_bitpix = 2*64 + 8
cor_process_expand_factor = 3.5
griddata_expand_factor = 20

# These are use to optimize parallelization until such time as
# ccdproc.combiner can be parallelized
num_ccdts = int((35 - (-10)) / 5)
num_dark_exptimes = 8
num_filts = 9
num_calibration_files = 11

data_root = '/data/io/IoIO'
raw_data_root = os.path.join(data_root, 'raw')
reduced_root = os.path.join(data_root, 'reduced')
calibration_root = os.path.join(reduced_root, 'Calibration')
calibration_scratch = os.path.join(calibration_root, 'scratch')
# string to append to processed files to avoid overwrite of raw data
outname_append = "_p"

# Lockfiles to prevent multiple upstream parallel processes from
# simultanously autoreducing calibration data
lockfile = '/tmp/calibration_reduce.lock'

# Raw (and reduced) data are stored in directories by UT date, but
# some have subdirectories that contain calibration files.
calibration_subdirs = ['Calibration', 'AutoFlat']

# Put the regular expressions for the biases, darks, and flats here so
# that they can be found quickly without having to do a ccd.Collection
# on a whold directory.  The later is the rock solid most reliable,
# but slow in big directories, since ccdproc.Collection has to read
# each file
bias_glob = ['Bias*', '*_bias.fit']
dark_glob = ['Dark*', '*_dark.fit']
flat_glob = '*Flat*'

# During the creation of master biases and darks files are grouped by
# CCD temperature.  This is the change in temperature seen as a
# function of time that is used to trigger the creation of a new group
dccdt_tolerance = 0.5
# During reduction of files, biases and darks need to be matched to
# each file by temperature.  This is the tolerance for that matching
ccdt_tolerance = 2
# When I was experimenting with bias collection on a per-night basis,
# I got lots of nights with a smattering of biases.  Discard these
min_num_biases = 7
min_num_flats = 3

# Accept as match darks with this much more exposure time
dark_exp_margin = 3

# Number of pixels to expand the ND filter over what CorObsData finds.
# This is the negative of the CorObsData edge_mask parameter, since
# that is designed to mask pixels inside the ND filter to make
# centering of object more reliable
nd_edge_expand = 40

######### CorMultiPipe object

class CorMultiPipe(CCDMultiPipe):
    def __init__(self,
                 calibration=None,
                 auto=False,
                 outname_append='_r',
                 naxis1=sx694.naxis1,
                 naxis2=sx694.naxis2,
                 process_expand_factor=cor_process_expand_factor,
                 **kwargs):
        self.calibration = calibration
        self.auto = auto
        super().__init__(outname_append=outname_append,
                         naxis1=naxis1,
                         naxis2=naxis2,
                         process_expand_factor=process_expand_factor,
                         **kwargs)

    def pre_process(self, data, **kwargs):
        """Add full-frame check permanently to pipeline"""
        kwargs = self.kwargs_merge(**kwargs)
        s = data.shape
        # Note Pythonic C index ordering
        if s != (self.naxis2, self.naxis1):
            return (None, kwargs)
        return super().pre_process(data, **kwargs)

    def data_process(self, data,
                     calibration=None,
                     auto=None,
                     **kwargs):
        kwargs = self.kwargs_merge(**kwargs)
        if calibration is None:
            calibration = self.calibration
        if auto is None:
            auto = self.auto
        data = cor_process(data,
                           calibration=calibration,
                           auto=auto,
                           **kwargs)
        return data

######### CorMultiPipe prepossessing routines
def full_frame(im,
               naxis1=sx694.naxis1,
               naxis2=sx694.naxis2,
               **kwargs):
    """CorMultiPipe pre-processing routine to select full-frame images (currently permanently installed into CorMultiPipe.pre_process) 
    """
    s = im.shape
    # Note Pythonic C index ordering
    if s != (naxis2, naxis1):
        return None
    return im

def light_image(im, light_tolerance=3, **kwargs):
    """CorMultiPipe pre-processing routine to reject light-contaminated bias & dark images
    """
    s = im.shape
    m = np.asarray(s)/2 # Middle of CCD
    q = np.asarray(s)/4 # 1/4 point
    m = m.astype(int)
    q = q.astype(int)
    # --> check lowest y too, since large filters go all the
    # --> way to the edge See 20200428 dawn biases
    dark_patch = im[m[0]-50:m[0]+50, 0:100]
    light_patch = im[m[0]-50:m[0]+50, q[1]:q[1]+100]
    mdp = np.median(dark_patch)
    mlp = np.median(light_patch)
    if (np.median(light_patch) - np.median(dark_patch) > light_tolerance):
        log.debug('light, dark patch medians ({:.4f}, {:.4f})'.format(mdp, mlp))
        return None
    return im

######### CorMultiPipe postpossessing routines
def mask_above_key(ccd_in, bmp_meta=None, key=None, margin=0.1, **kwargs):
    """CorMultiPipe post-processing routine to mask pixels > input key
    """
    if key is None:
        raise ValueError('key must be specified')
    masklevel = ccd_in.meta.get(key.lower())
    if masklevel is None:
        return ccd
    ccd = ccd_in.copy()
    # Saturation level is subject to overscan subtraction and
    # multiplication by gain, so don't do strict = testing, but give
    # ourselves a little margin.
    mask = ccd.data >= masklevel - margin
    n_masked = np.count_nonzero(mask)
    if n_masked > 0:
        log.info(f'Masking {n_masked} pixels above {key}')
    if len(key) > 6:
        h = 'HIERARCH '
    else:
        h = ''
    n_masked_key = h + 'N_' + key
    ccd.meta[n_masked_key] = (n_masked, f'masked pixels > {key}')
    # Avoid creating a mask of all Falses & supplement any existing mask
    if n_masked > 0:
        if ccd.mask is None:
            ccd.mask = mask
        else:
            ccd.mask = ccd.mask + mask
    if bmp_meta is not None:
        bmp_meta[n_masked_key] = n_masked
    return ccd

def mask_nonlin_sat(ccd, bmp_meta=None, margin=0.1, **kwargs):
    """CorMultiPipe post-processing routine to mask pixels > NONLIN and SATLEVEL
    """
    ccd = mask_above_key(ccd, bmp_meta=bmp_meta, key='SATLEVEL')
    ccd = mask_above_key(ccd, bmp_meta=bmp_meta, key='NONLIN')
    return ccd

def jd_meta(ccd, bmp_meta=None, **kwargs):
    """CorMultiPipe post-processing routine to return JD
    """
    tm = Time(ccd.meta['DATE-OBS'], format='fits')
    if bmp_meta is not None:
        bmp_meta['jd'] = tm.jd
    return ccd

def bias_stats(ccd, bmp_meta=None, gain=sx694.gain, **kwargs):
    """CorMultiPipe post-processing routine for bias_combine
    Returns dictionary of bias statistics for pandas dataframe
    """
    im = ccd.data
    hdr = ccd.meta
    # Calculate readnoise.  This is time-consuming
    diffs2 = (im[1:] - im[0:-1])**2
    rdnoise = np.sqrt(biweight_location(diffs2))
    # Skip uncertainty creation, since it is not used in any
    # subsequent calcs
    #uncertainty = np.multiply(rdnoise, np.ones(im.shape))
    #ccd.uncertainty = StdDevUncertainty(uncertainty)
    # Prepare to create a pandas data frame to track relevant
    # quantities
    tm = Time(ccd.meta['DATE-OBS'], format='fits')
    ccdt = ccd.meta['CCD-TEMP']
    tt = tm.tt.datetime
    # We have already subtracted overscan, so add it back in where
    # appropriate
    median = hdr['OVERSCAN_MEDIAN']
    stats = {'time': tt,
             'ccdt': ccdt,
             'median': median,
             'mean': np.mean(im) + median,
             'std': np.std(im)*gain,
             'rdnoise': rdnoise*gain,
             'min': np.min(im) + median,  
             'max': np.max(im) + median}
    if bmp_meta is not None:
        bmp_meta['bias_stats'] = stats
    return ccd

def nd_filter_mask(ccd_in, nd_edge_expand=nd_edge_expand, **kwargs):
    """CorMultiPipe post-processing routine to mask ND filter
    """
    # --> this will eventually get included in CorData or whatever I call it
    ccd = ccd_in.copy()
    hdul = ccd.to_hdu()
    obs_data = CorObsData(hdul, edge_mask=-nd_edge_expand)
    # Capture our ND filter metadata
    ccd.meta = hdul[0].header
    mask = np.zeros(ccd.shape, bool)
    mask[obs_data.ND_coords] = True
    if ccd.mask is None:
        ccd.mask = mask
    else:
        ccd.mask = ccd.mask + mask
    return ccd

def detflux(ccd_in, exptime_units=None, **kwargs):
    ccd = ccd_in.copy()
    if exptime_units is None:
        exptime_units = u.s
    exptime = ccd.meta['EXPTIME'] * exptime_units
    ccd = ccd.divide(exptime, handle_meta='first_found')
    satlevel = ccd.meta.get('satlevel')
    # --> really what I want here is my own CCDData that handles these!
    if satlevel is not None:
        satlevel /= exptime
    return ccd

######### cor_process routines
def kasten_young_airmass(hdr_in):
    """Record airmass considering curvature of Earth

Uses formula of F. Kasten and Young, A. T., “Revised optical air mass
tables and approximation formula”, Applied Optics, vol. 28,
pp. 4735–4738, 1989 found at
https://www.pveducation.org/pvcdrom/properties-of-sunlight/air-mass

    """
    if hdr_in.get('oairmass') is not None:
        # We have been here before, so exit quietly
        return hdr_in
    if hdr_in.get('objctalt') is None:
        # We have no alt to work with
        # --> double-check this
        return hdr_in
    hdr = hdr_in.copy()
    alt = float(hdr['OBJCTALT'])
    zd = 90 - alt
    airmass = hdr['AIRMASS']
    hdr.insert('AIRMASS',
               ('OAIRMASS', airmass, 'Original airmass'),
               after=True)
    denom = np.cos(np.radians(zd)) + 0.50572 * (96.07995 - zd)**(-1.6364)
    hdr['AIRMASS'] = (1/denom, 'Curvature-corrected (Kasten and Young 1989)')
    return(hdr)

def subtract_overscan(fname_or_ccd, oscan=None, *args, **kwargs):
    """Subtract overscan, estimating it, if necesesary, from image.
    Also subtracts overscan from SATLEVEL keyword

    Note: ccdproc's native subtract_overscan function can't be used
    because it assumes the overscan region is specified by a simple
    rectangle.

    """
    nccd = ccddata_read(fname_or_ccd)
    if nccd.meta.get('overscan_value') is not None:
        # We have been here before, so exit quietly
        return nccd
    if oscan is None:
        # Interface with sx694.overscan_estimate, which doesn't take
        # CCDData for the primary input data
        im = nccd.data
        hdr = nccd.meta
        # Fix problem where BUNIT is not recorded until file is
        # written
        hdr['BUNIT'] = nccd.unit.to_string()
        oscan = sx694.overscan_estimate(im, hdr, hdr_out=nccd.meta,
                                        *args, **kwargs)
    nccd = nccd.subtract(oscan*u.adu, handle_meta='first_found')
    nccd.meta['HIERARCH OVERSCAN_VALUE'] = (oscan, 'overscan value subtracted (ADU)')
    nccd.meta['HIERARCH SUBTRACT_OVERSCAN'] \
        = (True, 'Overscan has been subtracted')
    # Keep track of our precise saturation level
    satlevel = nccd.meta.get('satlevel')
    if satlevel is not None:
        satlevel -= oscan
        nccd.meta['SATLEVEL'] = satlevel # still in ADU
    return nccd

def cor_process(ccd,
                calibration=None,
                auto=False,
                imagetyp=None,
                ccd_meta=True,
                exp_correct=True,
                airmass_correct=True,
                oscan=None,
                trim=None,
                error=False,
                master_bias=None,
                dark_frame=None,
                master_flat=None,
                bad_pixel_mask=None,
                gain=None,
                gain_key=None,
                readnoise=None,
                readnoise_key=None,
                oscan_median=True,
                oscan_model=None,
                min_value=None,
                min_value_key=None,
                flat_norm_value=1,
                dark_exposure=None,
                data_exposure=None,
                exposure_key=None,
                exposure_unit=None,
                dark_scale=True,
                gain_corrected=True,
                *args, **kwargs):

    """Perform basic CCD processing/reduction of IoIO ccd data

    The following steps can be included:

    * add CCD metadata (:func:`sx694.metadata`)
    * correct CCD exposure time (:func:`sx694.exp_correct`)
    * overscan correction (:func:`subtract_overscan`)
    * trimming of the image (:func:`trim_image`)
    * create deviation frame (:func:`create_deviation`)
    * gain correction (:func:`gain_correct`)
    * add a mask to the data
    * subtraction of master bias (:func:`subtract_bias`)
    * subtraction of a dark frame (:func:`subtract_dark`)
    * correction of flat field (:func:`flat_correct`)

    The task returns a processed `~astropy.nddata.CCDData` object.

    Parameters
    ----------
    fname_or_ccd : str or `~astropy.nddata.CCDData`
        Filename or CCDData of image to be reduced.

    multi : bool

        Internal flat signaling that this call is being used as part
        of a multi-process run.  If True, assures input and output of
        CCDData are via files rather than CCDData objects

    calibration : `~Calibration`, bool, or None, optional
        Calibration object to be used to find best bias, dark, and
        flatfield files.  If True, a Calibration object is
        instantiated locally with no arguments (dangerous if
        calibration reductions have not been completed!)  
        Default is ``None``.

    auto : bool
        If True, do reduction automatically based on IMAGETYP
        keyword.  See imagetyp documentation.
        Default is ``False``

    imagetyp : bool, str, or None
        If True, do reduction based on IMAGETYP keyword.  If string,
        use that as IMAGETYP.  Requires calibration object
        bias -> oscan=True
        dark -> oscan=True, master_bias=True
        flat -> oscan=True, master_bias=True, dark_frame=True
        light-> oscan=True, error=True, master_bias=True,
                dark_frame=True, master_flat=True
        Default is ``None``

    ccd_meta : bool
        Add CCD metadata
        Default is ``True``

    exp_correct : bool
        Correct for exposure time problems
        Default is ``True``

    airmass_correct : bool
        Correct for curvature of earth airmass for very low elevation
        observations
        Default is ``True``

    oscan : number, bool, or None, optional
        Single pedistal value to subtract from image.  If True, oscan
        is estimated using :func:`sx694.overscan_estimate` and subtracted
        Default is ``None``.

    error : bool, optional
        If True, create an uncertainty array for ccd.
        Default is ``False``.

    master_bias : bool, str, `~astropy.nddata.CCDData` or None, optional
        Master bias frame to be subtracted from ccd image. The unit of the
        master bias frame should match the unit of the image **after
        gain correction** if ``gain_corrected`` is True.  If True,
        master_bias is determined using :func`Calibration.best_bias`.
        NOTE: master_bias RDNOISE card, if present, is propagated
        to output ccddata metadata.  This is helpful in systems where
        readnoise is measured on a per-masterbias basis and harmless
        when a manufacturer's value is used.
        Default is ``None``.

    dark_frame : bool, str, `~astropy.nddata.CCDData` or None, optional
        A dark frame to be subtracted from the ccd. The unit of the
        master dark frame should match the unit of the image **after
        gain correction** if ``gain_corrected`` is True.  If True,
        dark_frame is determined using :func`Calibration.best_dark`.
        Default is ``None``.

    master_flat : `~astropy.nddata.CCDData` or None, optional
        A master flat frame to be divided into ccd. The unit of the
        master flat frame should match the unit of the image **after
        gain correction** if ``gain_corrected`` is True.  If True,
        master_bias is determined using :func`Calibration.best_flat`.
        Default is ``None``.

    bad_pixel_mask : `numpy.ndarray` or None, optional
        A bad pixel mask for the data. The bad pixel mask should be in given
        such that bad pixels have a value of 1 and good pixels a value of 0.
        Default is ``None``.

    gain : `~astropy.units.Quantity`, bool or None, optional
        Gain value to multiple the image by to convert to electrons.
        If True, read metadata using gain_key
        Default is ``None``.

    gain_key :  `~ccdproc.Keyword`
    	Name of key in metadata that contains gain value.  
        Default is "GAIN" with units `~astropy.units.electron`/`~astropy.units.adu`

    readnoise : `~astropy.units.Quantity`, bool or None, optional
        Read noise for the observations. The read noise should be in
        electrons.  If True, read from the READNOISE keyword and
        associated with readnoise_unit
        Default is ``None``.

    readnoise_key : `astropy.units.core.UnitBase`
    	Name of key in metadata that contains gain value.  
        Default is "RDNOISE" with units `astropy.units.electron`

    min_value : float, bool, or None, optional
        Minimum value for flat field.  To avoid division by small
        number problems, all values in the flat below min_value will
        be replaced by this value.  If True, value read from FLAT_CUT
        keyword of flat.  If None, no replacement will be done.
        Default is ``None``.

    flat_norm_value : float
        Normalize flat by this value
        Default is 1 (no normalization -- flat is already normalized).

    dark_exposure : `~astropy.units.Quantity` or None, optional
        Exposure time of the dark image; if specified, must also provided
        ``data_exposure``.
        Default is ``None``.

    data_exposure : `~astropy.units.Quantity` or None, optional
        Exposure time of the science image; if specified, must also provided
        ``dark_exposure``.
        Default is ``None``.

    exposure_key : `~ccdp.Keyword`, str or None, optional
        Name of key in image metadata that contains exposure time.
        Default is ``None``.

    exposure_unit : `~astropy.units.Unit` or None, optional
        Unit of the exposure time if the value in the meta data does not
        include a unit.
        Default is ``None``.

    dark_scale : bool, optional
        If True, scale the dark frame by the exposure times.
        Default is ``True``.

    gain_corrected : bool, optional
        If True, the ``master_bias``, ``master_flat``, and ``dark_frame``
        have already been gain corrected.
        Default is ``True``.

    Returns
    -------
    ccd : `~astropy.nddata.CCDData`
	Processed image

    Examples --> fix these
    --------
    1. To overscan, trim and gain correct a data set::

        >>> import numpy as np
        >>> from astropy import units as u
        >>> from astropy.nddata import CCDData
        >>> from ccdproc import ccd_process
        >>> ccd = CCDData(np.ones([100, 100]), unit=u.adu)
        >>> nccd = ccd_process(ccd, oscan='[1:10,1:100]',
        ...                    trim='[10:100, 1:100]', error=False,
        ...                    gain=2.0*u.electron/u.adu)

    """

    if gain_key is None:
        gain_key = ccdp.Keyword('GAIN', u.electron/u.adu)
    if readnoise_key is None:
        readnoise_key = ccdp.Keyword('RDNOISE', u.electron)
    if min_value_key is None:
        min_value_key = ccdp.Keyword('FLAT_CUT', u.dimensionless_unscaled)
    if exposure_key is None:
        exposure_key = ccdp.Keyword('EXPTIME', u.s)

    # make a copy of the object
    nccd = ccd.copy()

    # Handle our calibration object
    if calibration is True:
        calibration  = Calibration()

    # Enable autocalibration through imagetyp keyword
    if auto:
        imagetyp = nccd.meta.get('imagetyp')
        if imagetyp is None:
            raise ValueError("CCD metadata contains no IMAGETYP keyword, can't proceed with automatic reduction")

    # Enable imagetyp to select reduction level
    if imagetyp is None:
        pass
    elif imagetyp.lower() == 'bias':
        oscan=True; error=True
    elif imagetyp.lower() == 'dark':
        oscan=True; gain=True; error=True; master_bias=True
    elif imagetyp.lower() == 'flat':
        oscan=True; gain=True; error=True; master_bias=True; dark_frame=True
    elif imagetyp.lower() == 'light':
        oscan=True; gain=True; error=True; master_bias=True; dark_frame=True; master_flat=True; min_value=True
    else:
        raise ValueError(f'Unknown IMAGETYP keyword {imagetyp}')

    # Convert "yes use this calibration" to calibration _filenames_
    if isinstance(calibration, Calibration):
        if master_bias is True:
            master_bias = calibration.best_bias(nccd)
        if dark_frame is True:
            dark_frame = calibration.best_dark(nccd)
        if master_flat is True:
            master_flat = calibration.best_flat(nccd.meta)

    if master_bias is True:
        raise ValueError('master_bias=True but no Calibration object supplied')
    if dark_frame is True:
        raise ValueError('dark_frame=True but no Calibration object supplied')
    if master_flat is True:
        raise ValueError('master_flat=True but no Calibration object supplied')

    if ccd_meta:
        # Put in our SX694 camera metadata
        nccd.meta = sx694.metadata(nccd.meta, *args, **kwargs)

    if exp_correct:
        # Correct exposure time for driver bug
        nccd.meta = sx694.exp_correct(nccd.meta, *args, **kwargs)
        
    if airmass_correct:
        nccd.meta = kasten_young_airmass(nccd.meta)
    # Apply overscan correction unique to the IoIO SX694 CCD.  This
    # adds our CCD metadata as a necessary step and uses the string
    # version of master_bias, if available for metadata
    if oscan is True:
        nccd = subtract_overscan(nccd, master_bias=master_bias,
                                       *args, **kwargs)
    elif oscan is None or oscan is False:
        pass
    else:
        # Hope oscan is a number...
        nccd = subtract_overscan(nccd, oscan=oscan,
                                 *args, **kwargs)

    # The rest of the code uses stock ccdproc routines for the most
    # part, so convert calibration filenames to CCDData objects,
    # capturing the names for metadata purposes
    if isinstance(master_bias, str):
        subtract_bias_keyword = \
            {'HIERARCH SUBTRACT_BIAS': 'subbias',
             'SUBBIAS': 'ccdproc.subtract_bias ccd=<CCDData>, master=BIASFILE',
             'BIASFILE': master_bias}
        master_bias = ccddata_read(master_bias)
    else:
        subtract_bias_keyword = None
    if isinstance(dark_frame, str):
        subtract_dark_keyword = \
            {'HIERARCH SUBTRACT_DARK': 'subdark',
             'SUBDARK': 'ccdproc.subtract_dark ccd=<CCDData>, master=DARKFILE',
             'DARKFILE': dark_frame}
        dark_frame = ccddata_read(dark_frame)
    else:
        subtract_dark_keyword = None
    if isinstance(master_flat, str):
        flat_correct_keyword = \
            {'HIERARCH FLAT_CORRECT': 'flatcor',
             'FLATCOR': 'ccdproc.flat_correct ccd=<CCDData>, master=FLATFILE',
             'FLATFILE': master_flat}
        master_flat = ccddata_read(master_flat)
    else:
        flat_correct_keyword = None


    # apply the trim correction
    if isinstance(trim, str):
        nccd = ccdp.trim_image(nccd, fits_section=trim)
    elif trim is None:
        pass
    else:
        raise TypeError('trim is not None or a string.')
    
    if isinstance(master_bias, CCDData):
        if master_bias.unit == u.electron:
            # Apply some knowledge of our reduction scheme to ease the
            # number of parameters to supply
            gain_corrected = True
        # Copy over measured readnoise, if present
        rdnoise = nccd.meta.get('rdnoise')
        if rdnoise is not None:
            nccd.meta['RDNOISE'] = rdnoise
            nccd.meta.comments['RDNOISE'] = master_bias.meta.comments['RDNOISE']

    if gain is True:
        gain = gain_key.value_from(nccd.meta)

    # Correct our SATLEVEL and NONLIN units if we are going to
    # gain-correct.
    satlevel = nccd.meta.get('satlevel')
    nonlin = nccd.meta.get('nonlin')
    if (isinstance(gain, u.Quantity)
        and satlevel is not None
        and nonlin is not None):
        nccd.meta['SATLEVEL'] = satlevel * gain.value
        nccd.meta.comments['SATLEVEL'] = 'saturation level (electron)'
        nccd.meta['NONLIN'] = nonlin * gain.value
        nccd.meta.comments['NONLIN'] = 'Measured nonlinearity point (electron)'

    if error and readnoise is None:
        # We want to make an error frame but the user has not
        # specified readnoise.  See if we can read from metadata
        readnoise = readnoise_key.value_from(nccd.meta)

    # Create the error frame.  Do this differently than ccdproc for
    # two reasons: (1) bias error should read the readnoise (2) I
    # can't trim my overscan, so there are lots of pixels at the
    # overscan level.  After overscan and bias subtraction, many of
    # them that are probably normal statitical outliers are negative
    # enough to overwhelm the readnoise in the deviation calculation.
    # But I don't want the error estimate on them to be NaN, since the
    # error is really the readnoise.
    if error and imagetyp is not None and imagetyp.lower() == 'bias':
        if gain is None:
            # We don't want to gain-correct, so we need to prepare to
            # convert readnoise (which is in electron) to ADU
            gain_for_bias = gain_key.value_from(nccd.meta)
        else:
            # Bias will be gain-corrected to read in electrons
            gain_for_bias = 1*u.electron
        readnoise_array = np.full_like(nccd,
                                       readnoise.value/gain_for_bias.value)
        nccd.uncertainty = StdDevUncertainty(readnoise_array,
                                             unit=nccd.unit,
                                             copy=False)
    else:
        if error and gain is not None and readnoise is not None:
            nccd = ccdp.create_deviation(nccd, gain=gain,
                                         readnoise=readnoise,
                                         disregard_nan=True)
        elif error and (gain is None or readnoise is None):
            raise ValueError(
                'gain and readnoise must be specified to create error frame.')

    # apply the bad pixel mask
    if isinstance(bad_pixel_mask, np.ndarray):
        nccd.mask = bad_pixel_mask
    elif bad_pixel_mask is None:
        pass
    else:
        raise TypeError('bad_pixel_mask is not None or numpy.ndarray.')
    
    # apply the gain correction
    if not (gain is None or isinstance(gain, u.Quantity)):
        raise TypeError('gain is not None or astropy.units.Quantity.')
    
    # Gain-correct now if bias, etc. are gain corrected (otherwise at end)
    if gain is not None and gain_corrected:
        nccd = ccdp.gain_correct(nccd, gain)

    # Subtract master bias, adding metadata that refers to bias
    # filename, if supplied
    if isinstance(master_bias, CCDData):
        nccd = ccdp.subtract_bias(nccd, master_bias,
                                  add_keyword=subtract_bias_keyword)
    elif master_bias is None:
        pass
    else:
        raise TypeError(
            'master_bias is not None, fname or a CCDData object.')
    
    # Correct OVERSCAN_MASTER_BIAS keyword, if possible
    hdr = nccd.meta
    osbias = hdr.get('osbias')
    biasfile = hdr.get('biasfile')
    if osbias is None or biasfile is None:
        pass
    elif osbias != biasfile:
        multi_logging('warning', pipe_meta,
                      'OSBIAS and BIASFILE are not the same')
    else:
        del hdr['OSBIAS']
        hdr['OVERSCAN_MASTER_BIAS'] = 'BIASFILE'

    # Subtract the dark frame.  Generally this will just use the
    # default exposure_key we create in our parameters to ccd_process
    if isinstance(dark_frame, CCDData):
        nccd = ccdp.subtract_dark(nccd, dark_frame,
                                  dark_exposure=dark_exposure,
                                  data_exposure=data_exposure,
                                  exposure_time=exposure_key,
                                  exposure_unit=exposure_unit,
                                  scale=dark_scale,
                                  add_keyword=subtract_dark_keyword)
    elif dark_frame is None:
        pass
    else:
        raise TypeError(
            'dark_frame is not None or a CCDData object.')
    
    if master_flat is None:
        pass
    else:
        if min_value is True:
            min_value = min_value_key.value_from(master_flat.meta)
            flat_correct_keyword['FLATCOR'] += f', min_value={min_value}'
        flat_correct_keyword['FLATCOR'] += f', norm_value={flat_norm_value}'
        nccd = ccdp.flat_correct(nccd, master_flat,
                                 min_value=min_value,
                                 norm_value=flat_norm_value,
                                 add_keyword=flat_correct_keyword)
        for i in range(2):
            for j in range(2):
                ndpar = master_flat.meta.get(f'ndpar{i}{j}')
                if ndpar is None:
                    break
                ndpar_comment = master_flat.meta.comments[f'NDPAR{i}{j}']
                ndpar_comment = 'FLAT ' + ndpar_comment
                nccd.meta[f'FNDPAR{i}{j}'] = (ndpar, ndpar_comment)

    # apply the gain correction only at the end if gain_corrected is False
    if gain is not None and not gain_corrected:
        nccd = ccdp.gain_correct(nccd, gain)

    return nccd

####### bias, dark, and flat generation routines
def add_history(header, text='', caller=1):
    """Add a HISTORY card to a FITS header with the caller's name inserted 

    Parameters
    ----------
    header : astropy.fits.Header object
        Header to write HISTORY into.  No default.

    text : str
        String to write.  Default '' indicates FITS-formatted current
        time will be used 

    caller : int or str
        If int, number of levels to go up the call stack to get caller
        name.  If str, string to use for caller name

    Raises
    ------
        ValueError if header not astropy.io.fits.Header

"""

    # if not isinstance(header, fits.Header):
    #     raise ValueError('Supply a valid FITS header object')

    # If not supplied, get our caller name from the stack
    # http://stackoverflow.com/questions/900392/getting-the-caller-function-name-inside-another-function-in-python
    # https://docs.python.org/3.6/library/inspect.html
    if type(caller) == int:
        try:
            caller = inspect.stack()[caller][3]
        except IndexError:
            caller = 'unknown'
    elif type(caller) != str:
        raise TypeError('Type of caller must be int or str')

    # If no text is supplied, put in the date in FITS format
    if text == '':
        now = Time.now()
        now.format = 'fits'
        text = now.value

    towrite = '(' + caller + ')' + ' ' + text
    # astropy.io.fits automatically wraps long entries
    #if len('HISTORY ') + len(towrite) > 80:
    #    log.warning('Truncating long HISTORY card: ' + towrite)

    header['HISTORY'] = towrite
    return


def fdict_list_collector(fdict_list_generator,
                         directory=None,
                         collection=None,
                         subdirs=None,
                         glob_include=None,
                         imagetyp=None,
                         **kwargs):

    if subdirs is None:
        subdirs = []
    if glob_include is None:
        # Trick to make loop on glob_include, below, pass None to
        # ccdp.ImageFileCollection
        glob_include = [None]
    fdict_list = []
    if collection is None:
        # Prepare to call ourselves recursively to build up a list of
        # fnames in the provided directory and optional subdirectories
        if not os.path.isdir(directory):
            # This is the end of our recursive line
            return fdict_list
        for sd in subdirs:
            subdir = os.path.join(directory, sd)
            sub_fdict_list = fdict_list_collector \
                (fdict_list_generator,
                 subdir,
                 imagetyp=imagetyp,
                 glob_include=glob_include,
                 **kwargs)
            for sl in sub_fdict_list:
                fdict_list.append(sl)
        # After processing our subdirs, process 'directory.'
        for gi in glob_include:
            # Speed things up considerably by allowing globbing.  As
            # per comment above, if None passed to glob_include, this
            # runs once with None passed to ccdp.ImageFileCollection's
            # glob_include
            # Avoid anoying warning abotu empty collection
            flist = glob.glob(os.path.join(directory, gi))
            if len(flist) == 0:
                continue
            collection = ccdp.ImageFileCollection(directory,
                                                  filenames=flist)
            #collection = ccdp.ImageFileCollection(directory,
            #                                      glob_include=gi)
            # Call ourselves recursively, but using the code below,
            # since collection is now defined
            gi_fdict_list = fdict_list_collector \
                (fdict_list_generator,
                 collection=collection,
                 imagetyp=imagetyp,
                 **kwargs)
            for gi in gi_fdict_list:
                fdict_list.append(gi)
        # Here is the end of our recursive line if directory and
        # optional subdirs were specified
        return fdict_list
    if collection.summary is None:
        # We were probably called on a glob_include that yielded no results
        return fdict_list

    # If we made it here, we have a collection, possibly from calling
    # ourselves recursively.  Hand off to our generator to do all the
    # work
    return fdict_list_generator(collection, imagetyp=imagetyp, **kwargs)

def bias_dark_fdict_generator(collection,
                              imagetyp=None,
                              dccdt_tolerance=dccdt_tolerance,
                              debug=False):
    # Create a summary table narrowed to our imagetyp
    our_imagetyp = collection.summary['imagetyp'] == imagetyp
    narrow_to_imagetyp = collection.summary[our_imagetyp]
    ts = narrow_to_imagetyp['ccd-temp']
    # ccd-temp is recorded as a string.  Convert it to a number so
    # we can sort +/- values properly
    ts = np.asarray(ts)
    # Get the sort indices so we can extract fnames in proper order
    tsort_idx = np.argsort(ts)
    # For ease of use, re-order everything in terms of tsort
    ts = ts[tsort_idx]
    narrow_to_imagetyp = narrow_to_imagetyp[tsort_idx]    
    # Spot jumps in t and translate them into slices into ts
    dts = ts[1:] - ts[0:-1]
    jump = np.flatnonzero(dts > dccdt_tolerance)
    tslices = np.append(0, jump+1)
    # Whew!  This was a tricky one!
    # https://stackoverflow.com/questions/509211/understanding-slice-notation
    # showed that I needed None and explicit call to slice(), below,
    # to be able to generate an array element in tslices that referred
    # to the last array element in ts.  :-1 is the next to the last
    # element because of how slices work.
    tslices = np.append(tslices, None)
    if debug:
        print(ts)
        print(dts)
        print(tslices)
    fdict_list = []
    for it in range(len(tslices)-1):
        these_ts = ts[slice(tslices[it], tslices[it+1])]
        mean_ccdt = np.mean(these_ts)
        # Create a new summary Table that inlcudes just these Ts
        narrow_to_t = narrow_to_imagetyp[tslices[it]:tslices[it+1]]
        exps = narrow_to_t['exptime']
        # These are sorted by increasing exposure time
        ues = np.unique(exps)
        for ue in ues:
            exp_idx = np.flatnonzero(exps == ue)
            files = narrow_to_t['file'][exp_idx]
            full_files = [os.path.join(collection.location, f) for f in files]
            fdict_list.append({'directory': collection.location,
                               'CCDT': mean_ccdt,
                               'EXPTIME': ue,
                               'fnames': full_files})
    return fdict_list

def discard_intermediate(out_fnames, sdir,
                         calibration_scratch, keep_intermediate):
    if not keep_intermediate:
        for f in out_fnames:
            try:
                os.remove(f)
            except Exception as e:
                # We do not expect this, since we created these with
                # our local process
                log.error(f'Unexpected!  Remove {f} failed: ' + str(e))
        # These we expect to fail until all of our other parallel
        # processes have finished
        try:
            os.rmdir(sdir)
        except Exception as e:
            pass
        try:
            os.rmdir(calibration_scratch)
        except Exception as e:
            pass


def bias_combine_one_fdict(fdict,
                           outdir=calibration_root,
                           calibration_scratch=calibration_scratch,
                           keep_intermediate=False,
                           show=False,
                           min_num_biases=min_num_biases,
                           dccdt_tolerance=dccdt_tolerance,
                           camera_description=sx694.camera_description,
                           gain=sx694.gain,
                           satlevel=sx694.satlevel,
                           readnoise=sx694.example_readnoise,
                           readnoise_tolerance=sx694.readnoise_tolerance,
                           gain_correct=False,
                           num_processes=max_num_processes,
                           mem_frac=max_mem_frac,
                           naxis1=sx694.naxis1,
                           naxis2=sx694.naxis2,
                           bitpix=max_ccddata_bitpix,
                           **kwargs):

    """Worker that allows the parallelization of calibrations taken at one
    temperature, exposure time, filter, etc.


    gain_correct : Boolean
        Effects unit of stored images.  True: units of electron.
        False: unit of ADU.  Default: False
    """

    fnames = fdict['fnames']
    num_files = len(fnames)
    mean_ccdt = fdict['CCDT']
    directory = fdict['directory']
    tmp = ccddata_read(fnames[0])
    tm = tmp.meta['DATE-OBS']
    this_dateb1, _ = tm.split('T')
    outbase = os.path.join(outdir, this_dateb1)
    bad_fname = outbase + '_ccdT_XXX' + '_bias_combined_bad.fits'
    if num_files < min_num_biases:
        log.warning(f"Not enough good biases found at CCDT = {mean_ccdt} C in {directory}")
        Path(bad_fname).touch()
        return False

    # Make a scratch directory that is the date of the first file.
    # Not as fancy as the biases, but, hey, it is a scratch directory
    sdir = os.path.join(calibration_scratch, this_dateb1)

    #mem = psutil.virtual_memory()
    #num_files_can_fit = \
    #    int(min(num_files,
    #            mem.available*mem_frac/ccddata_size))
    #num_can_process = min(num_processes, num_files_can_fit)
    #print('bias_combine_one_fdict: num_processes = {}, mem_frac = {}, num_files= {}, num_files_can_fit = {}, num_can_process = {}'.format(num_processes, mem_frac, num_files, num_files_can_fit, num_can_process))

    # Use CorMultiPipe to subtract the median from each bias and
    # create a dict of stats for a pandas dataframe
    cmp = CorMultiPipe(num_processes=num_processes,
                       mem_frac=mem_frac,
                       naxis1=naxis1,
                       naxis2=naxis2,
                       bitpix=bitpix,
                       outdir=sdir,
                       create_outdir=True,
                       overwrite=True,
                       pre_process_list=[light_image],
                       post_process_list=[bias_stats, jd_meta])
    #combined_base = outbase + '_bias_combined'
    pout = cmp.pipeline(fnames, **kwargs)
    pout, fnames = prune_pout(pout, fnames)
    if len(pout) == 0:
        log.warning(f"Not enough good biases {len(pout)} found at CCDT = {mean_ccdt} C in {directory}")
        Path(bad_fname).touch()
        return False
    out_fnames, pipe_meta = zip(*pout)
    if len(out_fnames) < min_num_biases:
        log.warning(f"Not enough good biases {len(pout)} found at CCDT = {mean_ccdt} C in {directory}")
        discard_intermediate(out_fnames, sdir,
                             calibration_scratch, keep_intermediate)
        Path(bad_fname).touch()
        return False

    stats = [m['bias_stats'] for m in pipe_meta]
    jds = [m['jd'] for m in pipe_meta]

    df = pd.DataFrame(stats)
    tm = Time(np.mean(jds), format='jd')
    this_date = tm.fits
    this_dateb = this_date.split('T')[0]
    if this_dateb != this_dateb1:
        log.warning(f"first bias is on {this_dateb1} but average is {this_dateb}")

    this_ccdt = '{:.1f}'.format(mean_ccdt)
    f = plt.figure(figsize=[8.5, 11])

    # In the absence of a formal overscan region, this is the best
    # I can do
    medians = df['median']
    overscan = np.mean(medians)

    ax = plt.subplot(6, 1, 1)
    plt.title('CCDT = {} C on {}'.format(this_ccdt, this_dateb))
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which='both', bottom=True, top=True, left=True, right=True)
    plt.plot(df['time'], df['ccdt'], 'k.')
    plt.ylabel('CCDT (C)')

    ax = plt.subplot(6, 1, 2)
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which='both', bottom=True, top=True, left=True, right=True)
    plt.plot(df['time'], df['max'], 'k.')
    plt.ylabel('max (ADU)')

    ax = plt.subplot(6, 1, 3)
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which='both', bottom=True, top=True, left=True, right=False)
    plt.plot(df['time'], df['median'], 'k.')
    plt.plot(df['time'], df['mean'], 'r.')
    plt.ylabel('median & mean (ADU)')
    plt.legend(['median', 'mean'])
    secax = ax.secondary_yaxis \
        ('right',
         functions=(lambda adu: (adu - overscan)*gain,
                    lambda e: e/gain + overscan))
    secax.set_ylabel('Electrons')

    ax=plt.subplot(6, 1, 4)
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which='both', bottom=True, top=True, left=True, right=True)
    plt.plot(df['time'], df['min'], 'k.')
    plt.ylabel('min (ADU)')

    ax=plt.subplot(6, 1, 5)
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which='both', bottom=True, top=True, left=True, right=True)
    plt.plot(df['time'], df['std'], 'k.')
    plt.ylabel('std (electron)')

    ax=plt.subplot(6, 1, 6)
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which='both', bottom=True, top=True, left=True, right=True)
    plt.plot(df['time'], df['rdnoise'], 'k.')
    plt.ylabel('rdnoise (electron)')

    plt.gcf().autofmt_xdate()

    # At the 0.5 deg level, there seems to be no correlation between T and bias level
    #plt.plot(df['ccdt'], df['mean'], 'k.')
    #plt.xlabel('ccdt')
    #plt.ylabel('mean')
    #plt.show()
        
    # Make sure outdir exists
    os.makedirs(outdir, exist_ok=True)
    outbase = os.path.join(outdir, this_dateb + '_ccdT_' + this_ccdt)
    out_fname = outbase + '_bias_combined.fits'
    plt.savefig((outbase + '_bias_vs_time.png'), transparent=True)
    if show:
        plt.show()
    plt.close()

    # Do a sanity check of readnoise
    av_rdnoise = np.mean(df['rdnoise'])            
    if (np.abs(av_rdnoise/sx694.example_readnoise - 1)
        > readnoise_tolerance):
        log.warning('High readnoise {}, skipping {}'.format(av_rdnoise, out_fname))
        Path(outbase + '_bad.fits').touch()
        return False

    # Use ccdp.combine since it enables memory management by breaking
    # up images to smaller chunks (better than throwing images away).
    # --> eventually it would be great to parallelize this primitive,
    # since it is very slow.  In the mean time I have parallelized all
    # the higher steps!
    mem = psutil.virtual_memory()
    im = \
        ccdp.combine(list(out_fnames),
                     method='average',
                     sigma_clip=True,
                     sigma_clip_low_thresh=5,
                     sigma_clip_high_thresh=5,
                     sigma_clip_func=np.ma.median,
                     sigma_clip_dev_func=mad_std,
                     mem_limit=mem.available*mem_frac)
    im.meta = sx694.metadata(im.meta)
    if gain_correct:
        im = ccdp.gain_correct(im, gain*u.electron/u.adu)
        im_gain = 1
    else:
        im_gain = gain
    im = mask_above_key(im, key='SATLEVEL')
    im = mask_above_key(im, key='NONLIN')
        
    # Collect image metadata.  For some reason, masked pixels
    # aren't ignored by std, etc. even though they output masked
    # arrays (which is annoying in its own right -- see example
    # commented mean).  So just create a new array, if needed, and
    # only put into it the good pixels
    if im.mask is None:
        # This is not a copy!  But don't worry, we don't change tim,
        # just collect info from it
        tim = im
    else:
        # This is a new array with fewer elements.  We will collect
        # stats and write the original im, below
        tim = im.data[im.mask == 0]
    std =  np.std(tim)*im_gain
    med =  np.median(tim)*im_gain
    #mean = np.asscalar(np.mean(tim).data  )
    mean = np.mean(tim)*im_gain
    tmin = np.min(tim)*im_gain
    tmax = np.max(tim)*im_gain
    print('std, mean, med, tmin, tmax (electron)')
    print(std, mean, med, tmin, tmax)
    im.meta['DATE-OBS'] = (this_date, 'Average of DATE-OBS from set of biases')
    im.meta['CCD-TEMP'] = (mean_ccdt, 'Average CCD temperature for combined biases')
    im.meta['RDNOISE'] = (av_rdnoise, 'Measured readnoise (electron)')
    im.meta['STD'] = (std, 'Standard deviation of image (electron)')
    im.meta['MEDIAN'] = (med, 'Median of image (electron)')
    im.meta['MEAN'] = (mean, 'Mean of image (electron)')
    im.meta['MIN'] = (tmin, 'Min of image (electron)')
    im.meta['MAX'] = (tmax, 'Max of image (electron)')
    im.meta['HIERARCH OVERSCAN_VALUE'] = (overscan, 'Average of raw bias medians (ADU)')
    im.meta['HIERARCH SUBTRACT_OVERSCAN'] = (True, 'Overscan has been subtracted')
    im.meta['NCOMBINE'] = (len(out_fnames), 'Number of biases combined')
    # Record each filename
    for i, f in enumerate(fnames):
        im.meta['FILE{0:02}'.format(i)] = f
    add_history(im.meta,
                'Combining NCOMBINE biases indicated in FILENN')
    add_history(im.meta,
                'SATLEVEL and NONLIN apply to pre-overscan subtraction')
    # Leave these large for fast calculations downstream and make
    # final results that primarily sit on disk in bulk small
    #im.data = im.data.astype('float32')
    #im.uncertainty.array = im.uncertainty.array.astype('float32')
    im.write(out_fname, overwrite=True)
    # Always display image in electrons
    impl = plt.imshow(im.multiply(im_gain), origin='lower',
                      cmap=plt.cm.gray,
                      filternorm=0, interpolation='none',
                      vmin=med-std, vmax=med+std)
    plt.title('CCDT = {} C on {} (electrons)'.format(this_ccdt, this_dateb))
    plt.savefig((outbase + '_bias_combined.png'), transparent=True)
    if show:
        plt.show()
    plt.close()
    discard_intermediate(out_fnames, sdir,
                         calibration_scratch, keep_intermediate)
                
def bias_combine(directory=None,
                 collection=None,
                 subdirs=calibration_subdirs,
                 glob_include=bias_glob,
                 dccdt_tolerance=dccdt_tolerance,
                 num_processes=max_num_processes,
                 mem_frac=max_mem_frac,
                 num_calibration_files=num_calibration_files,
                 naxis1=sx694.naxis1,
                 naxis2=sx694.naxis2,
                 bitpix=max_ccddata_bitpix,
                 process_expand_factor=cor_process_expand_factor,
                 **kwargs):
    """Combine biases in a directory

    Parameters
    ----------
    directory : string
        Directory in which to find biases.  Default: ``None``

    collection : ccdp.Collection
        Collection of directory in which to find calibration data.
        Default: ``None``

    subdirs : list
        List of subdirectories in which to search for calibration
        data.  Default: :value:`calibration_subdirs`

    glob_include : list
        List of `glob` expressions for calibration filenames

    dccdt_tolerance : float
        During the creation of master biases and darks files, are
        grouped by CCD temperature (ccdt).  This is the change in
        temperature seen as a function of time that is used to trigger
        the creation of a new group

    num_processes : int
        Number of processes available to this task for
        multiprocessing.  Default: :value:`max_num_processes`

    mem_frac : float
        Fraction of memory available to this task.  Default:
        :value:`max_mem_frac`

    **kwargs passed to bias_combine_one_fdict

    """
    fdict_list = \
        fdict_list_collector(bias_dark_fdict_generator,
                             directory=directory,
                             collection=collection,
                             subdirs=subdirs,
                             imagetyp='BIAS',
                             glob_include=glob_include,
                             dccdt_tolerance=dccdt_tolerance)
    if collection is not None:
        # Make sure 'directory' is a valid variable
        directory = collection.location
    nfdicts = len(fdict_list)
    if nfdicts == 0:
        log.debug('No biases found in: ' + directory)
        return False

    one_fdict_size = (num_calibration_files
                      * naxis1 * naxis2
                      * bitpix/8
                      * process_expand_factor)

    our_num_processes = num_can_process(nfdicts,
                                        num_processes=num_processes,
                                        mem_frac=mem_frac,
                                        process_size=one_fdict_size)

    num_subprocesses = int(num_processes / our_num_processes)
    subprocess_mem_frac = mem_frac / our_num_processes
    log.debug('bias_combine: {} num_processes = {}, mem_frac = {}, our_num_processes = {}, num_subprocesses = {}, subprocess_mem_frac = {}'.format(directory, num_processes, mem_frac, our_num_processes, num_subprocesses, subprocess_mem_frac))

    wwk = WorkerWithKwargs(bias_combine_one_fdict,
                           num_processes=num_subprocesses,
                           mem_frac=subprocess_mem_frac,
                           **kwargs)
    if nfdicts == 1:
        for fdict in fdict_list:
                wwk.worker(fdict)
    else:
        with NoDaemonPool(processes=our_num_processes) as p:
            p.map(wwk.worker, fdict_list)

def dark_combine_one_fdict(fdict,
                           outdir=calibration_root,
                           calibration_scratch=calibration_scratch,
                           outname_append=outname_append,
                           keep_intermediate=False,
                           show=False,
                           dccdt_tolerance=dccdt_tolerance,
                           mask_threshold=sx694.dark_mask_threshold,
                           num_processes=max_num_processes,
                           mem_frac=max_mem_frac,
                           naxis1=sx694.naxis1,
                           naxis2=sx694.naxis2,
                           bitpix=max_ccddata_bitpix,
                           **kwargs):
    """Worker that allows the parallelization of calibrations taken at one
    temperature, exposure time, filter, etc.

    """

    fnames = fdict['fnames']
    mean_ccdt = fdict['CCDT']
    exptime = fdict['EXPTIME']
    directory = fdict['directory']
    tmp = ccddata_read(fnames[0])
    tm = tmp.meta['DATE-OBS']
    this_dateb1, _ = tm.split('T')
    outbase = os.path.join(outdir, this_dateb1)
    bad_fname = outbase + '_ccdT_XXX' + '_bias_combined_bad.fits'

    # Make a scratch directory that is the date of the first file.
    # Not as fancy as the biases, but, hey, it is a scratch directory
    sdir = os.path.join(calibration_scratch, this_dateb1)

    cmp = CorMultiPipe(num_processes=num_processes,
                       mem_frac=mem_frac,
                       naxis1=naxis1,
                       naxis2=naxis2,
                       bitpix=bitpix,
                       outdir=sdir,
                       create_outdir=True,
                       overwrite=True,
                       pre_process_list=[light_image],
                       post_process_list=[jd_meta])
    pout = cmp.pipeline(fnames, **kwargs)
    pout, fnames = prune_pout(pout, fnames)
    if len(pout) == 0:
        log.warning(f"No good darks found at CCDT = {mean_ccdt} C in {directory}")
        Path(bad_fname).touch()
        return False

    out_fnames, pipe_meta = zip(*pout)
    jds = [m['jd'] for m in pipe_meta]

    tm = Time(np.mean(jds), format='jd')
    this_date = tm.fits
    this_dateb = this_date.split('T')[0]
    if this_dateb != this_dateb1:
        log.warning(f"first dark is on {this_dateb1} but average is {this_dateb}")
    this_ccdt = '{:.1f}'.format(mean_ccdt)
    outbase = '{}_ccdT_{}_exptime_{}s'.format(
        this_dateb, this_ccdt, exptime)
    
    mem = psutil.virtual_memory()
    out_fnames = list(out_fnames)
    if len(out_fnames) == 1:
        im = ccddata_read(out_fnames[0])
    else:
        im = \
            ccdp.combine(out_fnames,
                         method='average',
                         sigma_clip=True,
                         sigma_clip_low_thresh=5,
                         sigma_clip_high_thresh=5,
                         sigma_clip_func=np.ma.median,
                         sigma_clip_dev_func=mad_std,
                         mem_limit=mem.available*mem_frac)
    im = mask_above_key(im, key='SATLEVEL')
    im = mask_above_key(im, key='NONLIN')

    # Create a mask that blanks out all our pixels that are just
    # readnoise.  Multiply this in as zeros, not a formal mask,
    # otherwise subsequent operations with the dark will mask out
    # all but the dark current-affected pixels!
    measured_readnoise = im.meta['RDNOISE']
    is_dark_mask = im.data > measured_readnoise * mask_threshold
    n_dark_pix = np.count_nonzero(is_dark_mask)
    im.meta['NDARKPIX'] \
        = (n_dark_pix, 'number of pixels with dark current')
    if n_dark_pix > 0:
        im.data = im.data * is_dark_mask
        if im.uncertainty is not None:
            im.uncertainty.array = im.uncertainty.array*is_dark_mask

    # Collect image metadata.  For some reason, masked pixels
    # aren't ignored by std, etc. even though they output masked
    # arrays (which is annoying in its own right -- see example
    # commented mean).  So just create a new array, if needed, and
    # only put into it the good pixels
    bad_mask = is_dark_mask == 0
    if im.mask is not None:
        bad_mask = bad_mask | im.mask
    # Flip bad mask around so we get only the dark pixels in the
    # linear range
    tim = im.data[bad_mask == 0]

    std =  np.std(tim)
    #std =  np.asscalar(std.data  )
    med =  np.median(tim)
    #med =  np.asscalar(med.data  )
    mean = np.mean(tim)
    #mean = np.asscalar(mean.data  )
    tmin = np.min(tim)
    tmax = np.max(tim)
    rdnoise = np.sqrt(np.median((tim[1:] - tim[0:-1])**2))
    print('combined dark statistics for ' + outbase)
    print('std, rdnoise, mean, med, min, max, n_dark_pix')
    print(std, rdnoise, mean, med, tmin, tmax, n_dark_pix)
    im.meta['STD'] = (std, 'Standard deviation of image (electron)')
    im.meta['MEDIAN'] = (med, 'Median of image (electron)')
    im.meta['MEAN'] = (mean, 'Mean of image (electron)')
    im.meta['MIN'] = (tmin, 'Min of image (electron)')
    im.meta['MAX'] = (tmax, 'Max of image (electron)')
    im.meta['NCOMBINE'] = (len(out_fnames), 'Number of darks combined')
    add_history(im.meta,
                'Combining NCOMBINE biases indicated in FILENN')
    im.meta['HIERARCH MASK_THRESHOLD'] \
        = (mask_threshold, '*RDNOISE (electron)')
    add_history(im.meta,
                'Setting pixes below MASK_THRESHOLD to zero; prevents subtraction noise')
    # Record each filename
    for i, f in enumerate(fnames):
        im.meta['FILE{0:02}'.format(i)] = f
    # Prepare to write
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    outbase = os.path.join(outdir, outbase)
    out_fname = outbase + '_dark_combined.fits'
    # Leave these large for fast calculations downstream and make
    # final results that primarily sit on disk in bulk small
    #im.data = im.data.astype('float32')
    #im.uncertainty.array = im.uncertainty.array.astype('float32')
    im.write(out_fname, overwrite=True)
    if show:
        impl = plt.imshow(im, origin='lower', cmap=plt.cm.gray,
                          filternorm=0, interpolation='none',
                          vmin=med-std, vmax=med+std)
        plt.show()
        plt.close()
    discard_intermediate(out_fnames, sdir,
                         calibration_scratch, keep_intermediate)

def dark_combine(directory=None,
                 collection=None,
                 subdirs=calibration_subdirs,
                 glob_include=dark_glob,
                 dccdt_tolerance=dccdt_tolerance,
                 num_processes=max_num_processes,
                 mem_frac=max_mem_frac,
                 num_calibration_files=num_calibration_files,
                 naxis1=sx694.naxis1,
                 naxis2=sx694.naxis2,
                 bitpix=max_ccddata_bitpix,
                 process_expand_factor=cor_process_expand_factor,
                 **kwargs):
    fdict_list = \
        fdict_list_collector(bias_dark_fdict_generator,
                             directory=directory,
                             collection=collection,
                             subdirs=subdirs,
                             imagetyp='DARK',
                             glob_include=glob_include,
                             dccdt_tolerance=dccdt_tolerance)
    if collection is not None:
        # Make sure 'directory' is a valid variable
        directory = collection.location
    nfdicts = len(fdict_list)
    if len(fdict_list) == 0:
        log.debug('No darks found in: ' + directory)
        return False

    one_fdict_size = (num_calibration_files
                      * naxis1 * naxis2
                      * bitpix/8
                      * process_expand_factor)

    our_num_processes = num_can_process(nfdicts,
                                        num_processes=num_processes,
                                        mem_frac=mem_frac,
                                        process_size=one_fdict_size)
    num_subprocesses = int(num_processes / our_num_processes)
    subprocess_mem_frac = mem_frac / our_num_processes
    log.debug('dark_combine: {} num_processes = {}, mem_frac = {}, our_num_processes = {}, num_subprocesses = {}, subprocess_mem_frac = {}'.format(directory, num_processes, mem_frac, our_num_processes, num_subprocesses, subprocess_mem_frac))

    wwk = WorkerWithKwargs(dark_combine_one_fdict,
                           num_processes=num_subprocesses,
                           mem_frac=subprocess_mem_frac,
                           **kwargs)
    if nfdicts == 1:
        for fdict in fdict_list:
                wwk.worker(fdict)
    else:
        with NoDaemonPool(processes=our_num_processes) as p:
            p.map(wwk.worker, fdict_list)

def flat_process(ccd, bmp_meta=None,
                 init_threshold=100, # units of readnoise
                 nd_edge_expand=nd_edge_expand,
                 in_name=None,
                 **kwargs):
    # Use photutils.Background2D to smooth each flat and get a
    # good maximum value.  Mask edges and ND filter so as to
    # increase quality of background map
    mask = np.zeros(ccd.shape, bool)
    # Use the CorObsData ND filter stuff with a negative
    # edge_mask to blank out all of the fuzz from the ND filter cut
    obs_data = CorObsData(ccd.to_hdu(), edge_mask=-nd_edge_expand)
    mask[obs_data.ND_coords] = True
    rdnoise = ccd.meta['RDNOISE']
    mask[ccd.data < rdnoise * init_threshold] = True
    ccd.mask = mask
    bkg_estimator = MedianBackground()
    b = Background2D(ccd, 20, mask=mask, filter_size=5,
                     bkg_estimator=bkg_estimator)
    max_flat = np.max(b.background)
    if max_flat > ccd.meta['NONLIN']:
        log.debug(f'flat max value of {max_flat} too bright: {in_name}')
        return (None, {})
    ccd.mask = None
    ccd = ccd.divide(max_flat, handle_meta='first_found')
    ccd.meta['FLATDIV'] = (max_flat, 'Value used to normalize (smoothed max)')
    # Get ready to capture the mean DATE-OBS
    tm = Time(ccd.meta['DATE-OBS'], format='fits')
    if bmp_meta is not None:
        bmp_meta['jd'] = tm.jd 
    return ccd

def flat_combine_one_filt(this_filter,
                          collection=None,
                          outdir=calibration_root,
                          calibration_scratch=calibration_scratch,
                          keep_intermediate=False,
                          min_num_flats=min_num_flats,
                          num_processes=max_num_processes,
                          mem_frac=max_mem_frac,
                          naxis1=sx694.naxis1,
                          naxis2=sx694.naxis2,
                          bitpix=max_ccddata_bitpix,
                          show=False,
                          flat_cut=0.75,
                          nd_edge_expand=nd_edge_expand,
                          **kwargs):
    directory = collection.location
    fnames = collection.files_filtered(imagetyp='FLAT',
                                       filter=this_filter,
                                       include_path=True)

    if len(fnames) < min_num_flats:
        log.warning(f"Not enough good flats found for filter {this_filter} in {directory}")
        return False

    # Make a scratch directory that is the date of the first file.
    # Not as fancy as the biases, but, hey, it is a scratch directory
    tmp = ccddata_read(fnames[0])
    tm = tmp.meta['DATE-OBS']
    this_dateb1, _ = tm.split('T')
    sdir = os.path.join(calibration_scratch, this_dateb1)

    cmp = CorMultiPipe(num_processes=num_processes,
                       mem_frac=mem_frac,
                       naxis1=naxis1,
                       naxis2=naxis2,
                       bitpix=bitpix,
                       outdir=sdir,
                       create_outdir=True,
                       overwrite=True,
                       post_process_list=[flat_process, jd_meta],
                       nd_edge_expand=nd_edge_expand)
    pout = cmp.pipeline(fnames, **kwargs)
    pout, fnames = prune_pout(pout, fnames)
    if len(pout) == 0:
        log.warning(f"Not enough good flats found for filter {this_filter} in {directory}")
        return False
    out_fnames, pipe_meta = zip(*pout)
    if len(out_fnames) < min_num_biases:
        log.warning(f"Not enough good flats found for filter {this_filter} in {directory}")
        discard_intermediate(out_fnames, sdir,
                             calibration_scratch, keep_intermediate)
        return False

    jds = [m['jd'] for m in pipe_meta]

    # Combine our flats
    mem = psutil.virtual_memory()
    #print(f'flat_combine_one_filt: mem_frac {mem_frac}; num_processes {num_processes}')
    #print(f'flat_combine_one_filt: mem_limit {mem.available*mem_frac/2**20}')
    im = \
        ccdp.combine(list(out_fnames),
                     method='average',
                     sigma_clip=True,
                     sigma_clip_low_thresh=5,
                     sigma_clip_high_thresh=5,
                     sigma_clip_func=np.ma.median,
                     sigma_clip_dev_func=mad_std,
                     mem_limit=mem.available*mem_frac)
    im.meta['NCOMBINE'] = (len(out_fnames), 'Number of flats combined')
    # Record each filename
    for i, f in enumerate(fnames):
        im.meta['FILE{0:02}'.format(i)] = f
    add_history(im.meta,
                'Combining NCOMBINE biases indicated in FILENN')

    # Interpolate over our ND filter
    #print(f'flat_combine_one_filt pre CorObsData: mem available: {mem.available/2**20}')
    hdul = im.to_hdu()
    obs_data = CorObsData(hdul, edge_mask=-nd_edge_expand)
    # Capture our ND filter metadata
    im.meta = hdul[0].header
    # --> working on RedCorObsData
    good_pix = np.ones(im.shape, bool)
    good_pix[obs_data.ND_coords] = False
    points = np.where(good_pix)
    values = im[points]
    xi = obs_data.ND_coords
    log.debug(f'flat_combine_one_filt post CorObsData: mem available: {mem.available/2**20}')

    # Linear behaved much better
    nd_replacement = interpolate.griddata(points,
                                          values,
                                          xi,
                                          method='linear')
                                          #method='cubic')
    log.debug(f'flat_combine_one_filt post interpolate.griddata mem available: {mem.available/2**20}')
    im.data[xi] = nd_replacement
    # Do one last smoothing and renormalization
    bkg_estimator = MedianBackground()
    b = Background2D(im, 20, mask=(im.data<flat_cut), filter_size=5,
                     bkg_estimator=bkg_estimator)
    max_flat = np.max(b.background)
    log.debug(f'flat_combine_one_filt post Background2D mem available: {mem.available/2**20}')
    im = im.divide(max_flat, handle_meta='first_found')
    im.mask = im.data < flat_cut
    im.meta['FLAT_CUT'] = (flat_cut, 'Value below which flat is masked')

    # Prepare to write
    tm = Time(np.mean(jds), format='jd')
    this_date = tm.fits
    this_dateb = this_date.split('T')[0]
    if this_dateb != this_dateb1:
        log.warning(f"first flat is on {this_dateb1} but average is {this_dateb}")

    outbase = '{}_{}'.format(this_dateb, this_filter)
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    outbase = os.path.join(outdir, outbase)
    out_fname = outbase + '_flat.fits'
    im.write(out_fname, overwrite=True)
    if show:
        impl = plt.imshow(im, origin='upper', cmap=plt.cm.gray)
        plt.show()
    plt.close()
    discard_intermediate(out_fnames, sdir,
                         calibration_scratch, keep_intermediate)
    
def flat_combine(directory=None,
                 collection=None,
                 subdirs=calibration_subdirs,
                 glob_include=flat_glob,
                 num_processes=max_num_processes,
                 mem_frac=max_mem_frac,
                 num_calibration_files=num_calibration_files,
                 naxis1=sx694.naxis1,
                 naxis2=sx694.naxis2,
                 bitpix=max_ccddata_bitpix,
                 griddata_expand_factor=griddata_expand_factor,
                 **kwargs):
    if subdirs is None:
        subdirs = []
    if collection is None:
        if not os.path.isdir(directory):
            return False
        flist = glob.glob(os.path.join(directory, glob_include))
        if len(flist) == 0:
            return False
        collection = ccdp.ImageFileCollection(directory,
                                              filenames=flist)
    directory = collection.location
    if collection.summary is None:
        for sd in subdirs:
            newdir = os.path.join(directory, sd)
            return flat_combine(newdir,
                                subdirs=None,
                                glob_include=glob_include,
                                num_processes=max_num_processes,
                                mem_frac=mem_frac,
                                **kwargs)
        log.debug('No [matching] FITS files found in  ' + directory)
        return False
    # If we made it here, we have a collection with files in it
    filters = np.unique(collection.summary['filter'])
    nfilts = len(filters)
    if nfilts == 0:
        log.debug('No flats found in: ' + directory)
        return False

    one_filt_size = (num_calibration_files
                         * naxis1 * naxis2
                         * bitpix/8
                         * griddata_expand_factor)

    our_num_processes = num_can_process(nfilts,
                                        num_processes=num_processes,
                                        mem_frac=mem_frac,
                                        process_size=one_filt_size,
                                        error_if_zero=False)
    our_num_processes = max(1, our_num_processes)
    
    # Combining files is the slow part, so we want the maximum of
    # processes doing that in parallel
    log.debug(f'flat_combine: {directory}, nfilts = {nfilts}, our_num_processes = {our_num_processes}')
    # Number of sub-processes in each process we will spawn
    num_subprocesses = int(num_processes / our_num_processes)
    # Similarly, the memory fraction for each process we will spawn
    subprocess_mem_frac = mem_frac / our_num_processes
    log.debug('flat_combine: {} num_processes = {}, mem_frac = {}, our_num_processes = {}, num_subprocesses = {}, subprocess_mem_frac = {}'.format(directory, num_processes, mem_frac, our_num_processes, num_subprocesses, subprocess_mem_frac))
    wwk = WorkerWithKwargs(flat_combine_one_filt,
                           collection=collection,
                           num_processes=num_subprocesses,
                           mem_frac=subprocess_mem_frac,
                           **kwargs)
    if nfilts == 1 or our_num_processes == 1:
        for filt in filters:
                wwk.worker(filt)
    else:
        with NoDaemonPool(processes=our_num_processes) as p:
            p.map(wwk.worker, filters)


######### Calibration object

def get_dirs_dates(directory,
                   filt_list=None,
                   start=None,
                   stop=None):
    """Starting a root directory "directory," returns list of tuples
    (subdir, date) sorted by date.  Handles two cases of directory
    date formatting YYYYMMDD (ACP) and YYYY-MM-DD (MaxIm)

    Parameters
    ----------
    directory : string
        Directory in which to look for subdirectories
    filt_list : list of strings 
        Used to filter out bad directories (e.g. ["cloudy", "bad"]
        will omit listing of, e.g., 2018-02-02_cloudy and
        2018-02-03_bad_focus) 
    start : string YYYY-MM-DD
        Start date (inclusive).  Default = first date
    stop : string YYYY-MM-DD
        Stop date (inclusive).  Default = last date

    """
    assert os.path.isdir(directory)
    fulldirs = [os.path.join(directory, d) for d in os.listdir(directory)]
    # Filter out bad directories first
    dirs = [os.path.basename(d) for d in fulldirs
            if (not os.path.islink(d)
                and os.path.isdir(d)
                and (filt_list is None
                     or not np.any([filt in d for filt in filt_list])))]
    # Prepare to pythonically loop through date formats, trying each on 
    date_formats = ["%Y-%m-%d", "%Y%m%d"]
    ddlist = []
    for thisdir in dirs:
        d = thisdir
        dirfail = True
        for idf in date_formats:
            # The date formats are two characters shorter than the
            # length of the strings I am looking for (%Y is two
            # shorter than YYYY, but %M is the same as MM, etc.)
            d = d[0:min(len(d),len(idf)+2)]
            try:
                thisdate = datetime.datetime.strptime(d, idf)
                ddlist.append((thisdir, thisdate))
                dirfail = False
            except:
                pass
        if dirfail:
            pass
            #log.debug('Skipping non-date formatted directory: ' + thisdir)
    # Thanks to https://stackoverflow.com/questions/9376384/sort-a-list-of-tuples-depending-on-two-elements
    if len(ddlist) == 0:
        return []
    ddsorted = sorted(ddlist, key=lambda e:e[1])
    if start is None:
        start = ddsorted[0][1]
    elif isinstance(start, str):
        start = datetime.datetime.strptime(start, "%Y-%m-%d")
    elif isinstance(start, Time):
        start = start.datetime
    if stop is None:
        stop = ddsorted[-1][1]
    elif isinstance(stop, str):
        stop = datetime.datetime.strptime(stop, "%Y-%m-%d")
    elif isinstance(stop, Time):
        stop = stop.datetime
    if start > stop:
        log.warning('start date {} > stop date {}, returning empty list'.format(start, stop))
        return []
    ddsorted = [dd for dd in ddsorted
                if start <= dd[1] and dd[1] <= stop]
    dirs, dates = zip(*ddsorted)
    dirs = [os.path.join(directory, d) for d in dirs]
    return list(zip(dirs, dates))

def dir_has_calibration(directory, glob_include, subdirs=None):
    """Returns True if directory has calibration files matching pattern(s)
in glob_include.  Optionally checks subdirs"""
    if not os.path.isdir(directory):
        # This is the end of our recursive line
        return False
    if subdirs is None:
        subdirs = []
    for sd in subdirs:
        subdir = os.path.join(directory, sd)
        if dir_has_calibration(subdir, glob_include):
            return True
    # If we made it here, our subdirs had no calibration files or we
    # have been called recursively and are in one
    for gi in glob_include:
        flist = glob.glob(os.path.join(directory, gi))
        if len(flist) > 0:
            return True
    return False

class Lockfile():
    def __init__(self,
                 fname=None,
                 check_every=10):
        assert fname is not None
        self._fname = fname
        self.check_every = check_every

    @property
    def is_set(self):
        return os.path.isfile(self._fname)

    # --> could add a timeout and a user-specified optional message
    def wait(self):
        while self.is_set:
            with open(self._fname, "r") as f:
                log.error(f'lockfile {self._fname} detected for {f.read()}')
            time.sleep(self.check_every)

    def create(self):
        self.wait()
        with open(lockfile, "w") as f:
            f.write('PID: ' + str(os.getpid()))

    def clear(self):
        os.remove(lockfile)

class Calibration():
    """Class for conducting CCD calibrations"""
    def __init__(self,
                 reduce=False,
                 raw_data_root=raw_data_root,
                 calibration_root=calibration_root,
                 subdirs=calibration_subdirs,
                 keep_intermediate=False,
                 ccdt_tolerance=ccdt_tolerance,
                 dark_exp_margin=dark_exp_margin,
                 start_date=None,
                 stop_date=None,
                 gain_correct=True, # This is gain correcting the bias and dark
                 num_processes=max_num_processes,
                 mem_frac=max_mem_frac,
                 num_ccdts=num_ccdts,
                 num_dark_exptimes=num_dark_exptimes,
                 num_filts=num_filts,
                 num_calibration_files=num_calibration_files,
                 naxis1=sx694.naxis1,
                 naxis2=sx694.naxis2,
                 bitpix=max_ccddata_bitpix,
                 process_expand_factor=cor_process_expand_factor,
                 griddata_expand_factor=griddata_expand_factor,
                 bias_glob=bias_glob, 
                 dark_glob=dark_glob,
                 flat_glob=flat_glob,
                 lockfile=lockfile):
        self._raw_data_root = raw_data_root
        self._calibration_root = calibration_root
        self._subdirs = subdirs
        self.keep_intermediate = keep_intermediate
        self._ccdt_tolerance = ccdt_tolerance
        self._dark_exp_margin=dark_exp_margin
        self._bias_table = None
        self._dark_table = None
        self._flat_table = None
        # gain_correct is set only in the biases and propagated
        # through the rest of the pipeline in cor_process
        self._gain_correct = gain_correct
        self._bias_glob = bias_glob
        self._dark_glob = dark_glob
        self._flat_glob = flat_glob
        self._lockfile = lockfile
        self.num_processes = num_processes
        self.mem_frac = mem_frac
        self.num_ccdts = num_ccdts
        self.num_dark_exptimes = num_dark_exptimes
        self.num_filts = num_filts
        self.num_calibration_files = num_calibration_files
        self.naxis1 = naxis1
        self.naxis2 = naxis2
        self.bitpix = bitpix
        self.process_expand_factor = process_expand_factor
        self.griddata_expand_factor = griddata_expand_factor
        if start_date is None:
            self._start_date = datetime.datetime(1,1,1)
        else:
            self._start_date = datetime.datetime.strptime(start_date,
                                                          "%Y-%m-%d")
        if stop_date is None:
            # Make stop time tomorrow in case we are analyzing on the
            # UT boundary
            self._stop_date = datetime.datetime.today() + datetime.timedelta(days=1)
        else:
            self._stop_date = datetime.datetime.strptime(stop_date, "%Y-%m-%d")
        assert self._start_date <= self._stop_date
        # These need to be on a per-instantiation basis, since they
        # depend on our particular start-stop range.  These are also
        # important, since we don't take calibrations every night.  The
        # cost of checking for new reductions is relatively low, since
        # it is mostly a directory listing exercise
        self._bias_dirs_dates_checked = None
        self._dark_dirs_dates_checked = None
        self._flat_dirs_dates_checked = None
        if reduce:
            self.reduce()

    @property
    def gain_correct(self):
        return self._gain_correct

    def dirs_dates_to_reduce(self, table_creator,
                             glob_include,
                             dirs_dates_checked=None,
                             subdirs=None):
        to_check = get_dirs_dates(self._raw_data_root,
                                  start=self._start_date,
                                  stop=self._stop_date)
        # See if we have reduced/checked any/everything in this
        # instantiation.  This is not as efficient as it could be
        # since we have sorted lists, but we don't have many elements,
        # so there is not much point in getting fancier
        if dirs_dates_checked is not None:
            to_check = [dt for dt in to_check
                        if not dt in dirs_dates_checked]
            if len(to_check) == 0:
                return []
        # Take any reductions on disk out of the list.  Note, we check
        # for date only, since we have lost the original directory
        # information once reduced
        tbl = table_creator(autoreduce=False, rescan=True)
        if tbl is not None:
            reduced_ts = [tm.to_datetime() for tm in tbl['dates']]
            # Remove duplicates
            reduced_ts = list(set(reduced_ts))
            to_check = [dt for dt in to_check
                        if not dt[1] in reduced_ts]
            if len(to_check) == 0:
                return []
        to_reduce = [dt for dt in to_check
                     if dir_has_calibration(dt[0],
                                            glob_include,
                                            subdirs=subdirs)]
        # Remove duplicates
        return sorted(list(set(to_reduce)))

    def reduce_bias(self):
        dirs_dates = \
            self.dirs_dates_to_reduce(self.bias_table_create,
                                      self._bias_glob,
                                      self._bias_dirs_dates_checked,
                                      self._subdirs)
        ndirs_dates = len(dirs_dates)
        if ndirs_dates == 0:
            return

        # If we made it here, we have some real work to do
        # Set a simple lockfile so we don't have multiple processes reducing
        lock = Lockfile(self._lockfile)
        lock.create()

        one_fdict_size = (self.num_calibration_files
                          * self.naxis1 * self.naxis2
                          * self.bitpix/8
                          * self.process_expand_factor)

        ncp = num_can_process(self.num_ccdts,
                              num_processes=self.num_processes,
                              mem_frac=self.mem_frac,
                              process_size=self.num_ccdts * one_fdict_size)
        our_num_processes = max(1, ncp)
        num_subprocesses = int(self.num_processes / our_num_processes)
        subprocess_mem_frac = self.mem_frac / our_num_processes
        log.debug(f'Calibration.reduce_bias: ndirs_dates = {ndirs_dates}')
        log.debug('Calibration.reduce_bias: self.num_processes = {}, our_num_processes = {}, num_subprocesses = {}, subprocess_mem_frac = {}'.format(self.num_processes, our_num_processes, num_subprocesses, subprocess_mem_frac))
        #return
        wwk = WorkerWithKwargs(bias_combine,
                               subdirs=self._subdirs,
                               glob_include=self._bias_glob,
                               outdir=self._calibration_root,
                               auto=True, # A little dangerous, but just one place for changes
                               gain_correct=self._gain_correct,
                               num_processes=self.num_processes,
                               naxis1=self.naxis1,
                               naxis2=self.naxis2,
                               process_expand_factor=self.process_expand_factor,

                               num_calibration_files=self.num_calibration_files,
                               mem_frac=self.mem_frac,
                               keep_intermediate=self.keep_intermediate)

        dirs = [dt[0] for dt in dirs_dates]
        if our_num_processes == 1:
            for d in dirs:
                wwk.worker(d)
        else:
            with NoDaemonPool(processes=our_num_processes) as p:
                p.map(wwk.worker, dirs)

        self.bias_table_create(rescan=True, autoreduce=False)
        # This could potentially get set in dirs_dates_to_reduce, but
        # it seems better to set it after we have actually done the work
        all_dirs_dates = get_dirs_dates(self._raw_data_root,
                                  start=self._start_date,
                                  stop=self._stop_date)
        self._bias_dirs_dates_checked = all_dirs_dates
        lock.clear()

    def reduce_dark(self):
        dirs_dates = \
            self.dirs_dates_to_reduce(self.dark_table_create,
                                      self._dark_glob,
                                      self._dark_dirs_dates_checked,
                                      self._subdirs)
        ndirs_dates = len(dirs_dates)
        if ndirs_dates == 0:
            return

        # If we made it here, we have some real work to do
        # Set a simple lockfile so we don't have multiple processes reducing
        lock = Lockfile(self._lockfile)
        lock.create()

        one_fdict_size = (self.num_calibration_files
                          * self.naxis1 * self.naxis2
                          * self.bitpix/8
                          * self.process_expand_factor)

        ncp = num_can_process(self.num_ccdts,
                              num_processes=self.num_processes,
                              mem_frac=self.mem_frac,
                              process_size=self.num_ccdts * one_fdict_size)
        our_num_processes = max(1, ncp)
        num_subprocesses = int(self.num_processes / our_num_processes)
        subprocess_mem_frac = self.mem_frac / our_num_processes
        log.debug(f'Calibration.reduce_dark: ndirs_dates = {ndirs_dates}')
        log.debug('Calibration.reduce_dark: self.num_processes = {}, our_num_processes = {}, num_subprocesses = {}, subprocess_mem_frac = {}'.format(self.num_processes, our_num_processes, num_subprocesses, subprocess_mem_frac))
        #return
        wwk = WorkerWithKwargs(dark_combine,
                               subdirs=self._subdirs,
                               glob_include=self._dark_glob,
                               outdir=self._calibration_root,
                               calibration=self,
                               auto=True, # A little dangerous, but just one place for changes
                               num_processes=self.num_processes,
                               naxis1=self.naxis1,
                               naxis2=self.naxis2,
                               process_expand_factor=self.process_expand_factor,

                               num_calibration_files=self.num_calibration_files,
                               mem_frac=self.mem_frac,
                               keep_intermediate=self.keep_intermediate)

        dirs = [dt[0] for dt in dirs_dates]
        if our_num_processes == 1:
            for d in dirs:
                wwk.worker(d)
        else:
            with NoDaemonPool(processes=our_num_processes) as p:
                p.map(wwk.worker, dirs)

        self.dark_table_create(rescan=True, autoreduce=False)
        # This could potentially get set in dirs_dates_to_reduce, but
        # it seems better to set it after we have actually done the work
        all_dirs_dates = get_dirs_dates(self._raw_data_root,
                                  start=self._start_date,
                                  stop=self._stop_date)
        self._dark_dirs_dates_checked = all_dirs_dates
        lock.clear()

    def reduce_flat(self):
        dirs_dates = \
            self.dirs_dates_to_reduce(self.flat_table_create,
                                      self._flat_glob,
                                      self._flat_dirs_dates_checked,
                                      self._subdirs)
        ndirs_dates = len(dirs_dates)
        if ndirs_dates == 0:
            return

        # If we made it here, we have some real work to do
        # Set a simple lockfile so we don't have multiple processes reducing
        lock = Lockfile(self._lockfile)
        lock.create()

        one_filt_size = (self.num_calibration_files
                         * self.naxis1 * self.naxis2
                         * self.bitpix/8
                         * self.griddata_expand_factor)

        # Our sub-process can divide and conquer if necessary
        ncp = num_can_process(self.num_filts,
                              num_processes=self.num_processes,
                              mem_frac=self.mem_frac,
                              process_size=self.num_filts * one_filt_size,
                              error_if_zero=False)
        our_num_processes = max(1, ncp)
        num_subprocesses = int(self.num_processes / our_num_processes)
        subprocess_mem_frac = self.mem_frac / our_num_processes
        log.debug(f'Calibration.reduce_flat: ndirs_dates = {ndirs_dates}')
        log.debug('Calibration.reduce_flat: self.num_processes = {}, our_num_processes = {}, num_subprocesses = {}, subprocess_mem_frac = {}'.format(self.num_processes, our_num_processes, num_subprocesses, subprocess_mem_frac))
        wwk = WorkerWithKwargs(flat_combine,
                               subdirs=self._subdirs,
                               glob_include=self._flat_glob,
                               outdir=self._calibration_root,
                               calibration=self,
                               auto=True, # A little dangerous, but just one place for changes
                               num_processes=self.num_processes,
                               mem_frac=self.mem_frac,
                               num_calibration_files=self.num_calibration_files,
                               naxis1=self.naxis1,
                               naxis2=self.naxis2,
                               griddata_expand_factor=self.griddata_expand_factor,
                               keep_intermediate=self.keep_intermediate)
                               
        dirs = [dt[0] for dt in dirs_dates]
        if our_num_processes == 1:
            for d in dirs:
                wwk.worker(d)
        else:
            with NoDaemonPool(processes=our_num_processes) as p:
                p.map(wwk.worker, dirs)

        self.flat_table_create(rescan=True, autoreduce=False)
        # This could potentially get set in dirs_dates_to_reduce, but
        # it seems better to set it after we have actually done the work
        all_dirs_dates = get_dirs_dates(self._raw_data_root,
                                  start=self._start_date,
                                  stop=self._stop_date)
        self._flat_dirs_dates_checked = all_dirs_dates
        lock.clear()

    def reduce(self):
        self.reduce_bias()
        self.reduce_dark()
        self.reduce_flat()

    def bias_table_create(self,
                          rescan=False, # Set to True after new biases have been added
                          autoreduce=True): # Set to False to break recursion
                                            # when first looking for 
        """Create table of bias info from calibration directory"""
        if autoreduce:
            # By default always do auto reduction to catch the latest downloads
            self.reduce_bias()
            return self._bias_table
        # If we made it here, autoreduce is guaranteed to be false
        if rescan:
            self._bias_table = None
        if self._bias_table is not None:
            return self._bias_table
        if not os.path.isdir(self._calibration_root):
            # We haven't reduced any calibration images yet and we
            # don't want to automatically do so (just yet)
            return None
        fnames = glob.glob(os.path.join(self._calibration_root,
                                        '*_bias_combined*'))
        fnames = [f for f in fnames if '.fits' in f]
        if len(fnames) == 0:
            # Catch the not autoreduce case when we still have no files
            return None
        # If we made it here, we have files to populate our table
        dates = []
        ccdts = []
        bads = []
        for fname in fnames:
            bfname = os.path.basename(fname)
            sfname = bfname.split('_')
            date = Time(sfname[0], format='fits')
            bad = 'bad' in bfname
            if bad:
                ccdt = np.NAN
            else:
                ccdt = float(sfname[2])
            dates.append(date)
            ccdts.append(ccdt)
            bads.append(bad)
        self._bias_table = QTable([fnames, dates, ccdts, bads],
                                  names=('fnames', 'dates', 'ccdts', 'bad'),
                                  meta={'name': 'Bias information table'})
        return self._bias_table

    def dark_table_create(self,
                          rescan=False, # Set to True after new biases have been added
                          autoreduce=True): # Set to False to break recursion
                                            # when first looking for 
        """Create table of bias info from calibration directory"""
        if autoreduce:
            # By default always do auto reduction to catch the latest downloads
            self.reduce_dark()
            return self._dark_table
        # If we made it here, autoreduce is guaranteed to be false
        if rescan:
            self._dark_table = None
        if self._dark_table is not None:
            return self._dark_table
        if not os.path.isdir(self._calibration_root):
            # We haven't reduced any calibration images yet and we
            # don't want to automatically do so (just yet)
            return None
        fnames = glob.glob(os.path.join(self._calibration_root,
                                        '*_dark_combined*'))
        fnames = [f for f in fnames if '.fits' in f]
        if len(fnames) == 0:
            # Catch the not autoreduce case when we still have no files
            return None
        # If we made it here, we have files to populate our table
        dates = []
        ccdts = []
        exptimes = []
        bads = []
        for fname in fnames:
            bfname = os.path.basename(fname)
            sfname = bfname.split('_')
            date = Time(sfname[0], format='fits')
            bad = 'bad' in bfname
            if bad:
                ccdt = np.NAN
            else:
                ccdt = float(sfname[2])
            exptime = sfname[4]
            exptime = float(exptime[:-1])
            dates.append(date)
            ccdts.append(ccdt)
            exptimes.append(exptime)
            bads.append(bad)
        self._dark_table = \
            QTable([fnames, dates, ccdts, exptimes, bads],
                   names=('fnames', 'dates', 'ccdts', 'exptimes', 'bad'),
                   meta={'name': 'Dark information table'})
        return self._dark_table

    def flat_table_create(self,
                          rescan=False, # Set to True after new biases have been added
                          autoreduce=True): # Set to False to break recursion
                                            # when first looking for 
        """Create table of bias info from calibration directory"""
        if autoreduce:
            # By default always do auto reduction to catch the latest downloads
            self.reduce_flat()
            return self._flat_table
        # If we made it here, autoreduce is guaranteed to be false
        if rescan:
            self._flat_table = None
        if self._flat_table is not None:
            return self._flat_table
        if not os.path.isdir(self._calibration_root):
            # We haven't reduced any calibration images yet and we
            # don't want to automatically do so (just yet)
            return None
        fnames = glob.glob(os.path.join(self._calibration_root, '*_flat.fits'))
        if len(fnames) == 0:
            # Catch the not autoreduce case when we still have no files
            return None
        # If we made it here, we have files to populate our table
        dates = []
        filts = []
        bads = []
        for fname in fnames:
            bfname = os.path.basename(fname)
            sfname = bfname.split('_', 1)
            date = Time(sfname[0], format='fits')
            bad = 'bad' in bfname
            filttail = sfname[1]
            filt_tail = filttail.split('_flat')
            filt = filt_tail[0]
            dates.append(date)
            filts.append(filt)
            bads.append(bad)
        self._flat_table = \
            QTable([fnames, dates, filts, bads],
                   names=('fnames', 'dates', 'filters', 'bad'),
                   meta={'name': 'Flat information table'})
        return self._flat_table

    @property
    def bias_table(self):
        return self.bias_table_create()

    @property
    def dark_table(self):
        return self.dark_table_create()

    @property
    def flat_table(self):
        return self.flat_table_create()

    def best_bias(self, fname_ccd_or_hdr, ccdt_tolerance=None):
        """Returns filename of best-matched bias for a file"""
        if ccdt_tolerance is None:
            ccdt_tolerance = self._ccdt_tolerance
        if isinstance(fname_ccd_or_hdr, fits.Header):
            hdr = fname_ccd_or_hdr
        else:
            ccd = ccddata_read(fname_ccd_or_hdr)
            hdr = ccd.meta
        tm = Time(hdr['DATE-OBS'], format='fits')
        ccdt = hdr['CCD-TEMP']
        # This is the entry point for reduction 
        bad = self.bias_table['bad']
        dccdts = ccdt - self.bias_table['ccdts']
        within_tol = np.abs(dccdts) < ccdt_tolerance
        good = np.logical_and(within_tol, ~bad)
        good_ccdt_idx = np.flatnonzero(good)
        if len(good_ccdt_idx) == 0:
            log.warning(f'No biases found within {ccdt_tolerance} C, broadening by factor of 2')
            return self.best_bias(hdr, ccdt_tolerance=ccdt_tolerance*2)
        ddates = tm - self.bias_table['dates']
        best_ccdt_date_idx = np.argmin(np.abs(ddates[good_ccdt_idx]))
        # unwrap
        best_ccdt_date_idx = good_ccdt_idx[best_ccdt_date_idx]
        return self._bias_table['fnames'][best_ccdt_date_idx]

    def best_dark(self,
                  fname_ccd_or_hdr,
                  ccdt_tolerance=None,
                  dark_exp_margin=None):
        """Returns filename of best-matched dark for a file"""
        if ccdt_tolerance is None:
            ccdt_tolerance = self._ccdt_tolerance
        if dark_exp_margin is None:
            dark_exp_margin = self._dark_exp_margin
        
        if isinstance(fname_ccd_or_hdr, fits.Header):
            hdr = fname_ccd_or_hdr
        else:
            ccd = ccddata_read(fname_ccd_or_hdr)
            hdr = ccd.meta
        tm = Time(hdr['DATE-OBS'], format='fits')
        ccdt = hdr['CCD-TEMP']
        exptime = hdr['EXPTIME']
        # This is the entry point for reduction 
        bad = self.dark_table['bad']
        dccdts = ccdt - self.dark_table['ccdts']
        within_tol = np.abs(dccdts) < ccdt_tolerance
        good = np.logical_and(within_tol, ~bad)
        good_ccdt_idx = np.flatnonzero(good)
        if len(good_ccdt_idx) == 0:
            log.warning(f'No darks found within {ccdt_tolerance} C, broadening by factor of 2')
            return self.best_dark(hdr, ccdt_tolerance=ccdt_tolerance*2)
        # Find the longest exposure time in our collection of darks
        # that matches our exposure.  Prefer longer exposure times by
        # dark_exp_margin
        dexptimes = exptime - self.dark_table['exptimes']
        good_exptime_idx = np.flatnonzero(
            abs(dexptimes[good_ccdt_idx]) <  dark_exp_margin)
        if len(good_exptime_idx) == 0:
            log.warning(f'No darks found with exptimes within {dark_exp_margin} s, broadening margin by factor of 2')
            return self.best_dark(hdr,
                                  ccdt_tolerance=ccdt_tolerance,
                                  dark_exp_margin=dark_exp_margin*2)
        # unwrap
        good_exptime_idx = good_ccdt_idx[good_exptime_idx]
        ddates = tm - self.dark_table['dates']
        best_exptime_date_idx = np.argmin(np.abs(ddates[good_exptime_idx]))
        # unwrap
        best_exptime_date_idx = good_exptime_idx[best_exptime_date_idx]
        return self._dark_table['fnames'][best_exptime_date_idx]
    # --> TODO: possibly put in the number of darks as a factor as
    # --> well, weighted by difference in time

    def best_flat(self, fname_ccd_or_hdr):
        """Returns filename of best-matched flat for a file"""
        if isinstance(fname_ccd_or_hdr, fits.Header):
            hdr = fname_ccd_or_hdr
        else:
            ccd = ccddata_read(fname_ccd_or_hdr)
            hdr = ccd.meta
        tm = Time(hdr['DATE-OBS'], format='fits')
        filt = hdr['FILTER']
        # This is the entry point for reduction 
        bad = self.flat_table['bad']
        this_filt = filt == self.flat_table['filters']
        good = np.logical_and(this_filt, ~bad)
        good_filt_idx = np.flatnonzero(good)
        if len(good_filt_idx) == 0:
            raise ValueError(f'No {filt} flats found')
        ddates = tm - self.flat_table['dates']
        best_filt_date_idx = np.argmin(np.abs(ddates[good_filt_idx]))
        # unwrap
        best_filt_date_idx = good_filt_idx[best_filt_date_idx]
        return self._flat_table['fnames'][best_filt_date_idx]

#log.setLevel('DEBUG')
#
#c = Calibration(start_date='2020-07-07', stop_date='2020-08-22', reduce=True)
##fname = '/data/io/IoIO/raw/20200708/HD 118648-S001-R001-C001-Na_on.fts'
#fname = '/data/io/IoIO/raw/2020-07-15/HD87696-0016_Na_off.fit'
#cmp = CorMultiPipe(auto=True, calibration=c,
#                   post_process_list=[nd_filter_mask])
#pout = cmp.pipeline([fname], 
#                    outdir='/tmp', overwrite=True)
#out_fnames, pipe_meta = zip(*pout)
#
#print(pipe_meta)

#c = Calibration(start_date='2020-07-07', stop_date='2020-08-22', reduce=True,
#                keep_intermediate=False)
##fname = '/data/io/IoIO/raw/20200711/Bias-S005-R002-C001-B1.fts'
#fname = '/data/io/IoIO/raw/20200711/Dark-S005-R003-C010-B1.fts'
#ccd = ccddata_read(fname)
#ccd = cor_process(ccd, calibration=c, auto=True)
#ccd.write('/tmp/test.fits', overwrite=True)

#c = Calibration(start_date='2020-05-15', stop_date='2020-05-16', reduce=True,
#                keep_intermediate=False)
##fname = '/data/io/IoIO/raw/20200711/Bias-S005-R002-C001-B1.fts'
#fname = '/data/io/IoIO/raw/20200711/Dark-S005-R003-C010-B1.fts'
#ccd = ccddata_read(fname)
#ccd = cor_process(ccd, calibration=c, auto=True)
#ccd.write('/tmp/test.fits', overwrite=True)

#c = Calibration(start_date='2020-07-07', stop_date='2020-08-22', reduce=True,
#                keep_intermediate=False)
#flat_dir = '/data/io/IoIO/raw/2020-05-15'
#collection = ccdp.ImageFileCollection(flat_dir)
#flat_combine_one_filt('R', collection=collection, outdir='/tmp', keep_intermediate=True, calibration=c, auto=True)

#c = Calibration(start_date='2020-07-07', stop_date='2020-08-22', reduce=True)
##fname = '/data/io/IoIO/raw/20200708/HD 118648-S001-R001-C001-Na_on.fts'
#fname = '/data/io/IoIO/raw/2020-07-15/HD87696-0016_Na_off.fit'
#cmp = CorMultiPipe(auto=True,
#                   post_process_list=[nd_filter_mask])
#pout = cmp.pipeline([fname], calibration=c,
#                    outdir='/tmp', overwrite=True)
#out_fnames, pipe_meta = zip(*pout)
#
#print(pipe_meta)

#c = Calibration(start_date='2020-05-02', stop_date='2020-05-15', reduce=True)
#cmp = CorMultiPipe(auto=True, calibration=c,
#                   post_process_list=[detflux])
#fname1 = '/data/Mercury/raw/2020-05-27/Mercury-0005_Na-on.fit'
#fname2 = '/data/Mercury/raw/2020-05-27/Mercury-0005_Na_off.fit'
#pout = cmp.pipeline([fname1, fname2], outdir='/data/Mercury/analysis/2020-05-27/', overwrite=True)

#bias_combine('/data/io/IoIO/raw/20200711', show=False, auto=True, gain_correct=True)
#bias_combine('/data/io/IoIO/raw/20200708', show=False, auto=True, gain_correct=True)

#bname = '/data/io/IoIO/reduced/Calibration/2020-07-07_ccdT_-10.3_combined_bias.fits'
#dark_combine('/data/io/IoIO/raw/20200711', show=False,
#             oscan=True, gain=True, error=True,
#             master_bias=bname)

#c = Calibration(start_date='2020-07-08', stop_date='2020-07-11')
#c.reduce_bias()
#t = c.bias_table_create(autoreduce=False, rescan=True)
#dsample = '/data/io/IoIO/raw/20200711/Dark-S005-R004-C003-B1.fts'
#bias = c.best_bias(dsample)
#print(bias)
#

c = Calibration(start_date='2020-07-08', stop_date='2020-07-11', reduce=True)
