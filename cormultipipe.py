"""
The cormultipipe module implements the IoIO coronagraph data reduction
pipeline using ccdmultipipe/bigmultipipe as its base
"""

import os
import psutil

import numpy as np

from astropy import log
from astropy import units as u
from astropy.nddata import CCDData, StdDevUncertainty
from astropy.time import Time

from bigmultipipe import multi_proc, multi_logging

from ccdmultipipe import CCDMultiPipe

import sx694
from utils import assure_list, reduced_dir, get_dirs_dates, add_history
from cordata_base import overscan_estimate, CorDataBase, CorDataNDparams
from cordata import CorData
from cor_process import cor_process

# Processing global variables.  Since I avoid use of the global
# statement and don't reassign these at global scope, they stick to
# these values and provide handy defaults for routines and object
# inits.  It is also a way to be lazy about documenting all of the
# code :-o

# General FITS WCS reference:
# https://fits.gsfc.nasa.gov/fits_wcs.html
# https://heasarc.gsfc.nasa.gov/docs/fcg/standard_dict.html

# Tests with first iteration of pipeline showed that the real gain in
# speed is from the physical processors, not the logical processes
# (threads).  Threads automatically make the physical processes
# faster.  Going to num_processes greater than the number of physical
# processes does go faster, but only asymptotically, probably because
# wait times are minimized.  Rather than try to milk the asymptote for
# speed, just max out on physical processors to get the steepest gains
# and leave the asymptote for other jobs
MAX_NUM_PROCESSES = 0.8 * psutil.cpu_count(logical=False)
# Was 0.85 for 64Gb
MAX_MEM_FRAC = 0.92

# Calculate the maximum CCDdata size based on 64bit primary & uncert +
# 8 bit mask / 8 bits per byte.  It will be compared to
# psutil.virtual_memory() at runtime to optimize computational tasks
# is my do-it-yourself multiprocessing routines
MAX_CCDDATA_BITPIX = 2*64 + 8
COR_PROCESS_EXPAND_FACTOR = 3.5

IoIO_ROOT = '/data/IoIO'
RAW_DATA_ROOT = os.path.join(IoIO_ROOT, 'raw')
# string to append to processed files to avoid overwrite of raw data
OUTNAME_APPEND = "_p"

# Number of pixels to expand the ND filter over what CorData finds.
# This is the negative of the CorData edge_mask parameter, since
# that is designed to mask pixels inside the ND filter to make
# centering of object more reliable
ND_EDGE_EXPAND = 40

# Wed Mar 03 09:59:08 2021 EST  jpmorgen@snipe
# 2020 -- early 2021
NA_OFF_ON_RATIO = 4.74
SII_OFF_ON_RATIO = 4.87

######### CorMultiPipe object

class CorMultiPipeBase(CCDMultiPipe):
    """Base class for CorMultiPipe system.  Avoids excessive ND_params and
obj_center calculations by using CorDataBase as CCDData

    """
    ccddata_cls = CorDataBase
    def __init__(self,
                 calibration=None,
                 auto=False,
                 outname_append=OUTNAME_APPEND,
                 naxis1=sx694.naxis1,
                 naxis2=sx694.naxis2,
                 process_expand_factor=COR_PROCESS_EXPAND_FACTOR,
                 **kwargs):
        self.calibration = calibration
        self.auto = auto
        super().__init__(outname_append=outname_append,
                         naxis1=naxis1,
                         naxis2=naxis2,
                         process_expand_factor=process_expand_factor,
                         **kwargs)

    # --> This might fail with multi files
    def pre_process(self, data, **kwargs):
        """Add full-frame check permanently to pipeline."""
        kwargs = self.kwargs_merge(**kwargs)
        if full_frame(data, **kwargs) is None:
                return (None, {})
        return super().pre_process(data, **kwargs)

    def data_process(self, data,
                     calibration=None,
                     auto=None,
                     **kwargs):
        """Process data, adding RAWFNAME keyword"""
        kwargs = self.kwargs_merge(**kwargs)
        if calibration is None:
            calibration = self.calibration
        if auto is None:
            auto = self.auto
        if isinstance(data, CCDData):
            data = add_raw_fname(data, **kwargs)
            data = cor_process(data,
                               calibration=calibration,
                               auto=auto,
                               **kwargs)
            return data
        # Allow processing of individual CCDData in the case where an
        # input file is actually a list (of lists...) of input files
        return [self.data_process(d,
                                  calibration=calibration,
                                  auto=auto,
                                  **kwargs)
                for d in data]

class CorMultiPipeNDparams(CorMultiPipeBase):
    """Uses CorDataNDparams to ensure ND_params calculations locate the ND
    filter to high precision.  Requires a decent amount of skylight

    """
    ccddata_cls = CorDataNDparams

class CorMultiPipe(CorMultiPipeNDparams):
    """Full CorMultiPipe system intended for coronagraphic observations

    """
    
    ccddata_cls = CorData

class FixFnameCorMultipipe(CorMultiPipe):
    def outname_create(self, *args,
                       **kwargs):
        outname = super().outname_create(*args, **kwargs)
        outname = outname.replace('Na-on', 'Na_on')
        return outname


######### CorMultiPipe prepossessing routines
def full_frame(data,
               naxis1=sx694.naxis1,
               naxis2=sx694.naxis2,
               **kwargs):

    """CorMultiPipe pre-processing routine to select full-frame images.
        In the case where data is a list, if any ccd is not full
        frame, the entire list fails.  Currently permanently installed
        into CorMultiPipe.pre_process.

    """
    if isinstance(data, CCDData):
        s = data.shape
        # Note Pythonic C index ordering
        if s != (naxis2, naxis1):
            return None
        return data
    for ccd in data:
        ff = full_frame(ccd, naxis1=naxis1,
                        naxis2=naxis2,
                        **kwargs)
        if ff is None:
            return None
    return data

def im_med_min_max(im):
    """Returns median values of representative dark and light patches
    of images recorded by the IoIO coronagraph"""
    s = np.asarray(im.shape)
    m = s/2 # Middle of CCD
    q = s/4 # 1/4 point
    m = m.astype(int)
    q = q.astype(int)
    # Note Y, X.  Use the left middle to avoid any first and last row
    # issues in biases
    dark_patch = im[m[0]-50:m[0]+50, 0:100]
    light_patch = im[m[0]-50:m[0]+50, q[1]:q[1]+100]
    mdp = np.median(dark_patch)
    mlp = np.median(light_patch)
    return (mdp, mlp)

def light_image(im, light_tolerance=3, **kwargs):
    """CorMultiPipe pre-processing routine to reject light-contaminated bias & dark images
    """
    mdp, mlp = im_med_min_max(im)
    if (mlp - mdp > light_tolerance):
        log.debug('light, dark patch medians ({:.4f}, {:.4f})'.format(mdp, mlp))
        return None
    return im

######### CorMultiPipe post-processing routines
def add_raw_fname(ccd_in, in_name=None, **kwargs):
    ccd = ccd_in.copy()
    ccd.meta['RAWFNAME'] = in_name
    return ccd

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

def combine_masks(data, **kwargs):
    """Combine CCDData masks in a list of CCDData"""
    # Avoid working with large arrays if we don't have to
    newmask = None
    for ccd in data:
        if ccd.mask is None:
            continue
        if newmask is None:
            newmask = ccd.mask
            continue
        newmask += ccd.mask
    if newmask is None:
        return data
    # Write mask into ccds
    newdata = []
    for ccd in data:
        ccd.mask = newmask
        newdata.append(ccd)
    return data

def multi_filter_proc(data, **kwargs):
    #return multi_proc(nd_filter_mask, **kwargs)(data)
    return multi_proc(nd_filter_mask,
                      element_type=CCDData,
                      **kwargs)(data)

def nd_filter_mask(ccd_in, nd_edge_expand=ND_EDGE_EXPAND, **kwargs):
    """CorMultiPipe post-processing routine to mask ND filter
    """
    ccd = ccd_in.copy()
    mask = np.zeros(ccd.shape, bool)
    # Return a copy of ccd with the edge_mask property adjusted.  Do
    # it this way to keep ccd's ND filt parameters intact
    emccd = RedCorData(ccd, edge_mask=-nd_edge_expand)
    mask[emccd.ND_coords] = True
    if ccd.mask is None:
        ccd.mask = mask
    else:
        ccd.mask = ccd.mask + mask
    return ccd

def detflux(ccd_in, exptime_unit=None, **kwargs):
    # --> put in a check for flux units
    ccd = ccd_in.copy()
    # The exptime_unit stuff may become obsolete with Card Quantities
    exptime_unit = exptime_unit or u.s
    exptime = ccd.meta['EXPTIME'] * exptime_unit
    exptime_uncertainty = ccd.meta.get('EXPTIME-UNCERTAINTY')
    if exptime_uncertainty is None:
        ccd = ccd.divide(exptime, handle_meta='first_found')
    else:
        exptime_array = np.full_like(ccd, exptime.value)
        exptime_uncertainty_array = \
            np.full_like(ccd, exptime_uncertainty)
        exptime_uncertainty_std = \
            StdDevUncertainty(exptime_uncertainty_array,
                              unit=exptime_unit,
                              copy=False)
        exp_ccd = CCDData(exptime_array,
                          uncertainty=exptime_uncertainty_std,
                          unit=exptime_unit)
        ccd = ccd.divide(exp_ccd, handle_meta='first_found')
    # Make sure to scale rdnoise, which doesn't participate in
    # additive operations
    rdnoise = ccd.meta.get('rdnoise')
    if rdnoise is not None:
        rdnoise /= exptime.value
        ccd.meta['RDNOISE'] = (rdnoise, '(electron/s)')
    return ccd

