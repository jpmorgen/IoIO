"""
The cormultipipe module implements the IoIO coronagraph data reduction
pipeline using ccdmultipipe/bigmultipipe as its base
"""

import os
import psutil
import argparse

import numpy as np

from astropy import log
from astropy import units as u
from astropy.nddata import CCDData, StdDevUncertainty
from astropy.time import Time
from astropy.coordinates import SkyCoord

from bigmultipipe.argparse_handler import ArgparseHandler, BMPArgparseMixin

from ccdmultipipe import CCDMultiPipe, CCDArgparseMixin

import IoIO.sx694 as sx694
from IoIO.utils import add_history, im_med_min_max

from IoIO.cordata_base import (SMALL_FILT_CROP, overscan_estimate,
                               CorDataBase, CorDataNDparams)
from IoIO.cordata import CorData
from IoIO.cor_process import cor_process, standardize_filt_name

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
MAX_NUM_PROCESSES = int(0.8 * psutil.cpu_count(logical=False))
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
MIN_CENTER_QUALITY = 5


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

    def outname_create(self, *args,
                       outname_ext='.fits',
                       **kwargs):
        """Default to .fits as output extension, fix inconsistent Na_on
        filenames recorded with MaxIm

        """
        outname = super().outname_create(*args,
                                         outname_ext=outname_ext,
                                         **kwargs)
        outname = outname.replace('Na-on', 'Na_on')
        outname = outname.replace('WASP-136b', 'WASP-163b')
        return outname

class CorMultiPipeBinnedOK(CorMultiPipeBase, CCDMultiPipe):
    """Enable binned images on the main camera to be processed"""

    def pre_process(self, *args, **kwargs):
        return CCDMultiPipe(self).pre_process(*args, **kwargs)

######### CorMultiPipe prepossessing routines
def full_frame(data,
               naxis1=sx694.naxis1,
               naxis2=sx694.naxis2,
               **kwargs):

    """CorMultiPipe pre-processing routine to select full-frame images.
        In the case where data is a list, if any ccd is not full
        frame, the entire list fails.  Currently permanently installed
        into CorMultiPipe.pre_process.

    NOTE: The ND filter is slightly off-center.  Early data were
    recorded a slightly less than full-frame in the X-direction in an
    attempty to have robotic software automatically center on Jupiter.
    It didn't work, but be liberal in what we accept for processing
    purposes.

    """
    if isinstance(data, CCDData):
        s = data.shape
        # Note Pythonic C index ordering
        full = np.asarray((sx694.naxis2, sx694.naxis1))
        if np.any(s < 0.8*full):
            return None
        #if s != (naxis2, naxis1):
        #    return None
        return data
    for ccd in data:
        ff = full_frame(ccd, naxis1=naxis1,
                        naxis2=naxis2,
                        **kwargs)
        if ff is None:
            return None
    return data

def light_image(im, light_tolerance=3, **kwargs):
    """CorMultiPipe pre-processing routine to reject light-contaminated bias & dark images
    """
    mdp, mlp = im_med_min_max(im)
    if (mlp - mdp > light_tolerance):
        log.debug('light, dark patch medians ({:.4f}, {:.4f})'.format(mdp, mlp))
        return None
    return im

######### CorMultiPipe post-processing routines
def reject_center_quality_below(ccd_in, bmp_meta=None,
                                min_center_quality=MIN_CENTER_QUALITY,
                                **kwargs):
    # NOTE: unlike the general pattern of post-processing routines,
    # this does not return a copy of the ccd and it expects to
    # sometimes return None
    if bmp_meta is None:
        bmp_meta = {}
    if isinstance(ccd_in, list):
        result = [reject_center_quality_below(
            ccd, bmp_meta=bmp_meta,
            min_center_quality=min_center_quality, **kwargs)
                  for ccd in ccd_in]
        if None in result:
            bmp_meta.clear()
            return None
        return result
    if ccd_in.center_quality >= min_center_quality:
        return ccd_in
    bmp_meta.clear()
    return None
    
def add_raw_fname(ccd_in, in_name=None, **kwargs):
    ccd = ccd_in.copy()
    if isinstance(in_name, str):
        ccd.meta['RAWFNAME'] = in_name
    elif isinstance(in_name, list):
        # Generally I process multiple files in inverse order so
        # metadata, etc., of primary file is easiest to grab
        ccd.meta['RAWFNAME'] = in_name[-1]
    else:
        raise ValueError(f'cannot convert to RAWFNAME in_name {in_name}')
    return ccd

def mask_above_key(ccd_in, bmp_meta=None, key=None, margin=0.1, **kwargs):
    """CorMultiPipe post-processing routine to mask pixels > input key
    """
    if isinstance(ccd_in, list):
        return [mask_above_key(ccd, bmp_meta=bmp_meta, key=key,
                               margin=margin, **kwargs)
                for ccd in ccd_in]
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
        log.debug(f'Masking {n_masked} pixels above {key}')
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
            ccd.mask = np.ma.mask_or(ccd.mask, mask)
    if bmp_meta is not None:
        bmp_meta[n_masked_key] = n_masked
    return ccd

def mask_nonlin_sat(ccd_in, bmp_meta=None, margin=0.1, **kwargs):
    """CorMultiPipe post-processing routine to mask pixels > NONLIN
    and SATLEVEL.  Convenient for calling without needing to provide a keyword

    """
    if isinstance(ccd_in, list):
        return [mask_nonlin_sat(ccd, bmp_meta=bmp_meta, 
                               margin=margin, **kwargs)
                for ccd in ccd_in]
    ccd = ccd_in.copy()
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
            newmask = ccd.mask.copy()
            continue
        newmask = np.logical_or(newmask, ccd.mask)
    if newmask is None:
        return data
    # Write mask into copied versions of ccds
    newdata = []
    for ccd_in in data:
        ccd = ccd_in.copy()        
        ccd.mask = newmask
        newdata.append(ccd)
    return newdata

def nd_filter_mask(ccd_in, nd_edge_expand=ND_EDGE_EXPAND, **kwargs):
    """CorMultiPipe post-processing routine to mask ND filter.  Does
    so individually in the case of a list of ccds
    """
    if isinstance(ccd_in, list):
        return [nd_filter_mask(ccd, nd_edge_expand=ND_EDGE_EXPAND, **kwargs)
                for ccd in ccd_in]
    nd_edge_expand = np.asarray(nd_edge_expand)
    if nd_edge_expand.size == 1:
        nd_edge_expand = np.append(nd_edge_expand, -nd_edge_expand)
    ccd = ccd_in.copy()
    mask = np.zeros(ccd.shape, bool)
    # Return a copy of ccd with the edge_mask property adjusted.  Do
    # it this way to keep ccd's ND filt parameters intact
    emccd = ccd.copy()
    # We are expanding.  edge_mask contracts
    emccd.edge_mask = -nd_edge_expand
    mask[emccd.ND_coords] = True
    if ccd.mask is None:
        ccd.mask = mask
    else:
        ccd.mask = np.ma.mask_or(ccd.mask, mask)
    return ccd

#def multi_filter_proc(data, **kwargs):
#    """CorMultiPipe post-processing routine to mask ND filter when a list
#    of files is being processed for each datum (e.g. na_back)
#
#    """
#    # As per documentation in multi_proc, I have to have a separate
#    # top-level function for each call to multi_proc
#    #return multi_proc(nd_filter_mask, **kwargs)(data)
#    return multi_proc(nd_filter_mask,
#                      element_type=CCDData,
#                      **kwargs)(data)

def detflux(ccd_in, exptime_unit=None, **kwargs):
    if isinstance(ccd_in, list):
        return [detflux(ccd, exptime_unit=None, **kwargs)
                for ccd in ccd_in]
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

def objctradec_to_obj_center(ccd_in, bmp_meta=None, **kwargs):
    """Sets obj_center from OBJCTRA and OBJCTDEC keywords

    Only operates if ccd.wcs is set.

    NOTE! OBJCTRA and OBJCTDEC are assumed to be set from an ephemeris
    calculation and thus center_quality is set to 10

    """
    if isinstance(ccd_in, list):
        return [objctradec_to_obj_center(ccd, bmp_meta=bmp_meta,
                                         **kwargs)
                for ccd in ccd_in]
    if ccd_in.wcs is None:
        return ccd_in
    objctra = ccd_in.meta.get('OBJCTRA')
    objctdec = ccd_in.meta.get('OBJCTDEC')
    if objctra is None or objctdec is None:
        return ccd_in        
    ccd = ccd_in.copy()
    bmp_meta = bmp_meta or {}
    cent = SkyCoord(objctra, objctdec, unit=(u.hour, u.deg))
    x, y = ccd.wcs.world_to_pixel(cent)
    # Check for pre-processed ccd
    ocenter = (ccd.meta.get('OBJ_CR1'), ccd.meta.get('OBJ_CR0'))
    ocenter = np.asarray(ocenter)
    oquality = ccd.meta.get('CENTER_QUALITY')
    if ocenter.all() == None:
        # Calculate
        ocenter = ccd.obj_center.copy()
        oquality = ccd.center_quality
    ccd.obj_center = (y, x)
    ccd.center_quality = 10
    dcent = ccd.obj_center - ocenter
    norm_dcent = np.linalg.norm(dcent)
    ccd.meta['OOBJ_CR0'] = (ocenter[1], 'Old OBJ_CR0')
    ccd.meta['OOBJ_CR1'] = (ocenter[0], 'Old OBJ_CR1')
    ccd.meta['HIERARCH OCENTER_QUALITY'] = (
        oquality, 'Old center quality')
    ccd.meta['DOBJ_CR0'] = (dcent[1], 'Delta original to current OBJ_CR0')
    ccd.meta['DOBJ_CR1'] = (dcent[0], 'Delta original to current OBJ_CR1')
    ccd.meta['DOBJ_CRN'] = (norm_dcent, 'norm of DOBJ_CR0 and DOBJ_CR1')
    add_history(ccd.meta, 'Used OBJCTRA and OBJCTDEC keywords to calculate obj_cent')
    bmp_meta['dobj_center'] = dcent
    return ccd

def small_filter_crop(ccd_in, small_filt_crop=SMALL_FILT_CROP, **kwargs):
    """Crops ccd image to ((y0, y1), (x0, x1)) set in small_filt_crop,
which defaults to the same values used for CorDataNDparams"""
    if isinstance(ccd_in, list):
        result = [small_filter_crop(
            ccd, small_filt_crop=small_filt_crop, **kwargs)
                  for ccd in ccd_in]
        return result
    if full_frame(ccd_in) is None:
        raise ValueError(f'Input ccd is not full-frame')
    vert = np.asarray(small_filt_crop)
    ccd = ccd_in[vert[0,0]:vert[1,0], vert[0,1]:vert[1,1]]
    return ccd    

######### Argparse mixin
class CorArgparseMixin:
    # This modifies BMPArgparseMixin outname_append
    outname_append = OUTNAME_APPEND

    def add_log_level(self, 
                      default='DEBUG',
                      help=None,
                      **kwargs):
        option = 'log_level'
        if help is None:
            help = f'astropy.log level (default: {default})'
        self.parser.add_argument('--' + option, 
                            default=default, help=help, **kwargs)

    def add_raw_data_root(self, 
                          default=RAW_DATA_ROOT,
                          help=None,
                          **kwargs):
        option = 'raw_data_root'
        if help is None:
            help = f'raw data root (default: {default})'
        self.parser.add_argument('--' + option, 
                            default=default, help=help, **kwargs)
        
    # These might go away or morph if I use astropy Tables exclusively
    def add_read_csvs(self, 
                      default=False,
                      help=None,
                      **kwargs):
        option = 'read_csvs'
        if help is None:
            help = (f'Read CSV files')
        self.parser.add_argument('--' + option,
                                 action=argparse.BooleanOptionalAction,
                                 default=default,
                                 help=help, **kwargs)

    def add_write_csvs(self, 
                      default=False,
                      help=None,
                      **kwargs):
        option = 'write_csvs'
        if help is None:
            help = (f'Write CSV files')
        self.parser.add_argument('--' + option,
                                 action=argparse.BooleanOptionalAction,
                                 default=default,
                                 help=help, **kwargs)

class CorArgparseHandler(CorArgparseMixin, CCDArgparseMixin,
                         BMPArgparseMixin, ArgparseHandler):
    """Adds basic argparse options relevant to cormultipipe system""" 

    def add_all(self):
        """Add options used in cmd"""
        self.add_log_level()
        self.add_create_outdir(default=True) 
        self.add_outname_append()
        self.add_fits_fixed_ignore(default=True)
        super().add_all()

    def cmd(self, args):
        log.setLevel(args.log_level)

