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
from astropy.table import QTable, vstack
from astropy.coordinates import SkyCoord, solar_system_ephemeris, get_body

from bigmultipipe import WorkerWithKwargs, NestablePool
from bigmultipipe.argparse_handler import ArgparseHandler, BMPArgparseMixin

from ccdmultipipe import CCDMultiPipe, CCDArgparseMixin

import IoIO.sx694 as sx694
from IoIO.utils import (add_history, im_med_min_max, sum_ccddata, cached_csv,
                        dict_to_ccd_meta, pixel_per_Rj)

from IoIO.cordata_base import CorDataBase
from IoIO.cordata import CorData
from IoIO.cor_process import cor_process

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
# This is not intended for production use.  Everything should go into
# its own specific directory (e.g. Calibration, StandardStar, etc.)
REDUCED_ROOT = os.path.join(IoIO_ROOT, 'reduced')

# string to append to processed files to avoid overwrite of raw data
OUTNAME_APPEND = '_p'
OUTNAME_EXT = '.fits'

# Number of pixels to expand the ND filter over what CorData finds.
# This is the negative of the CorData edge_mask parameter, since that
# is designed to mask pixels inside the ND filter to make centering of
# object more reliable
# --> For optimal processing of U and u_sdss, this needs to be some
# sort of function that checks against a list of filters that are out
# of focus.  This function would process a list of tuples that would
# contain (filter_name, nd_edge_expand) pairs.
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
                 outname_ext=OUTNAME_EXT,
                 naxis1=sx694.naxis1,
                 naxis2=sx694.naxis2,
                 process_expand_factor=COR_PROCESS_EXPAND_FACTOR,
                 **kwargs):
        self.calibration = calibration
        self.auto = auto
        super().__init__(outname_append=outname_append,
                         outname_ext=outname_ext,
                         naxis1=naxis1,
                         naxis2=naxis2,
                         process_expand_factor=process_expand_factor,
                         **kwargs)

    # --> This might fail with multi files
    def pre_process(self, data, **kwargs):
        """Add full-frame and check for filter name permanently to pipeline."""
        kwargs = self.kwargs_merge(**kwargs)
        if has_filter(data, **kwargs) is None:
            return (None, {})            
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
        result = [self.data_process(d,
                                    calibration=calibration,
                                    auto=auto,
                                    **kwargs)
                  for d in data]
        if None in result:
            return None
        return result

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
def has_filter(ccd_in, **kwargs):
    if isinstance(ccd_in, list):
        result = [has_filter(ccd, kwargs=kwargs) for ccd in ccd_in]
        if None in result:
            return None
        return ccd_in
    imagetyp = ccd_in.meta.get('imagetyp')
    imagetyp = imagetyp.lower() 
    if imagetyp in ['bias', 'dark']:
        return ccd_in
    if ccd_in.meta.get('filter') is None:
        return None
    return ccd_in

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
def mean_image(ccd, bmp_meta=None, **kwargs):
    if bmp_meta is None:
        bmp_meta = {}
    m = np.nanmean(ccd) * ccd.unit
    bmp_meta['mean_image'] = m
    return ccd

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
    n_masked_key = 'N_' + key
    meta_key = h + n_masked_key
    ccd.meta[meta_key] = (n_masked, f'masked pixels > {key}')
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
    ccd = mask_above_key(ccd_in, bmp_meta=bmp_meta, key='SATLEVEL')
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

def tavg_to_bmp_meta(ccd_in, bmp_meta=None, **kwargs):
    if isinstance(ccd_in, list):
        return [tavg_to_bmp_meta(ccd, bmp_meta=bmp_meta,
                                 **kwargs)
                for ccd in ccd_in]
    if bmp_meta is None:
        bmp_meta = {}
    bmp_meta['tavg'] = ccd_in.tavg
    bmp_meta['tavg_uncertainty'] = ccd_in.meta['DATE-AVG-UNCERTAINTY'] * u.s
    bmp_meta['exptime'] = ccd_in.meta['EXPTIME'] * u.s
    bmp_meta['exptime-uncertainty'] = ccd_in.meta['EXPTIME-UNCERTAINTY'] * u.s
    return ccd_in

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
    if bmp_meta is None:
        bmp_meta = {}
    cent = SkyCoord(objctra, objctdec, unit=(u.hour, u.deg))
    x, y = ccd.wcs.world_to_pixel(cent)
    ccd.obj_center = (y, x)
    ccd.center_quality = 10
    add_history(ccd.meta, 'Used OBJCTRA and OBJCTDEC keywords to calculate obj_cent')
    # This is causing more trouble than it is worth.  There are
    # several special cases here that just aren't worth pursuing 
    ## Check for pre-processed ccd
    #ocenter = (ccd.meta.get('OBJ_CR1'), ccd.meta.get('OBJ_CR0'))
    #ocenter = np.asarray(ocenter)
    #oquality = ccd.meta.get('CENTER_QUALITY')
    #if ocenter.all() == None:
    #    # Calculate
    #    ocenter = ccd.obj_center.copy()
    #    oquality = ccd.center_quality
    #    ccd.obj_center = (y, x)
    #    ccd.center_quality = 10
    #    dcent = ccd.obj_center - ocenter
    #    norm_dcent = np.linalg.norm(dcent)
    #    ccd.meta['OOBJ_CR0'] = (ocenter[1], 'Old OBJ_CR0')
    #    ccd.meta['OOBJ_CR1'] = (ocenter[0], 'Old OBJ_CR1')
    #    ccd.meta['HIERARCH OCENTER_QUALITY'] = (
    #        oquality, 'Old center quality')
    #    ccd.meta['DOBJ_CR0'] = (dcent[1], 'Delta original to current OBJ_CR0')
    #    ccd.meta['DOBJ_CR1'] = (dcent[0], 'Delta original to current OBJ_CR1')
    #    ccd.meta['DOBJ_CRN'] = (norm_dcent, 'norm of DOBJ_CR0 and DOBJ_CR1')
    #    add_history(ccd.meta, 'Used OBJCTRA and OBJCTDEC keywords to calculate obj_cent')
    #    bmp_meta['dobj_center'] = dcent
    return ccd

def calc_obj_to_ND(ccd_in, bmp_meta=None, **kwargs):
    """obj_to_ND gets messed up when rotating unless it is calculated
ahead of time.  Put this early in the pipeline

    """
    if bmp_meta is None:
        bmp_meta = {}
    if isinstance(ccd_in, list):
        return [calc_obj_to_ND(ccd, bmp_meta=bmp_meta, **kwargs)
                for ccd in ccd_in]
    # This does everything
    ccd = ccd_in.copy()
    bmp_meta['obj_to_ND'] = ccd.obj_to_ND
    return ccd

def crop_ccd(ccd_in, crop_ccd_coord=None,
             crop_from_center=None,
             crop_from_desired_center=None,
             crop_from_obj_center=None,
             **kwargs):
    """Crops ccd image

    Parameters
    ----------
    ccd_in : ccd or list

    crop_ccd_coord : tuple of tuple or None
        Coordinates of crop vertices ((y0, x0), (y1, x1)) 

    crop_from_center : tuple or None
        Pixels +/- from center of image used to create crop_ccd_coord

    crop_from_desired_center : tuple or None
        Pixels from desired center used to create crop_ccd_coord

    """
    if isinstance(ccd_in, list):
        result = [crop_ccd(ccd,
                           crop_ccd_coord=crop_ccd_coord,
                           crop_from_center=crop_from_center,
                           crop_from_desired_center=crop_from_desired_center,
                           crop_from_obj_center=crop_from_obj_center,
                           **kwargs)
                  for ccd in ccd_in]
        return result
    if full_frame(ccd_in) is None:
        raise ValueError(f'Input ccd is not full-frame')
    if ((crop_ccd_coord is not None)
        + (crop_from_center is not None)
        + (crop_from_desired_center is not None)
        + (crop_from_obj_center is not None) != 1):
        raise ValueError('Specify only one of crop_ccd_coord or crop_from_center')
    #_ = ccd_in.obj_center
    if crop_from_obj_center is not None:
        s = np.asarray(ccd_in.shape)
        c = np.round(ccd_in.obj_center).astype(int)
        crop_ccd_coord = ((c[0] - crop_from_center[0],
                           c[0] + crop_from_center[0]),
                          (c[1] - crop_from_center[1],
                           c[1] + crop_from_center[1]))
        return crop_ccd(ccd_in, crop_ccd_coord=crop_ccd_coord)
    
    if crop_from_desired_center is not None:
        c = np.round(ccd_in.desired_center).astype(int)
        crop_ccd_coord = ((c[0] - crop_from_desired_center[0],
                           c[0] + crop_from_desired_center[0]),
                          (c[1] - crop_from_desired_center[1],
                           c[1] + crop_from_desired_center[1]))
        return crop_ccd(ccd_in, crop_ccd_coord=crop_ccd_coord)
    if crop_from_center is not None:
        s = np.asarray(ccd_in.shape)
        c = np.round(s/2).astype(int)
        crop_ccd_coord = ((c[0] - crop_from_center[0],
                           c[0] + crop_from_center[0]),
                          (c[1] - crop_from_center[1],
                           c[1] + crop_from_center[1]))
        return crop_ccd(ccd_in, crop_ccd_coord=crop_ccd_coord)
    vert = np.asarray(crop_ccd_coord)
    ccd = ccd_in[vert[0,0]:vert[1,0], vert[0,1]:vert[1,1]]
    
    return ccd    

def reject_center_quality_below(ccd_in, bmp_meta=None,
                                min_center_quality=MIN_CENTER_QUALITY,
                                **kwargs):
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
    if ccd_in.center_quality is None:
        rawfname = ccd_in.meta.get('RAWFNAME')
        log.error(f'center_quality was None for {rawfname}')
        bmp_meta.clear()
        return None
    if ccd_in.center_quality < min_center_quality:
        bmp_meta.clear()
        return None
    return ccd_in
    
def angle_to_major_body(ccd, body_str):
    """Returns angle between pointing direction and solar system major
    body

    Build-in astropy geocentric ephemeris is used for the solar system
    major body (moon, planets, sun).  The ccd.sky_coord is used to
    construct the CCD's pointing.  The angle between these two
    directions is returned.  For a more accurate pointing direction
    from the perspective of the observatory, the astroquery.horizons
    module can be used.

    Parameters
    ----------
    ccd : astropy.nddata.CCDData

    body : str
        solar system major body name

    Returns
    -------
    angle : astropy.units.Quantity
        angle between pointing direction and major body

    """
    with solar_system_ephemeris.set('builtin'):
        body_coord = get_body(body_str, ccd.tavg, ccd.obs_location)
    # https://docs.astropy.org/en/stable/coordinates/common_errors.html
    # notes that separation is order dependent, with the calling
    # object's frame (e.g., body_coord) used as the reference frame
    # into which the argument (e.g., ccd.sky_coord) is transformed
    return body_coord.separation(ccd.sky_coord)

def planet_to_object(ccd_in, bmp_meta=None, planet=None, **kwargs):
    """Update OBJECT keyword if CCD RA and DEC point at planet, reject otherwise"""
    # OBJECT was not always set and sometimes set wrong in the early
    # Jupiter observations.  Similarly with Mercury.  This is designed
    # to be called on a per-planet pipeline basis so that it doesn't
    # take resources in other pipelines
    if planet is None:
        return ccd_in
    if bmp_meta is None:
        bmp_meta = {}
    if isinstance(ccd_in, list):
        result = [planet_to_object(ccd, bmp_meta=bmp_meta,
                                   planet=planet, **kwargs)
                  for ccd in ccd_in]
        if None in result:
            bmp_meta.clear()
            return None
        return result
    a = angle_to_major_body(ccd_in, planet.lower())
    if a > 1*u.deg:
        # This should be accurate enough for our pointing
        log.warning(f'Expected planet {planet} not found within 1 deg of {ccd_in.sky_coord}')
        bmp_meta.clear()
        return None
    ccd = ccd_in.copy()
    ccd.meta['object'] = planet.capitalize()
    return ccd

def obj_surface_bright(ccd_in, bmp_meta=None, **kwargs):
    """Calculates Jupiter surface brightness using a box 0.5 Rj on a side"""
    if bmp_meta is None:
        bmp_meta = {}
    ccd = ccd_in.copy()
    center = ccd.obj_center*u.pixel
    pix_per_Rj = pixel_per_Rj(ccd)
    side = 0.5 * u.R_jup
    b = np.round(center[0] - side/2 * pix_per_Rj).astype(int)
    t = np.round(center[0] + side/2 * pix_per_Rj).astype(int)
    l = np.round(center[1] - side/2 * pix_per_Rj).astype(int)
    r = np.round(center[1] + side/2 * pix_per_Rj).astype(int)
    subim = ccd[b.value:t.value, l.value:r.value]
    jup_sum, jup_area = sum_ccddata(subim)
    sb = jup_sum / jup_area
    if ccd.uncertainty.uncertainty_type == 'std':
        dev = np.nansum(subim.uncertainty.array**2)
    else:
        dev = np.nansum(subim.uncertainty.array)
    sb_err = dev**-0.5*ccd.unit*u.pixel**2 / jup_area
    osb_dict = {'obj_surf_bright': sb,
                'obj_surf_bright_err': sb_err}
    ccd = dict_to_ccd_meta(ccd, osb_dict)
    bmp_meta.update(osb_dict)
    return ccd

class table_stacker_obj:
    """Combines QTables or lists of dict into summary_table property.

    In the case of QTable, handles Time column so as to dance around
    issue that time.location doesn't vstack properly.  Just copies off
    first location, sets location to None and copies location back
    again.  Does simple but not exhaustive sanity check that location
    is consistent

    Parameters
    ----------
    colname : str
        Name of column containing Time object.  This could be expanded
        and generalized if more Time columns or other Mixin columns
        cause problems

    """

    def __init__(self, colname='tavg'):
        self.summary_table = None
        self.colname = colname
        self._location = None

    # Hack for combining Time mixin columns of different lengths
    @property
    def location(self):
        return self._location

    @location.setter
    def location(self, value):
        # Assure we have only one location
        if self._location is None:
            self._location = value
        assert value == self._location
    
    def clear_loc(self, t):
        """Clear location in input QTable and store it in property.  Quietly
        ignore cases where input is not QTable

        """
        if not isinstance(t, QTable) or len(t) == 0:
            return
        if t[0][self.colname].location is None:
            return
        self.location = t[0][self.colname].location.copy()
        t[self.colname].location = None

    def set_loc(self, t):
        """Set location in input QTable from stored property"""
        if not isinstance(t, QTable) or len(t) == 0:
            return
        t[self.colname].location = self.location

    def append(self, t):
        """Append input t to summary_table, respecting type"""
        if t is None or len(t) == 0:
            return
        self.clear_loc(t)
        self.clear_loc(self.summary_table)
        if self.summary_table is None:
            self.summary_table = t
        elif isinstance(t, list):
            self.summary_table.append(t)
        elif isinstance(t, QTable):
            self.summary_table = vstack([self.summary_table, t])
        else:
            raise ValueError(f'Unknown cached_csv return type {t}')
        self.set_loc(self.summary_table)

def parallel_cached_csvs(dirs,
                         code=None,
                         collector=None,
                         files_per_process=1, # set to 2--3 for on_off_pipeline
                         max_num_processes=MAX_NUM_PROCESSES,
                         read_csvs=False,
                         **cached_csv_args):

    wwk = WorkerWithKwargs(cached_csv,
                           code=code,
                           read_csvs=read_csvs,
                           **cached_csv_args)

    running_nprocesses = 0
    collection_list = []
    table_stacker = table_stacker_obj()
    for directory in dirs:

        if read_csvs:
            # Try to read cache
            t = cached_csv(directory,
                           code=None,
                           read_csvs=read_csvs,
                           **cached_csv_args)
            if t is not None:
                # Cache was successfully read
                table_stacker.append(t)
                continue

        # If we made it here, we need to run our code.  But first
        # generate a collection so we can look inside to see how many
        # files are there to maximize parallelization
        collection = collector(directory, **cached_csv_args)
        
        # Reserve a process for empty directories
        nfiles = np.max((1, len(collection.files)))

        nprocesses = int(nfiles / files_per_process)
        if nprocesses >= max_num_processes:
            # Directory has lots of files.  Let cormultipipe regulate
            # the number of processes
            t = cached_csv(directory,
                           code=code,
                           read_csvs=read_csvs,
                           **cached_csv_args)
            table_stacker.append(t)
            continue
            
        if nprocesses + running_nprocesses < max_num_processes:
            # Keep building our list across multiple directories
            running_nprocesses += nprocesses
            collection_list.append(collection)
            continue

        if nprocesses + running_nprocesses == max_num_processes:
            # We hit is just right
            collection_list.append(collection)
            to_process = collection_list
            running_nprocesses = 0
            collection_list = []
        else:
            # We went over a bit.  Process what we have & start at the
            # current directory
            to_process = collection_list
            running_nprocesses = nprocesses
            collection_list = [collection]

        with NestablePool(processes=max_num_processes) as p:
            tlist = p.map(wwk.worker, to_process)

        for t in tlist:
            table_stacker.append(t)
            
    if len(collection_list) > 0:
        # Handle last (set of) directories
        to_process = collection_list
        with NestablePool(processes=max_num_processes) as p:
            tlist = p.map(wwk.worker, to_process)
        for t in tlist:
            table_stacker.append(t)

    return table_stacker.summary_table

######### Argparse mixin
class CorArgparseMixin:
    # This modifies BMPArgparseMixin outname_append
    outname_append = OUTNAME_APPEND

    def add_log_level(self,
                      option='log_level',
                      default='DEBUG',
                      help=None,
                      **kwargs):
        if help is None:
            help = f'astropy.log level (default: {default})'
        self.parser.add_argument('--' + option, 
                            default=default, help=help, **kwargs)

    def add_raw_data_root(self, 
                          option='raw_data_root',
                          default=RAW_DATA_ROOT,
                          help=None,
                          **kwargs):
        if help is None:
            help = f'raw data root (default: {default})'
        self.parser.add_argument('--' + option, 
                            default=default, help=help, **kwargs)
        
    def add_reduced_root(self,
                         option='reduced_root',
                         default=REDUCED_ROOT,
                         help=None,
                         **kwargs):
        if help is None:
            help = f'root of reduced file directory tree (default: {default})'
        self.parser.add_argument('--' + option, 
                                 default=default, help=help, **kwargs)

    def add_base(self, 
                 option='base',
                 default=None,
                 help=None,
                 **kwargs):
        if help is None:
            help = f'base of output filename (default: {default})'
        self.parser.add_argument('--' + option, 
                            default=default, help=help, **kwargs)

    def add_start(self,
                  option='start',
                  default=None,
                  help=None,
                  **kwargs):
        if help is None:
            help = 'start directory/date (default: earliest)'
        self.parser.add_argument('--' + option, 
                                 default=default, help=help, **kwargs)

    def add_stop(self, 
                 option='stop',
                 default=None,
                 help=None,
                 **kwargs):
        if help is None:
            help = 'stop directory/date (default: latest)'
        self.parser.add_argument('--' + option, 
                                 default=default, help=help, **kwargs)

    def add_read_csvs(self, 
                      option='read_csvs',
                      default=False,
                      help=None,
                      **kwargs):
        if help is None:
            help = (f'Read CSV files')
        self.parser.add_argument('--' + option,
                                 action=argparse.BooleanOptionalAction,
                                 default=default,
                                 help=help, **kwargs)

    def add_write_csvs(self, 
                       option='write_csvs',
                       default=False,
                       help=None,
                       **kwargs):
        if help is None:
            help = (f'Write CSV files')
        self.parser.add_argument('--' + option,
                                 action=argparse.BooleanOptionalAction,
                                 default=default,
                                 help=help, **kwargs)

    def calc_highest_product(self, 
                             option='calc_highest_product',
                             default=True,
                             help=None,
                             **kwargs):
        if help is None:
            help = (f'Calculate highest product (e.g. all columns in QTable)')
        self.parser.add_argument('--' + option,
                                 action=argparse.BooleanOptionalAction,
                                 default=default,
                                 help=help, **kwargs)

    def add_show(self, 
                 option='show',
                 default=False,
                 help=None,
                 **kwargs):
        if help is None:
            help = (f'Show plots interactively')
        self.parser.add_argument('--' + option,
                                 action=argparse.BooleanOptionalAction,
                                 default=default,
                                 help=help, **kwargs)

class CorArgparseHandler(CorArgparseMixin, CCDArgparseMixin,
                         BMPArgparseMixin, ArgparseHandler):
    """Adds basic argparse options relevant to cormultipipe system""" 

    # This modifies CCDArgparseMixin outname_ext
    outname_ext = OUTNAME_EXT

    def add_all(self):
        """Add options used in cmd"""
        self.add_log_level()
        self.add_create_outdir(default=True) 
        self.add_outname_append()
        self.add_outname_ext()
        self.add_fits_fixed_ignore(default=True)
        self.add_num_processes(default=MAX_NUM_PROCESSES)
        self.add_mem_frac(default=MAX_MEM_FRAC)
        super().add_all()

    def cmd(self, args):
        # This is the base of all cormultipipe cmd super() calls
        log.setLevel(args.log_level)

if __name__ == '__main__':
    from calibration import Calibration
    ccd = CorData.read('/data/IoIO/raw/2018-05-08/SII_on-band_022.fits')
    c = Calibration(reduce=True)
    #ccd = cor_process(ccd, calibration=c, auto=True)
    ccd.obj_center
    ccd = crop_ccd(ccd, crop_from_desired_center=(150, 300))
