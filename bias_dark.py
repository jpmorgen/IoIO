#!/usr/bin/python3

"""
Module to reduce biases and darks in a directory
"""
    
import inspect
import os
import time
import datetime
import glob
import psutil

# For NoDaemonPool
import multiprocessing
# We must import this explicitly, it is not imported by the top-level
# multiprocessing module.
import multiprocessing.pool
from multiprocessing import Process, Pool

import numpy as np
import numpy.ma as ma
from scipy import signal, stats, interpolate

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import matplotlib.transforms as transforms
import pandas as pd

from astropy import log
from astropy import units as u
from astropy.io import fits
from astropy.nddata import CCDData, StdDevUncertainty
from astropy.table import QTable
from astropy.time import Time, TimeDelta
from astropy.stats import mad_std, biweight_location
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize

from photutils import Background2D, MedianBackground

import ccdproc as ccdp

# From pip3 install https://github.com/kinderhead/mp_tools/releases/download/v1.1/mp_tools-1.1.0-py3-none-any.whl
from mp_tools.process_limiter import p_limit
#from mp_tools_devel.process_limiter import p_limit
from IoIO import CorObsData

# Record in global variables Starlight Xpress Trius SX694 CCD
# characteristics.  Note that CCD was purchased in 2017 and is NOT the
# "Pro" version, which has a different gain but otherwise similar
# characteristics

sx694_camera_description = 'Starlight Xpress Trius SX694 mono, 2017 model version'

# naxis1 = fastest changing axis in FITS primary image = X in
# Cartesian thought
# naxis1 = next to fastest changing axis in FITS primary image = Y in
# Cartesian thought
sx694_naxis1 = 2750
sx694_naxis2 = 2200

# 16-bit A/D converter, stored in SATLEVEL keyword
sx694_satlevel = 2**16-1
sx694_satlevel_comment = 'Saturation level (ADU)'

# Gain measured in /data/io/IoIO/observing/Exposure_Time_Calcs.xlsx.
# Value agrees well with Trius SX-694 advertised value (note, newer
# "PRO" model has a different gain value).  Stored in GAIN keyword
sx694_gain = 0.3
sx694_gain_comment = 'Measured gain (electron/ADU)'

# Sample readnoise measured as per ioio.notebk
# Tue Jul 10 12:13:33 2018 MCT jpmorgen@byted 

# Readnoies is measured regularly as part of master bias creation and
# stored in the RDNOISE keyword.  This is used as a sanity check.
sx694_example_readnoise = 15.475665 * sx694_gain
sx694_example_readnoise_comment = '2018-07-10 readnoise (electron)'
sx694_readnoise_tolerance = 0.5 # Units of electrons

# Measurement in /data/io/IoIO/observing/Exposure_Time_Calcs.xlsx of
# when camera becomes non-linear.  Stored in NONLIN keyword.  Raw
# value of 42k was recorded with a typical overscan value.  Helps to
# remember ~40k is absolute max raw ADU to shoot for.  This is
# suspiciously close to the full-well depth in electrons of 17,000
# (web) - 18,000 (user's manual) provided by the manufacturer
# --> could do a better job of measuring the precise high end of this,
# since it could be as high as 50k
sx694_nonlin = 42000 - 1811
sx694_nonlin_comment = 'Measured nonlinearity point (ADU)'

# Exposure times at or below this value are counted on the camera and
# not in MaxIm.  There is a bug in the SX694 MaxIm driver seems to
# consistently add about sx694_exposure_correct seconds to the
# exposure time before asking the camera to read the CCD out.
# Measured in /data/io/IoIO/observing/Exposure_Time_Calcs.xlsx
# --> NEEDS TO BE VERIFIED WITH PHOTOMETRY FROM 2019 and 2020
# Corrected as part of local version of ccd_process
sx694_max_accurate_exposure = 0.7 # s
sx694_exposure_correct = 1.7 # s

# The SX694 and similar interline transfer CCDs have such low dark
# current that it is not really practical to map out the dark current
# in every pixel (a huge number of darks would be needed to beat down
# the readnoise).  Rather, cut everything below this threshold, which
# reads in units of readnoise (e.g., value in electrons is
# 3 * sx694_example_readnoise).  The remaining few pixels generate
# more dark current, so include only them in the dark images
sx694_dark_mask_threshold = 3



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
max_ccddata_size = (sx694_naxis1 * sx694_naxis2
                    * (2 * 64 + 8)) / 8
griddata_expansion_factor = 100

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



# Adapted from various source on the web
# https://stackoverflow.com/questions/6974695/python-process-pool-non-daemonic
class NoDaemonProcess(multiprocessing.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class NoDaemonPool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess

class PoolWorkerKwargs():
    """Class to hold static kwargs for use with Pool.map or Pool.starmap.

    """
    def __init__(self,
                 function,
                 **kwargs):
        self.function = function
        self.kwargs = kwargs
    def worker(self, *args):
        return self.function(*args, **self.kwargs)

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

def ccd_metadata(hdr_in,
                 camera_description=sx694_camera_description,
                 gain=sx694_gain,
                 gain_comment=sx694_gain_comment,
                 satlevel=sx694_satlevel,
                 satlevel_comment=sx694_satlevel_comment,
                 nonlin=sx694_nonlin,
                 nonlin_comment=sx694_nonlin_comment,
                 readnoise=sx694_example_readnoise,
                 readnoise_comment=sx694_example_readnoise_comment,
                 *args, **kwargs):
    """Record [SX694] CCD metadata in FITS header object"""
    if hdr_in.get('camera') is not None:
        # We have been here before, so exit quietly
        return hdr_in
    hdr = hdr_in.copy()
    # Clean up double exposure time reference to avoid confusion
    if hdr.get('exposure') is not None:
        del hdr['EXPOSURE']
    hdr.insert('INSTRUME',
                    ('CAMERA', camera_description),
                    after=True)
    hdr['GAIN'] = (gain, gain_comment)
    # This gets used in ccdp.cosmicray_lacosmic
    hdr['SATLEVEL'] = (satlevel, satlevel_comment)
    # This is where the CCD starts to become non-linear and is
    # used for things like rejecting flats recorded when
    # conditions were too bright
    hdr['NONLIN'] = (nonlin, nonlin_comment)
    hdr['RDNOISE'] = (readnoise, readnoise_comment)
    return hdr

def ccd_exp_correct(hdr_in,
                    max_accurate_exposure=sx694_max_accurate_exposure,
                    exposure_correct=sx694_exposure_correct,
                    *args, **kwargs):
    """Correct exposure time for [SX694] CCD driver problem
     --> REFINE THIS ESTIMATE BASED ON MORE MEASUREMENTS
    """
    if hdr_in.get('OEXPTIME') is not None:
        # We have been here before, so exit quietly
        return hdr_in
    hdr = hdr_in.copy()
    exptime = hdr['EXPTIME']
    if exptime > max_accurate_exposure:
        hdr.insert('EXPTIME', 
                         ('OEXPTIME', exptime,
                          'original exposure time (seconds)'),
                         after=True)
        exptime += exposure_correct
        hdr['EXPTIME'] = (exptime,
                                'corrected exposure time (seconds)')
        hdr.insert('OEXPTIME', 
                         ('HIERARCH EXPTIME_CORRECTION',
                          exposure_correct, '(seconds)'),
                         after=True)
        #add_history(hdr,
        #            'Corrected exposure time for SX694 MaxIm driver bug')
    return hdr

def ccddata_read(fname_or_ccd,
                 raw_unit=u.adu,
                 *args, **kwargs):
    """Convenience function to read a FITS file into a CCDData object.

    Catches the case where the raw FITS file does not have a BUNIT
    keyword, which otherwise causes CCDData.read() to crash.  In this
    case, ccddata_read assigns ccddata units of ``raw_unit``.  Also
    ads comment to BUNIT "physical units of the array values," which
    is curiously omitted in the astropy fits writing system.  Comment
    is from official FITS documentation
    https://heasarc.gsfc.nasa.gov/docs/fcg/standard_dict.html where
    BUNIT is in the same family as BZERO and BSCALE

    Parameters
    ----------
    fname_or_ccd : str or `~astropy.nddata.CCDData`
        If str, assumed to be a filename, which is read into a
        CCDData.  If ccddata, return a copy of the CCDData with BUNIT
        keyword possibly added

    raw_unit : str or `astropy.units.core.UnitBase`
        Physical unit of pixel in case none is specified 
        Default is `astropy.units.adu`

    *args and **kwargs passed to CCDData.read()

    Returns
    -------
    ccd : `~astropy.nddata.CCDData`
        CCDData with units set to raw_unit if none specified in FITS file

    """
    if isinstance(fname_or_ccd, str):
        try:
            ccd = CCDData.read(fname_or_ccd, *args, **kwargs)
        except Exception as e: 
            ccd = CCDData.read(fname_or_ccd, *args,
                               unit=raw_unit, **kwargs)
    else:
        ccd = fname_or_ccd.copy()
    assert isinstance(ccd, CCDData)
    if ccd.unit is None:
        log.warning('ccddata_read: CCDData.read read a file and did not assign any units to it.  Not sure why.  Setting units to' + raw_unit.to_string())
        ccd.unit = raw_unit
    # Setting ccd.unit to something does not set the BUNIT keyword
    # until file write.  So to write the comment before we write the
    # file, we need to set BUNIT ourselves.  If ccd,unit changes
    # during our calculations (e.g. gain correction), the BUNIT
    # keyword is changed but the comment is not.
    ccd.meta['BUNIT'] = (ccd.unit.to_string(),
                         'physical units of the array values')
    return ccd

def full_frame(im,
               naxis1=sx694_naxis1,
               naxis2=sx694_naxis2,
               **kwargs):
    """cor_pipeline pre-processing routine to select full-frame images
    """
    s = im.shape
    # Note Pythonic C index ordering
    if s != (naxis2, naxis1):
        return (None, {})
    return (im, {})

#def full_frame(im,
#               naxis1=sx694_naxis1,
#               naxis2=sx694_naxis2):
#    """Returns true if image is a full frame, as defined by naxis1 and naxis2.
#
#    Helps spot binned and guider images"""
#    
#    s = im.shape
#    # Note Pythonic C index ordering
#    if s != (naxis2, naxis1):
#        return False
#    return True    

def light_image(im, light_tolerance=3, **kwargs):
    """cor_pipeline pre-processing routine to reject light-contaminated bias & dark images
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
        return (None, {})
    return (im, {})

def check_oscan(ccd, pipe_meta, **kwargs):
    hdr = ccd.meta
    osbias = hdr.get('osbias')
    biasfile = hdr.get('biasfile')
    if osbias is None or biasfile is None:
        return (ccd, {})
    if osbias != biasfile:
        multi_logging('warning', pipe_meta,
                      'OSBIAS and BIASFILE are not the same')
    else:
        del hdr['OSBIAS']
        hdr['OVERSCAN_MASTER_BIAS'] = 'BIASFILE'
    return (ccd, {})

def mask_above(ccd_in, key, margin=0.1):
    ccd = ccd_in.copy()
    masklevel = ccd.meta[key]
    # Saturation level is subject to overscan subtraction and
    # multiplication by gain, so don't do strict = testing, but give
    # ourselves a little margin.
    mask = ccd.data >= masklevel - margin
    n_masked = np.count_nonzero(mask)
    if n_masked > 0:
        log.info('Masking {} pixels above {}'.format(n_masked, key))
    if len(key) > 6:
        h = 'HIERARCH '
    else:
        h = ''
    ccd.meta[h + 'N_' + key] \
        = (n_masked, 'number of pixels > {}'.format(key))
    if n_masked > 0:
        # Avoid creating a mask of all Falses
        ccd.mask = ccd.mask or mask
    return ccd, n_masked
    
#def mask_saturated(ccd):
#

    """Record the number of saturated pixels in image metadata and log a warning if > 0"""
#    satlevel = ccd.meta['SATLEVEL']
#    # Saturation level is subject to overscan subtraction and multiplication by gain, so don't do strict = testing, but give ourselves a little margin.
#    mask = ccd.data >= satlevel - 0.1
#    n_saturated = np.count_nonzero(mask)
#    if n_saturated > 0:
#        log.warning('There are {} saturated pixels'.format(n_saturated))
#    ccd.meta['N_SAT'] \
#        = (n_saturated, 'number of saturated pixels')
#    return (n_saturated, mask)
#
#def mask_nonlin(ccd):
#    """Record the number of pixels > nonlinearity point in image metadata and log a warning if > 0"""
#    nonlin = ccd.meta['NONLIN']
#    mask = ccd.data > nonlin
#    n_nonlin = np.count_nonzero(mask)
#    if n_nonlin > 0:
#        log.warning('There are {} pixels > nonlinearity point'.format(n_nonlin))
#    ccd.meta['N_NONLIN'] \
#        = (n_nonlin, 'number of pixels > nonlinearity point')
#    return (n_nonlin, mask)

# --> Could make something general that does all of the code below to
# --> the "If we made it here".  It would have a file_grouper or
# --> something like that.  I am using fdict_list, so maybe I should
# --> just call it an fdict_creator.  There would be
# --> bias_dark_fdict_creator and flat_fdict_creator

# Maybe it should be subdir searcher or something like that

#def subdir_processor(directory=None,
#                     subdirs=None,
#                     sp_output_creator=None,
#                     debug=False,
#                     **kwargs):
#    """
#    Runs output_creator in `directory` and on one level of `subdirs`.
#    Collects results in a list
#    """
#    if directory is None:
#        directory = os.getcwd()
#    if subdirs is None:
#        subdirs = []
#    if subdirs is True:
#        subdirs = [s for s in os.listdir(directory)
#                   if os.path.isdir(os.path.join(directory, s))]
#    out_list = []
#    # Prepare to call ourselves recursively to build up a list of
#    # products in the provided directory and optional subdirectories
#    if not os.path.isdir(directory):
#        # This is the end of our recursive line
#        return out_list
#    for sd in subdirs:
#        subdir = os.path.join(directory, sd)
#        sub_out_list = subdir_processor(subdir,
#                                        output_creator=output_creator,
#                                        debug=debug,
#                                        **kwargs)
#        for sl in sub_out_list:
#            out_list.append(sl)
#    # After processing our subdirs, process 'directory.'
#    root_out_list = output_creator(directory, debug=debug, **kwargs)
#    for sl in root_out_list:
#        out_list.append(sl)
#    return out_list
#
#def glob_list_processor(directory
#                       glob_include=flat_glob):
#    for gi in glob_include:


def fname_by_imagetyp_ccdt_exp(directory=None,
                               collection=None,
                               subdirs=None,
                               glob_include=None,
                               imagetyp=None,
                               dccdt_tolerance=dccdt_tolerance,
                               debug=False):
    """For a given IMAGETYP, returns a list of dictionaries with keys T (CCD-TEMP), EXPTIME, and fnames"""
    assert imagetyp is not None
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
            sub_fdict_list = fname_by_imagetyp_ccdt_exp \
                (subdir,
                 imagetyp=imagetyp,
                 glob_include=glob_include,
                 dccdt_tolerance=dccdt_tolerance,
                 debug=debug)
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
            gi_fdict_list = fname_by_imagetyp_ccdt_exp \
                (collection=collection,
                 imagetyp=imagetyp,
                 dccdt_tolerance=dccdt_tolerance,
                 debug=debug)
            for gi in gi_fdict_list:
                fdict_list.append(gi)
        # Here is the end of our recursive line if directory and
        # optional subdirs were specified
        return fdict_list
    if collection.summary is None:
        # We were probably called on a glob_include that yielded no results
        return fdict_list

    # If we made it here, we have a collection, possibly from calling
    # ourselves recursively
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

def jd_meta(ccd, pipe_meta, **kwargs):
    tm = Time(ccd.meta['DATE-OBS'], format='fits')
    return (ccd, {'jd': tm.jd})

def bias_stats(ccd, pipe_meta, gain=sx694_gain, **kwargs):
    """cor_pipeline post-processing routine for bias_combine
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
    return (ccd, {'stats': stats})

#def bias_dataframe(fname, gain):
#    """Worker routine to enable parallel processing of time-consuming
#    matrix calculation"""
#
#    ccd = ccddata_read(fname, add_metadata=True)
#    if not full_frame(ccd):
#        log.debug('bias wrong shape: ' + fname)
#        return {'good': False}
#    if light_image(ccd):
#        log.debug('bias recorded during light conditions: ' +
#                  fname)
#        return {'good': False}
#    im = ccd.data
#    # Create uncertainty image
#    diffs2 = (im[1:] - im[0:-1])**2
#    rdnoise = np.sqrt(biweight_location(diffs2))
#    #uncertainty = np.multiply(rdnoise, np.ones(im.shape))
#    #ccd.uncertainty = StdDevUncertainty(uncertainty)
#    # Prepare to create a pandas data frame to track relevant
#    # quantities
#    tm = Time(ccd.meta['DATE-OBS'], format='fits')
#    ccdt = ccd.meta['CCD-TEMP']
#    tt = tm.tt.datetime
#    dataframe = {'time': tt,
#                 'ccdt': ccdt,
#                 'median': np.median(im),
#                 'mean': np.mean(im),
#                 'std': np.std(im)*gain,
#                 'rdnoise': rdnoise*gain,
#                 'min': np.min(im),  
#                 'max': np.max(im)}
#    return {'good': True,
#            'fname': fname,
#            'dataframe': dataframe,
#            'jd': tm.jd}

#def bias_subtract_one_median(fname, med, **kwargs):
#    return ccd_process(fname, oscan=med, **kwargs)


def bias_combine_one_fdict(fdict,
                           outdir=calibration_root,
                           calibration_scratch=calibration_scratch,
                           keep_intermediate=False,
                           show=False,
                           min_num_biases=min_num_biases,
                           dccdt_tolerance=dccdt_tolerance,
                           camera_description=sx694_camera_description,
                           gain=sx694_gain,
                           satlevel=sx694_satlevel,
                           readnoise=sx694_example_readnoise,
                           readnoise_tolerance=sx694_readnoise_tolerance,
                           gain_correct=False,
                           num_processes=max_num_processes,
                           mem_frac=max_mem_frac,
                           ccddata_size=max_ccddata_size):

    """Worker that allows the parallelization of calibrations taken at one
    temperature, exposure time, filter, etc.


    gain_correct : Boolean
        Effects unit of stored images.  True: units of electron.
        False: unit of ADU.  Default: False
    """

    num_files = len(fdict['fnames'])
    mean_ccdt = fdict['CCDT']
    directory = fdict['directory']
    if num_files < min_num_biases:
        log.debug(f"Not enough good biases found at CCDT = {mean_ccdt} C in {directory}")
        return False

    # Make a scratch directory that is the date of the first file.
    # Not as fancy as the biases, but, hey, it is a scratch directory
    tmp = ccddata_read(fdict['fnames'][0])
    tm = tmp.meta['DATE-OBS']
    this_dateb1, _ = tm.split('T')
    sdir = os.path.join(calibration_scratch, this_dateb1)

    #mem = psutil.virtual_memory()
    #num_files_can_fit = \
    #    int(min(num_files,
    #            mem.available*mem_frac/ccddata_size))
    #num_can_process = min(num_processes, num_files_can_fit)
    #print('bias_combine_one_fdict: num_processes = {}, mem_frac = {}, num_files= {}, num_files_can_fit = {}, num_can_process = {}'.format(num_processes, mem_frac, num_files, num_files_can_fit, num_can_process))

    # Use the cor_pipeline to subtract the median from each bias and
    # create a dict of stats for a pandas dataframe
    pout = cor_pipeline(fdict['fnames'],
                        num_processes=num_processes,
                        mem_frac=mem_frac,
                        process_size=ccddata_size,
                        outdir=sdir,
                        create_outdir=True,
                        overwrite=True,
                        pre_process_list=[full_frame, light_image],
                        post_process_list=[bias_stats, jd_meta],
                        oscan=True)
    pout = [p for p in pout if p[0] is not None]
    if len(pout) < min_num_biases:
        log.debug(f"Not enough good biases {len(pout)} found at CCDT = {mean_ccdt} C in {directory}")
        return False

    out_fnames, pipe_meta = zip(*pout)
    stats = [m['stats'] for m in pipe_meta]
    jds = [m['jd'] for m in pipe_meta]

    #if num_can_process == 1:
    #    ccd_dfdlist = [bias_dataframe(ccd, gain=gain, **kwargs)
    #try:
    #    with Pool(processes=num_can_process) as p:
    #        dfdlist = p.starmap(bias_dataframe,
    #                            zip(fdict['fnames'], gains))
    #except Exception as e:
    #    log.debug('Single-process mode: ' + str(e))
    #    dfdlist = [bias_dataframe(d, gain) for d in fdict['fnames']]
    #
    ## Unpack        
    #good_fnames = [dfd['fname'] for dfd in dfdlist if dfd['good']]
    #stats = [dfd['dataframe'] for dfd in dfdlist if dfd['good']]
    #jds = [dfd['jd'] for dfd in dfdlist if dfd['good']]
    #
    #if len(stats) < min_num_biases:
    #    log.debug(f"Not enough good biases {len(stats)} found at CCDT = {mean_ccdt} C in {directory}")
    #    return False

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
    out_fname = outbase + '_combined_bias.fits'
    plt.savefig((outbase + '_vs_time.png'), transparent=True)
    if show:
        plt.show()
    plt.close()

    # Do a sanity check of readnoise
    av_rdnoise = np.mean(df['rdnoise'])            
    if (np.abs(av_rdnoise/sx694_example_readnoise - 1)
        > readnoise_tolerance):
        log.warning('High readnoise {}, skipping {}'.format(av_rdnoise, out_fname))
        return False

    ## "Overscan subtract."  Go through each image and subtract the
    ## median, since that value wanders as a function of ambient
    ## (not CCD) temperature.  This leaves just the bias pattern.
    ## To use ccd.subtract, unit must match type of array.  Note
    ## that we need to save our images to disk to avoid
    ## overwhelming memory when we have lots of biases.  Also note that
    ## we can't use the ccd_process_multi machinery because we have an
    ## argument, medians, which changes with each bias. 
    #
    ## Make a date subdirectory in our calibration scratch dir
    #sdir = os.path.join(calibration_scratch, this_dateb)
    #nfnames = len(good_fnames)
    #mem = psutil.virtual_memory()
    #max_mem = mem.available*mem_frac
    #max_nframes = int(max_mem / ccddata_size)
    #our_num_processes = min(nfnames, num_processes, max_nframes)
    #pwk = PoolWorkerKwargs(bias_subtract_one_median,
    #                       outdir=sdir,
    #                       create_outdir=True,
    #                       overwrite=True,
    #                       return_outname=True)
    #with NoDaemonPool(processes=num_processes) as p:
    #    out_fnames = p.starmap(pwk.worker, zip(good_fnames, medians))
        
    #os.makedirs(sdir, exist_ok=True)
    #out_fnames = []
    #for fname, m in zip(good_fnames, medians):
    #    ccd = ccddata_read(fname, add_metadata=True)
    #    ccd = ccd.subtract(m*u.adu, handle_meta='first_found')
    #    # Get our filename only, hack off extension, add _os.fits
    #    os_fname = os.path.split(fname)[1]
    #    os_fname = os.path.splitext(os_fname)[0] + '_os.fits'
    #    os_fname = os.path.join(sdir, os_fname)
    #    ccd.write(os_fname, overwrite=True)
    #    out_fnames.append(os_fname)

    ### Use ccdproc Combiner object iteratively as per example to
    ### mask out bad pixels
    ##combiner = ccdp.Combiner(out_fnames)
    ##old_n_masked = -1  # dummy value to make loop execute at least once
    ##new_n_masked = 0
    ##while (new_n_masked > old_n_masked):
    ##    #combiner.sigma_clipping(low_thresh=2, high_thresh=5,
    ##    #                        func=np.ma.median)
    ##    # Default to 1 std and np.ma.mean for func
    ##    combiner.sigma_clipping()
    ##    old_n_masked = new_n_masked
    ##    new_n_masked = combiner.data_arr.mask.sum()
    ##    print(old_n_masked, new_n_masked)
    ###Just one iteration for testing
    ###combiner.sigma_clipping(low_thresh=2, high_thresh=5, func=np.ma.median)
    ##combiner.sigma_clipping(low_thresh=1, high_thresh=1, func=np.ma.mean)
    ## I prefer average to get sub-ADU accuracy.  Median outputs in
    ## double-precision anyway, so it doesn't save space
    #combined_average = combiner.average_combine()

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
    im.meta = ccd_metadata(im.meta)
    if gain_correct:
        im = ccdp.gain_correct(im, gain*u.electron/u.adu)
        im_gain = 1
    else:
        im_gain = gain
    im, _ = mask_above(im, 'SATLEVEL')
    im, _ = mask_above(im, 'NONLIN')
        
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
    for i, f in enumerate(fdict['fnames']):
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
    plt.savefig((outbase + '_combined_bias.png'), transparent=True)
    if show:
        plt.show()
    plt.close()
    if not keep_intermediate:
        for f in out_fnames:
            try:
                os.remove(f)
            except Exception as e:
                # We do not expect this, since we created these with
                # our local process
                log.debug(f'Unexpected!  Remove {f} failed: ' + str(e))
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
                


#def process_runner(plist, num_processes):
#    nps = len(plist)
#    for i in range(min(nps, num_processes)):
#        plist.start()
#    for i in range(nps):
        

def bias_combine(directory=None,
                 collection=None,
                 subdirs=calibration_subdirs,
                 glob_include=bias_glob,
                 dccdt_tolerance=dccdt_tolerance,
                 num_processes=max_num_processes,
                 mem_frac=max_mem_frac,
                 num_calibration_files=num_calibration_files,
                 max_ccddata_size=max_ccddata_size,
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
        fname_by_imagetyp_ccdt_exp(directory=directory,
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

    one_fdict_size = num_calibration_files * max_ccddata_size
    our_num_processes = num_can_process(nfdicts,
                                        num_processes=num_processes,
                                        mem_frac=mem_frac,
                                        process_size=one_fdict_size)

    num_subprocesses = int(num_processes / our_num_processes)
    subprocess_mem_frac = mem_frac / our_num_processes
    log.debug('bias_combine: {} num_processes = {}, mem_frac = {}, our_num_processes = {}, num_subprocesses = {}, subprocess_mem_frac = {}'.format(directory, num_processes, mem_frac, our_num_processes, num_subprocesses, subprocess_mem_frac))

    pwk = PoolWorkerKwargs(bias_combine_one_fdict,
                           num_processes=num_subprocesses,
                           mem_frac=subprocess_mem_frac,
                           **kwargs)
    if nfdicts == 1:
        for fdict in fdict_list:
                pwk.worker(fdict)
    else:
        with NoDaemonPool(processes=our_num_processes) as p:
            p.map(pwk.worker, fdict_list)


    ## Combining files is the slow part, so we want the maximum of
    ## processes doing that in parallel
    #our_num_processes = min(nfdicts, num_processes)
    #log.debug(f'bias_combine: {directory}, nfdicts = {nfdicts}, our_num_processes = {our_num_processes}')
    #elif nfdicts == 1 or our_num_processes == 1:
    #    for fdict in fdict_list:
    #        bias_combine_one_fdict(fdict,
    #                               num_processes=num_processes,
    #                               mem_frac=mem_frac,
    #                               **kwargs)
    #else:
    #    # Number of sub-processes in each process we will spawn
    #    num_subprocesses = int(num_processes / our_num_processes)
    #    # Similarly, the memory fraction for each process we will spawn
    #    subprocess_mem_frac = mem_frac / our_num_processes
    #    log.debug('bias_combine: {} num_processes = {}, mem_frac = {}, our_num_processes = {}, num_subprocesses = {}, subprocess_mem_frac = {}'.format(directory, num_processes, mem_frac, our_num_processes, num_subprocesses, subprocess_mem_frac))
    #    # Initiate all our processes
    #    plist = [Process(target=bias_combine_one_fdict,
    #                     args=(fdict,),
    #                     kwargs={"num_processes": num_subprocesses,
    #                             "mem_frac": subprocess_mem_frac,
    #                             **kwargs},
    #                     daemon=False) # Let subprocesses create more children
    #             for fdict in fdict_list]
    #    p_limit(plist, our_num_processes)
            

def hist_of_im(im, binsize=1, show=False):
    """Returns a tuple of the histogram of image and index into *centers* of
bins."""
    # Code from west_aux.py, maskgen.
    # Histogram bin size should be related to readnoise
    hrange = (im.data.min(), im.data.max())
    nbins = int((hrange[1] - hrange[0]) / binsize)
    hist, edges = np.histogram(im, bins=nbins,
                               range=hrange, density=False)
    # Convert edges of histogram bins to centers
    centers = (edges[0:-1] + edges[1:])/2
    if show:
        plt.plot(centers, hist)
        plt.show()
        plt.close()
    return (hist, centers)

def overscan_estimate(ccd_in, meta=None, master_bias=None,
                      binsize=None, min_width=1, max_width=8, box_size=100,
                      min_hist_val=10,
                      show=False, *args, **kwargs):
    """Estimate overscan in ADU in the absense of a formal overscan region

    For biases, returns in the median of the image.  For all others,
    uses the minimum of: (1) the first peak in the histogram of the
    image or (2) the minimum of the median of four boxes at the
    corners of the image.

    Works best if bias shape (particularly bias ramp) is subtracted
    first.  Will subtract bias if bias is supplied and has not been
    subtracted.

    Parameters
    ----------
    ccd_in : `~astropy.nddata.CCDData` or filename
        Image from which to extract overscan estimate

    meta : `astropy.io.fits.header` or None
        referece to metadata of ccd into which to write OVERSCAN_* cards.
        If None, no metadata will be returned

    master_bias : `~astropy.nddata.CCDData`, filename, or None
        Bias to subtract from ccd before estimate is calculated.
        Improves accruacy by removing bias ramp.  Bias can be in units
        of ADU or electrons and is converted using the specified gain.
        If bias has already been subtracted, this step will be skipped
        but the bias header will be used to extract readnoise and gain
        using the *_key keywords.  Default is ``None``.

    binsize: float or None, optional
        The binsize to use for the histogram.  If None, binsize is 
        (readnoise in ADU)/4.  Default = None

    min_width : int, optional
        Minimum width peak to search for in histogram.  Keep in mind
        histogram bins are binsize ADU wide.  Default = 1

    max_width : int, optional
        See min_width.  Default = 8

    box_size : int
        Edge size of square box used to extract biweight median location
        from the corners of the image for this method of  overscan
        estimation.  Default = 100

    show : boolean
       Show image with min/max set to highlight overscan pixels and
       histogram with overscan chopped  histogram.  Default is False [consider making this boolean or name of plot file]

    """
    # This returns a copy of ccd_in if it is not a filename.  This is
    # important, since we mess with both the ccd.data and .metadata
    ccd = ccddata_read(ccd_in)
    ccd.meta = ccd_metadata(ccd.meta)
    if meta is None:
        meta = ccd.meta
    if ccd.unit != u.adu:
        # For now don't get fancy with unit conversion
        raise ValueError('CCD units must be in ADU for overscan estimation')
    if ccd.meta['IMAGETYP'] == "BIAS":
        overscan = np.median(ccd)
        meta['HIERARCH OVERSCAN_MEDIAN'] = (overscan, 'ADU')
        meta['HIERARCH OVERSCAN_METHOD'] = ('median',
                                            'Method used for overscan estimation')
        return overscan

    # Prepare for histogram method of overscan estimation.  These
    # keywords are guaranteed to be in meta because we put there there
    # in ccd_metadata
    readnoise = ccd.meta['RDNOISE']
    gain = ccd.meta['GAIN']
    if ccd.meta.get('subtract_bias') is None and master_bias is not None:
        # Bias has not been subtracted and we have a bias around to be
        # able to do that subtraction
        bias = ccddata_read(master_bias)
        # Improve our readnoise (measured) and gain (probably not
        # re-measured) values
        readnoise = bias.meta['RDNOISE']
        gain = bias.meta['GAIN']
        if bias.unit is u.electron:
            # Convert bias back to ADU for subtraction
            bias = bias.divide(gain*u.electron/u.adu)
        ccd = ccdp.subtract_bias(ccd, bias)
        if isinstance(master_bias, str):
            meta['HIERARCH OVERSCAN_MASTER_BIAS'] = 'OSBIAS'
            meta['OSBIAS'] = master_bias
        else:
            meta['HIERARCH OVERSCAN_MASTER_BIAS'] = 'CCDData object provided'
    if ccd.meta.get('subtract_bias') is None:
        log.warning('overscan_estimate: bias has not been subtracted, which can lead to inaccuracy of overscan estimate')
    # The coronagraph creates a margin of un-illuminated pixels on the
    # CCD.  These are great for estimating the bias and scattered
    # light for spontanous subtraction.
    # Corners method
    s = ccd.shape
    bs = box_size
    c00 = biweight_location(ccd[0:bs,0:bs])
    c10 = biweight_location(ccd[s[0]-bs:s[0],0:bs])
    c01 = biweight_location(ccd[0:bs,s[1]-bs:s[1]])
    c11 = biweight_location(ccd[s[0]-bs:s[0],s[1]-bs:s[1]])
    corners_method = min(c00, c10, c01, c11)
    # Histogram method.  The first peak is the bias, the second is the
    # ND filter.  Note that the 1.25" filters do a better job at this
    # than the 2" filters but with carefully chosen parameters, the
    # first small peak can be spotted.
    if binsize is None:
        # Calculate binsize based on readnoise in ADU, but oversample
        # by 4.  Note need to convert from Quantity to float
        binsize = readnoise/gain/4.
    im_hist, im_hist_centers = hist_of_im(ccd, binsize)
    # Note that after bias subtraction, there is sometimes some noise
    # at low counts.  We expect a lot of pixels in the histogram, so filter
    good_idx = np.flatnonzero(im_hist > min_hist_val)
    im_hist = im_hist[good_idx]
    im_hist_centers = im_hist_centers[good_idx]
    # The arguments to linspace are the critical parameters I played
    # with together with binsize to get the first small peak to be recognized
    im_peak_idx = signal.find_peaks_cwt(im_hist,
                                        np.linspace(min_width, max_width))
    hist_method = im_hist_centers[im_peak_idx[0]]
    overscan_methods = ['corners', 'histogram']
    overscan_values = np.asarray((corners_method, hist_method))
    meta['HIERARCH OVERSCAN_CORNERS'] = (corners_method, 'ADU')
    meta['HIERARCH OVERSCAN_HISTOGRAM'] = (hist_method, 'ADU')
    o_idx = np.argmin(overscan_values)
    overscan = overscan_values[o_idx]
    meta['HIERARCH OVERSCAN_METHOD'] = (overscan_methods[o_idx],
                                       'Method used for overscan estimation')
    if show:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 9))
        ccds = ccd.subtract(1000*u.adu)
        range = 5*readnoise/gain
        vmin = overscan - range - 1000
        vmax = overscan + range - 1000
        ax1.imshow(ccds, origin='lower', cmap=plt.cm.gray, filternorm=0,
                   interpolation='none', vmin=vmin, vmax=vmax)
        ax1.set_title('Image minus 1000 ADU')
        ax2.plot(im_hist_centers, im_hist)
        ax2.set_yscale("log")
        ax2.set_xscale("log")
        ax2.axvline(overscan, color='r')
        # https://stackoverflow.com/questions/13413112/creating-labels-where-line-appears-in-matplotlib-figure
        # the x coords of this transformation are data, and the
        # y coord are axes
        trans = transforms.blended_transform_factory(
            ax2.transData, ax2.transAxes)
        ax2.set_title('Histogram')
        ax2.text(overscan+20, 0.05, overscan_methods[o_idx]
                 + ' overscan = {:.2f}'.format(overscan),
                 rotation=90, transform=trans,
                 verticalalignment='bottom')
        plt.show()
    return overscan

def subtract_overscan(fname_or_ccd, oscan=None, *args, **kwargs):
    """Subtract overscan, estimating it, if necesesary, from image.
    Also subtracts overscan from SATLEVEL keyword

    Note: ccdproc's native subtract_overscan function can't be used
    because it assumes the overscan region is specified by a simple
    rectangle.

    """
    nccd = ccddata_read(fname_or_ccd)
    if oscan is None:
        oscan = overscan_estimate(nccd, meta=nccd.meta, *args, **kwargs)
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

#def get_calibration(ccd, auto=False,
#                imagetyp=None,
#                master_bias=None,
#                dark_frame=None,
#                master_flat=None,
#
#
#def get_imagetyp(ccd, **kwargs):
#    imagetyp = nccd.meta.get('imagetyp')
#    outkwargs = {'imagetyp': imagetyp}
#    return (ccd, outkwargs)
#
#def autocalibrate(ccd, imagetyp=None, **kwargs):
#    if imagetyp is None:
#        return (ccd, {})
#    elif imagetyp.lower() == 'bias':
#        outkwargs = {'oscan': True, 'gain': True}
#    elif imagetyp.lower() == 'dark':
#        outkwargs = {'oscan': True, 'gain': True, 'master_bias': True}
#    elif imagetyp.lower() == 'flat':
#        outkwargs = {'oscan': True, 'gain': True, 'master_bias': True,
#                     'dark_frame': True}
#    elif imagetyp.lower() == 'light':
#        outkwargs = {'oscan': True, 'gain': True, 'master_bias': True,
#                     'dark_frame': True, 'master_flat': True}
#    else:
#        raise ValueError(f'Unknown IMAGETYP keyword {imagetyp}')
#    return (ccd, outkwargs)

def prune_pout(pout, fnames):
    pruned_pout = []
    pruned_fnames = []
    for i in range(len(pout)):
        if pout[i][0] is None:
            # ccd is None
            continue
        pruned_pout.append(pout[i])
        pruned_fnames.append(fnames[i])
    return (pruned_pout, pruned_fnames)


def multi_logging(level, pipe_meta, message):
    """Implements logging on a per-process basis in ccd_pipeline post-processing"""
    # Work directly with the pipe_meta dictionary, thus a return value
    # is not needed
    if level in pipe_meta:
        pipe_meta[level].append(message)
    else:
        pipe_meta[level] = [message]

def outname_create(fname, ccd, pipe_meta,
                   outdir=None,
                   create_outdir=False,
                   outname_append='_ccdmp',
                   **kwargs):
    if not (isinstance(fname, str)
            and isinstance(outname_append, str)):
        raise ValueError("Not enough information provided to create output filename.  Specify outname or use an input filename and specify a string to append to that output filename to assure input is not overwritten")
    if outdir is None:
        outdir = os.getcwd()
    if create_outdir:
        os.makedirs(outdir, exist_ok=True)
    if not os.path.isdir(outdir):
        raise ValueError(f"outdir {outdir} does not exist.  Create directory or use create_outdir=True")
    bname = os.path.basename(fname)
    prepend, ext = os.path.splitext(bname)
    outbname = prepend + outname_append + ext
    outname = os.path.join(outdir, outbname)
    return outname

def num_can_process(num_to_process,
                    num_processes=max_num_processes,
                    mem_frac=max_mem_frac,
                    process_size=max_ccddata_size,
                    error_if_zero=True):
    mem = psutil.virtual_memory()
    max_mem = mem.available*mem_frac
    max_n = int(max_mem / process_size)
    if error_if_zero and max_n == 0:
        raise EnvironmentError(f'Current memory {max_mem/2**20} MiB insufficient for process size {process_size/2**20} MiB')

    return min(num_to_process, num_processes, max_n)


def ccd_pipeline(fnames,
                 num_processes=max_num_processes,
                 mem_frac=max_mem_frac,
                 process_size=max_ccddata_size,
                 **kwargs):
    ncp = num_can_process(len(fnames),
                          num_processes=num_processes,
                          mem_frac=mem_frac,
                          process_size=process_size)
    if ncp == 1:
        retvals = [ccd_process_file(f, **kwargs)
                   for f in fnames]
        return retvals

    pwk = PoolWorkerKwargs(ccd_process_file,
                           **kwargs)
    with NoDaemonPool(processes=ncp) as p:
        retvals = p.map(pwk.worker, fnames)
    return retvals

def ccd_process_file(fname,
                     ccddata_reader=None,
                     outname_creator=None,
                     overwrite=False,
                     **kwargs):
    if ccddata_reader is None:
        ccddata_reader = ccddata_read
    if outname_creator is None:
        outname_creator = outname_create
    ccd = ccddata_reader(fname)
    ccd, pipe_meta = ccd_post_process(ccd, **kwargs)
    if ccd is None:
        return (None, pipe_meta)
    outname = outname_creator(fname, ccd, pipe_meta, **kwargs)
    if outname is not None:
        ccd.write(outname, overwrite=overwrite)
    return (outname, pipe_meta)
    
def ccd_post_process(ccd,
                     post_process_list=None,
                     **kwargs):
    if post_process_list is None:
        post_process_list = []
    ccd = ccd_pre_process(ccd, **kwargs)
    pipe_meta = {}
    if ccd is None:
        return (None, pipe_meta)
    for pp in post_process_list:
        ccd, this_meta = pp(ccd, pipe_meta, **kwargs)
        pipe_meta.update(this_meta)
    return (ccd, pipe_meta)

def ccd_pre_process(ccd,
                    pre_process_list=None,
                    ccd_processor=None,
                    **kwargs):
    if pre_process_list is None:
        pre_process_list = []
    if ccd_processor is None:
        ccd_processor = ccdp.ccd_process
    for pp in pre_process_list:
        ccd, these_kwargs = pp(ccd, **kwargs)
        kwargs.update(these_kwargs)
    if ccd is None:
        return None
    return ccd_processor(ccd, **kwargs)

def cor_process(ccd,
                calibration=None,
                auto=False,
                imagetyp=None,
                ccd_meta=True,
                exp_correct=True,
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
                outdir=None,
                create_outdir=False,
                *args, **kwargs):
#def ccd_process(fname_or_ccd, calibration=None,
#                oscan=None, error=False, master_bias=None,
#                gain=None, gain_corrected=True, readnoise=None,
#                dark_frame=None, master_flat=None, bad_pixel_mask=None,
#                *args, **kwargs):
    """Perform basic CCD processing/reduction of IoIO ccd data

    The following steps can be included:

    * add CCD metadata (:func:`ccd_metadata`)
    * correct CCD exposure time (:func:`ccd_exp_correct`)
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

    filter_func_list : list
        List of functions to run on image just after it is read used to
        reject image.  Functions must return True to result in
        rejection.  Rejected images result in a ccd_process return
        value of None.
        Default is [:func:`not_full_frame`]

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

    imagetyp : bool or str
        If True, do reduction based on IMAGETYP keyword.  If string,
        use that as IMAGETYP.  Requires calibration object
        bias -> oscan=True
        dark -> oscan=True, master_bias=True
        flat -> oscan=True, master_bias=True, dark_frame=True
        light-> oscan=True, error=True, master_bias=True,
                dark_frame=True, master_flat=True

    ccd_meta : bool
        Add CCD metadata
        Default is ``True``

    exp_correct : bool
        Correct for exposure time problems
        Default is ``True``

    oscan : number, bool, or None, optional
        Single pedistal value to subtract from image.  If True, oscan
        is estimated using :func:`overscan_estimate` and subtracted
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
        Default is "GAIN" with units `astropy.units.electron/astropy.units.adu`

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

    outname : str, optional
        Name of output file.  If outdir is supplied, full outname will
        be constructed by joining outdir and outname.
        Default ``None``

    outdir : str, optional
        Directory into which to write outfile.  
        Default is ``None``

    create_outdir : bool, optional
        Create outdir including any needed leaf directories if it does
        not exist.  
        Default is ``False``

    outname_append : str, optional
        String appended to input filename to create outname.  Only
        used if outdir is specified and outname is not.  
        Default is ``_p``

    overwrite : bool, optional
        If writing file, overwrite existing file.
        Default is ``False``

    return_outname : bool
        If True, return outname rather than reduced CCDData.
        Automatically set to True if multi is True.
    	Default is ``False``

    post_process_list : list
        List of functions to run on ccd object just before it is
        returned.  Each function must accept the current CCDData
        object, and all input keywords of ccd_process.  Return value
        of functions must be of type CCDData.  Subsequent functions
        in the list will start with the return value of the previous.
        Default is ``[]``

    meta_process_list : list
        List of functions to run just before ccd_process returns.
        Each process must accept all keywords of ccd_process and
        return a dictionary-like object, "meta".  Dictionaries are
        combined using the dictionary update() method.  If this
        keyword is not ``[]``, the return value of ccd_process will be
        a tuple (ccd_or_outname, meta)
        Default is ``[]``


    Returns
    -------
    ccd : `~astropy.nddata.CCDData`
        Reduded ccd if `return_outname` is False and meta_process_list
        is empty

    - or -

    outname : str
    	Full filename of file into which ccd was saved if
    	`return_outname` is True

    - or -

    (ccd_or_outname, meta) : `tuple`
    	A tuple consisting of one of the above and a dict-like object
    	that is the output of the function(s) in meta_process_list

    - or -

    None if one of the filters in filter_func_list fails

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
        oscan=True
    elif imagetyp.lower() == 'dark':
        oscan=True; gain=True; master_bias=True
    elif imagetyp.lower() == 'flat':
        oscan=True; gain=True; master_bias=True; dark_frame=True
    elif imagetyp.lower() == 'light':
        oscan=True; gain=True; error=True; master_bias=True; dark_frame=True; master_flat=True
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
        nccd.meta = ccd_metadata(nccd.meta, *args, **kwargs)

    if exp_correct:
        # Correct exposure time for driver bug
        nccd.meta = ccd_exp_correct(nccd.meta, *args, **kwargs)
        
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
        nccd = trim_image(nccd, fits_section=trim)
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

    # Create the error frame.  I can't trim my overscan, so there are
    # lots of pixels at the overscan level.  After overscan and bias
    # subtraction, many of them that are probably normal statitical
    # outliers are negative enough to overwhelm the readnoise in the
    # deviation calculation.  But I don't want the error estimate on
    # them to be NaN, since the error is really the readnoise.
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
            min_value = min_value_key.value_from(nccd.meta)
        nccd = ccdp.flat_correct(nccd, master_flat,
                                 min_value=min_value,
                                 norm_value=flat_norm_value,
                                 add_keyword=flat_correct_keyword)

    # apply the gain correction only at the end if gain_corrected is False
    if gain is not None and not gain_corrected:
        nccd = ccdp.gain_correct(nccd, gain)

    return nccd
    

    #### make a copy of the object
    ###nccd = ccd.copy()

    #### apply the overscan correction
    ###if isinstance(oscan, CCDData):
    ###    nccd = subtract_overscan(nccd, overscan=oscan,
    ###                             median=oscan_median,
    ###                             model=oscan_model)
    ###elif isinstance(oscan, str):
    ###    nccd = subtract_overscan(nccd, fits_section=oscan,
    ###                             median=oscan_median,
    ###                             model=oscan_model)
    ###elif oscan is None:
    ###    pass
    ###else:
    ###    raise TypeError('oscan is not None, a string, or CCDData object.')
    ###
    #### apply the trim correction
    ###if isinstance(trim, str):
    ###    nccd = trim_image(nccd, fits_section=trim)
    ###elif trim is None:
    ###    pass
    ###else:
    ###    raise TypeError('trim is not None or a string.')
    ###
    #### create the error frame
    ###if error and gain is not None and readnoise is not None:
    ###    nccd = create_deviation(nccd, gain=gain, readnoise=readnoise)
    ###elif error and (gain is None or readnoise is None):
    ###    raise ValueError(
    ###        'gain and readnoise must be specified to create error frame.')
    ###
    #### apply the bad pixel mask
    ###if isinstance(bad_pixel_mask, np.ndarray):
    ###    nccd.mask = bad_pixel_mask
    ###elif bad_pixel_mask is None:
    ###    pass
    ###else:
    ###    raise TypeError('bad_pixel_mask is not None or numpy.ndarray.')
    ###
    #### apply the gain correction
    ###if not (gain is None or isinstance(gain, Quantity)):
    ###    raise TypeError('gain is not None or astropy.units.Quantity.')
    ###
    ###if gain is not None and gain_corrected:
    ###    nccd = gain_correct(nccd, gain)
    ###
    #### subtracting the master bias
    ###if isinstance(master_bias, CCDData):
    ###    nccd = subtract_bias(nccd, master_bias)
    ###elif master_bias is None:
    ###    pass
    ###else:
    ###    raise TypeError(
    ###        'master_bias is not None or a CCDData object.')
    ###
    #### subtract the dark frame
    ###if isinstance(dark_frame, CCDData):
    ###    nccd = subtract_dark(nccd, dark_frame, dark_exposure=dark_exposure,
    ###                         data_exposure=data_exposure,
    ###                         exposure_time=exposure_key,
    ###                         exposure_unit=exposure_unit,
    ###                         scale=dark_scale)
    ###elif dark_frame is None:
    ###    pass
    ###else:
    ###    raise TypeError(
    ###        'dark_frame is not None or a CCDData object.')
    ###
    #### test dividing the master flat
    ###if isinstance(master_flat, CCDData):
    ###    nccd = flat_correct(nccd, master_flat, min_value=min_value)
    ###elif master_flat is None:
    ###    pass
    ###else:
    ###    raise TypeError(
    ###        'master_flat is not None or a CCDData object.')
    ###
    #### apply the gain correction only at the end if gain_corrected is False
    ###if gain is not None and not gain_corrected:
    ###    nccd = gain_correct(nccd, gain)
    ###
    ###return nccd

def cor_pipeline(fnames,
                 pre_process_list=None,
                 post_process_list=None,
                 ccd_processor=None,
                 **kwargs):
    if pre_process_list is None:
        pre_process_list = [full_frame]
    if post_process_list is None:
        post_process_list=[check_oscan]
    if ccd_processor is None:
        ccd_processor = cor_process
    return ccd_pipeline(fnames,
                        pre_process_list=pre_process_list,
                        post_process_list=post_process_list,
                        ccd_processor=ccd_processor,
                        **kwargs)

def dark_process_one_file(fname,
                          calibration_scratch=calibration_scratch,
                          create_calibration_scratch=False,
                          outname_append=outname_append,
                          outname=None,
                          **kwargs):

    ccd = ccddata_read(fname)
    if not full_frame(ccd):
        log.debug('dark wrong shape: ' + fname)
        return {'good': False}
    if light_image(ccd):
        log.debug('dark recorded during light conditions: ' +
                  fname)
        return {'good': False}
    scratch_outname = create_outname(fname,
                                     outdir=calibration_scratch,
                                     create_outdir=create_calibration_scratch,
                                     outname_append=outname_append,
                                     outname=outname)
    ccd = ccd_process(ccd, **kwargs)
    ccd.write(scratch_outname, overwrite=True)
    # Get ready to capture the mean DATE-OBS
    tm = Time(ccd.meta['DATE-OBS'], format='fits')
    return {'good': True,
            'fname': scratch_outname,
            'jd': tm.jd}

def dark_combine_one_fdict(fdict,
                           outdir=calibration_root,
                           calibration_scratch=calibration_scratch,
                           outname_append=outname_append,
                           keep_intermediate=False,
                           show=False,
                           dccdt_tolerance=dccdt_tolerance,
                           mask_threshold=sx694_dark_mask_threshold,
                           num_processes=max_num_processes,
                           mem_frac=max_mem_frac,
                           ccddata_size=max_ccddata_size,
                           **kwargs):
    """Worker that allows the parallelization of calibrations taken at one
    temperature, exposure time, filter, etc.

    """

    mean_ccdt = fdict['CCDT']
    exptime = fdict['EXPTIME']
    directory = fdict['directory']

    # Make a scratch directory that is the date of the first file.
    # Not as fancy as the biases, but, hey, it is a scratch directory
    tmp = ccddata_read(fdict['fnames'][0])
    tm = tmp.meta['DATE-OBS']
    this_dateb1, _ = tm.split('T')
    sdir = os.path.join(calibration_scratch, this_dateb1)

    pout = cor_pipeline(fdict['fnames'],
                        num_processes=num_processes,
                        mem_frac=mem_frac,
                        process_size=max_ccddata_size,
                        outdir=sdir,
                        create_outdir=True,
                        overwrite=True,
                        pre_process_list=[full_frame, light_image],
                        post_process_list=[jd_meta, check_oscan],
                        **kwargs)
    pout = [p for p in pout if p[0] is not None]
    if len(pout) == 0:
        log.debug(f"No good darks found at CCDT = {mean_ccdt} C in {directory}")
        return False

    out_fnames, pipe_meta = zip(*pout)
    jds = [m['jd'] for m in pipe_meta]


    #dfdlist = [dark_process_one_file(d, calibration_scratch=sdir,
    #                                 create_calibration_scratch=create_calibration_scratch,
    #                                 outname_append=outname_append,
    #                                 **kwargs)
    #           for d in fdict['fnames']]
    #
    #
    #ncp = num_can_process(fdict['fnames'],
    #                     num_processes=num_processes,
    #                     mem_frac=mem_frac,
    #                     ccddata_size=ccddata_size)
    #
    #pwk = PoolWorkerKwargs(dark_process_one_file,
    #                       calibration_scratch=sdir,
    #                       create_calibration_scratch=create_calibration_scratch,
    #                       outname_append=outname_append,
    #                       **kwargs)
    #with NoDaemonPool(processes=ncp) as p:
    #    dfdlist = p.map(pwk.worker, fdict['fnames'])
    ## Unpack        
    #out_fnames = [dfd['fname'] for dfd in dfdlist if dfd['good']]
    #jds = [dfd['jd'] for dfd in dfdlist if dfd['good']]

    tm = Time(np.mean(jds), format='jd')
    this_date = tm.fits
    this_dateb = this_date.split('T')[0]
    if this_dateb != this_dateb1:
        log.warning(f"first dark is on {this_dateb1} but average is {this_dateb}")
    this_ccdt = '{:.1f}'.format(mean_ccdt)
    outbase = '{}_ccdT_{}_exptime_{}s'.format(
        this_dateb, this_ccdt, exptime)
    
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
    im, _ = mask_above(im, 'SATLEVEL')
    im, _ = mask_above(im, 'NONLIN')

    # Create a mask that blanks out all our pixels that are just
    # readnoise.  Multiply this in as zeros, not a formal mask,
    # otherwise subsequent operations with the dark will mask out
    # all but the dark current-affected pixels!
    measured_readnoise = im.meta['RDNOISE']
    is_dark_mask = im.data > measured_readnoise * mask_threshold
    n_dark_pix = np.count_nonzero(is_dark_mask)
    im.meta['NDARKPIX'] \
        = (n_dark_pix, 'number of pixels with dark current')
    im = im.multiply(is_dark_mask, handle_meta='first_found')

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
    for i, f in enumerate(fdict['fnames']):
        im.meta['FILE{0:02}'.format(i)] = f
    # Prepare to write
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    outbase = os.path.join(outdir, outbase)
    out_fname = outbase + '_combined_dark.fits'
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
    if not keep_intermediate:
        for f in out_fnames:
            try:
                os.remove(f)
            except Exception as e:
                log.debug(f'Unexpected!  Remove {f} failed: ' + str(e))
        try:
            os.rmdir(sdir)
        except Exception as e:
            pass
        try:
            os.rmdir(calibration_scratch)
        except Exception as e:
            pass

#    print('IN dark_combine_one_fdict')
#    lccds = []
#    jds = []
#    for fname in fdict['fnames']:
#        ccd = ccddata_read(fname, add_metadata=True)
#        if not full_frame(ccd):
#            log.debug('dark wrong shape: ' + fname)
#            return False
#        if light_image(ccd):
#            log.debug('dark recorded during light conditions: ' +
#                      fname)
#            return False
#        ccd = ccd_process(ccd, oscan=True, master_bias=master_bias,
#                          gain=True, error=True, calibration=calibration)
#        lccds.append(ccd)
#        # Get ready to capture the mean DATE-OBS
#        tm = Time(ccd.meta['DATE-OBS'], format='fits')
#        tt = tm.tt.datetime
#        jds.append(tm.jd)
#    mean_ccdt = fdict['CCDT']
#    exptime = fdict['EXPTIME']
#    directory = fdict['directory']
#    if len(jds) == 0:
#        log.debug("No good darks found at CCDT = {mean_ccdt} C in {directory}")
#        return False
#    tm = Time(np.mean(jds), format='jd')
#    this_date = tm.fits
#    this_dateb = this_date.split('T')[0]
#    this_ccdt = '{:.1f}'.format(mean_ccdt)
#    outbase = '{}_ccdT_{}_exptime_{}s'.format(
#        this_dateb, this_ccdt, exptime)
#
#    mem = psutil.virtual_memory()
#    im = \
#        ccdp.combine(lccds,
#                     method='average',
#                     sigma_clip=True,
#                     sigma_clip_low_thresh=5,
#                     sigma_clip_high_thresh=5,
#                     sigma_clip_func=np.ma.median,
#                     sigma_clip_dev_func=mad_std,
#                     mem_limit=mem.available*mem_frac)
#    im, _ = mask_above(im, 'SATLEVEL')
#    im, _ = mask_above(im, 'NONLIN')
#
#    # Create a mask that blanks out all our pixels that are just
#    # readnoise.  Multiply this in as zeros, not a formal mask,
#    # otherwise subsequent operations with the dark will mask out
#    # all but the dark current-affected pixels!
#    measured_readnoise = im.meta['RDNOISE']
#    is_dark_mask = im.data > measured_readnoise * mask_threshold
#    n_dark_pix = np.count_nonzero(is_dark_mask)
#    im.meta['NDARKPIX'] \
#        = (n_dark_pix, 'number of pixels with dark current')
#    im = im.multiply(is_dark_mask, handle_meta='first_found')
#
#    # Collect image metadata.  For some reason, masked pixels
#    # aren't ignored by std, etc. even though they output masked
#    # arrays (which is annoying in its own right -- see example
#    # commented mean).  So just create a new array, if needed, and
#    # only put into it the good pixels
#    bad_mask = is_dark_mask == 0
#    if im.mask is not None:
#        bad_mask = bad_mask | im.mask
#    # Flip bad mask around so we get only the dark pixels in the
#    # linear range
#    tim = im.data[bad_mask == 0]
#
#    std =  np.std(tim)
#    #std =  np.asscalar(std.data  )
#    med =  np.median(tim)
#    #med =  np.asscalar(med.data  )
#    mean = np.mean(tim)
#    #mean = np.asscalar(mean.data  )
#    tmin = np.min(tim)
#    tmax = np.max(tim)
#    rdnoise = np.sqrt(np.median((tim[1:] - tim[0:-1])**2))
#    print('combined dark statistics for ' + outbase)
#    print('std, rdnoise, mean, med, min, max, n_dark_pix')
#    print(std, rdnoise, mean, med, tmin, tmax, n_dark_pix)
#    im.meta['STD'] = (std, 'Standard deviation of image (electron)')
#    im.meta['MEDIAN'] = (med, 'Median of image (electron)')
#    im.meta['MEAN'] = (mean, 'Mean of image (electron)')
#    im.meta['MIN'] = (tmin, 'Min of image (electron)')
#    im.meta['MAX'] = (tmax, 'Max of image (electron)')
#    im.meta['NCOMBINE'] = (len(lccds), 'Number of darks combined')
#    add_history(im.meta,
#                'Combining NCOMBINE biases indicated in FILENN')
#    im.meta['HIERARCH MASK_THRESHOLD'] \
#        = (mask_threshold, '*RDNOISE (electron)')
#    add_history(im.meta,
#                'Setting pixes below MASK_THRESHOLD to zero; prevents subtraction noise')
#    # Record each filename
#    for i, f in enumerate(fdict['fnames']):
#        im.meta['FILE{0:02}'.format(i)] = f
#    # Prepare to write
#    if not os.path.exists(outdir):
#        os.mkdir(outdir)
#    outbase = os.path.join(outdir, outbase)
#    out_fname = outbase + '_combined_dark.fits'
#    # Leave these large for fast calculations downstream and make
#    # final results that primarily sit on disk in bulk small
#    #im.data = im.data.astype('float32')
#    #im.uncertainty.array = im.uncertainty.array.astype('float32')
#    im.write(out_fname, overwrite=True)
#    if show:
#        impl = plt.imshow(im, origin='lower', cmap=plt.cm.gray,
#                          filternorm=0, interpolation='none',
#                          vmin=med-std, vmax=med+std)
#        plt.show()
#        plt.close()


def dark_combine(directory=None,
                 collection=None,
                 subdirs=calibration_subdirs,
                 glob_include=dark_glob,
                 dccdt_tolerance=dccdt_tolerance,
                 num_processes=max_num_processes,
                 mem_frac=max_mem_frac,
                 num_calibration_files=num_calibration_files,
                 max_ccddata_size=max_ccddata_size,
                 **kwargs):
    fdict_list = \
        fname_by_imagetyp_ccdt_exp(directory=directory,
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

    one_fdict_size = num_calibration_files * max_ccddata_size
    our_num_processes = num_can_process(nfdicts,
                                        num_processes=num_processes,
                                        mem_frac=mem_frac,
                                        process_size=one_fdict_size)
    num_subprocesses = int(num_processes / our_num_processes)
    subprocess_mem_frac = mem_frac / our_num_processes
    log.debug('dark_combine: {} num_processes = {}, mem_frac = {}, our_num_processes = {}, num_subprocesses = {}, subprocess_mem_frac = {}'.format(directory, num_processes, mem_frac, our_num_processes, num_subprocesses, subprocess_mem_frac))

    pwk = PoolWorkerKwargs(dark_combine_one_fdict,
                           num_processes=num_subprocesses,
                           mem_frac=subprocess_mem_frac,
                           **kwargs)
    if nfdicts == 1:
        for fdict in fdict_list:
                pwk.worker(fdict)
    else:
        with NoDaemonPool(processes=our_num_processes) as p:
            p.map(pwk.worker, fdict_list)

def flat_process(ccd, pipe_meta,
                 init_threshold=100, # units of readnoise
                 edge_mask=-40, # CorObsData parameter for ND filter coordinate
                 **kwargs): 
    # Use photutils.Background2D to smooth each flat and get a
    # good maximum value.  Mask edges and ND filter so as to
    # increase quality of background map
    mask = np.zeros(ccd.shape, bool)
    # Use the CorObsData ND filter stuff with a negative
    # edge_mask to blank out all of the fuzz from the ND filter cut
    obs_data = CorObsData(ccd.to_hdu(), edge_mask=edge_mask)
    mask[obs_data.ND_coords] = True
    rdnoise = ccd.meta['RDNOISE']
    mask[ccd.data < rdnoise * init_threshold] = True
    ccd.mask = mask
    bkg_estimator = MedianBackground()
    b = Background2D(ccd, 20, mask=mask, filter_size=5,
                     bkg_estimator=bkg_estimator)
    max_flat = np.max(b.background)
    if max_flat > ccd.meta['NONLIN']:
        log.warning(f'flat max value of {max_flat} too bright: {fname}')
        return (None, {})
    ccd.mask = None
    ccd = ccd.divide(max_flat, handle_meta='first_found')
    ccd.meta['FLATDIV'] = (max_flat, 'Value used to normalize (smoothed max)')
    # Get ready to capture the mean DATE-OBS
    tm = Time(ccd.meta['DATE-OBS'], format='fits')
    return (ccd, {'jd': tm.jd})

def flat_combine_one_filt(this_filter,
                          collection=None,
                          outdir=calibration_root,
                          calibration_scratch=calibration_scratch,
                          keep_intermediate=False,
                          min_num_flats=min_num_flats,
                          num_processes=max_num_processes,
                          mem_frac=max_mem_frac,
                          ccddata_size=max_ccddata_size,
                          show=False,
                          flat_cut=0.75,
                          edge_mask=-40, # CorObsData parameter for ND filter coordinates    
                          **kwargs):
    flat_fnames = collection.files_filtered(imagetyp='FLAT',
                                            filter=this_filter,
                                            include_path=True)

    if len(flat_fnames) < min_num_flats:
        log.debug(f"Not enough good flats found for filter {this_filter} in {directory}")
        return False

    # Make a scratch directory that is the date of the first file.
    # Not as fancy as the biases, but, hey, it is a scratch directory
    tmp = ccddata_read(flat_fnames[0])
    tm = tmp.meta['DATE-OBS']
    this_dateb1, _ = tm.split('T')
    sdir = os.path.join(calibration_scratch, this_dateb1)

    pout = cor_pipeline(flat_fnames,
                        num_processes=num_processes,
                        mem_frac=mem_frac,
                        process_size=ccddata_size,
                        outdir=sdir,
                        create_outdir=True,
                        overwrite=True,
                        pre_process_list=[full_frame],
                        post_process_list=[flat_process, jd_meta, check_oscan],
                        **kwargs)
    pout, flat_fnames = prune_pout(pout, flat_fnames)
    if len(pout) < min_num_flats:
        log.debug(f"Not enough good flats found for filter {this_filter} in {directory}")
        return False

    out_fnames, pipe_meta = zip(*pout)
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
    for i, f in enumerate(flat_fnames):
        im.meta['FILE{0:02}'.format(i)] = f
    add_history(im.meta,
                'Combining NCOMBINE biases indicated in FILENN')

    # Interpolate over our ND filter
    #print(f'flat_combine_one_filt pre CorObsData: mem available: {mem.available/2**20}')
    hdul = im.to_hdu()
    obs_data = CorObsData(hdul, edge_mask=edge_mask)
    # Capture our ND filter metadata
    im.meta = hdul[0].header
    good_pix = np.ones(im.shape, bool)
    good_pix[obs_data.ND_coords] = False
    points = np.where(good_pix)
    values = im[points]
    xi = obs_data.ND_coords
    print(f'flat_combine_one_filt post CorObsData: mem available: {mem.available/2**20}')

    # Linear behaved much better
    nd_replacement = interpolate.griddata(points,
                                          values,
                                          xi,
                                          method='linear')
                                          #method='cubic')
    print(f'flat_combine_one_filt post interpolate.griddata mem available: {mem.available/2**20}')
    im.data[xi] = nd_replacement
    # Do one last smoothing and renormalization
    bkg_estimator = MedianBackground()
    b = Background2D(im, 20, mask=(im.data<flat_cut), filter_size=5,
                     bkg_estimator=bkg_estimator)
    max_flat = np.max(b.background)
    print(f'flat_combine_one_filt post Background2D mem available: {mem.available/2**20}')
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
    if not keep_intermediate:
        for f in out_fnames:
            try:
                os.remove(f)
            except Exception as e:
                # We do not expect this, since we created these with
                # our local process
                log.debug(f'Unexpected!  Remove {f} failed: ' + str(e))
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
    
def flat_combine(directory=None,
                 collection=None,
                 subdirs=calibration_subdirs,
                 glob_include=flat_glob,
                 num_processes=max_num_processes,
                 mem_frac=max_mem_frac,
                 num_calibration_files=num_calibration_files,
                 max_ccddata_size=max_ccddata_size,
                 griddata_expansion_factor=griddata_expansion_factor,
                 **kwargs):
    if subdirs is None:
        subdirs = []
    if collection is None:
        if not os.path.isdir(directory):
            return False
        collection = ccdp.ImageFileCollection(directory,
                                              glob_include=glob_include)
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

    one_filt_size = max(num_calibration_files * max_ccddata_size,
                        max_ccddata_size * griddata_expansion_factor)
    our_num_processes = num_can_process(nfilts,
                                        num_processes=num_processes,
                                        mem_frac=mem_frac,
                                        process_size=one_filt_size)
    # Combining files is the slow part, so we want the maximum of
    # processes doing that in parallel
    log.debug(f'flat_combine: {directory}, nfilts = {nfilts}, our_num_processes = {our_num_processes}')
    # Number of sub-processes in each process we will spawn
    num_subprocesses = int(num_processes / our_num_processes)
    # Similarly, the memory fraction for each process we will spawn
    subprocess_mem_frac = mem_frac / our_num_processes
    log.debug('flat_combine: {} num_processes = {}, mem_frac = {}, our_num_processes = {}, num_subprocesses = {}, subprocess_mem_frac = {}'.format(directory, num_processes, mem_frac, our_num_processes, num_subprocesses, subprocess_mem_frac))
    pwk = PoolWorkerKwargs(flat_combine_one_filt,
                           collection=collection,
                           num_processes=num_subprocesses,
                           mem_frac=subprocess_mem_frac,
                           **kwargs)
    if nfilts == 1 or our_num_processes == 1:
        for filt in filters:
                pwk.worker(filt)
    else:
        with NoDaemonPool(processes=our_num_processes) as p:
            p.map(pwk.worker, filters)
    
        ## Initiate all our processes
        #plist = [Process(target=flat_combine_one_filt,
        #                 args=(filt,),
        #                 kwargs={"collection": collection,
        #                         "num_processes": num_subprocesses,
        #                         "mem_frac": subprocess_mem_frac,
        #                         **kwargs},
        #                 daemon=False) # Let subprocesses create more children
        #         for filt in filters]
        #p_limit(plist, our_num_processes)





## --> This loop could be parallelized
#filters = np.unique(collection.summary['filter'])
#for this_filter in filters:
#    flat_fnames = collection.files_filtered(imagetyp='FLAT',
#                                            filter=this_filter,
#                                            include_path=True)
#    lccds = []
#    jds = []
#    for fname in flat_fnames:
#        ccd = ccddata_read(fname, add_metadata=True)
#        ccd = ccd_process(ccd,
#                          oscan=True,
#                          master_bias=master_bias,
#                          gain=True,
#                          error=True,
#                          dark_frame=dark_frame)
#        # Use photutils.Background2D to smooth each flat and get a
#        # good maximum value.  Mask edges and ND filter so as to
#        # increase quality of background map
#        mask = np.zeros(ccd.shape, bool)
#        # Use the CorObsData ND filter stuff with a negative
#        # edge_mask to blank out all of the fuzz from the ND filter cut
#        obs_data = CorObsData(ccd.to_hdu(), edge_mask=edge_mask)
#        mask[obs_data.ND_coords] = True
#        rdnoise = ccd.meta['RDNOISE']
#        mask[ccd.data < rdnoise * init_threshold] = True
#        ccd.mask = mask
#        bkg_estimator = MedianBackground()
#        b = Background2D(ccd, 20, mask=mask, filter_size=5,
#                         bkg_estimator=bkg_estimator)
#        max_flat = np.max(b.background)
#        if max_flat > ccd.meta['NONLIN']:
#            log.warning('flat max value of {} too bright: {}'.format(
#                max_flat, fname))
#            continue
#        ccd.mask = None
#        ccd = ccd.divide(max_flat, handle_meta='first_found')
#        lccds.append(ccd)
#        # Get ready to capture the mean DATE-OBS
#        tm = Time(ccd.meta['DATE-OBS'], format='fits')
#        tt = tm.tt.datetime
#        jds.append(tm.jd)
#    
#    if len(jds) < min_num_flats:
#        log.debug('Not enough good flats found for filter = {} in {}'.format(this_filter, directory))
#        continue
#    # Combine our flats
#    mem = psutil.virtual_memory()
#    im = \
#        ccdp.combine(lccds,
#                     method='average',
#                     sigma_clip=True,
#                     sigma_clip_low_thresh=5,
#                     sigma_clip_high_thresh=5,
#                     sigma_clip_func=np.ma.median,
#                     sigma_clip_dev_func=mad_std,
#                     mem_limit=mem.available*mem_frac)
#    # Interpolate over our ND filter
#    hdul = ccd.to_hdu()
#    obs_data = CorObsData(hdul, edge_mask=edge_mask)
#    # Capture our ND filter metadata
#    im.meta = hdul[0].header
#    good_pix = np.ones(ccd.shape, bool)
#    good_pix[obs_data.ND_coords] = False
#    points = np.where(good_pix)
#    values = im[points]
#    xi = obs_data.ND_coords
#    # Linear behaved much better
#    nd_replacement = interpolate.griddata(points,
#                                          values,
#                                          xi,
#                                          method='linear')
#                                          #method='cubic')
#    im.data[xi] = nd_replacement
#    # Do one last smoothing and renormalization
#    bkg_estimator = MedianBackground()
#    b = Background2D(im, 20, mask=(im.data<flat_cut), filter_size=5,
#                     bkg_estimator=bkg_estimator)
#    max_flat = np.max(b.background)
#    im = im.divide(max_flat, handle_meta='first_found')
#    im.mask = im.data < flat_cut
#    im.meta['FLAT_CUT'] = (flat_cut, 'Value below which flat is masked')
#
#    # Prepare to write
#    tm = Time(np.mean(jds), format='jd')
#    this_date = tm.fits
#    this_dateb = this_date.split('T')[0]
#    outbase = '{}_{}'.format(this_dateb, this_filter)
#    if not os.path.exists(outdir):
#        os.mkdir(outdir)
#    outbase = os.path.join(outdir, outbase)
#    out_fname = outbase + '_flat.fits'
#    im.write(out_fname, overwrite=True)
#    if show:
#        impl = plt.imshow(im, origin='upper', cmap=plt.cm.gray)
#        plt.show()


def bias_analyze(directory='/data/io/IoIO/reduced/bias_dark'):
    collection = ccdp.ImageFileCollection(directory)
    s = collection.summary
    f = plt.figure(figsize=[8.5, 11])
    #good_idx = s['date-obs'] > '2020-04-17T00:00'
    #s = s[good_idx]
    plt.plot(s['ccd-temp'], s['median'], 'k.')
    plt.plot(s['ccd-temp'], s['mean'], 'r.')
    plt.xlabel('CCD Temperature (C)')
    plt.ylabel('median & mean (ADU)')
    plt.legend(['median', 'mean'])
    plt.show()
    plt.close

# From
# https://mwcraig.github.io/ccd-as-book/03-01-Dark-current-The-ideal-case.html 
def plot_dark_with_distributions(image, rn, dark_rate, 
                                 n_images=1,
                                 exposure=1,
                                 gain=1,
                                 show_poisson=True, 
                                 show_gaussian=True):
    """
    Plot the distribution of dark pixel values, optionally overplotting the expected Poisson and
    normal distributions corresponding to dark current only or read noise only.
    
    Parameters
    ----------
    
    image : numpy array
        Dark frame to histogram.
    
    rn : float
        The read noise, in electrons.
        
    dark_rate : float
        The dark current in electrons/sec/pixel.
    
    n_images : float, optional
        If the image is formed from the average of some number of dark frames then 
        the resulting Poisson distribution depends on the number of images, as does the 
        expected standard deviation of the Gaussian.
        
    exposure : float
        Exosure time, in seconds.
        
    gain : float, optional
        Gain of the camera, in electron/ADU.
        
    show_poisson : bool, optional
        If ``True``, overplot a Poisson distribution with mean equal to the expected dark
        counts for the number of images.
    
    show_gaussian : bool, optional
        If ``True``, overplot a normal distribution with mean equal to the expected dark
        counts and standard deviation equal to the read noise, scaled as appropiate for 
        the number of images.
    """
    
    #h = plt.hist(image.flatten(), bins=20, align='mid', 
    #             density=True, label="Dark frame");
    h = plt.hist(image.flatten(), bins=20000, align='mid', 
                 density=True, label="Dark frame");

    bins = h[1]
    
    expected_mean_dark = dark_rate * exposure / gain
    
    pois = stats.poisson(expected_mean_dark * n_images)

    pois_x = np.arange(0, 300, 1)

    new_area = np.sum(1/n_images * pois.pmf(pois_x))

    if show_poisson:
        plt.plot(pois_x / n_images, pois.pmf(pois_x) / new_area, 
                 label="Poisson dsitribution, mean of {:5.2f} counts".format(expected_mean_dark)) 

    if show_gaussian:
        # The expected width of the Gaussian depends on the number of images.
        expected_scale = rn / gain * np.sqrt(n_images)
        
        # Mean value is same as for the Poisson distribution 
        expected_mean = expected_mean_dark * n_images
        gauss = stats.norm(loc=expected_mean, scale=expected_scale)
        
        gauss_x = np.linspace(expected_mean - 5 * expected_scale,
                              expected_mean + 5 * expected_scale,
                              num=100)
        plt.plot(gauss_x / n_images, gauss.pdf(gauss_x) * n_images, label='Gaussian, standard dev is read noise in counts') 
        
    plt.xlabel("Dark counts in {} sec exposure".format(exposure))
    plt.ylabel("Fraction of pixels (area normalized to 1)")
    plt.grid()
    plt.legend()

def quick_show(im):
    std =  np.std(im)
    med =  np.median(im)
    mean = np.mean(im)
    tmin = np.min(im)
    tmax = np.max(im)
    rdnoise = np.sqrt(np.median((im.data[1:] - im.data[0:-1])**2))
    print('image statistics')
    print('std, rdnoise, mean, med, min, max')
    print(std, rdnoise, mean, med, tmin, tmax)
    #impl = plt.imshow(im, origin='lower', cmap=plt.cm.gray, filternorm=0,
    #                  interpolation='none', vmin=med-std, vmax=med+std)
    impl = plt.imshow(im, origin='upper', cmap=plt.cm.gray, filternorm=0,
                      interpolation='none', vmin=med-std, vmax=med+std)
    if isinstance(im, CCDData):
        plt.title(im.unit)
    else:
        plt.title('units not specified')    
    plt.show()
    plt.close()
    

log.setLevel('DEBUG')

## bias_fname = '/data/io/IoIO/reduced/bias_dark/2020-07-11_ccdT_-5.3_combined_bias.fits'
## master_bias = ccddata_read(bias_fname)
## dark_fname = '/data/io/IoIO/reduced/bias_dark/2020-07-11_ccdT_-5.3_exptime_100.0s_combined_dark.fits'
## dark_frame = ccddata_read(dark_fname)
## flat_fname = '/data/io/IoIO/reduced/bias_dark/2020-10-01_R_flat.fits'
## master_flat = ccddata_read(flat_fname)
## fname = '/data/io/IoIO/raw/20190814/0029P-S001-R001-C001-R_dupe-4.fts'
## raw = ccddata_read(fname)
## ccd = ccd_process(raw, oscan=True, master_bias=master_bias,
##                   gain=True, error=True, dark_frame=dark_frame,
##                   master_flat=master_flat)
## #ccd.write('/tmp/test.fits', overwrite=True)
## ccd.write('/home/jpmorgen/Proposals/NSF/Landslides_2020/20190814_0029P_R.fits', overwrite=True)
## impl = plt.imshow(ccd, origin='upper', cmap=plt.cm.gray, filternorm=0,
##                       interpolation='none', vmin=0, vmax=500)
## plt.show()

#bias_fname = '/data/io/IoIO/reduced/bias_dark/2020-07-11_ccdT_-5.3_combined_bias.fits'
#master_bias = ccddata_read(bias_fname)
#dark_fname = '/data/io/IoIO/reduced/bias_dark/2020-07-11_ccdT_-5.3_exptime_100.0s_combined_dark.fits'
#dark_frame = ccddata_read(dark_fname)
#flat_fname = '/data/io/IoIO/reduced/bias_dark/2020-10-01_R_flat.fits'
#master_flat = ccddata_read(flat_fname)
#fname = '/data/io/IoIO/raw/20190820/0029P-S001-R001-C001-R_dupe-4.fts'
#raw = ccddata_read(fname)
#ccd = ccd_process(raw, oscan=True, master_bias=master_bias,
#                  gain=True, error=True, dark_frame=dark_frame,
#                  master_flat=master_flat)
##ccd.write('/tmp/test.fits', overwrite=True)
#ccd.write('/home/jpmorgen/Proposals/NSF/Landslides_2020/20190820_0029P_R.fits', overwrite=True)
#impl = plt.imshow(ccd, origin='upper', cmap=plt.cm.gray, filternorm=0,
#                      interpolation='none', vmin=0, vmax=500)
#plt.show()

#bias_fname = '/data/io/IoIO/reduced/bias_dark/2020-07-11_ccdT_-5.3_combined_bias.fits'
#master_bias = ccddata_read(bias_fname)
#dark_fname = '/data/io/IoIO/reduced/bias_dark/2020-07-11_ccdT_-5.3_exptime_100.0s_combined_dark.fits'
#dark_frame = ccddata_read(dark_fname)
#flat_fname = '/data/io/IoIO/reduced/bias_dark/2020-10-01_R_flat.fits'
#master_flat = ccddata_read(flat_fname)
#fname = '/data/io/IoIO/raw/20190816/0029P-S001-R001-C001-R_dupe-4.fts'
#raw = ccddata_read(fname)
#ccd = ccd_process(raw, oscan=True, master_bias=master_bias,
#                  gain=True, error=True, dark_frame=dark_frame,
#                  master_flat=master_flat)
##ccd.write('/tmp/test.fits', overwrite=True)
#ccd.write('/home/jpmorgen/Proposals/NSF/Landslides_2020/20190816_0029P_R.fits', overwrite=True)
#impl = plt.imshow(ccd, origin='upper', cmap=plt.cm.gray, filternorm=0,
#                      interpolation='none', vmin=0, vmax=500)
#plt.show()

#bias_fname = '/data/io/IoIO/reduced/bias_dark/2020-08-21_ccdT_-10.2_combined_bias.fits'
#bias = ccddata_read(bias_fname)
#dark_fname = '/data/io/IoIO/reduced/bias_dark/2020-08-21_ccdT_-10.2_exptime_3.0s_combined_dark.fits'
#dark_frame = ccddata_read(dark_fname)
#flat_fname = '/data/io/IoIO/reduced/bias_dark/2020-10-01_R_flat.fits'
#master_flat = ccddata_read(flat_fname)
## --> remove this
##master_flat.meta['FLAT_CUT'] = 0.75
#fname = '/data/io/IoIO/raw/20200926/0029P-S001-R001-C001-R_dupe-4.fts'
#raw = ccddata_read(fname)
#ccd = ccd_process(raw, oscan=True, master_bias=master_bias,
#                  gain=True, error=True, dark_frame=dark_frame,
#                  master_flat=master_flat)
##ccd.write('/tmp/test.fits', overwrite=True)
#ccd.write('/home/jpmorgen/Proposals/NSF/Landslides_2020/20200926_0029P_R.fits', overwrite=True)
#impl = plt.imshow(ccd, origin='upper', cmap=plt.cm.gray, filternorm=0,
#                      interpolation='none', vmin=0, vmax=500)
#plt.show()



#bias_fname = '/data/io/IoIO/reduced/bias_dark/2020-04-21_ccdT_-20.2_combined_bias.fits'
#master_bias = ccddata_read(bias_fname)
##dark_fname = '/data/io/IoIO/reduced/bias_dark/2020-04-21_ccdT_-20.1_exptime_300.0s_combined_dark.fits'
#dark_fname = '/data/io/IoIO/reduced/bias_dark/2020-04-21_ccdT_-20.1_exptime_100.0s_combined_dark.fits'
#dark_frame = ccddata_read(dark_fname)
#flat_fname = '/data/io/IoIO/reduced/bias_dark/2020-05-17_R_flat.fits'
#master_flat = ccddata_read(flat_fname)
## --> remove this
##master_flat.meta['FLAT_CUT'] = 0.75
#fname = '/data/io/IoIO/raw/20200514/CK17T020-S001-R001-C001-R_dupe-4.fts'
#raw = ccddata_read(fname)
#ccd = ccd_process(raw, oscan=True, master_bias=master_bias,
#                  gain=True, error=True, dark_frame=dark_frame,
#                  master_flat=master_flat)
##ccd.write('/tmp/test.fits', overwrite=True)
#ccd.write('/home/jpmorgen/Proposals/NSF/Landslides_2020/CK17T020_R.fits', overwrite=True)
#impl = plt.imshow(ccd, origin='upper', cmap=plt.cm.gray, filternorm=0,
#                      interpolation='none', vmin=0, vmax=500)
#plt.show()
#fname = '/data/io/IoIO/raw/20200514/CK19Y010-S001-R001-C001-R_dupe-4.fts'
#raw = ccddata_read(fname)
#ccd = ccd_process(raw, oscan=True, master_bias=master_bias,
#                  gain=True, error=True, dark_frame=dark_frame,
#                  master_flat=master_flat)
##ccd.write('/tmp/test.fits', overwrite=True)
#ccd.write('/home/jpmorgen/Proposals/NSF/Landslides_2020/CK19Y010_R.fits', overwrite=True)
#impl = plt.imshow(ccd, origin='upper', cmap=plt.cm.gray, filternorm=0,
#                      interpolation='none', vmin=0, vmax=500)
#plt.show()



# bias_fname = '/data/io/IoIO/reduced/bias_dark/2020-07-11_ccdT_-5.27_combined_bias.fits'
# master_bias = ccddata_read(bias_fname)
# #dark_fname = '/data/io/IoIO/reduced/bias_dark/2020-07-11_ccdT_-5.25_exptime_300.0s_combined_dark.fits'
# dark_fname = '/data/io/IoIO/reduced/bias_dark/2020-07-11_ccdT_-5.25_exptime_10.0s_combined_dark.fits'
# dark_frame = ccddata_read(dark_fname)
# #flat_fname = '/data/io/IoIO/reduced/bias_dark/2020-07-07_Na_on_flat.fits'
# #master_flat = ccddata_read(flat_fname)
# # --> remove this
# #master_flat.meta['FLAT_CUT'] = 0.75
# fname = '/data/io/IoIO/raw/2020-07-08/NEOWISE-0007_Na-on.fit'
# raw = ccddata_read(fname)
# ccd = ccd_process(raw, oscan=True, master_bias=master_bias,
#                   gain=True, error=True, dark_frame=dark_frame)
# 
# readnoise = ccd.meta['RDNOISE']
# binsize = readnoise/4.
# im_hist, im_hist_centers = hist_of_im(ccd, binsize)
# min_hist_val=10
# min_width=10
# max_width=20
# good_idx = np.flatnonzero(im_hist > min_hist_val)
# im_hist = im_hist[good_idx]
# im_hist_centers = im_hist_centers[good_idx]
# # The arguments to linspace are the critical parameters I played
# # with together with binsize to get the first small peak to be recognized
# im_peak_idx = signal.find_peaks_cwt(im_hist,
#                                     np.linspace(min_width, max_width))
# print(im_hist_centers)
# sky = im_hist_centers[im_peak_idx[1]]
# print(sky) # expecting ~90
# mask = ccd.data < 0.25 * sky
# ccd.mask = mask
#ccd.write('/tmp/test.fits', overwrite=True)
#impl = plt.imshow(ccd, origin='upper', cmap=plt.cm.gray, filternorm=0,
#                      interpolation='none', vmin=0, vmax=500)
#plt.show()
#quick_show(ccd)

#bias_fname = '/data/io/IoIO/reduced/bias_dark/2020-07-11_ccdT_-5.27_combined_bias.fits'
#master_bias = ccddata_read(bias_fname)
#dark_fname = '/data/io/IoIO/reduced/bias_dark/2020-07-11_ccdT_-5.25_exptime_300.0s_combined_dark.fits'
##dark_fname = '/data/io/IoIO/reduced/bias_dark/2020-07-11_ccdT_-5.25_exptime_10.0s_combined_dark.fits'
#dark_frame = ccddata_read(dark_fname)
#flat_fname = '/data/io/IoIO/reduced/bias_dark/2020-07-07_Na_on_flat.fits'
#master_flat = ccddata_read(flat_fname)
## --> remove this
##master_flat.meta['FLAT_CUT'] = 0.75
#fname = '/data/io/IoIO/raw/2020-07-08/NEOWISE-0007_Na-on.fit'
#raw = ccddata_read(fname)
#ccd = ccd_process(raw, oscan=True, master_bias=master_bias,
#                  gain=True, error=True, dark_frame=dark_frame,
#                  master_flat=master_flat)
#ccd.write('/tmp/test.fits', overwrite=True)
#impl = plt.imshow(ccd, origin='upper', cmap=plt.cm.gray, filternorm=0,
#                      interpolation='none', vmin=0, vmax=500)
#plt.show()
##quick_show(ccd)


#bias_fname = '/data/io/IoIO/reduced/bias_dark/2020-07-11_ccdT_-5.3_combined_bias.fits'
#master_bias = ccddata_read(bias_fname)
#dark_fname = '/data/io/IoIO/reduced/bias_dark/2020-07-11_ccdT_-5.3_exptime_300.0s_combined_dark.fits'
#dark_frame = ccddata_read(dark_fname)
#flat_combine('/data/io/IoIO/raw/2020-07-07/',
#             master_bias=master_bias,
#             dark_frame=dark_frame,
#             show=False)

#bias_fname = '/data/io/IoIO/reduced/bias_dark/2020-04-21_ccdT_-20.2_combined_bias.fits'
#master_bias = ccddata_read(bias_fname)
#dark_fname = '/data/io/IoIO/reduced/bias_dark/2020-04-21_ccdT_-20.1_exptime_300.0s_combined_dark.fits'
#dark_frame = ccddata_read(dark_fname)
#flat_combine('/data/io/IoIO/raw/2020-05-17/',
#             master_bias=master_bias,
#             dark_frame=dark_frame,
#             show=False)

#bias_fname = '/data/io/IoIO/reduced/bias_dark/2020-10-01_ccdT_-19.9_combined_bias.fits'
#master_bias = ccddata_read(bias_fname)
#dark_fname = '/data/io/IoIO/reduced/bias_dark/2020-10-01_ccdT_-19.8_exptime_100.0s_combined_dark.fits'
#dark_frame = ccddata_read(dark_fname)
#flat_combine('/data/io/IoIO/raw/2020-10-01/',
#             master_bias=master_bias,
#             dark_frame=dark_frame,
#             show=False)

#fname = '/data/io/IoIO/raw/20200708/SII_on-band_004.fits'
#master_bias = ccddata_read(bias_fname)
#ccd = ccddata_read(fname)
#b = ccd_process(ccd, oscan=True, master_bias=master_bias,
#                gain=True, error=True)
#quick_show(b)


#dark_fname = '/data/io/IoIO/reduced/bias_dark/2020-07-11_ccdT_-5.25_exptime_300.0s_combined_dark.fits'
##dark_fname = '/data/io/IoIO/reduced/bias_dark/2020-07-11_ccdT_-5.25_exptime_300.0s_combined_dark.fits'
#ccd = ccddata_read(dark_fname)
#rn = ccd.meta['RDNOISE']
#exptime = ccd.meta['EXPTIME']
#n_images = ccd.meta['NCOMBINE']
#plt.figure(figsize=(10, 8))
#image=ccd.data
##h = plt.hist(image.flatten(), bins=20000, align='mid', 
##             density=True, label="Dark frame");
#plot_dark_with_distributions(image, rn, 1E-4, exposure=exptime,
#                             show_poisson=True, show_gaussian=True,
#                             n_images=n_images)
##plt.xlim(-20, 30)
#plt.show()

#bias_combine('/data/io/IoIO/raw/20200711', show=True, gain_correct=True)
#bias_combine('/data/io/IoIO/raw/20200711', show=False, gain_correct=True)
#bias_combine('/data/io/IoIO/raw/20200711', show=True, gain_correct=False)
#bias_combine('/data/io/IoIO/raw/20200711', show=False, gain_correct=False)


#bias_combine('/data/io/IoIO/raw/20200421', show=True, gain_correct=True)
#bias_combine('/data/io/IoIO/raw/20200711', show=True, gain_correct=True)

#bias_combine('/data/io/IoIO/raw/20200821', show=True, gain_correct=True)
#bias_combine('/data/io/IoIO/raw/20201001', show=True, gain_correct=True)

#bias_combine('/data/io/IoIO/raw/20201129', show=True, gain_correct=True)


# bias_fname = '/data/io/IoIO/reduced/bias_dark/2020-07-11_ccdT_-5.3_combined_bias.fits'
# bias = ccddata_read(bias_fname)
# dark_dir = '/data/io/IoIO/raw/20200711'
# dark_combine(dark_dir, master_bias=bias, show=True)
# #dark_combine(dark_dir, master_bias=bias, show=False)
# 
# bias_fname = '/data/io/IoIO/reduced/bias_dark/2020-04-21_ccdT_-20.2_combined_bias.fits'
# bias = ccddata_read(bias_fname)
# dark_dir = '/data/io/IoIO/raw/20200421'
# dark_combine(dark_dir, master_bias=bias, show=True)
# 
# 
# bias_fname = '/data/io/IoIO/reduced/bias_dark/2020-08-21_ccdT_-10.2_combined_bias.fits'
# bias = ccddata_read(bias_fname)
# dark_dir = '/data/io/IoIO/raw/20200821'
# dark_combine(dark_dir, master_bias=bias, show=True)
# #dark_combine(dark_dir, master_bias=bias, show=False)
# 
# bias_fname = '/data/io/IoIO/reduced/bias_dark/2020-10-01_ccdT_-19.9_combined_bias.fits'
# bias = ccddata_read(bias_fname)
# dark_dir = '/data/io/IoIO/raw/20201001'
# dark_combine(dark_dir, master_bias=bias, show=True)
# #dark_combine(dark_dir, master_bias=bias, show=False)
# 
# bias_fname = '/data/io/IoIO/reduced/bias_dark/2020-11-29_ccdT_-30.2_combined_bias.fits'
# bias = ccddata_read(bias_fname)
# dark_dir = '/data/io/IoIO/raw/20201129'
# dark_combine(dark_dir, master_bias=bias, show=True)
# #dark_combine(dark_dir, master_bias=bias, show=False)


## 
## im = CCDData.read('/data/io/IoIO/reduced/bias_dark/2020-07-11_ccdT_-5.27_combined_bias.fits')
## data = im.data.copy()
## 
## im = im.uncertainty.array.copy()
## #im = data - im
#std =  np.std(im)
#med =  np.median(im)
##mean = np.asscalar(np.mean(im).data  )
#mean = np.mean(im)
#tmin = np.min(im)
#tmax = np.max(im)
#impl = plt.imshow(im, origin='lower', cmap=plt.cm.gray, filternorm=0,
#                  interpolation='none', vmin=med-std, vmax=med+std)
#im.dtype
#plt.show()
#plt.close()

#print(fname_by_imagetyp_ccdt_exp('/data/io/IoIO/raw/20201129', imagetyp='BIAS',
#                        glob_include=['Bias*', '*_bias.fit'], debug=True))

bname = '2020-10-01_ccdT_-10.3_combined_bias.fits'

#bias_dark_dir = '/data/io/IoIO/reduced/bias_dark'
#bnames = glob.glob(bias_dark_dir + '/*_combined_bias.fits')

#Ts = []
#dates = []
#for bname in bnames:
#    bbname = os.path.basename(bname)
#    sbname = bbname.split('_')
#    date = Time(sbname[0], format='fits')
#    T = float(sbname[2])
#    dates.append(date)
#    Ts.append(T)
#dates = np.asarray(dates)
#Ts = np.asarray(Ts)
#
#sample = '/data/io/IoIO/raw/20201001/Dark-S001-R003-C009-B1.fts'
#ccd = ccddata_read(sample)
#tm = Time(ccd.meta['DATE-OBS'], format='fits')
#ccdt = ccd.meta['CCD-TEMP']
#dTs = ccdt - Ts
#good_T_idx = np.flatnonzero(np.abs(dTs) < 2)
##good_T_idx = np.flatnonzero(np.abs(dTs) < 0.0000003)
#if len(good_T_idx) == 0:
#    print('yuck')
#ddates = tm - dates
#best_T_date_idx = np.argmin(ddates[good_T_idx])
## unwrap
#best_T_date_idx = good_T_idx[best_T_date_idx]
#print(bnames[best_T_date_idx])
#
#def time_to_date(time):
#    timestring = time.fits
#    return timestring.split('T')[0]
###############################


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
                log.debug(f'lockfile {self._fname} detected for {f.read()}')
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
                 max_ccddata_size=max_ccddata_size,
                 griddata_expansion_factor=griddata_expansion_factor,
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
        self.max_ccddata_size = max_ccddata_size
        self.griddata_expansion_factor = griddata_expansion_factor
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

        one_fdict_size = self.num_calibration_files * self.max_ccddata_size
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
        pwk = PoolWorkerKwargs(bias_combine,
                               subdirs=self._subdirs,
                               glob_include=self._bias_glob,
                               outdir=self._calibration_root,
                               gain_correct=self._gain_correct,
                               num_processes=self.num_processes,
                               max_ccddata_size=self.max_ccddata_size,
                               num_calibration_files=self.num_calibration_files,
                               mem_frac=self.mem_frac,
                               keep_intermediate=self.keep_intermediate)

        dirs = [dt[0] for dt in dirs_dates]
        if our_num_processes == 1:
            for d in dirs:
                pwk.worker(d)
        else:
            with NoDaemonPool(processes=our_num_processes) as p:
                p.map(pwk.worker, dirs)

        self.bias_table_create(rescan=True, autoreduce=False)
        # This could potentially get set in dirs_dates_to_reduce, but
        # it seems better to set it after we have actually done the work
        all_dirs_dates = get_dirs_dates(self._raw_data_root,
                                  start=self._start_date,
                                  stop=self._stop_date)
        self._bias_dirs_dates_checked = all_dirs_dates
        lock.clear()

        ## Make sure each directory we process has enough processes to
        ## work with to parallelize the slowest step (combining images)
        #our_num_processes = min(ndirs_dates,
        #                        int(self.num_processes / self.num_ccdts))
        #if ndirs_dates == 0:
        #    pass
        #elif ndirs_dates == 1 or our_num_processes == 1:
        #    for dt in dirs_dates:
        #        bias_combine(dt[0],
        #                     subdirs=self._subdirs,
        #                     glob_include=self._bias_glob,
        #                     outdir=self._calibration_root,
        #                     gain_correct=self._gain_correct,
        #                     num_processes=self.num_processes,
        #                     mem_frac=self.mem_frac,
        #                     keep_intermediate=self.keep_intermediate)
        #else:
        #    num_subprocesses = int(self.num_processes / our_num_processes)
        #    subprocess_mem_frac = self.mem_frac / our_num_processes
        #    log.debug(f'Calibration.reduce_bias: ndirs_dates = {ndirs_dates}')
        #    log.debug('Calibration.reduce_bias: self.num_processes = {}, our_num_processes = {}, num_subprocesses = {}, subprocess_mem_frac = {}'.format(self.num_processes, our_num_processes, num_subprocesses, subprocess_mem_frac))
        #    #return
        #
        #    plist = \
        #        [Process(target=bias_combine,
        #                 args=(dt[0],),
        #                 kwargs= {'subdirs': self._subdirs,
        #                          'glob_include': self._bias_glob,
        #                          'outdir': self._calibration_root,
        #                          'gain_correct': self._gain_correct,
        #                          'num_processes': num_subprocesses,
        #                          'mem_frac': subprocess_mem_frac},
        #                 daemon=False) # Let subprocesses create more children
        #         for dt in dirs_dates]
        #    p_limit(plist, our_num_processes)

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

        one_fdict_size = self.num_calibration_files * self.max_ccddata_size
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
        pwk = PoolWorkerKwargs(dark_combine,
                               subdirs=self._subdirs,
                               glob_include=self._dark_glob,
                               outdir=self._calibration_root,
                               calibration=self,
                               auto=True, # A little dangerous, but just one place for changes
                               num_processes=self.num_processes,
                               max_ccddata_size=self.max_ccddata_size,
                               num_calibration_files=self.num_calibration_files,
                               mem_frac=self.mem_frac,
                               keep_intermediate=self.keep_intermediate)

        dirs = [dt[0] for dt in dirs_dates]
        if our_num_processes == 1:
            for d in dirs:
                pwk.worker(d)
        else:
            with NoDaemonPool(processes=our_num_processes) as p:
                p.map(pwk.worker, dirs)

        self.dark_table_create(rescan=True, autoreduce=False)
        # This could potentially get set in dirs_dates_to_reduce, but
        # it seems better to set it after we have actually done the work
        all_dirs_dates = get_dirs_dates(self._raw_data_root,
                                  start=self._start_date,
                                  stop=self._stop_date)
        self._dark_dirs_dates_checked = all_dirs_dates
        lock.clear()


        #if ndirs_dates == 1 or our_num_processes == 1:
        #    for dt in dirs_dates:
        #        dark_combine(dt[0],
        #                     subdirs=self._subdirs,
        #                     glob_include=self._dark_glob,
        #                     outdir=self._calibration_root,
        #                     calibration=self,
        #                     auto=True, # A little dangerous, but just one place for changes
        #                     num_processes=self.num_processes,
        #                     mem_frac=self.mem_frac,
        #                     keep_intermediate=self.keep_intermediate)
        #else:
        #    num_subprocesses = int(self.num_processes / our_num_processes)
        #    subprocess_mem_frac = self.mem_frac / our_num_processes
        #
        #    plist = \
        #        [Process(target=dark_combine,
        #                 args=(dt[0],),
        #                 kwargs= {'subdirs': self._subdirs,
        #                          'glob_include': self._dark_glob,
        #                          'outdir': self._calibration_root,
        #                          'calibration': self,
        #                          'auto': True,
        #                          'num_processes': num_subprocesses,
        #                          'mem_frac': subprocess_mem_frac},
        #                 daemon=False) # Let subprocesses create more children
        #         for dt in dirs_dates]
        #    p_limit(plist, our_num_processes)
        #
        #self.dark_table_create(rescan=True, autoreduce=False)
        ## This could potentially get set in dirs_dates_to_reduce, but
        ## it seems better to set it after we have actually done the work
        #all_dirs_dates = get_dirs_dates(self._raw_data_root,
        #                          start=self._start_date,
        #                          stop=self._stop_date)
        #self._dark_dirs_dates_checked = all_dirs_dates
        #lock.clear()

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

        one_filt_size = max(self.num_calibration_files * self.max_ccddata_size,
                            max_ccddata_size * self.griddata_expansion_factor)
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
        pwk = PoolWorkerKwargs(flat_combine,
                               subdirs=self._subdirs,
                               glob_include=self._flat_glob,
                               outdir=self._calibration_root,
                               calibration=self,
                               auto=True, # A little dangerous, but just one place for changes
                               num_processes=self.num_processes,
                               mem_frac=self.mem_frac,
                               num_calibration_files=self.num_calibration_files,
                               max_ccddata_size=self.max_ccddata_size,
                               griddata_expansion_factor=self.griddata_expansion_factor,
                               keep_intermediate=self.keep_intermediate)
                               
        dirs = [dt[0] for dt in dirs_dates]
        if our_num_processes == 1:
            for d in dirs:
                pwk.worker(d)
        else:
            with NoDaemonPool(processes=our_num_processes) as p:
                p.map(pwk.worker, dirs)

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
        fnames = glob.glob(os.path.join(self._calibration_root, '*_combined_bias.fits'))
        if len(fnames) == 0:
            # Catch the not autoreduce case when we still have no files
            return None
        # If we made it here, we have files to populate our table
        dates = []
        ccdts = []
        for fname in fnames:
            bfname = os.path.basename(fname)
            sfname = bfname.split('_')
            date = Time(sfname[0], format='fits')
            ccdt = float(sfname[2])
            dates.append(date)
            ccdts.append(ccdt)
        self._bias_table = QTable([fnames, dates, ccdts],
                                  names=('fnames', 'dates', 'ccdts'),
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
        fnames = glob.glob(os.path.join(self._calibration_root, '*_combined_dark.fits'))
        if len(fnames) == 0:
            # Catch the not autoreduce case when we still have no files
            return None
        # If we made it here, we have files to populate our table
        dates = []
        ccdts = []
        exptimes = []
        for fname in fnames:
            bfname = os.path.basename(fname)
            sfname = bfname.split('_')
            date = Time(sfname[0], format='fits')
            ccdt = float(sfname[2])
            exptime = sfname[4]
            exptime = float(exptime[:-1])
            dates.append(date)
            ccdts.append(ccdt)
            exptimes.append(exptime)
        self._dark_table = \
            QTable([fnames, dates, ccdts, exptimes],
                   names=('fnames', 'dates', 'ccdts', 'exptimes'),
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
        for fname in fnames:
            bfname = os.path.basename(fname)
            sfname = bfname.split('_', 1)
            date = Time(sfname[0], format='fits')
            filttail = sfname[1]
            filt_tail = filttail.split('_flat.fits')
            filt = filt_tail[0]
            dates.append(date)
            filts.append(filt)
        self._flat_table = \
            QTable([fnames, dates, filts],
                   names=('fnames', 'dates', 'filters'),
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
        dccdts = ccdt - self.bias_table['ccdts']
        good_ccdt_idx = np.flatnonzero(np.abs(dccdts) < ccdt_tolerance)
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
        dccdts = ccdt - self.dark_table['ccdts']
        good_ccdt_idx = np.flatnonzero(np.abs(dccdts) < ccdt_tolerance)
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
        good_filt_idx = np.flatnonzero(filt == self.flat_table['filters'])
        if len(good_filt_idx) == 0:
            raise ValueError(f'No {filt} flats found')
        ddates = tm - self.flat_table['dates']
        best_filt_date_idx = np.argmin(np.abs(ddates[good_filt_idx]))
        # unwrap
        best_filt_date_idx = good_filt_idx[best_filt_date_idx]
        return self._flat_table['fnames'][best_filt_date_idx]



#sample = '/data/io/IoIO/raw/20201001/Dark-S001-R003-C009-B1.fts'

#bias_dark_dir = '/data/io/IoIO/reduced/bias_dark'
#c = Calibration(calibration_root=bias_dark_dir)
#print(c.best_bias(sample))
##print(c.dirs_to_reduce(c.bias_table_create, c._bias_glob))
#
#d = Calibration(start_date='2018-01-01', stop_date='2019-01-01')
#print(d.dirs_to_reduce(d.bias_table_create, c._bias_glob))
#
#d = Calibration(start_date='2017-01-01', stop_date='2018-01-01')
#print(d.dirs_to_reduce(d.bias_table_create, c._bias_glob))

#c = Calibration(start_date='2020-07-01', stop_date='2020-08-01')
#print(c.dirs_to_reduce(c.bias_table_create, c._bias_glob))


#gi = ['Bias*', '*_bias.fit', 'Dark*', '*_dark.fit', '*Flat*']
#print(dir_has_calibration('/data/io/IoIO/raw/20200711', gi))
#print(dir_has_calibration('/data/io/IoIO/raw/20200712', gi))
#print(dir_has_calibration('/data/io/IoIO/raw/202007122', gi))

#c = Calibration(start_date='2020-07-11', stop_date='2020-07-12')
#print(c.dirs_to_reduce(c.bias_table_create, c._bias_glob))
#print(c.best_bias(sample))

#bias_combine('/data/io/IoIO/raw/20200711', show=True, gain_correct=True)
#bias_combine('/data/io/IoIO/raw/20200821', show=True, gain_correct=True)

#c = Calibration(start_date='2020-07-11', stop_date='2020-07-12')
#sample = '/data/io/IoIO/raw/20200711/Dark-S005-R001-C009-B1.fts'
#print(c.dirs_to_reduce(c.bias_table_create, c._bias_glob))
#print(c.best_bias(sample))
#fname = sample
#raw = ccddata_read(fname)
#ccd = ccd_process(raw, oscan=True, master_bias=True,
#                  gain=True, error=True, calibration=c)
#ccd.write('/tmp/test.fits', overwrite=True)

#c = Calibration(start_date='2020-07-11', stop_date='2020-07-12')
#dark_dir = '/data/io/IoIO/raw/20200711'
#dark_combine(dark_dir, master_bias=True, calibration=c,
#             show=True)

#sample = '/data/io/IoIO/raw/20200711/Dark-S005-R001-C009-B1.fts'
#c = Calibration(start_date='2020-07-11', stop_date='2020-07-12')
#print(c.best_bias(sample))

#sample = '/data/io/IoIO/raw/20200711/Dark-S005-R001-C009-B1.fts'
#c = Calibration(start_date='2020-07-01', stop_date='2020-09-01')
#print(c.best_bias(sample))

#c = Calibration(start_date='2020-07-11', stop_date='2020-07-12')
#print(c.dirs_to_reduce(c.bias_table_create, c._bias_glob))
#t = c.bias_table_create(autoreduce=False)

#c = Calibration(start_date='2020-07-11', stop_date='2020-07-12', reduce=True)
#c = Calibration(start_date='2020-08-21', stop_date='2020-08-22', reduce=True)

#c = Calibration(start_date='2020-07-11', stop_date='2020-08-22', reduce=True)

#bias_combine('/data/io/IoIO/raw/20200818', show=True, gain_correct=True)
#print(fname_by_imagetyp_ccdt_exp('/data/io/IoIO/raw/20200818',
#                                 imagetyp='BIAS',
#                                 debug=True))
#print(fname_by_imagetyp_ccdt_exp('/data/io/IoIO/raw/20200711',
#                                 imagetyp='BIAS',
#                                 debug=True))

#sample = '/data/io/IoIO/raw/20200711/Dark-S005-R008-C001-B1.fts'
#c = Calibration(start_date='2020-07-11', stop_date='2020-08-22')
#ccd = ccd_process(sample, calibration=c, oscan=True, master_bias=True,
#                  gain=True, error=True)
#ccd.write('/tmp/test.fits', overwrite=True)

#bias_combine('/data/io/IoIO/raw/20200711', show=False, gain_correct=True)

#c = Calibration(start_date='2020-07-11', stop_date='2020-08-22')
##print(c.dirs_dates_to_reduce(c.bias_table_create, c._bias_glob))
#
#print('DIRS TO REDUCE PRE')
#print(c.dirs_dates_to_reduce(c.bias_table_create,
#                             c._bias_glob,
#                             c._bias_dirs_dates_checked,
#                             c._subdirs))
#c.reduce()
#print('DIRS TO REDUCE POST')
#print(c.dirs_dates_to_reduce(c.bias_table_create,
#                             c._bias_glob,
#                             c._bias_dirs_dates_checked,
#                             c._subdirs))
#

#print('MADE IT HERE')
#print(c._bias_dirs_dates_checked)

#l = Lockfile(lockfile)
#l.create()
#


#c = Calibration(start_date='2020-07-11', stop_date='2020-08-22')
#sample = '/data/io/IoIO/raw/20200711/Dark-S005-R008-C001-B1.fts'
#ccd = ccd_process(sample, calibration=c, oscan=True, master_bias=True,
#                  gain=True, error=True)
#ccd.write('/tmp/test.fits', overwrite=True)
#
#print('SECOND TRY')
#ccd = ccd_process(sample, calibration=c, oscan=True, master_bias=True,
#                  gain=True, error=True)
#ccd.write('/tmp/test.fits', overwrite=True)

#c = Calibration(start_date='2020-07-11', stop_date='2020-08-22')
#sample = '/data/io/IoIO/raw/20200711/Dark-S005-R008-C001-B1.fts'
##ccd.write('/tmp/test.fits', overwrite=True)
#
#print(c.best_bias(sample))

#c = Calibration(start_date='2020-07-11', stop_date='2020-08-22')
#dark_dir = '/data/io/IoIO/raw/20200711'
##dark_combine(dark_dir, calibration=c, master_bias=True, show=True)
##dark_combine(dark_dir, calibration=c, master_bias=True, show=True, num_processes=1)
#dark_combine(dark_dir, calibration=c, master_bias=True, show=True)

#c = Calibration(start_date='2020-07-11', stop_date='2020-08-22', reduce=True)
#dark_dir = '/data/io/IoIO/raw/20200711'
##dark_combine(dark_dir, calibration=c, master_bias=True, show=True)
#dark_combine(dark_dir, calibration=c, master_bias=True, show=False)

#bsample = '/data/io/IoIO/raw/20200711/Bias-S005-R002-C001-B1.fts'
#b = ccd_process(bsample)
#ccd = subtract_overscan(bsample, overscan=100000)
#b = ccd_process(bsample, overscan=100000)
#b = ccd_process(bsample, oscan=100000)
#print(os.path.basename(bsample))

#fnames = glob.glob('/data/io/IoIO/raw/20200711/Bias*')
#print(ccd_process_multi(fnames, '/tmp'))

#bsample = '/data/io/IoIO/raw/20200711/Bias-S005-R002-C001-B1.fts'
#b = ccd_process(bsample, outname='/tmp/test.fits', overwrite=True)
#b = ccd_process(bsample, outdir='/tmp', outname='test.fits', overwrite=True)
#b = ccd_process(bsample, outdir='/tmp', overwrite=True)
#b = ccd_process(bsample, outdir='/tmp')
#b = ccd_process(bsample, oscan=True, outdir='/tmp', overwrite=True)
#b = ccd_process(bsample, oscan=True, gain=True, master_bias=True,
#                outdir='/tmp', overwrite=True)

#bsample = '/data/io/IoIO/raw/20200711/Bias-S005-R002-C001-B1.fts'
#c = Calibration(start_date='2020-07-11', stop_date='2020-08-22', reduce=True)
#b = ccd_process(bsample, calibration=c, oscan=True, gain=True, master_bias=True,
#                outdir='/tmp', overwrite=True)
#b = ccd_process(bsample, calibration=c, auto=True,
#                outdir='/tmp', overwrite=True)
#b = ccd_process(bsample, calibration=c, outdir='/tmp', overwrite=True)
#b = ccd_process(bsample, calibration=c, auto=True, outdir='/tmp', overwrite=True)

#c = Calibration(start_date='2020-07-11', stop_date='2020-08-22', reduce=True)
#fnames = glob.glob('/data/io/IoIO/raw/20200711/Bias*')
#outnames = ccd_process_multi(fnames, calibration=c, auto=True,
#                             outdir='/tmp', overwrite=True,
#                             return_outname=True)
#

#bsample = '/data/io/IoIO/raw/20200711/Bias-S005-R002-C001-B1.fts'
#print(bias_subtract_one_median(bsample, 10000,
#                               outdir='/tmp'))


#dark_dir = '/data/io/IoIO/raw/20200711'
#c = Calibration(start_date='2020-07-11', stop_date='2020-08-22', reduce=True)
#dark_combine(dark_dir, calibration=c, auto=True, create_outdir=True, show=False)

#c = Calibration(start_date='2020-07-11', stop_date='2020-08-22', reduce=True)
#dsample = '/data/io/IoIO/raw/20200711/Dark-S005-R004-C003-B1.fts'
#print(create_outname(dsample, outdir=calibration_scratch, create_outdir=True))
#dfd = dark_process_one_file(dsample, calibration_scratch='/tmp', calibration=c, auto=True, create_outdir=True, show=False)

#dark_dir = '/data/io/IoIO/raw/20200711'
#c = Calibration(start_date='2020-07-11', stop_date='2020-08-22', reduce=True)
#dark_combine(dark_dir, calibration=c, auto=True, create_outdir=True, show=False)

#dsample = '/data/io/IoIO/raw/20200711/Dark-S005-R004-C003-B1.fts'
##c = Calibration(start_date='2020-07-11', stop_date='2020-07-12')
#c = Calibration(start_date='2020-07-11', stop_date='2020-08-22')
##t = c.bias_table_create(autoreduce=False)
#t = c.dark_table_create(autoreduce=False)
#print(c.best_bias(dsample))
#print(c.best_dark(dsample))

#c = Calibration(start_date='2020-07-11', stop_date='2020-08-22')
#fname = '/data/io/IoIO/raw/2020-07-07/Sky_Flat-0002_R.fit'
#f = ccd_process(fname, calibration=c, auto=True)
#f.write('/tmp/test.fits')

#print(overscan_estimate('/tmp/test.fits'))
#fsample = '/data/io/IoIO/raw/2020-07-07/Sky_Flat-0002_R.fit'
#print(overscan_estimate(fsample)
#c = Calibration(start_date='2020-07-11', stop_date='2020-08-22')
#bias = c.best_bias(fsample)
#print(overscan_estimate(fsample, master_bias=bias))

#bsample = '/data/io/IoIO/raw/20200711/Bias-S005-R002-C001-B1.fts'
#bsample1 = '/data/io/IoIO/raw/20200711/Bias-S005-R001-C001-B1.fts'
#print(overscan_estimate(bsample))
#ccd = ccddata_read(bsample)
#print(overscan_estimate(bsample, meta=ccd.meta))
#ccd = subtract_overscan(bsample)
#b = ccd_process(bsample)
#b = ccd_process(bsample, auto=True,
#                filter_func_list=[not_full_frame, light_image])
#b, m = ccd_process(bsample, auto=True,
#                   filter_func_list=[not_full_frame, light_image],
#                   meta_process_list=[bias_dataframe, jd_meta])
#outlist = cor_process([bsample, bsample1], auto=True,
#                      filter_func_list=[not_full_frame, light_image],
#                      meta_process_list=[bias_dataframe, jd_meta])

#bsample = ccddata_read('/data/io/IoIO/raw/20200711/Bias-S005-R002-C001-B1.fts')
#bsample1 = ccddata_read('/data/io/IoIO/raw/20200711/Bias-S005-R001-C001-B1.fts')
#c = Calibration(start_date='2020-07-11', stop_date='2020-08-22')
#ccd = cor_process(bsample, calibration=c, auto=True)
#ccd = cor_process(bsample, calibration=c, auto=True, gain=True)
#ccd = ccd_pre_process(bsample, ccd_processor=cor_process,
#                      calibration=c, auto=True, gain=True,
#                      pre_process_list=[full_frame, light_image])
#ccd, meta = ccd_post_process(bsample, ccd_processor=cor_process,
#                             calibration=c, auto=True, gain=True,
#                             pre_process_list=[full_frame, light_image])
#ccd, meta = ccd_post_process(bsample, ccd_processor=cor_process,
#                             calibration=c, auto=True, gain=True,
#                             pre_process_list=[full_frame, light_image],
#                             post_process_list=[bias_dataframe])
#ccd, meta = ccd_post_process(bsample, ccd_processor=cor_process,
#                             calibration=c, auto=True, gain=True,
#                             pre_process_list=[full_frame, light_image],
#                             post_process_list=[bias_dataframe, jd_meta])
#bfname = '/data/io/IoIO/raw/20200711/Bias-S005-R002-C001-B1.fts'
#bfname1 = '/data/io/IoIO/raw/20200711/Bias-S005-R001-C001-B1.fts'
#pout = ccd_pipeline([bfname, bfname1], ccd_processor=cor_process,
#                    calibration=c, auto=True, gain=True,
#                    pre_process_list=[full_frame, light_image],
#                    post_process_list=[bias_dataframe, jd_meta],
#                    outdir='/tmp')
#pout = ccd_pipeline([bfname, bfname1], ccd_processor=cor_process,
#                    calibration=c, auto=True, gain=True,
#                    pre_process_list=[full_frame, light_image],
#                    post_process_list=[bias_dataframe, jd_meta],
#                    outdir='/tmp', overwrite=True)
#onames, ometa = zip(*pout)

#bias_combine('/data/io/IoIO/raw/20200711', show=True, gain_correct=True,
#                        keep_intermediate=True)
#bias_combine('/data/io/IoIO/raw/20200711', show=False, gain_correct=True,
#                        keep_intermediate=True)
#bias_combine('/data/io/IoIO/raw/20200711', show=False, gain_correct=True)
#bias_combine('/data/io/IoIO/raw/20200711', show=True, gain_correct=True)

#dark_dir = '/data/io/IoIO/raw/20200711'
#c = Calibration(start_date='2020-07-11', stop_date='2020-08-22', reduce=True)
#
#dark_combine(dark_dir, calibration=c, auto=True, show=False)

#bias_fname = '/data/io/IoIO/reduced/bias_dark/2020-07-11_ccdT_-5.3_combined_bias.fits'
#master_bias = ccddata_read(bias_fname)
#dsample = '/data/io/IoIO/raw/20200711/Dark-S005-R003-C010-B1.fts'
#meta = master_bias.meta.copy()
#o = overscan_estimate(dsample, master_bias=master_bias, meta=meta)
##o = overscan_estimate(dsample, master_bias=bias_fname, meta=meta)
#master_bias.meta = meta
##master_bias.write('/tmp/test.fits', overwrite=True)

#dark_dir = '/data/io/IoIO/raw/20200711'
#c = Calibration(start_date='2020-07-11', stop_date='2020-08-22', reduce=True)
#
#dark_combine(dark_dir, calibration=c, auto=True, show=False)

#c = Calibration(start_date='2020-07-11', stop_date='2020-08-22', reduce=True)
#fsample = '/data/io/IoIO/raw/2020-07-07/Sky_Flat-0002_R.fit'
#pout = cor_pipeline([fsample], auto=True, calibration=c)

#flat_dir = '/data/io/IoIO/raw/2020-07-07'
#collection = ccdp.ImageFileCollection(flat_dir)
#flat_combine_one_filt('R', collection=collection, outdir='/tmp', keep_intermediate=True, calibration=c, auto=True)

#flat_combine(flat_dir, calibration=c, auto=True, show=False)

#print(num_can_process(5000, process_size=50000E99))

#c = Calibration(start_date='2020-07-07', stop_date='2020-08-22', reduce=True)
#c = Calibration(start_date='2020-07-07', stop_date='2020-08-22')
#fname = '/data/io/IoIO/raw/2020-07-08/NEOWISE-0007_Na-on.fit'
#print(c.best_flat(fname))

#print(c.dirs_dates_to_reduce(c.flat_table_create,
#                             c._flat_glob,
#                             c._flat_dirs_dates_checked,
#                             c._subdirs))

#print(c.flat_table_create( autoreduce=False))

#c.reduce_flat()

c = Calibration(start_date='2020-07-07', stop_date='2020-08-22', reduce=True)
fname = '/data/io/IoIO/raw/2020-07-08/NEOWISE-0007_Na-on.fit'
ccd = ccddata_read(fname)
ccd = cor_process(ccd, calibration=c, auto=True)
ccd.write('/tmp/test.fits', overwrite=True)

