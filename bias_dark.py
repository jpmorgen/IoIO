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
import csv

# For NestablePool
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
# stored in the RDNOISE keyword.  This is used as a sanity check 
sx694_example_readnoise = 15.475665 * sx694_gain
sx694_example_readnoise_comment = '2018-07-10 readnoise (electron)'

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

# Processing global variables.  Since I avoid use of the global
# statement and don't reassign these at global scope, they stick to
# these values and provide handy defaults for routines and object
# inits.


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

# These are use to optimize parallelization until such time as
# ccdproc.combiner can be parallelized
num_ccdts = int((35 - (-10)) / 5)
num_dark_exptimes = 8
num_filts = 9


data_root = '/data/io/IoIO'
raw_data_root = os.path.join(data_root, 'raw')
reduced_root = os.path.join(data_root, 'reduced')
calibration_root = os.path.join(reduced_root, 'Calibration')
calibration_scratch = os.path.join(calibration_root, 'scratch')

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
flat_glob = ['*Flat*']

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
class NestablePool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess


def get_dirs_dates(directory,
                   filt_list=None,
                   start=None,
                   stop=None):
    """Returns list of tuples (subdir, date) sorted by date.  Handles two
    cases of directory date formatting YYYYMMDD (ACP) and YYYY-MM-DD
    (MaxIm)

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

def ccd_metadata(ccd,
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
    """Record [SX694] CCD metadata for all the reductions"""
    if ccd.meta.get('camera') is not None:
        # We have been here before, so exit quietly
        return
    # Clean up double exposure time reference to avoid confusion
    if ccd.meta.get('exposure') is not None:
        del ccd.meta['EXPOSURE']
    ccd.meta.insert('INSTRUME',
                    ('CAMERA', camera_description),
                    after=True)
    ccd.meta['GAIN'] = (gain, gain_comment)
    # This gets used in ccdp.cosmicray_lacosmic
    ccd.meta['SATLEVEL'] = (satlevel, satlevel_comment)
    # This is where the CCD starts to become non-linear and is
    # used for things like rejecting flats recorded when
    # conditions were too bright
    ccd.meta['NONLIN'] = (nonlin, nonlin_comment)
    ccd.meta['RDNOISE'] = (readnoise, readnoise_comment)

def ccddata_read(fname_or_ccd, add_metadata=False, *args, **kwargs):
    """Convenience function to read a FITS file into a CCDData object.  

    Catches the case where the raw FITS file does not have a BUNIT
    keyword, which otherwise causes CCDData.read to crash.  In this
    case, ccddata_read assumes raw data are in units of ADU.  Optionally 
    supplements metadata with externally measured quantities such as
    gain, nonlinearity level, and readnoise

    Adds following FITS card if no BUNIT keyword present in metadata
        BUNIT = 'ADU' / physical units of the array values

    Parameters
    ----------
    fname_or_ccd : str or `~astropy.nddata.CCDData`

        If str, assumed to be a filename, which is read into a
        CCDData.  If ccddata, simply return the CCDData with BUNIT
        keyword possibly added

    add_metadata : bool
        If True, add [SX694] metadata.  Default: False
        
    Returns
    -------
    ccd : `~astropy.nddata.CCDData`
        CCDData with units set to ADU if none had been specified

    """
    if isinstance(fname_or_ccd, str):
        try:
            # This SOMETIMES fails if no units are specified
            ccd = CCDData.read(fname_or_ccd, *args, **kwargs)
        except Exception as e: 
            ccd = CCDData.read(fname_or_ccd, *args, unit="adu", **kwargs)
    else:
        ccd = fname_or_ccd
    assert isinstance(ccd, CCDData)
    if ccd.unit is None:
        log.warning('ccddata_read: CCDData.read read a file and did not assign any units to it.  Not sure why.  Setting units to ADU')
        ccd.unit = u.adu
    if ccd.meta.get('BUNIT') is None and ccd.unit is u.adu:
        # This comment is from the official FITS definition.  Not sure
        # why astropy stuff doesn't write it.  BUNIT is in the same
        # family is BZERO and BSCALE
        # https://heasarc.gsfc.nasa.gov/docs/fcg/standard_dict.html
        ccd.meta['BUNIT'] = ('ADU', 'physical units of the array values')
    if add_metadata:
        ccd_metadata(ccd, **kwargs)
    return ccd

def full_frame(im,
               naxis1=sx694_naxis1,
               naxis2=sx694_naxis2):
    """Returns true if image is a full frame, as defined by naxis1 and naxis2.

    Helps spot binned and guider images"""
    
    s = im.shape
    # Note Pythonic C index ordering
    if s != (naxis2, naxis1):
        return False
    return True    

def light_image(im, tolerance=3):
    
    """Returns True if light detected in image"""
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
    if (np.median(light_patch) - np.median(dark_patch) > tolerance):
        log.debug('light, dark patch medians ({:.4f}, {:.4f})'.format(mdp, mlp))
        return True
    return False

def mask_above(ccd, key, margin=0.1):
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
    return n_masked
    
#def mask_saturated(ccd):
#    """Record the number of saturated pixels in image metadata and log a warning if > 0"""
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

def fname_by_imagetyp_ccdt_exp(directory=None,
                               collection=None,
                               subdirs=None,
                               imagetyp=None,
                               glob_include=None,
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
        mean_t = np.mean(these_ts)
        # Create a new summary Table that inlcudes just these Ts
        narrow_to_t = narrow_to_imagetyp[tslices[it]:tslices[it+1]]
        exps = narrow_to_t['exptime']
        # These are sorted by increasing exposure time
        ues = np.unique(exps)
        for ue in ues:
            exp_idx = np.flatnonzero(exps == ue)
            files = narrow_to_t['file'][exp_idx]
            full_files = [os.path.join(collection.location, f) for f in files]
            fdict_list.append({'T': mean_t,
                               'EXPTIME': ue,
                               'fnames': full_files})
    return fdict_list

def bias_dataframe(fname, gain):
    """Worker routine to enable parallel processing of time-consuming
    matrix calculation"""

    ccd = ccddata_read(fname, add_metadata=True)
    if not full_frame(ccd):
        log.debug('bias wrong shape: ' + fname)
        return {'good': False}
    if light_image(ccd):
        log.debug('bias recorded during light conditions: ' +
                  fname)
        return {'good': False}
    im = ccd.data
    # Create uncertainty image
    diffs2 = (im[1:] - im[0:-1])**2
    rdnoise = np.sqrt(biweight_location(diffs2))
    uncertainty = np.multiply(rdnoise, np.ones(im.shape))
    ccd.uncertainty = StdDevUncertainty(uncertainty)
    # Prepare to create a pandas data frame to track relevant
    # quantities
    tm = Time(ccd.meta['DATE-OBS'], format='fits')
    ccdt = ccd.meta['CCD-TEMP']
    tt = tm.tt.datetime
    dataframe = {'time': tt,
                 'ccdt': ccdt,
                 'median': np.median(im),
                 'mean': np.mean(im),
                 'std': np.std(im)*gain,
                 'rdnoise': rdnoise*gain,
                 'min': np.min(im),  
                 'max': np.max(im)}
    return {'good': True,
            'fname': fname,
            'dataframe': dataframe,
            'jd': tm.jd}

def bias_combine_one_fdict(fdict,
                           directory,
                           outdir=calibration_root,
                           calibration_scratch=calibration_scratch,
                           keep_intermediate=False,
                           show=False,
                           min_num_biases=min_num_biases,
                           dccdt_tolerance=dccdt_tolerance,
                           num_processes=max_num_processes,
                           mem_frac=max_mem_frac,
                           camera_description=sx694_camera_description,
                           gain=sx694_gain,
                           satlevel=sx694_satlevel,
                           readnoise=sx694_example_readnoise,
                           readnoise_tolerance=0.5, # units of readnoise (e.g., electrons)
                           gain_correct=False):

    """Worker that allows the parallelization of calibrations taken at one
    temperature


    gain_correct : Boolean
        Effects unit of stored images.  True: units of electron.
        False: unit of ADU.  Default: False
    """

    # Parallelize collection of stats.  Make sure we don't read in
    # too many files for our instantaneous memory available.  Also
    # make sure we don't use too many processors.  Using the
    # global max_ccddata_size assures this is a reaosnably
    # conservative calculation, though I haven't check the details
    # of intermediate calculations
    num_files = len(fdict['fnames'])
    mean_t = fdict['T']
    if num_files < min_num_biases:
        log.debug(f'Not enough good biases found at CCDT = {mean_t} C in {directory}')
        return
    mem = psutil.virtual_memory()
    num_files_can_fit = \
        int(min(num_files,
                mem.available*mem_frac/max_ccddata_size))
    num_can_process = min(num_processes, num_files_can_fit)
    #print('bias_combine_one_fdict: num_processes = {}, mem_frac = {}, num_files= {}, num_files_can_fit = {}, num_can_process = {}'.format(num_processes, mem_frac, num_files, num_files_can_fit, num_can_process))
    gains = [gain] * num_files
    try:
        with Pool(processes=num_can_process) as p:
            dfdlist = p.starmap(bias_dataframe,
                                zip(fdict['fnames'], gains))
    except Exception as e:
        log.debug('Single-process mode: ' + str(e))
        dfdlist = [bias_dataframe(d, gain) for d in fdict['fnames']]

    # Unpack        
    good_fnames = [dfd['fname'] for dfd in dfdlist if dfd['good']]
    stats = [dfd['dataframe'] for dfd in dfdlist if dfd['good']]
    jds = [dfd['jd'] for dfd in dfdlist if dfd['good']]

    if len(stats) < min_num_biases:
        log.debug(f'Not enough good biases {len(stats)} found at CCDT = {mean_t} C in {directory}')
        return
    df = pd.DataFrame(stats)
    tm = Time(np.mean(jds), format='jd')
    this_date = tm.fits
    this_dateb = this_date.split('T')[0]
    this_ccdt = '{:.1f}'.format(mean_t)
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
        return

    # "Overscan subtract."  Go through each image and subtract the
    # median, since that value wanders as a function of ambient
    # (not CCD) temperature.  This leaves just the bias pattern.
    # To use ccd.subtract, unit must match type of array.  Note
    # that we need to save our images to disk to avoid
    # overwhelming memory when we have lots of biases

    # Make a date subdirectory in our calibration scratch dir
    sdir = os.path.join(calibration_scratch, this_dateb)
    os.makedirs(sdir, exist_ok=True)
    os_fnames = []
    for fname, m in zip(good_fnames, medians):
        ccd = ccddata_read(fname, add_metadata=True)
        ccd = ccd.subtract(m*u.adu, handle_meta='first_found')
        # Get our filename only, hack off extension, add _os.fits
        os_fname = os.path.split(fname)[1]
        os_fname = os.path.splitext(os_fname)[0] + '_os.fits'
        os_fname = os.path.join(sdir, os_fname)
        ccd.write(os_fname, overwrite=True)
        os_fnames.append(os_fname)

    ### Use ccdproc Combiner object iteratively as per example to
    ### mask out bad pixels
    ##combiner = ccdp.Combiner(os_fnames)
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

    # Use ccdp.combine since it enables memory management by
    # breaking up images to smaller chunks (better than throwing
    # images away).  --> eventually it would be great to
    # parallelize this primitive, since it is very slow
    mem = psutil.virtual_memory()
    im = \
        ccdp.combine(os_fnames,
                     method='average',
                     sigma_clip=True,
                     sigma_clip_low_thresh=5,
                     sigma_clip_high_thresh=5,
                     sigma_clip_func=np.ma.median,
                     sigma_clip_dev_func=mad_std,
                     mem_limit=mem.available*mem_frac)
    ccd_metadata(im)
    if gain_correct:
        im = ccdp.gain_correct(im, gain*u.electron/u.adu)
        im_gain = 1
    else:
        im_gain = gain
    mask_above(im, 'SATLEVEL')
    mask_above(im, 'NONLIN')
        
    # Collect image metadata.  For some reason, masked pixels
    # aren't ignored by std, etc. even though they output masked
    # arrays (which is annoying in its own right -- see example
    # commented mean).  So just create a new array, if needed, and
    # only put into it the good pixels
    if im.mask is None:
        tim = im
    else:
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
    im.meta['CCD-TEMP'] = (mean_t, 'Average CCD temperature for combined biases')
    im.meta['RDNOISE'] = (av_rdnoise, 'Measured readnoise (electron)')
    im.meta['STD'] = (std, 'Standard deviation of image (electron)')
    im.meta['MEDIAN'] = (med, 'Median of image (electron)')
    im.meta['MEAN'] = (mean, 'Mean of image (electron)')
    im.meta['MIN'] = (tmin, 'Min of image (electron)')
    im.meta['MAX'] = (tmax, 'Max of image (electron)')
    im.meta['HIERARCH OVERSCAN_VALUE'] = (overscan, 'Average of raw bias medians (ADU)')
    im.meta['HIERARCH SUBTRACT_OVERSCAN'] = (True, 'Overscan has been subtracted')
    im.meta['NCOMBINE'] = (len(good_fnames), 'Number of biases combined')
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
        for f in os_fnames:
            try:
                os.remove(f)
            except Exception as e:
                log.debug(f'Remove {f} failed: ' + str(e))
        try:
            os.rmdir(sdir)
        except Exception as e:
            log.debug(f'Remove {calibration_scratch} failed: ' + str(e))
        try:
            os.rmdir(calibration_scratch)
        except Exception as e:
            log.debug(f'Remove {calibration_scratch} failed: ' + str(e))
                


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
    our_num_processes = min(nfdicts, num_processes)
    log.debug(f'bias_combine: {directory}, nfdicts = {nfdicts}, our_num_processes = {our_num_processes}')
    if nfdicts == 0:
        log.debug('No biases found in: ' + directory)
        return False
    elif nfdicts == 1 or our_num_processes == 1:
        for fdict in fdict_list:
            bias_combine_one_fdict(fdict, directory,
                                   num_processes=num_processes,
                                   mem_frac=mem_frac,
                                   **kwargs)
    else:
        # Number of sub-processes in each process we will spawn
        num_subprocesses = int(num_processes / our_num_processes)
        # Similarly, the memory fraction for each process we will spawn
        subprocess_mem_frac = mem_frac / our_num_processes
        log.debug('bias_combine: {} num_processes = {}, mem_frac = {}, our_num_processes = {}, num_subprocesses = {}, subprocess_mem_frac = {}'.format(directory, num_processes, mem_frac, our_num_processes, num_subprocesses, subprocess_mem_frac))
        # Initiate all our processes
        plist = [Process(target=bias_combine_one_fdict,
                         args=(fdict, directory),
                         kwargs={"num_processes": num_subprocesses,
                                 "mem_frac": subprocess_mem_frac,
                                 **kwargs},
                         daemon=False) # Let subprocesses create more children
                 for fdict in fdict_list]
        p_limit(plist, num_processes)
            

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
                      readnoise=sx694_example_readnoise,
                      gain=sx694_gain, binsize=None,
                      min_width=1, max_width=8, box_size=100,
                      min_hist_val=10,
                      show=False, *args, **kwargs):
    """Estimate overscan in ADU in the absense of a formal overscan region

    Uses the minimum of: (1) the first peak in the histogram of the
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
        referece to metadata of ccd into which to write OVERSCAN_* cards

    master_bias : `~astropy.nddata.CCDData`, filename, or None
        Bias to subtract from ccd before estimate is calculated.
        Improves accruacy by removing bias ramp.  Bias can be in units
        of ADU or electrons and is converted using the specified gain.
        If bias has already been subtracted, this step will be skipped
        but the bias header will be used to extract readnoise and gain
        using the *_key keywords.  Default is ``None``.

    readnoise : float
        If bias supplied, its value of the RDNOISE keyword is used
        Default = ``sx694_example_readnoise``.

    gain :  float
        If bias supplied, its value of the GAIN keyword is used
        Default = ``sx_gain``.

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
    # Originally in IoIO.py as back_level
    ccd = ccddata_read(ccd_in, add_metadata=True)
    if meta is None:
        meta = ccd.meta
    # For now don't get fancy with unit conversion
    assert ccd.unit is u.adu
    if master_bias is None:
        bias = None
    elif isinstance(master_bias, CCDData):
        # Make a copy because we are going to mess with it
        bias = master_bias.copy()
    else:
        bias = ccddata_read(master_bias)
    if isinstance(bias, CCDData):
        readnoise = bias.meta['RDNOISE']
        gain = bias.meta['GAIN']
    if isinstance(bias, CCDData):
        # Make sure bias hasn't been subtracted before
        if ccd.meta.get('subtract_bias') is None:
            if bias.unit is u.electron:
                # Convert bias back to ADU for subtraction, if needed
                bias = bias.divide(gain*u.electron/u.adu)
            ccd = ccdp.subtract_bias(ccd, bias)
    elif ccd.meta.get('subtract_bias') is None:
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

def subtract_overscan(ccd_in, *args, **kwargs):
    """Subtract overscan for IoIO coronagraph and add CCD metadata.
    Overscan value is subtracted from the SATLEVEL keyword

    This is a wrapper around overscan_estimate in case I want to make
    overscan estimation more complicated by linking files within a
    directory.  Note: ccdproc's native subtract_overscan function can't be
    used because it assumes the overscan region is specified by a
    simple rectangle.

    All processing except bias_combine needs to run through this
    point, so this is a good place to add our common metadata

    """
    ccd = ccddata_read(ccd_in, add_metadata=True)
    overscan = overscan_estimate(ccd, meta=ccd.meta, *args, **kwargs)
    ccd = ccd.subtract(overscan*u.adu, handle_meta='first_found')
    ccd.meta['HIERARCH OVERSCAN_VALUE'] = (overscan, 'overscan value subtracted (ADU)')
    ccd.meta['HIERARCH SUBTRACT_OVERSCAN'] \
        = (True, 'Overscan has been subtracted')
    # Keep track of our precise saturation level
    satlevel = ccd.meta['SATLEVEL']
    satlevel -= overscan
    ccd.meta['SATLEVEL'] = satlevel # still in ADU
    ccd.meta
    return ccd

# Copy and tweak ccdp.ccd_process
def ccd_process(fname_or_ccd, calibration=None,
                oscan=None, trim=None, error=False, master_bias=None,
                dark_frame=None, master_flat=None, bad_pixel_mask=None,
                gain=None, readnoise=None, oscan_median=True, oscan_model=None,
                min_value=None, dark_exposure=None, data_exposure=None,
                exposure_key=None, exposure_unit=None,
                dark_scale=False, gain_corrected=True,
                *args, **kwargs):
#def ccd_process(fname_or_ccd, calibration=None,
#                oscan=None, error=False, master_bias=None,
#                gain=None, gain_corrected=True, readnoise=None,
#                dark_frame=None, master_flat=None, bad_pixel_mask=None,
#                *args, **kwargs):
    """Perform basic processing on IoIO ccd data.  Uses ccd_process
    for all steps except overscan subtraction

    The following steps can be included:

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
        Frame to be reduced.

    oscan : `~astropy.nddata.CCDData`, str or None, optional
        For no overscan correction, set to None. Otherwise provide a region
        of ccd from which the overscan is extracted, using the FITS
        conventions for index order and index start, or a
        slice from ccd that contains the overscan.
        Default is ``None``.

    trim : str or None, optional
        For no trim correction, set to None. Otherwise provide a region
        of ccd from which the image should be trimmed, using the FITS
        conventions for index order and index start.
        Default is ``None``.

    error : bool, optional
        If True, create an uncertainty array for ccd.
        Default is ``False``.

    master_bias : `~astropy.nddata.CCDData` or None, optional
        A master bias frame to be subtracted from ccd. The unit of the
        master bias frame should match the unit of the image **after
        gain correction** if ``gain_corrected`` is True.
        Default is ``None``.

    dark_frame : `~astropy.nddata.CCDData` or None, optional
        A dark frame to be subtracted from the ccd. The unit of the
        master dark frame should match the unit of the image **after
        gain correction** if ``gain_corrected`` is True.
        Default is ``None``.

    master_flat : `~astropy.nddata.CCDData` or None, optional
        A master flat frame to be divided into ccd. The unit of the
        master flat frame should match the unit of the image **after
        gain correction** if ``gain_corrected`` is True.
        Default is ``None``.

    bad_pixel_mask : `numpy.ndarray` or None, optional
        A bad pixel mask for the data. The bad pixel mask should be in given
        such that bad pixels have a value of 1 and good pixels a value of 0.
        Default is ``None``.

    gain : `~astropy.units.Quantity` or None, optional
        Gain value to multiple the image by to convert to electrons.
        Default is ``None``.

    readnoise : `~astropy.units.Quantity` or None, optional
        Read noise for the observations. The read noise should be in
        electrons.
        Default is ``None``.

    oscan_median : bool, optional
        If true, takes the median of each line. Otherwise, uses the mean.
        Default is ``True``.

    oscan_model : `~astropy.modeling.Model` or None, optional
        Model to fit to the data. If None, returns the values calculated
        by the median or the mean.
        Default is ``None``.

    min_value : float or None, optional
        Minimum value for flat field. The value can either be None and no
        minimum value is applied to the flat or specified by a float which
        will replace all values in the flat by the min_value.
        Default is ``None``.

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
        Default is ``False``.

    gain_corrected : bool, optional
        If True, the ``master_bias``, ``master_flat``, and ``dark_frame``
        have already been gain corrected.  Default is ``True``.

    Returns
    -------
    occd : `~astropy.nddata.CCDData`
        Reduded ccd.

    Examples
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
    if isinstance(fname_or_ccd, str):
        nccd = ccddata_read(fname_or_ccd)
    else:
        # make a copy of the object
        nccd = fname_or_ccd.copy()

    # Put in our common metadata
    ccd_metadata(nccd)

    # Correct exposure time
    # --> REFINE THIS ESTIMATE BASED ON MORE MEASUREMENTS
    exptime = nccd.meta['EXPTIME']
    if exptime > sx694_max_accurate_exposure:
        nccd.meta.insert('EXPTIME', 
                         ('OEXPTIME', exptime, 'original exposure time (seconds)'),
                         after=True)
        exptime += sx694_exposure_correct
        nccd.meta['EXPTIME'] = (exptime, 'corrected exposure time (seconds)')
        nccd.meta.insert('OEXPTIME', 
                         ('HIERARCH EXPTIME_CORRECTION',
                          sx694_exposure_correct, '(seconds)'),
                         after=True)
        add_history(nccd.meta,
                    'Corrected exposure time for SX694 MaxIm driver bug')

    # Handle our calibration object
    if calibration is True:
        # --> Document this
        calibration  = Calibration()

    if (isinstance(calibration, Calibration)
        and master_bias is True):
        # --> Document this
        master_bias = calibration.best_bias(nccd)

    # Make our master_bias easy to input and capture the name for
    # metadata purposes
    if isinstance(master_bias, str):
        subtract_bias_keyword = \
            {'HIERARCH SUBTRACT_BIAS': 'subbias',
             'SUBBIAS': 'ccdproc.subtract_bias ccd=<CCDData>, master=BIASFILE',
             'BIASFILE': master_bias}
        master_bias = ccddata_read(master_bias)
    else:
        subtract_bias_keyword = None
        
    # Apply overscan correction unique to the IoIO SX694 CCD.  This
    # also adds our CCD metadata
    if oscan is True:
        nccd = subtract_overscan(nccd, master_bias=master_bias,
                                 *args, **kwargs)
    elif oscan is None or oscan is False:
        pass
    else:
        raise TypeError('oscan is not None, True or False')

    if master_bias is not None:
        nccd.meta['RDNOISE'] = master_bias.meta['RDNOISE']
        nccd.meta.comments['RDNOISE'] = master_bias.meta.comments['RDNOISE']
        # Extract values used for further ccd_process
        # --> document these True options in help
        if gain is True:
            gain = master_bias.meta['GAIN']
            if master_bias.unit == u.electron/u.adu:
                gain_corrected = True
        if error is True:
            readnoise = master_bias.meta['RDNOISE']

    # Correct our SATLEVEL and NONLIN units if we are going to gain
    # correct
    if gain is not None:
        nccd.meta['SATLEVEL'] = nccd.meta['SATLEVEL'] * gain
        nccd.meta.comments['SATLEVEL'] = 'saturation level (electron)'
        nccd.meta['NONLIN'] = nccd.meta['NONLIN'] * gain
        nccd.meta.comments['NONLIN'] = 'Measured nonlinearity point (electron)'
        gain = gain * u.electron/u.adu
            
    # Create the error frame.  I can't trim my overscan, so there are
    # lots of pixels at the overscan level.  After overscan and bias
    # subtraction, many of them that are probably normal statitical
    # outliers are negative enough to overwhelm the readnoise in the
    # deviation calculation.  But I don't want the error estimate on
    # them to be NaN, since the error is really the readnoise.
    if error and gain is not None and readnoise is not None:
        nccd = ccdp.create_deviation(nccd, gain=gain,
                                     readnoise=readnoise*u.electron,
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
    
    if gain is not None and gain_corrected:
        nccd = ccdp.gain_correct(nccd, gain)

    # subtract the master bias
    if isinstance(master_bias, CCDData):
        nccd = ccdp.subtract_bias(nccd, master_bias,
                             add_keyword=subtract_bias_keyword)
    elif master_bias is None:
        pass
    else:
        raise TypeError(
            'master_bias is not None, fname or a CCDData object.')
    
    # Make it convenient to just specify dark_frame to do dark
    # subtraction the way I want
    if isinstance(calibration, Calibration) and dark_frame is True:
        # --> Document this
        dark_frame = ccddata_read(calibration.best_dark(nccd.meta))

    if isinstance(dark_frame, CCDData):
        exposure_key = ccdp.Keyword('EXPTIME', u.s)
        dark_scale = True
    else:
        exposure_key=None
        dark_scale=False

    # subtract the dark frame
    if isinstance(dark_frame, CCDData):
        nccd = subtract_dark(nccd, dark_frame, dark_exposure=dark_exposure,
                             data_exposure=data_exposure,
                             exposure_time=exposure_key,
                             exposure_unit=exposure_unit,
                             scale=dark_scale)
    elif dark_frame is None:
        pass
    else:
        raise TypeError(
            'dark_frame is not None or a CCDData object.')
    

    #nccd = ccdp.ccd_process(nccd, master_bias=master_bias,
    #                        gain=gain*u.electron/u.adu,
    #                        dark_frame=dark_frame,
    #                        exposure_key=exposure_key,
    #                        dark_scale=dark_scale,
    #                        gain_corrected=gain_corrected,
    #                        *args, **kwargs)

    if isinstance(calibration, Calibration) and master_flat is True:
        # --> Document this
        master_flat = ccddata_read(calibration.best_flat(nccd.meta))

    if master_flat is None:
        return nccd
    min_value = master_flat.meta['FLAT_CUT']
    nccd = ccdp.flat_correct(nccd, master_flat,
                             min_value=min_value, norm_value=1)
    # My flats look different that most, so just divide and call that
    # good enough
    #master_flat.mask=None
    #nccd = nccd.divide(master_flat, handle_meta='first_found')
    #nccd.meta['HIERARCH FLAT_CORRECT'] = True
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

def dark_combine(directory=None,
                 collection=None,
                 subdirs=calibration_subdirs,
                 glob_include=dark_glob,
                 master_bias=None, # This is going to have to be True or something like that to trigger search for optimum
                 calibration=None,
                 num_processes=max_num_processes,
                 mem_frac=max_mem_frac,
                 outdir=calibration_root,
                 show=False,
                 dccdt_tolerance=dccdt_tolerance,
                 mask_threshold=3): #Units of readnoise
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
    if len(fdict_list) == 0:
            log.debug('No darks found in: ' + directory)
            return False
    log.debug('Darks found in ' + directory)

    for fdict in fdict_list:
        lccds = []
        jds = []
        for fname in fdict['fnames']:
            ccd = ccddata_read(fname, add_metadata=True)
            if not full_frame(ccd):
                log.debug('dark wrong shape: ' + fname)
                continue
            if light_image(ccd):
                log.debug('dark recorded during light conditions: ' +
                          fname)
                continue
            ccd = ccd_process(ccd, oscan=True, master_bias=master_bias,
                              gain=True, error=True, calibration=calibration)
            lccds.append(ccd)
            # Get ready to capture the mean DATE-OBS
            tm = Time(ccd.meta['DATE-OBS'], format='fits')
            tt = tm.tt.datetime
            jds.append(tm.jd)
        mean_t = fdict['T']
        exptime = fdict['EXPTIME']
        if len(jds) == 0:
            log.debug('No good darks found at CCDT = {} C EXPTIME = {} in {}'.format(mean_t, exptime, directory))
            continue
        tm = Time(np.mean(jds), format='jd')
        this_date = tm.fits
        this_dateb = this_date.split('T')[0]
        this_ccdt = '{:.1f}'.format(mean_t)
        outbase = '{}_ccdT_{}_exptime_{}s'.format(
            this_dateb, this_ccdt, exptime)

        mem = psutil.virtual_memory()
        im = \
            ccdp.combine(lccds,
                         method='average',
                         sigma_clip=True,
                         sigma_clip_low_thresh=5,
                         sigma_clip_high_thresh=5,
                         sigma_clip_func=np.ma.median,
                         sigma_clip_dev_func=mad_std,
                         mem_limit=mem.available*mem_frac)
        mask_above(im, 'SATLEVEL')
        mask_above(im, 'NONLIN')

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
        im.meta['NCOMBINE'] = (len(lccds), 'Number of darks combined')
        add_history(im.meta,
                    'Combining NCOMBINE biases indicated in FILENN')
        im.meta['HIERARCH MASK_THRESHOLD'] \
            = (mask_threshold, 'Units of readnoise')
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

def flat_combine(directory=None,
                 collection=None,
                 subdir=calibration_subdirs,
                 glob_include=flat_glob,
                 master_bias=None, # This is going to have to be True or something like that to trigger search for optimum
                 dark_frame=None, # similarly here
                 outdir=calibration_root,
                 min_num_flats=min_num_flats,
                 num_processes=max_num_processes,
                 mem_frac=max_mem_frac,
                 show=False,
                 flat_cut=0.75,
                 init_threshold=100, # units of readnoise
                 edge_mask=-40): # CorObsData parameter for ND filter coordinates    
    if collection is None:
        if not os.path.isdir(directory):
            log.debug('No directory ' + directory)
            return False
        collection = ccdp.ImageFileCollection(directory,
                                              glob_include=glob_include)
    directory = collection.location
    if collection.summary is None:
        if subdir is not None:
            newdir = os.path.join(directory, subdir)
            return flat_combine(newdir,
                                subdir=None,
                                glob_include=glob_include,
                                master_bias=master_bias,
                                dark_frame=dark_frame, # similarly here
                                outdir=outdir,
                                min_num_flats=min_num_flats,
                                show=show)
        log.debug('No [matching] FITS files found in  ' + directory)
        return False
    # If we made it here, we have a collection with files in it
    # --> This loop could be parallelized
    filters = np.unique(collection.summary['filter'])
    for this_filter in filters:
        flat_fnames = collection.files_filtered(imagetyp='FLAT',
                                                filter=this_filter,
                                                include_path=True)
        lccds = []
        jds = []
        for fname in flat_fnames:
            ccd = ccddata_read(fname, add_metadata=True)
            ccd = ccd_process(ccd,
                              oscan=True,
                              master_bias=master_bias,
                              gain=True,
                              error=True,
                              dark_frame=dark_frame)
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
                log.warning('flat max value of {} too bright: {}'.format(
                    max_flat, fname))
                continue
            ccd.mask = None
            ccd = ccd.divide(max_flat, handle_meta='first_found')
            lccds.append(ccd)
            # Get ready to capture the mean DATE-OBS
            tm = Time(ccd.meta['DATE-OBS'], format='fits')
            tt = tm.tt.datetime
            jds.append(tm.jd)
        
        if len(jds) < min_num_flats:
            log.debug('Not enough good flats found for filter = {} in {}'.format(this_filter, directory))
            continue
        # Combine our flats
        mem = psutil.virtual_memory()
        im = \
            ccdp.combine(lccds,
                         method='average',
                         sigma_clip=True,
                         sigma_clip_low_thresh=5,
                         sigma_clip_high_thresh=5,
                         sigma_clip_func=np.ma.median,
                         sigma_clip_dev_func=mad_std,
                         mem_limit=mem.available*mem_frac)
        # Interpolate over our ND filter
        hdul = ccd.to_hdu()
        obs_data = CorObsData(hdul, edge_mask=edge_mask)
        # Capture our ND filter metadata
        im.meta = hdul[0].header
        good_pix = np.ones(ccd.shape, bool)
        good_pix[obs_data.ND_coords] = False
        points = np.where(good_pix)
        values = im[points]
        xi = obs_data.ND_coords
        # Linear behaved much better
        nd_replacement = interpolate.griddata(points,
                                              values,
                                              xi,
                                              method='linear')
                                              #method='cubic')
        im.data[xi] = nd_replacement
        # Do one last smoothing and renormalization
        bkg_estimator = MedianBackground()
        b = Background2D(im, 20, mask=(im.data<flat_cut), filter_size=5,
                         bkg_estimator=bkg_estimator)
        max_flat = np.max(b.background)
        im = im.divide(max_flat, handle_meta='first_found')
        im.mask = im.data < flat_cut
        im.meta['FLAT_CUT'] = (flat_cut, 'Value below which flat is masked')

        # Prepare to write
        tm = Time(np.mean(jds), format='jd')
        this_date = tm.fits
        this_dateb = this_date.split('T')[0]
        outbase = '{}_{}'.format(this_dateb, this_filter)
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        outbase = os.path.join(outdir, outbase)
        out_fname = outbase + '_flat.fits'
        im.write(out_fname, overwrite=True)
        if show:
            impl = plt.imshow(im, origin='upper', cmap=plt.cm.gray)
            plt.show()


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
                 start_date=None,
                 stop_date=None,
                 gain_correct=True, # This is gain correcting the bias and dark
                 num_processes=max_num_processes,
                 mem_frac=max_mem_frac,
                 num_ccdts=num_ccdts,
                 num_dark_exptimes=num_dark_exptimes,
                 num_filts=num_filts,
                 bias_glob=bias_glob, 
                 dark_glob=dark_glob,
                 flat_glob=flat_glob,
                 lockfile=lockfile):
        self._raw_data_root = raw_data_root
        self._calibration_root = calibration_root
        self._subdirs = subdirs
        self.keep_intermediate = keep_intermediate
        self._ccdt_tolerance = ccdt_tolerance
        self._bias_table = None
        self._dark_table = None
        self._flat_table = None
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

        #to_reduce = [dt[0] for dt in to_check
        #             if dir_has_calibration(dt[0],
        #                                    glob_include,
        #                                    subdirs=subdirs)]
        ## Remove duplicates
        #return sorted(list(set(to_reduce)))



        ## Find next date to reduce
        #if t is None:
        #    next_to_reduce = datetime.datetime(1,1,1)
        #else:
        #    # Work in astropy.Time
        #    start_time = Time(self._start_date)
        #    stop_time = Time(self._stop_date)
        #    # Find the first unreduced directory in our start-stop range
        #    reduced_idx = np.logical_and(start_time <= t['dates'],
        #                                 t['dates'] <= stop_time)
        #
        #
        #    # Find last reduction within our desired star-stop range.
        #    # Table reads in astropy Time objects...
        #    good_idx = np.logical_and(start_time <= t['dates'],
        #                               t['dates'] <= stop_time)
        #    next_to_reduce = np.max(t['dates'][good_idx])
        #    # Don't re-reduce our last day
        #    next_to_reduce = next_to_reduce + TimeDelta(1, format='jd')
        #    if next_to_reduce > stop_time:
        #        print('No reductions need to be done')
        #        return []
        #    next_to_reduce = next_to_reduce.to_datetime()
        ## Usual case of running pipeline on latest files, don't
        ## re-reduce, but we can skip ahead if we want
        #start = max(next_to_reduce, self._start_date)
        #if start > self._stop_date:
        #    # Catch case where we want to work with older data
        #    start = self._start_date
        #to_check = get_dirs(self._raw_data_root,
        #                    start=start,
        #                    stop=self._stop_date)
        #to_reduce = [d for d in to_check
        #             if dir_has_calibration(d, glob_include, subdirs=subdirs)]
        ## Remove duplicates
        #return sorted(list(set(to_reduce)))

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

        # Make sure each directory we process has enough processes to
        # work with to parallelize the slowest step (combining images)
        our_num_processes = min(ndirs_dates,
                                int(self.num_processes / self.num_ccdts))
        if ndirs_dates == 0:
            pass
        elif ndirs_dates == 1 or our_num_processes == 1:
            for dt in dirs_dates:
                bias_combine(dt[0],
                             subdirs=self._subdirs,
                             glob_include=self._bias_glob,
                             outdir=self._calibration_root,
                             gain_correct=self._gain_correct,
                             num_processes=self.num_processes,
                             mem_frac=self.mem_frac,
                             keep_intermediate=self.keep_intermediate)
        else:
            num_subprocesses = int(self.num_processes / our_num_processes)
            subprocess_mem_frac = self.mem_frac / our_num_processes
            log.debug(f'Calibration.reduce_bias: ndirs_dates = {ndirs_dates}')
            log.debug('Calibration.reduce_bias: self.num_processes = {}, our_num_processes = {}, num_subprocesses = {}, subprocess_mem_frac = {}'.format(self.num_processes, our_num_processes, num_subprocesses, subprocess_mem_frac))
            #return

            plist = \
                [Process(target=bias_combine,
                         args=(dt[0],),
                         kwargs= {'subdirs': self._subdirs,
                                  'glob_include': self._bias_glob,
                                  'outdir': self._calibration_root,
                                  'gain_correct': self._gain_correct,
                                  'num_processes': num_subprocesses,
                                  'mem_frac': subprocess_mem_frac},
                         daemon=False) # Let subprocesses create more children
                 for dt in dirs_dates]
            p_limit(plist, our_num_processes)

        self.bias_table_create(rescan=True, autoreduce=False)
        # This could potentially get set in dirs_dates_to_reduce, but
        # it seems better to set it after we have actually done the work
        all_dirs_dates = get_dirs_dates(self._raw_data_root,
                                  start=self._start_date,
                                  stop=self._stop_date)
        self._bias_dirs_dates_checked = all_dirs_dates
        lock.clear()

    def reduce(self):
        self.reduce_bias()
        #self.reduce_dark()
        #self.reduce_flat()

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
        ccdts = []
        dates = []
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

    @property
    def bias_table(self):
        return self.bias_table_create()

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
            log.warning('No biases found within {}C, broadening by factor of 2'.format(ccdt_tolerance))
            return self.best_bias(hdr, ccdt_tolerance=ccdt_tolerance*2)
        ddates = tm - self.bias_table['dates']
        best_ccdt_date_idx = np.argmin(ddates[good_ccdt_idx])
        # unwrap
        best_ccdt_date_idx = good_ccdt_idx[best_ccdt_date_idx]
        return self._bias_table['fnames'][best_ccdt_date_idx]


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
c = Calibration(start_date='2020-07-11', stop_date='2020-08-22')
sample = '/data/io/IoIO/raw/20200711/Dark-S005-R008-C001-B1.fts'
ccd = ccd_process(sample, calibration=c, oscan=True, master_bias=True,
                  gain=True, error=True)
ccd.write('/tmp/test.fits', overwrite=True)

print('SECOND TRY')
ccd = ccd_process(sample, calibration=c, oscan=True, master_bias=True,
                  gain=True, error=True)
ccd.write('/tmp/test.fits', overwrite=True)

