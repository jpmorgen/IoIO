"""Utilities for the IoIO data collection and reduction system"""

import gc
import inspect
import os
import re
import glob
import fnmatch
import time
import datetime
from pathlib import Path

import csv

import numpy as np
from numpy.polynomial import Polynomial

from scipy.signal import medfilt # not as good as median_filter
from scipy.ndimage import median_filter

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib.colors import LogNorm
import matplotlib.dates as mdates

from astropy import log
import astropy.units as u
from astropy.time import Time, TimeDelta
from astropy.stats import biweight_location, mad_std
from astropy.table import Table, QTable, vstack
from astropy.convolution import (convolve, interpolate_replace_nans,
                                 Gaussian1DKernel)
from astropy.coordinates import SkyCoord
from astropy.nddata import CCDData

from ccdproc import ImageFileCollection

from bigmultipipe import assure_list, outname_creator

from IoIO.photometry import rot_to
from IoIO.juno import JunoTimes, PJAXFormatter, juno_pj_axis

# MaxIM, ACP, FITS
FITS_GLOB_LIST = ['*.fit', '*.fts', '*.fits']

# These are MaxIm and ACP day-level raw data directories, respectively
DATE_FORMATS = ["%Y-%m-%d", "%Y%m%d"]


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
        if not self.is_set:
            return
        while self.is_set:
            with open(self._fname, "r") as f:
                log.error(f'lockfile {self._fname} detected for {f.read()}')
            time.sleep(self.check_every)
        log.error(f'(error cleared) lockfile {self._fname} removed')

    def create(self):
        self.wait()
        with open(self._fname, "w") as f:
            f.write('PID: ' + str(os.getpid()))

    def clear(self):
        os.remove(self._fname)

def dict_to_ccd_meta(ccd_in, d):
    ccd = ccd_in.copy()
    for k in d.keys():
        if len(k) > 8:
            kname = f'HIERARCH {k}'
        else:
            kname = k
        if np.isnan(d[k]):
            ccd.meta[kname] = 'NAN'
        elif np.isinf(d[k]):
            ccd.meta[kname] = 'INF'
        elif isinstance(d[k], u.Quantity):
            ccd.meta[kname] = (d[k].value, f'[{d[k].unit}]')
        else:
            ccd.meta[kname] = d[k]
    return ccd

def sum_ccddata(ccd):
    """Annoying that sum and np.ma.sum don't work out of the box, but
    understandable given the various choices.  This returns a tuple
    with units of the sum of the ccd pixel values (e.g. flux) and
    the number unmasked pixels which can be used to calculate surface
    brightness (flux/solid angle)

    """
    a = np.ma.array(ccd.data, mask=ccd.mask)
    return (np.ma.sum(a) * ccd.unit * u.pixel**2,
            np.sum(~ccd.mask * u.pixel**2))

def nan_biweight(a):
    """Fix special case of `~astropy.stats.biweight_location` where NAN
    return value ignores units

    """
    b = biweight_location(a, ignore_nan=True)
    if np.isnan(b) and isinstance(a, u.Quantity):
        b *= a.unit
    return b

def nan_mad(a):
    """Fix special case of `~astropy.stats.mad_std` where NAN return value
    ignores units

    """
    b = mad_std(a, ignore_nan=True)
    if np.isnan(b) and isinstance(a, u.Quantity):
        b *= a.unit
    return b

def multi_glob(directory, glob_list=None, glob_exclude_list=None):
    """Returns list of files matching one or more regexp

    Parameters
    ----------
    directory : str
        Directory in which to search for files

    glob_list : str or list
        (list of) regexp to pass to `glob.glob` used to construct flist

    glob_exclude_list : str or list
        (list of) regexp to use to exclude from final list of filenames

    Returns
    -------
    flist : list
        List of filenames
    """
    glob_list = assure_list(glob_list)
    flist = []
    for gi in glob_list:
        flist += glob.glob(os.path.join(directory, gi))
    glob_exclude_list = assure_list(glob_exclude_list)
    for ge in glob_exclude_list:
        flist = [f for f in flist
                 if not fnmatch.fnmatch(f, ge)]
    return flist

def datetime_dir(directory,
                 date_formats=DATE_FORMATS):
    """Returns datetime object corresponding to date found in directory

    Paramters
    ---------
    directory : str
        directory basename.  Possible to have a string following the
        date, e.g. _cloudy, but may be buggy if date_formats expands
        to more than ["%Y-%m-%d", "%Y%m%d"]

    date_formats : list
        list of datetime formats representing valid data-containing
        directories.  E.g. YYYY-MM-DD (MaxIm) and YYYYMMDD (ACP)
        ["%Y-%m-%d", "%Y%m%d"]

    Returns
    -------
    thisdate : datetime.datetime or False

    """
    for idf in date_formats:
        try:
            # This is a total ugly hack dependent on the particular
            # date formats I have.  But it allows directories with
            # date formats and other things like _cloudy to be
            # considered as valid data.  It works by noting the date
            # formats are two characters shorter than the length of
            # the strings I am looking for (%Y is two shorter than
            # YYYY, but %M is the same as MM, etc.)
            d = directory
            d = d[0:min(len(d),len(idf)+2)]
            thisdate = datetime.datetime.strptime(d, idf)
            return thisdate
        except:
            pass
    return False
    
def reduced_dir(rawdir, reddir_top,
                create=False,
                date_formats=DATE_FORMATS):
    """Create a parallel directory to a rawdir rooted in reddir_top into
    which reduced files will be deposited.

    Paramters
    ---------
    rawdir : str
        Name of raw directory from which to start.  If this is at or
        below a night directory (see date_formats), a name of the same
        type will be returned

    reddir_top : str
        The name of the top-level reduced directory

    create : bool
        Create reduced directory (and parents) if they don't exist
        Default is ``False``

    date_formats : list
        list of datetime formats representing valid data-containing
        directories.  E.g. YYYY-MM-DD (MaxIm) and YYYYMMDD (ACP)
        ["%Y-%m-%d", "%Y%m%d"]

    Returns
    -------
    reddir: str
        Directory name in reduced directory tree structure

    """
    ps = os.path.sep
    # List all elements of the path to the raw directory
    rawlist = []
    rparent = rawdir
    if rparent[-1] == ps:
        # Remove trailing path sep, which causes loop problems
        rparent = rparent[0:-1]
    d = True
    while d:
        rparent, d = os.path.split(rparent)
        rawlist.append(d)
    # Get all directories underneath a date_format directory, keeping
    # in mind rawlist would list these first
    redlist = []
    date_dir = False
    for d in rawlist:
        if datetime_dir(d):
            date_dir = d
            break
        redlist.append(d)
    reddir = reddir_top
    for d in reversed(redlist):
        reddir += ps + d
    if date_dir:
        reddir = reddir + ps + date_dir
    else:
        reddir = reddir_top
    if create:
        os.makedirs(reddir, exist_ok=True)
    return reddir

def get_dirs_dates(directory,
                   filt_list=None,
                   start=None,
                   stop=None,
                   date_formats=DATE_FORMATS):
    """Starting a root directory "directory," returns list of tuples
    (subdir, date) sorted by date.  Handles all cases of directory
    date formatting in the date_formats list

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

    date_formats : list
        list of regexp representing valid data-containing
        directories.  E.g. YYYY-MM-DD (MaxIm) and YYYYMMDD (ACP)
        ["%Y-%m-%d", "%Y%m%d"]
    """
    assert os.path.isdir(directory)
    fulldirs = [os.path.join(directory, d) for d in os.listdir(directory)]
    # Filter out bad directories first
    dirs = [os.path.basename(d) for d in fulldirs
            if (not os.path.islink(d)
                and os.path.isdir(d)
                and (filt_list is None
                     or not np.any([filt in d for filt in filt_list])))]
    ddlist = []
    for thisdir in dirs:
        thisdate = datetime_dir(thisdir, date_formats=date_formats)
        if thisdate:
            ddlist.append((thisdir, thisdate))
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
    if len(ddsorted) == 0:
        return []
    dirs, dates = zip(*ddsorted)
    dirs = [os.path.join(directory, d) for d in dirs]
    return list(zip(dirs, dates))

def multi_row_selector(table, keyword, value_list,
                       row_selector=None, **kwargs):
    """Returns lists of indices into a table, one list per value of `keyword`

    Parameters
    ----------
    table : `~astropy.table.Table`
        Usually a `ccdproc.ImageFileCollection.summary`, hence the use
        of "keyword" instead of "tag".

    keyword : str
        The primary keyword used for selection (e.g. 'FILTER')

    value_list : list
        List of allowed values of keyword.  One list of indices into
        `table` will be returned for each member of value_list

    row_selector : func
        Function applied on a per-row basis to filter out unwanted
        rows.  `row_selector` must accept one argument, a
        `~astropy.table.Table` row and return a `bool` value.  If
        `row_selector` is `None` no filtering is done
        Default is `None`
    """
    if row_selector is None:
        row_selector = (lambda x : True)
    retval = {}
    for value in value_list:
        idx = [i for i, r in enumerate(table)
               if (r[keyword.lower()] == value
                 and row_selector(r, **kwargs))]
        retval[value] = idx
    return retval
    #return [[i for i, r in enumerate(summary_table)
    #         if (r[keyword.lower()] == value
    #             and row_selector(r, **kwargs))]
    #        for value in value_list]
        

def closest_in_time(collection, value_pair,
                    row_selector=None,
                    keyword='filter',
                    directory=None):
    """Returns list of filename pairs.  In all pairs, the second
    observation is the closest in time to the first observation,
    relative to all other second observations.  Example use: for each
    on-band image, find the off-band image recorded closest in time
    
    Parameters
    ----------
    collection : `~astropy.ccdproc.ImageFileCollection`

    value_pair : tuple
        Values of `keyword` used to construct pairs

    keyword : str
        FITS keyword used to select pairs
        Default is ``filter``

    TODO
    ----
    Could possibly be generalized to finding groups

    """

    directory = collection.location or directory
    if directory is None:
        raise ValueError('Collection does not have a location.  Specify directory')
    st = collection.summary
    if st is None:
        return []

    row_dict = multi_row_selector(st, keyword, value_pair, row_selector)

    pair_list = []
    for i_on in row_dict[value_pair[0]]:
        t_on = Time(st[i_on]['date-obs'], format='fits')
        # Seach all second values
        t_offs = Time(st[row_dict[value_pair[1]]]['date-obs'],
                      format='fits')
        if len(t_offs) == 0:
            continue
        dts = [t_on - T for T in t_offs]
        idx_best1 = np.argmin(np.abs(dts))
        # Unwrap
        i_off = row_dict[value_pair[1]][idx_best1]
        pair = [os.path.join(directory,
                             st[i]['file'])
                             for i in (i_on, i_off)]
        pair_list.append(pair)
    return pair_list

def closest_in_coord(collection, value_pair,
                     row_selector=None,
                     keyword='filter',
                     directory=None):
    directory = collection.location or directory
    if directory is None:
        raise ValueError('Collection does not have a location.  Specify directory')
    st = collection.summary
    if st is None:
        return []

    row_dict = multi_row_selector(st, keyword, value_pair, row_selector)

    pair_list = []
    for i_on in row_dict[value_pair[0]]:
        try:
            coord_on = SkyCoord(st[i_on]['objctra'],
                                st[i_on]['objctdec'],
                                unit=(u.hourangle, u.deg))
        except Exception as e:
            fname = os.path.join(directory, st[i_on]['file'])
            log.warning(f'Problem getting SkyCoord for {fname}: {e}')
            continue
        # Seach all second values
        coord_offs = SkyCoord(st[row_dict[value_pair[1]]]['objctra'],
                              st[row_dict[value_pair[1]]]['objctdec'],
                              unit=(u.hourangle, u.deg))
        if len(coord_offs) == 0:
            continue
        seps = coord_offs.separation(coord_on)
        idx_best1 = np.argmin(seps)
        # Unwrap
        i_off = row_dict[value_pair[1]][idx_best1]
        pair = [os.path.join(directory,
                             st[i]['file'])
                             for i in (i_on, i_off)]
        pair_list.append(pair)
    return pair_list

def valid_long_exposure(r):
    """Inspects FITS header or ImageFileCollection row for condition"""
    valid = np.logical_and(r['imagetyp'] == 'LIGHT',
                           r['xbinning'] == 1)
    valid = np.logical_and(valid,
                           r['ybinning'] == 1)
    valid = np.logical_and(valid,
                           r['exptime'] > 10) # s
    return valid

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

# Note, this already exists in a more general form in astropy
# https://docs.astropy.org/en/stable/modeling/example-fitting-line.html#fit-using-uncertainties

def iter_polyfit(x, y, poly_class=None, deg=1, max_resid=None,
                 **kwargs):
    """Performs least squares fit iteratively to discard bad points

    If you actually know the statistical weights on the points,
    just use poly_class.fit directly.

    Parameters
    ----------
    x, y : array-like
        points to be fit

    poly_class : numpy.polynomial.polynomial series
        Default: `~numpy.polynomial.polynomial.Polynomial`

    deg : int
       Degree of poly_class
       Default is 1

    max_resid : float or None
        Points falling > max_resid from fit at any time are discarded.
        If `None`, ignored
        Default is `None`

    **kwargs passed to poly_class.fit

    returns : poly
       Best fit `~numpy.polynomial.polynomial.Polynomial`

    """
    if poly_class is None:
        poly_class = Polynomial
    x = np.asarray(x); y = np.asarray(y)
    # https://stackoverflow.com/questions/28647172/numpy-polyfit-doesnt-handle-nan-values
    idx = np.isfinite(x) & np.isfinite(y)
    x = x[idx]; y = y[idx]
    # Let polyfit report errors in x and y
    poly = poly_class.fit(x, y, deg=deg, **kwargs)
    # We are done if we have just two points
    if len(x) == deg + 1:
        return poly
    
    # Our first fit may be significantly pulled off by bad
    # point(s), particularly if the number of points is small.
    # Construct a repeat until loop the Python way with
    # while... break to iterate to squeeze bad points out with
    # low weights
    last_redchi2 = None
    iterations = 1
    while True:
        # Calculate weights roughly based on chi**2, but not going
        # to infinity
        yfit = poly(x)
        resid = (y - yfit)
        if resid.all == 0:
            break
        # Add 1 to avoid divide by zero error
        resid2 = resid**2 + 1
        # Use the residual as the variance + do the algebra
        redchi2 = np.sum(1/(resid2))
        # Converge to a reasonable epsilon
        if last_redchi2 and last_redchi2 - redchi2 < np.finfo(float).eps*10:
            break
        poly = poly_class.fit(x, y, deg=deg, w=1/resid2, **kwargs)
        last_redchi2 = redchi2
        iterations += 1

    if max_resid is not None:
        # The next level of cleanliness is to exclude any points above
        # max_resid from the fit.  But don't over-specify if too many
        # points are thrown out
        goodc = np.flatnonzero(np.abs(resid) < max_resid)
        if len(goodc) >= deg + 1:
            poly = iter_polyfit(x[goodc], y[goodc],
                                poly_class=poly_class,
                                deg=deg, max_resid=None,
                                **kwargs)
    return poly

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

def csvname_creator(directory_or_collection, *args,
                    outdir_root=None,
                    csv_base=None,
                    **kwargs):
    """Create full path to [e]csv file in outdir_root

    Parameters
    ----------
    directory_or_collection : str or ccdproc.ImageFileCollection
        Provides raw directory location (e.g. /data/IoIO/raw/20210604)

    outdir_root : str
        Root of output directory (e.g., /data/IoIO/Torus)

    csv_base : str
        Basename of [e]csv file (e.g. Torus.ecsv)
        
    Returns
    -------
    csvname : str
        Full path to [e]csv file

    """
    assert outdir_root is not None and csv_base is not None
    if isinstance(directory_or_collection, ImageFileCollection):
        directory = directory_or_collection.location
    else:
        directory = directory_or_collection
    rd = reduced_dir(directory, outdir_root, create=False)
    return os.path.join(rd, csv_base)

#def cached_csv_exists(*args, csvnames=cvnames, **kwargs):
#    """Returns true if all expected csvs exist"""
#    if callable(csvnames):
#        csvnames = csvnames(*args, **kwargs)
#    csvnames = assure_list(csvnames)
#    exists = True
#    for csvname in csvnames:
#        exists = exists and os.path.exists(csvname)
#    return exists
    
def cached_csv(*args,
               code=None,
               csvnames=None,
               read_csvs=False,
               write_csvs=False,
               create_outdir=False,
               newline='',
               quoting=csv.QUOTE_NONNUMERIC,
               **kwargs):
    """Write/read `~astropy.table.QTable`(s) or list(s) of dict to [e]csv
    file(s)

    Parameters
    ----------
    code : function
        Function that generates `~astropy.table.QTable`(s) or list(s)
        of dict(s) if `read_csvs` is ``False`` or `csvnames` cannot be
        read.  Use code=None and read_csvs=True to avoid running code
        and return only cached values.  In that case None return value
        indicates no cache(s)

    csvnames : str, list of str or callable
        Filename(s) to be read/written.  There must be on filename per
        astropy table or list of dict returned by code.  If callable,
        takes *args and **kwargs passed to cached_csv

    read_csvs : bool
        If `True` read [e]csv(s) from `csvnames`.  If there is an
        error in reading the files (e.g. they do not exist) ``code``
        is run
        Default is `False`

    write_csvs : bool
        If `True` write `~astropy.table.QTable`(s) or list(s)
        of dict(s) generated by ``code`` to filename(s) listed in `csvnames`
        Default is `False`

    create_outdir : bool, optional
        If ``True``, create any needed parent directories.
        Does not raise an error if outdir already exists.
        Default is ``False``

    newline : str
        For CSV writing:  Default is ''

    quoting : int
        For CSV writing: csv quoting type

    **kwargs : keyword arguments to pass to `code`

    Returns
    -------
    dict_lists : None or list of dict
        List of dictionaries read from csv file(s), one list per
        filename in csvnames.  If code is None and no cache files are
        found, None is returned

    """
    
    if callable(csvnames):
        csvnames = csvnames(*args, **kwargs)

    single_table_or_dictlist = isinstance(csvnames, str)
    if single_table_or_dictlist:
        csvnames = [csvnames]

    list_of_table_or_dicts = None
    if read_csvs:
        try:
            for csvname in csvnames:
                _, ext = os.path.splitext(csvname)
                if ext == '.ecsv':
                    if os.path.getsize(csvname) == 0:
                        # [Q]Table.read generates an error rather than
                        # an empty table when an empty file is read,
                        # so we have to dance around a bit.
                        # The "if" generates an error it csvname
                        # doesn't exist.
                        # If we made it here, the code ran
                        # presumably successfully, so [] is our cache
                        if list_of_table_or_dicts is None:
                            list_of_table_or_dicts = []
                        continue
                    
                    if list_of_table_or_dicts is None:
                        list_of_table_or_dicts = []
                    list_of_table_or_dicts.append(QTable.read(csvname))
                else:
                    dict_list = []
                    with open(csvname, newline=newline) as csvfile:
                        csvr = csv.DictReader(csvfile, quoting=quoting)
                        for row in csvr:
                            dict_list.append(row)
                    if list_of_table_or_dicts is None:
                        list_of_table_or_dicts = []
                    list_of_table_or_dicts.append(dict_list)
            if single_table_or_dictlist and len(list_of_table_or_dicts) > 0:
                list_of_table_or_dicts = list_of_table_or_dicts[0]
            return list_of_table_or_dicts
        except Exception as e:
            d = os.path.dirname(csvname)
            if code is not None:
                log.debug(f'Running code on {d} because received exception {e}')
            pass

    if code is None:
        return list_of_table_or_dicts

    # If we made it here, we need to generate our list(s) of dicts
    list_of_table_or_dicts = code(*args, **kwargs)

    if (write_csvs
        and (isinstance(list_of_table_or_dicts, list)
             or isinstance(list_of_table_or_dicts, QTable))):
        # Make sure we have a cacheable return result, e.g. not the
        # ImageFileCollection that is returned for processing multiple
        # directories in parallel.  We can't turn off caching
        # entirely, since then zero caches aren't written properly
        if single_table_or_dictlist:
            list_of_table_or_dicts = [list_of_table_or_dicts]
        for csvname, table_or_dict_list \
            in zip(csvnames, list_of_table_or_dicts):
            if create_outdir:
                os.makedirs(os.path.dirname(csvname), exist_ok=True)
            if len(table_or_dict_list) == 0:
                # Signal we have been here and found nothing
                Path(csvname).touch()
                continue
            _, ext = os.path.splitext(csvname)
            if ext == '.ecsv':
                table_or_dict_list.write(csvname, overwrite=True)
            else:
                fieldnames = list(table_or_dict_list[0].keys())
                with open(csvname, 'w', newline=newline) as csvfile:
                    csvdw = csv.DictWriter(csvfile, fieldnames=fieldnames,
                                           quoting=quoting)
                    csvdw.writeheader()
                    for d in table_or_dict_list:
                        csvdw.writerow(d)
        if single_table_or_dictlist:
            list_of_table_or_dicts = list_of_table_or_dicts[0]
    elif write_csvs:
        log.debug(f'Uncachable code return value of type '
                  f'{type(list_of_table_or_dicts)}')
    return list_of_table_or_dicts

def savefig_overwrite(fname, **kwargs):
    if os.path.exists(fname):
        os.remove(fname)
    plt.savefig(fname, **kwargs)

def finish_stripchart(outdir, outbase, show=False):
    plt.tight_layout()
    savefig_overwrite(os.path.join(outdir, outbase))
    if show:
        plt.show()
    plt.close()

# --> This is probably obsolete
def stripchart(to_plot,
               fig=None,
               ax=None,
               show=False, # not sure if I want these
               fig_close=False,
               **kwargs):
    """Plots a single plot in a stripchart 
    
    Parameters
    ----------
    to_plot : function
        Function that accepts **kwargs to plot desired stripchart element

    fig : pyplot.figure
        Defaults to creating a new figure with no args

    ax : pyplot.Axes
        Defaults to plt.subplot()
    """

    if fig is None:
        fig = plt.figure()
    if ax is None:
        ax = plt.subplot()

    to_plot(fig=None, ax=None, **kwargs)

    if show:
        plt.show()
    if fig_close:
        plt.close()


def filled(val, fill_value=np.NAN, unmask=False):
    """Fill astropy Mask or numpy ma.masked value(s) with NAN.
    Ignores if not one of these types

    Parameters
    ----------
    fill_value : compatible with type of val
        Desired fill value
        Default is `np.NAN'

    unmask : boolean
        If column contains a masked datatype, run the unmask method to
        return a non-masked version

    """
    if hasattr(val, 'filled'):
        val = val.filled(fill_value)
    if unmask:
        # This handles astropy masks, which aren't quite bugged
        # workede out:
        # https://docs.astropy.org/en/stable/utils/masked/index.html
        # TypeError: no implementation found for 'numpy.pad' on types
        # that implement __array_function__: [<class
        # 'astropy.utils.masked.core.MaskedNDArray'>]
        val = getattr(val, 'unmasked', val)
    return val    

def filled_columns(t, colnames, **kwargs):
    """Fill all masked values in desired table columns with fill_value

    Parameters
    ----------
    t : Table or list of Table
        Input Table(s) in which to fill values

    colnames : list of str
        Column names on which to operate

    **kwargs to pass to filled

    """
    colnames = assure_list(colnames)
    for colname in colnames:
        t[colname] = filled(t[colname], **kwargs)
    
def qtable2df(t, index=None):
    # If you need to filter or sort, do that first!
    if isinstance(t, str):
        t = QTable.read(t)
    filled_columns(t, t.colnames, unmask=True)
    one_d_colnames = [cn for cn in t.colnames
                      if len(t[cn].shape) <= 1]
    df = t[one_d_colnames].to_pandas()
    if index is not None:
        df = df.set_index(df[index])
    return df

# DataFrames may be a better way to do this
def interpolate_replace_nans_columns(t, colnames, kernel=None,
                                     boundary='extend',
                                     suffix='_interp'):
    """Uses interpolate_replace_nans to fill any NAN or masked
    values.  Adds new columns ending in "_interp"

    Parameters
    ----------
    t : Table or list of Table
        Input Table(s) to add interpolated columns to
    """
    filled_columns(t, colnames, fill_value=np.NAN, unmask=True)
    for colname in colnames:
        vals = t[colname]
        vals = interpolate_replace_nans(vals, kernel,
                                        boundary=boundary)
        t[f'{colname}{suffix}'] = vals

def contiguous_sections(t,
                        time_col,
                        max_time_gap):
    """Break a table up into sections with contiguous time coverage

    Parameters
    ----------
    t : QTable
        Table with time and possibly other columns.  Table must be
        sorted in time

    time_col : str
        name of time-like column

    max_time_gap : TimeDelta-like
        maximum length of time between table elements before a new
        segment will be formed.  Must be compatible with time_col

    Returns
    ------
    contig_tables : list
        list of QTables containing contiguous sections of time_col, as
        defined by max_time_gap
    """
    deltas = t[time_col][1:] - t[time_col][0:-1]
    last_contig_idx = np.flatnonzero(deltas > max_time_gap)
    if len(last_contig_idx) == 0:
        return [t]
    # Turn these into proper slices
    rights = last_contig_idx + 1
    rights = list(rights)
    # This is the only way to reference the slice to the end of an array/list
    rights.append(None)
    left = 0
    contig_tables = []
    for right in rights:
        contig_tables.append(t[left:right])
        left = right
    return contig_tables
    
def add_itime_col(summary_table,
                  time_col=None,
                  itime_col=None,
                  utc_offset=0,
                  dt=1):
    """Add an integer time column to a summary table

    Parameters
    ----------
    summary_table : QTable
        Table with time column to which integer time column will be added

    time_col : str
        Name of time column.  Time column must be astropy.time.Time or
        otherwise have a jd property
        Default is `tavg'

    itime_col : str
        Name of new column to contain potentially shifted and scaled JD
        Default is `ijdlt'

    utc_offset : TimeDelta
        UT offset of observatory.  Used for calculating JD local time
        (JDLT) to shift the JD scale
        Default is `-7*u.hr'

    dt : float
        Desired timestep
        Default is `1'
    """
    utc_offset = utc_offset.to(u.day)
    jds = summary_table[time_col].jd
    djds = jds - jds[0]
    idjds = djds / dt
    idjds = idjds.astype(int)
    ijds = idjds*dt + jds[0] + utc_offset.value 
    summary_table[itime_col] = ijds

def daily_biweight(qtable,
                   day_col=None,
                   data_col=None,
                   biweight_col=None,
                   std_col=None,
                   ignore_nan=True):
    """Compute biweight location and mad_std on a daily basis.  Used
    by add_daily_biweight

    Parameters
    ----------
    qtable : astropy.table.QTable
        qtable to which computed columns will be placed/added

    day_col : str
        column name containing days (or other quantity over which to
        collect biweight groups).  Column can't be a Time, so use, e.g.,
        itime_col from add_itime_col or other number-like column

    data_col : str
        column name of data

    biweight and std_col : str
        column names in into which biweight and mad_std will be placed.
        They will be created, if necessary

    ignore_nan : bool
        Ignore np.NAN in data
        default = ``True''

    """
    if isinstance(qtable[data_col], u.Quantity):
        unit = qtable[data_col].unit
    else:
        unit = 1
    
    vals = qtable[data_col]
    if isinstance(vals, Time):
        # Translate Time into JD so we can do complex math operations
        # on it.  Generally better form to demand number-like columns
        # are input to this routine and deal with special cases outside
        vals = vals.jd
    new_vals = vals.copy()
    new_std = vals.copy()
    unique_days = list(set(qtable[day_col]))
    for day in unique_days:
        mask = qtable[day_col] == day
        tvals = vals[mask]
        tbiweight = biweight_location(tvals, ignore_nan=ignore_nan)
        tstd = mad_std(tvals, ignore_nan=ignore_nan)
        new_vals[mask] = tbiweight
        new_std[mask] = tstd
    qtable[biweight_col] = new_vals
    qtable[std_col] = new_std
    if isinstance(qtable[data_col], Time):
        # This assumes that this time column is defined for all rows
        # (e.g. is the original time column).  Translate biweight
        # column back into the original Time format and put units on std
        qtable[biweight_col] = Time(
            qtable[biweight_col], format='jd',
            location=qtable[data_col][0].location.copy())
        qtable[biweight_col].format=qtable[data_col][0].format
        qtable[std_col] = qtable[std_col]*u.day

def add_daily_biweights(summary_table,
                        day_col='ijdlt',
                        colnames=None,
                        **kwargs):
    """Add a set of daily biweight location and MAD std columns to
    summary_table

    Parameters
    ----------
    summary_table : QTable
        Table with time and data columns

    day_col : str
        Name of column containing integer timesteps
        Default is `ijdlt'

    colnames : list
        List of column names for whcih daily biweight and MAD will be
        calculated.  Added columns have the form 'biweight_*'

    **kwargs passed to daily_biweight

    Returns
    -------
    added_colnames : list
        column names added to summary_table
    """
    added_colnames = []
    for col in colnames:
        biweight_col='biweight_' + col
        std_col='std_' + col
        daily_biweight(summary_table,
                       day_col=day_col,
                       data_col=col,
                       biweight_col=biweight_col,
                       std_col=std_col,
                       **kwargs)
        added_colnames.extend([biweight_col, std_col])
    return added_colnames        
        
def interp_table(t, x, xp_col, **kwargs):
    """Interpolates all rows in a Table to new axix x

    Parameters
    ----------
    t : Table
        Input table.  Must be sorted in xp_col

    x : array_like
        The x-coordinates at which to evaluate the interpolated values        

    xp_col : str
        Column name containing original x data points

    **kwargs passed to astropy.utils.masked.function_helpers.interp

    Returns
    -------
    itable : Table
        New table with all columns interpolated
    
    """
    if len(t) < 3:
        return t.copy()
    x_span = t[xp_col][-1] - t[xp_col][0]
    itable = QTable()
    for col in t.colnames:
        # numpy interp is overridden when operating on an astropy
        # masked array to handle the astropy difference.  See
        # https://docs.astropy.org/en/stable/utils/masked/index.html
        itable[col] = np.interp(x,
                                t[xp_col], t[col],
                                **kwargs)
    return itable

def linspace_table(t, x_col, xp_col=None, dx=1, b=0, algorithm=None, **kwargs):
    """Returns a table sampled at regular intervals

    Parameters
    ----------
    t : Table
        Input table.  Must be sorted in x_col

    x_col : str
        Name of column containing x-axis to make regular

    xp_col : str or None
        Name of column containing x-axis measurement points.  This
        covers the case where x_col is an integerized version of
        xp_col, where the xp_col are the actual measurement points
        which are grouped within dx.  If None, x_col is used
        Default is `None'

    dx : number-like
        Desired sample spacing
        Default is `1'

    b : number-like
        Starting point for spacing relative to first x_col point.
        This allows the regular sample scale to be shifted to line up
        with the optimal position determined from the xp_col.  See
        linspace_day_table for application example
        Default is `0'

    algorithm : str
        'interpolate_replace_nans' - use
        astropy.convolution.interpolate_replace_nans to replace NANs
        in the table.  This runs the algorithm='nan' case first and
        results in the best performance

        'nan' - do not interpolate, just put NANs in place of missing
        value

        'interp' - use np.interp (overmapped in astropy ecosystem to
        deal with astropy Mask) to do the interpolation.  This
        doesn't perform well when there are large gaps in the data

    **kwargs passed to interpolate_replace_nans_columns

    Returns
    -------
    ltable : Table
        Table sampled at regular intervals, interpolating to fill
        missing value where necessary
    """
    if xp_col is None:
        xp_col = x_col
    x_span = t[x_col][-1] - t[x_col][0]
    nrows = x_span / dx
    nrows = round(nrows)
    # Extend before and after range just to make sure we get all the
    # interpolates.  We'll mark anything outside of x_col's range with
    # a NAN for removal
    if algorithm == 'interpolate_replace_nans':
        ltable = linspace_table(t, x_col, xp_col=xp_col, dx=dx, b=b,
                                algorithm='nan', **kwargs)
        interpolate_replace_nans_columns(
            ltable, ltable.colnames, suffix='', **kwargs)
    elif algorithm == 'interp':
        x = t[x_col][0] + np.arange(-1, nrows+2) * dx + b
        ltable = interp_table(t, x, xp_col,
                              left=np.NAN, right=np.NAN)
        mask = np.isnan(ltable[x_col])
        ltable = ltable[~mask]
    elif algorithm == 'nan':
        ltable = t.copy()
        # Start with an X axis that matches.
        x = t[x_col][0] + np.arange(-1, nrows+2) * dx
        match_mask = False
        for x_orig in t[x_col]:
            tmask = np.isclose(x,
                               np.full_like(x, x_orig),
                               rtol=0, atol=dx/2)
            match_mask = np.logical_or(tmask, match_mask)
        new_row_mask = ~match_mask
        for i in np.arange(len(x)-len(t)):
            ltable.add_row()
        for col in t.colnames:
            if col == x_col:
                # Add our offset here
                ltable[col] = x + b
                continue
            if col == xp_col:
                # Match our xp_col to x_col
                ltable[col] = ltable[x_col]
                continue
            ltable[col][match_mask] = t[col]
            ltable[col][new_row_mask] = np.NAN
    else:
        raise ValueError(f'Unrecognized algorithm {algorithm}')
    return ltable

def linspace_day_table(day_table,
                       day_col='ijdlt',
                       time_col='biweight_jd',
                       time_col_offset=None,
                       dt=1*u.day,
                       max_gap_fraction=None,
                       **kwargs):
    """Linearly spaces a table on day increments, interpolating using
    a specified kernel to add rows if necessary.  Time increment is
    set by dt, but optimal offset of time axis is calculated using the
    ensemble of points in day_col

    Parameters
    ----------
    day_table : Table or list of Table
        Input Table with integer day column (day_col) and precise time of
        measurement column (time_col).  If list, operates on each
        Table in list

    day_col : str
        Name of column containing integer days corresponding to the
        real times in the time_col.  day_col can be created by add_itime_col
        Default is `ijdlt'

    time_col : str
        Name of column containing precise measurement times of
        datapoints stored in the rest of the columns in the Table
        (e.g. added by add_daily_biweights)
        Default is `biweight_jd'

    time_col_offset : TimeDelta or None
        Offset of time_col from UT.  If None, calculated based on the
        biweight location of the difference between the time_col and
        day_col values.
        Default is `-7*u.hr'

    dt : TimeDelta
        Desired increment in time between the rows of the
        output table. 
        Default is `1*u.day'

    max_gap_fraction : float or None
        If not None, calculates interpolation kernel using a
        Gaussian1DKernel with a width of max_gap_fraction * maximum
        gap in time_col

    based
        on the maxium difference between adjacent points times
        max_gap_fraction.  If None, **kwarg take care of passing
        needed arguments 
        Default is `None'
    
    **kwargs passed to linspace_table

    Returns
    -------
    Table or list of Tables that have had all of their column values
    interpolated so as to be found on a regularly spaced grid in time
    with no gaps

    """
    if isinstance(day_table, list):
        ldts = []
        for t in day_table:
            lst = linspace_day_table(
                t, day_col=day_col, time_col=time_col,
                time_col_offset=time_col_offset,
                dt=dt, max_gap_fraction=max_gap_fraction,
                **kwargs)
            ldts.append(lst)
        return ldts
    if len(day_table) < 3:
        log.warning('length of input table is too short for interpolation')
        return day_table
    if time_col_offset is None:
        time_col_offsets = day_table[time_col] - day_table[day_col]
        time_col_offset = biweight_location(time_col_offsets)
    if hasattr(time_col_offset, 'to'):
        time_col_offset = time_col_offset.to(u.day).value        
    # I am not sure if I am shooting myself in the foot by doing this.
    # The plus side is that interpolation is easiest and most accurate
    # for all of the points, the minus side is that different segments
    # of a list of tables will have different offsets and may be hard
    # to compare with a continously sampled model across the icdts.
    # OTOH, I interpoolate the model to the datapoints anyway
    if max_gap_fraction:
        deltas = day_table[time_col][1:] - day_table[time_col][0:-1]
        sigma = max(deltas)*max_gap_fraction
        kernel = Gaussian1DKernel(sigma)
    ddt = dt.to(u.day).value
    return linspace_table(day_table, day_col, time_col, dx=ddt,
                          b=time_col_offset, kernel=kernel, **kwargs)

# --> Note that astropy.utils.masked.function_helpers doesn't have an analog
def nan_median_filter(data, mask=False, **kwargs):
    """Median filter data with masked values and/or NANs.  Note that
    the quality of the median filtering is best if the data are
    sampled on a regular interval and have no NAN gaps.  The mask
    feature is made available for a quick-and-dirty look into data
    
    Parameters
    ---------
    data : ndarray-like
        data to be median filtered.  Can be a astropy.units.Quantity
    mask : bool
        default = ``False''
    **kwargs : dict
        passed on to scipy.ndimage median_filter

    Returns
    -------
    ndata : ndarray-like
        Median filtered unmasked & non-NAN values copied into
        original-length data array with missing values marked as NANs
    """
    ndata = data.copy()
    mask = np.logical_or(mask, np.isnan(data))
    meds = median_filter(data[~mask], **kwargs)
    if isinstance(ndata, u.Quantity):
        meds *= ndata.unit
    ndata[~mask] = meds
    return ndata

def add_medfilt_columns(t, colnames, medfilt_width=None, **kwargs):
    """Add columns produced by scipy. median_filter to table(s)

    Parameters
    ----------
    t : Table or list of Table
        Input Table(s) to add median filter columns to

    colnames : list of str
        list of column names on which to compute median filter.
        Added columns will be of the form `colname_medfilt'

    medfilt_width : int
        Width of the filter in table rows

    **kwargs : dict
        kwargs passed to scipy. median_filter
        
    """
    if isinstance(t, list):
        for tt in t:
            add_medfilt_columns(tt, colnames,
                                medfilt_width=medfilt_width,
                                **kwargs)
        return
    for colname in colnames:
        meds = nan_median_filter(t[colname],
                                 size=medfilt_width,
                                 **kwargs)
        t[f'medfilt_{colname}'] = meds
        
    
# --> THIS IS OBSOLETE EXCEPT IN na_meso
def daily_convolve(qtable,
                   day_col,
                   data_col,
                   convolve_col,
                   kernel,
                   all_days=None):
    """Returns a new QTable with a convolution column added to it.  This
    assumes data are sampled evenly in time.  The all_days input,
    together with an astropy convolution kernel that handles NANs, is
    used to properly handle missing time steps

    Parameters
    ----------
    qtable : QTable
        input

    day_col, data_col, convolve_col : str
        Names of corresponding columns in qtable.  convolve_col will
        be created if it is not in qtable

    kernel : astropy.convolution.kernels kernal object
        Kernel to convolve through data_col.  e.g. Box1DKernel(20)

    all_days : ndarray
        Array of day_col values that provides, e.g., a complete set of
        time samples

    Returns
    -------
    qtable : QTable
        New QTable

    """
    if all_days is not None:
        # Prepare to create a table of missing days that is filled
        # with NANs.  Don't assume we have just 3 input columns
        missing_days = [day for day in all_days
                        if day not in qtable[day_col]]
        if convolve_col in qtable.colnames:
            names = qtable.colnames
        else:
            names = qtable.colnames + [convolve_col]
        n_nancols = len(qtable.colnames)
        nan_col = np.full(len(missing_days), np.NAN)
        if isinstance(qtable[data_col], u.Quantity):
            unit = qtable[data_col].unit
        else:
            unit = 1
        nan_cols = n_nancols * [nan_col*unit]
        mt = QTable([missing_days] + nan_cols,
                    names=names)
        if len(mt) > 0:
            qtable = vstack([qtable, mt])
    qtable.sort(day_col)
    #qtable[convolve_col] = convolve(qtable[data_col], kernel)
    qtable[convolve_col] = medfilt(qtable[data_col], 105)*qtable[data_col].unit
    return qtable

class ColnameEncoder:
    """Properties and methods to encode and decode quantities into
    strings useful for QTable column headings

    Parameters
    ----------
    colbase : str
        Base string of column, e.g. 'Na_sum'
        Default is ``None''

    formatter : str
        Numberic format string for encoding, e.g. '.1f'
        Default is ``None''
    """
    def __init__(self,
                 colbase=None,
                 formatter=None):
        self.colbase = colbase
        self.formatter = formatter
        self._colbase_regexp = None
        self._colbase_middle_regexp = None

    def to_colname(self, rad):
        """Encode `astropy.units.Quantity` into string

        Parameters 
        ----------
        rad : Quantity
            `astropy.units.Quantity` to be encoded

        Returns
        -------
        string in the form 'colbase_value_unit' where value is
        formatted numerically according to formatter property

        """
        return f'{self.colbase}_{rad.value:{self.formatter}}_{rad.unit}'

    def from_colname(self, colname):
        """Decode string into quantity

        Parameters 
        ----------
        rad : Quantity
            `astropy.units.Quantity` to be encoded

        """
        s = colname.split('_')
        return float(s[-2])*u.Unit(s[-1])

    @property
    def colbase_regexp(self):
        """regexp with colbase anchored at beginning of regexp
        """
        if self._colbase_regexp:
            return self._colbase_regexp
        return re.compile(self.colbase + '_.*')

    @property
    def colbase_middle_regexp(self):
        """regexp with colbase in the middle of regexp
        """
        if self._colbase_middle_regexp:
            return self._colbase_middle_regexp
        return re.compile('.*_' + self.colbase + '_.*')

    def colbase_list(self, from_list):
        return list(filter(self.colbase_regexp.match, from_list))

    def colbase_middle_list(self, from_list):
        return list(filter(self.colbase_middle_regexp.match, from_list))

    def largest_colbase(self, from_list):
        """Returns element of list that has the largest encoded Quantity"""
        largest = None
        from_list = self.colbase_list(from_list)
        for col in from_list:
            val = self.from_colname(col)
            if largest is None:
                largest = val
            elif val > largest:
                largest = val
        return self.to_colname(largest)

def flexi_slice(im, b, t, l, r, fill_value=0):
    """Performs a slice of im, inserting it into a larger shaped array, if necessary
    
    Parameters
    ----------
    im : ndarray or CCDData

    b, t, l, r : int 
        Slice coords.  These can be outside the original array
        bounds, resulting in a larger return array/CCDData padded with zeros

    Returns
    -------
    nim : ndarray or CCDData

    """
    if isinstance(im, CCDData):
        nccd = im.copy()
        nccd.data = flexi_slice(im.data, b, t, l, r)
        fname = im.meta['RAWFNAME']                        
        if im.uncertainty is not None:
            nccd.uncertainty.array = flexi_slice(im.uncertainty.array,
                                                 b, t, l, r)
        if im.mask is not None:
            nccd.mask = flexi_slice(im.mask, b, t, l, r, fill_value=True)
        return nccd

    nim = np.full((t-b, r-l), fill_value, im.dtype)

    if b < 0:
        orig_b = 0
        new_b = -b
    else:
        orig_b = b
        new_b = 0
    if t > im.shape[0]:
        orig_t = im.shape[0]
        new_t = new_b + orig_t - orig_b
    else:
        orig_t = t
        new_t = orig_t + new_b

    if l < 0:
        orig_l = 0
        new_l = -l    
    else:
        orig_l = l
        new_l = 0
    if r > im.shape[1]:
        orig_r = im.shape[1]
        new_r = new_l + orig_r - orig_l
    else:
        orig_r = r
        new_r = orig_r + new_l
    nim[new_b:new_t, new_l:new_r] = im[orig_b:orig_t, orig_l:orig_r]
    return nim
    
def pixel_per_Rj(ccd):
    Rj_arcsec = ccd.meta['Jupiter_ang_width'] * u.arcsec / 2
    cdelt = ccd.wcs.proj_plane_pixel_scales() # tuple
    lcdelt = [c.value for c in cdelt]
    pixscale = np.mean(lcdelt) * cdelt[0].unit
    pixscale = pixscale.to(u.arcsec) / u.pixel
    return Rj_arcsec / pixscale / u.R_jup 

def fill_plot_col(vals, ngaps):
    vals = filled(vals, unmask=True)
    vals = np.append(vals, (np.NAN, ) * ngaps)
    return vals


def plot_column(t,
                time_col='tavg',
                max_time_gap=15*u.day,
                colname=None,
                err_colname=None,
                fmt='',
                fig=None,
                ax=None,
                tlim=None,
                **kwargs):

    """Plot a column from a astropy.table.Table or pandas.DataFrame as
    a function of time during the Juno mission.  Use juno_pj_axis(ax)
    to plot the Juno perijove axis on the top axis

    Parameters
    ----------
    t : astropy.table.Table or pandas.DataFrame
        Input table or dataframe.  dataframe index must be a datetime

    time_col : str
        Name of time column
        Default is `tavg'

    colname : str
        Data column name

    err_colname : str
        Data column name (optional).  If not None, ax.errorbar will be
        used

    max_time_gap : timdelta-like
        For line plots, gaps in time larger than this will cause the
        plotting pen to be lifted and started again for the next
        segment    
        Default is `15*u.day'
    """
    if fig is None:
        fig = plt.figure()
    if ax is None:
        ax = fig.add_subplot()

    # Play a little fast-and-lose with [Q]Table and DataFrame possible
    # inputs and prepare for fact that ax.errorbar does not like
    # astropy Times
    if isinstance(t, Table):
        datetimes = t[time_col]
        datetimes = datetimes.to_datetime()
    else:
        # pandas.DataFrames handle time as the index property.
        # Hopefully that has been set up properly with
        if hasattr(t, 'index'):
            datetimes = t.index.to_pydatetime()
        else:
            # Hope for the best
            datetimes = t[time_col]
    # When plotting lines, insert NANs into the values so that the
    # plotting pen picks up.  If not plotting lines, this doesn't
    # hurt.  Note that we need to work in datetimes because of our
    # conversion, above.
    if isinstance(max_time_gap, u.Quantity):
        max_time_gap = TimeDelta(max_time_gap).to_datetime()
    deltas = datetimes[1:] - datetimes[0:-1]
    med_dt = np.median(deltas)
    last_contig_idx = np.flatnonzero(deltas > max_time_gap)
    datetimes = np.append(
        datetimes, datetimes[last_contig_idx] + med_dt)
    sort_idx = np.argsort(datetimes)
    vals = t[colname]
    ngaps = len(last_contig_idx)
    npts = len(vals) + ngaps
    vals = fill_plot_col(vals, ngaps)
    datetimes = datetimes[sort_idx]
    vals = vals[sort_idx]
    if err_colname is None:
        # plot returns an artist
        h = ax.plot(datetimes, vals,
                     fmt, **kwargs)
        h = h[0]
    else:
        if isinstance(err_colname, tuple):
            errs = np.empty((2, npts))
            for ic, ec in enumerate(err_colname):
                err = fill_plot_col(t[ec], ngaps)
                err = err[sort_idx]
                errs[ic] = err
        else:
            errs = fill_plot_col(t[err_colname], ngaps)
            errs = errs[sort_idx]
        h = ax.errorbar(datetimes, vals, errs, fmt=fmt, **kwargs)
        
    ax.set_xlabel('Date')
    ax.set_xlim(tlim)
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    fig.autofmt_xdate()
    ax.format_coord = PJAXFormatter(datetimes, vals)
    return h

def plot_planet_subim(ccd_in,
                      fig=None,
                      ax=None,
                      plot_planet_rot_from_key=None,
                      pix_per_planet_radius=pixel_per_Rj,
                      in_name=None,
                      outname=None,
                      planet_subim_axis_label=r'R$\mathrm{_J}$',
                      planet_subim_dx=None,
                      planet_subim_dy=None,
                      planet_subim_vmin=30,
                      planet_subim_vmax=5000,
                      planet_subim_figsize=[5, 2.5],
                      plot_planet_cmap='gist_heat',
                      planet_subim_backcalc=None,
                      plot_planet_overlay=None,
                      bmp_meta=None,
                      **kwargs):
    # https://stackoverflow.com/questions/4931376/generating-matplotlib-graphs-without-a-running-x-server
    # --> This might help with the X crashes, but it has to be
    # imported before import matplotlib.pyplot 
    # import matplotlib as mpl
    # mpl.use('Agg')

    if fig is None:
        fig = plt.figure(figsize=planet_subim_figsize)
    if ax is None:
        ax = fig.add_subplot()

    if in_name is None or isinstance(in_name, list):
        # A bit of a hack to deal with the most common use of this
        # function, where in_name is a list of two files: on-band and
        # off-band.  Although I generally standardize on-band to be
        # the second file in in_name, this is a little safer and only
        # assumes that I am plotting a file that has been processed by
        # cormultipipe and has RAWFNAME
        in_name = os.path.basename(ccd_in.meta['RAWFNAME'])
        
    # This will pick up outdir, if specified
    outname = outname_creator(in_name, outname=outname, **kwargs)
    if outname is None:
        raise ValueError('in_name or outname must be specified')
    if planet_subim_backcalc is None:
        planet_subim_backcalc = lambda *args, **kwargs:0*ccd_in.unit
    background = planet_subim_backcalc(in_name=in_name,
                                       outname=outname,
                                       bmp_meta=bmp_meta, 
                                       **kwargs)
        
    # We are occationally using this to plot images that have aready
    # been rotated.  rot_to is a bit brittle to multiple rotations, so
    # don't call it if we aren't actually rotating.  This would cause
    # a problem if we wanted to do WCS centering as part of this plot,
    # but generally we already have our center set by this time
    if plot_planet_rot_from_key:
        ccd = rot_to(ccd_in, rot_angle_from_key=plot_planet_rot_from_key)
    else:
        ccd = ccd_in.copy()
    ccd = ccd.subtract(background, handle_meta='first_found') 
    pix_per_Rp = pix_per_planet_radius(ccd)
    center = ccd.wcs.world_to_pixel(ccd.sky_coord)*u.pixel
    # Trying to have the axes always read the same valuees
    center = np.floor(center).astype(int)
    if planet_subim_dx is None:
        l, r = 0*u.pixel, ccd.shape[1]*u.pixel
    else:
        l = np.floor(center[0] - planet_subim_dx * pix_per_Rp)
        r = np.ceil(center[0] + planet_subim_dx * pix_per_Rp)
    if planet_subim_dy is None:
        b, t = 0*u.pixel, ccd.shape[0]*u.pixel
    else:
        b = np.floor(center[1] - planet_subim_dy * pix_per_Rp)
        t = np.ceil(center[1] + planet_subim_dy * pix_per_Rp)
    l = l.astype(int)
    r = r.astype(int)
    b = b.astype(int)
    t = t.astype(int)
    

    try:
        subim = flexi_slice(ccd, b.value, t.value, l.value, r.value)
        nr, nc = subim.shape
        subim.data[subim.data < planet_subim_vmin] = planet_subim_vmin
        x = (np.arange(nc)*u.pixel - (center[0] - l)) / pix_per_Rp
        y = (np.arange(nr)*u.pixel - (center[1] - b)) / pix_per_Rp
        X, Y = np.meshgrid(x.value, y.value)
        pcm = ax.pcolormesh(X, Y, subim,
                            norm=LogNorm(vmin=planet_subim_vmin,
                                         vmax=planet_subim_vmax),
                            cmap=plot_planet_cmap,
                            shading='auto')
    except Exception as e:
        log.error(f'RAWFNAME of problem: {ccd.meta["RAWFNAME"]} {e}')
        return ccd_in
    ax.set_ylabel(planet_subim_axis_label, labelpad=0)
    ax.set_xlabel(planet_subim_axis_label, labelpad=0)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.axis('scaled')
    ax.margins(x=0, y=0)
    #plt.axis('equal')
    cbar = fig.colorbar(pcm, ax=ax)
    cbar.ax.set_xlabel(ccd.unit.to_string())
    date_obs, time_obs = ccd.tavg.fits.split('T')
    time_obs, _ = time_obs.split('.')
    #plt.title(f'{date_obs} {os.path.basename(outname)}')
    #fig.suptitle(f'{date_obs} {time_obs} UT')
    ax.set_title(f'{date_obs} {time_obs} UT')
    if plot_planet_overlay:
        plot_planet_overlay(ax, **kwargs)

    plt.tight_layout()
    outroot, _ = os.path.splitext(outname)
    d = os.path.dirname(outname)
    os.makedirs(d, exist_ok=True)
    savefig_overwrite(outroot + '.png')
    # See note in standard_star.py about the gc.collect().  Hoping
    # that this solves X freeze-ups
    plt.close()
    gc.collect()

    return ccd_in

