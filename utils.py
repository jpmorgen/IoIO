"""Utilities for the IoIO data collection and reduction system"""

import inspect
import glob
import os
import time
import datetime
from pathlib import Path

import csv

import numpy as np
from numpy.polynomial import Polynomial

import matplotlib.pyplot as plt

from astropy import log

import astropy.units as u
from astropy.time import Time

from bigmultipipe import assure_list

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

def multi_glob(directory, glob_list=None):
    """Returns list of files matching one or more regexp

    Parameters
    ----------
    directory : str
        Directory in which to search for files

    glob_list : str or list
        (list of) regexp to pass to `glob.glob` used to construct flist

    Returns
    -------
    flist : list
        List of filenames
    """
    glob_list = assure_list(glob_list)
    flist = []
    for gi in glob_list:
        flist += glob.glob(os.path.join(directory, gi))
    return flist
    
def is_flux(unit):
    """Determine if we are in flux units or not"""
    unit = unit.decompose()
    if isinstance(unit, u.IrreducibleUnit):
        return False
    # numpy tests don't work with these objects, so do it by hand
    spower = [p for un, p in zip(unit.bases, unit.powers)
              if un == u.s]
    if len(spower) == 0 or spower[0] != -1:
        return False
    return True

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

def valid_long_exposure(r):
    """Inspects FITS header or ImageFileCollection row for condition"""
    return (r['imagetyp'].lower() == 'light'
            and r['xbinning'] == 1
            and r['ybinning'] == 1
            and r['exptime'] > 10)

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

def iter_polyfit(x, y, poly_class=Polynomial, deg=1, max_resid=None,
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
                                poly_class=Polynomial,
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

def cached_csv(dict_list_code,
               csvnames=None,
               read_csvs=False,
               write_csvs=False,
               create_outdir=False,
               newline='',
               quoting=csv.QUOTE_NONNUMERIC,
               **kwargs):
    """Write/read list(s) of dict to csv file(s)

    Parameters
    ----------
    dict_list_code : function
        Function that generates list(s) of dict(s)  if `read_csvs`
        is ``False`` or `poutname` cannot be read

    csvnames : str or list of str
        Filename(s) to be read/written.  There must be on filename per
        list of dict returned by dict_list_code

    read_csvs : bool
        If `True` read csv(s) from `csvnames`
        Default is `False`

    write_pout : bool
        If `True` write csv(s) to filename(s) listed in `csvnames`
        Default is `False`

    create_outdir : bool, optional
        If ``True``, create any needed parent directories.
        Does not raise an error if outdir already exists.
        Default is ``False``

    newline : str
        Default is ''

    quoting : int
        csv quoting type

    **kwargs : keyword arguments to pass to `dict_list_code`

    Returns
    -------
    dict_lists : list of dict
        List of dictionaries read from csv file(s), one list per
        filename in csvnames

    """
    
    single_dictlist = isinstance(csvnames, str)
    if single_dictlist:
        csvnames = [csvnames]

    if read_csvs:
        try:
            dict_lists = []
            for csvname in csvnames:
                dict_list = []
                with open(csvname, newline=newline) as csvfile:
                    csvr = csv.DictReader(csvfile, quoting=quoting)
                    for row in csvr:
                        dict_list.append(row)
                dict_lists.append(dict_list)
            if single_dictlist:
                dict_lists = dict_lists[0]
            return dict_lists
        except Exception as e:
            log.debug(f'Running code because received exception {e}')
            pass

    # If we made it here, we need to generate our list(s) of dicts
    dict_lists = dict_list_code(**kwargs)

    if write_csvs:
        if single_dictlist:
            dict_lists = [dict_lists]
        for csvname, dict_list in zip(csvnames, dict_lists):
            if create_outdir:
                os.makedirs(os.path.dirname(csvname), exist_ok=True)
            if len(dict_list) == 0:
                # Signal we have been here and found nothing
                Path(csvname).touch()
            else:
                fieldnames = list(dict_list[0].keys())
                with open(csvname, 'w', newline=newline) as csvfile:
                    csvdw = csv.DictWriter(csvfile, fieldnames=fieldnames,
                                           quoting=quoting)
                    csvdw.writeheader()
                    for d in dict_list:
                        csvdw.writerow(d)
    if single_dictlist:
        dict_lists = dict_lists[0]
    return dict_lists    

# https://stackoverflow.com/questions/27704490/interactive-pixel-information-of-an-image-in-python
class CCDImageFormatter(object):
    """Provides the x,y,z formatting I like for CCD images in the
    interactive pyplot window"""
    def __init__(self, im):
        self.im = im
    def __call__(self, x, y):
        s = self.im.shape
        col = int(x + 0.5)
        row = int(y + 0.5)
        if col >= 0 and col < s[1] and row >= 0 and row < s[0]:
            z = self.im[row, col]
            return 'x=%1.1f, y=%1.1f, z=%1.1f' % (x, y, z)
        else:
            return 'x=%1.1f, y=%1.1f' % (x, y)        

def simple_show(im, **kwargs):
    fig, ax = plt.subplots()
    ax.imshow(im, origin='lower',
              cmap=plt.cm.gray,
              filternorm=0, interpolation='none',
              **kwargs)
    ax.format_coord = CCDImageFormatter(im.data)
    plt.show()

def savefig_overwrite(fname, **kwargs):
    if os.path.exists(fname):
        os.remove(fname)
    plt.savefig(fname, **kwargs)

def location_to_dict(loc):
    """Useful for JPL horizons"""
    return {'lon': loc.lon.value,
            'lat': loc.lat.value,
            'elevation': loc.height.to(u.km).value}
