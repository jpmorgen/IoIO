"""Utilities for the IoIO data collection and reduction system"""

import inspect
import os
import datetime

import matplotlib.pyplot as plt

from astropy.time import Time

# These are MaxIm and ACP day-level raw data directories, respectively
DATE_FORMATS = ["%Y-%m-%d", "%Y%m%d"]

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
    ax.imshow(im.data, origin='lower',
              cmap=plt.cm.gray,
              filternorm=0, interpolation='none',
              **kwargs)
    ax.format_coord = CCDImageFormatter(im.data)
    plt.show()

def assure_list(x):
    """Assures x is type `list`"""
    if x is None:
        x = []
    if not isinstance(x, list):
        x = [x]
    return x

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
        row_selector = True
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
    summary_table = collection.summary
    row_dict = multi_row_selector(summary_table,
                                  keyword, value_pair,
                                  row_selector)

    pair_list = []
    for i_on in row_dict[value_pair[0]]:
        t_on = Time(summary_table[i_on]['date-obs'], format='fits')
        # Seach all second values
        t_offs = Time(summary_table[row_dict[value_pair[1]]]['date-obs'],
                      format='fits')
        if len(t_offs) == 0:
            continue
        dts = [t_on - T for T in t_offs]
        idx_best1 = np.argmin(np.abs(dts))
        # Unwrap
        i_off = row_dict[value_pair[1]][idx_best1]
        pair = [os.path.join(directory,
                             summary_table[i]['file'])
                             for i in (i_on, i_off)]
        pair_list.append(pair)
    return pair_list

def valid_long_exposure(r):
    """Inspects FITS header or ImageFileCollection row for condition"""
    return (r['imagetyp'].lower() == 'light'
            and r['xbinning'] == 1
            and r['ybinning'] == 1
            and r['exptime'] > 10)

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

