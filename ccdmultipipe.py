import os
import psutil

import multiprocessing
# For NoDaemonPool we must import this explicitly, it is not
# imported by the top-level multiprocessing module.
import multiprocessing.pool

from astropy import units as u
from astropy.nddata import CCDData
import ccdproc as ccdp

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
    """Subclass of multiprocessing.pool that allows child processes to spawn"""
    Process = NoDaemonProcess

class PoolWorkerKwargs():
    """Class to hold static kwargs for use with Pool.map or Pool.starmap.

    Parameters
    ----------
    function : function
    	Function called by worker() 

    **kwargs : kwargs
    	kwargs to be passed to function

    Methods
    -------
    worker(*args)
    	Call function(*args, **kwargs)

    Example
    -------
        pwk = PoolWorkerKwargs(my_favorite_function,
                               kwarg1='Hello world',
                               kwarg2=42)
        result = pwk.worker(1, 2, 3)

        is equivalent to

        result = my_favorite_function(1, 2, 3, 
		 		      kwarg1='Hello world', 
				      kwarg2=42)



    """
    def __init__(self,
                 function,
                 **kwargs):
        self.function = function
        self.kwargs = kwargs
    def worker(self, *args):
        return self.function(*args, **self.kwargs)

def prune_pout(pout, input_fnames):
    """Removes entries in a ccd_pipeline output list

    Parameters
    ----------
    pout : list of tuples (`str` or None, `dict`)
    	Output of a ccd_pipeline run.  The `str` correspond to
        pipeline output filenames, the `dict` is a dictionary
        containing output metadata.

    input_fnames : list of `str`
    	Input fnames to a ccd_pipeline run.  There will be one pout
    	for each input_fname

    Returns
    -------
    (pruned_pout, pruned_input_fnames) : list of tuples (`str`, `dict`)
        Pruned output.  To mark input/output pair for pruning, set the
        output fname in ccd_pipeline to None.  This can be done at any
        level: pre_process_list, ccd_processor, or post_process_list
    """
    pruned_pout = []
    pruned_fnames = []
    for i in range(len(pout)):
        if pout[i][0] is None:
            # ccd is None
            continue
        pruned_pout.append(pout[i])
        pruned_fnames.append(input_fnames[i])
    return (pruned_pout, pruned_fnames)


def multi_logging(level, pipe_meta, message):
    """Implements logging on a per-process basis in ccd_pipeline post-processing"""
    # Work directly with the pipe_meta dictionary, thus a return value
    # is not needed
    if level in pipe_meta:
        pipe_meta[level].append(message)
    else:
        pipe_meta[level] = [message]

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
                    num_processes=None,
                    mem_frac=0.8,
                    process_size=None,
                    error_if_zero=True):
    if num_processes is None:
        num_processes = psutil.cpu_count(logical=False)
    mem = psutil.virtual_memory()
    max_mem = mem.available*mem_frac
    if process_size is None:
        max_n = num_processes
    else:
        max_n = int(max_mem / process_size)
    if error_if_zero and max_n == 0:
        raise EnvironmentError(f'Current memory {max_mem/2**20} MiB insufficient for process size {process_size/2**20} MiB')

    return min(num_to_process, num_processes, max_n)

def ccd_pipeline(fnames,
                 num_processes=None,
                 mem_frac=0.8,
                 process_size=None,
                 PoolClass=None,
                 **kwargs):
    if PoolClass is None:
        PoolClass = multiprocessing.Pool
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
    with PoolClass(processes=ncp) as p:
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
