"""Provides tools for parallel pipeline processing of large data structures

The bigmultipipe module uses a three-stream approach to address the
problem of handling large data structures in a Python multiprocessing
environment while simultaneously minimizing the amount of disk I/O.
Stream (1) is the data, which begins and ends on disk, thus
side-stepping the issues of interprocess communication discussed in
the Background Section.  Stream (2) is control.  Using Python's
flexible **kwarg feature, keyword arguments that control the
underlying machinery of the pipeline can be adjusted on a per-file
basis.  Alternately, the object can be subclassed and tailored as
necessary.  Stream (3) is metadata.  This could be anything, but is
generally a reasonably-sized reduction product from the pipeline run
of each file.  Stream (3) is not written to disk, but, along with the
output filename of the processed file, is returned to the caller for
use in subsequent processing steps.

Example code is found in bigmultipipe_test.py

BigMultiPipe that allows the end user to specify their own file reader
and writer.  For ease of workflow, processing of the data is divided
into three stages: (1) pre-processing, (2) processing, and (3)
post-processing.

This module is best suited for the simple case of one input file to
one output file.  For more complicated pipeline structures, the Mpipe
module may be useful: https://vmlaker.github.io/mpipe/

Background: The parallel pipeline processing of large data structures
is best done in a multithreaded environment, which enables the data to
be easily shared between threads executing in the same process.
Unfortunately, Python's Global Interpreter Lock (GIL) prevents
multiple threads from running at the same time, except in
circumstances such as I/O wait and certain functions in cpython
packages such as numpy.  Python's multiprocessing module provides a
partial solution to the GIL dilema by providing tools that launch
multiple independent Python processes.  The multiprocessing module
also provides tools such as Queue and Pipe objects that enable
communication between these processes.  Unfortunately, these
interprocess communication solutions rely on pickle, which is usually
not suitable for large data structures.  Thus, a multiprocessing
solution for big data needs to implement the disk read of raw data and
write of processed data within each independent process.  This enables
filenames to be passed between processes rather than the large data
structures themselves.  The bigmultipipe module expands on this
minimum of information exchange by providing additional control and
metadata channels, as discussed above.

"""

import os
import psutil

import multiprocessing
# For NoDaemonPool we must import this explicitly, it is not
# imported by the top-level multiprocessing module.
import multiprocessing.pool

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

class WorkerWithKwargs():
    """
    Class to hold static kwargs for use with, e.g., multiprocessing.Pool.map

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
        wwk = WorkerWithKwargs(my_favorite_function,
                               kwarg1='Hello world',
                               kwarg2=42)
        result = wwk.worker(1, 2, 3)

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

def num_can_process(num_to_process=None,
                    num_processes=None,
                    mem_available=None,
                    mem_frac=0.8,
                    process_size=None,
                    error_if_zero=True):
    """
    Calculates maximum number of processes that can run simultaneously

    Parameters
    ----------
    num_to_process : int or None, optional
        Total number of items to process.  This number is returned if
        it is less than the maximum possible simultanous processes.
	If None, not used in calculation.
        Default is ``None``

    num_processes : int or None, optional
        Maximum number of parallel processes.  If None, set to the
        number of physical (not logical) cores available using
        :func:`psutil.cpu_count(logical=False)`
        Default is ``None``

    mem_available : int or None. optional
    	Amount of memory available in bytes for the total set of 
        processes.  If None, mem_frac parameter is used.
    	Default is ``None``

    mem_frac : float, optional
        Maximum fraction of current memory available that total set of 
        processes is allowed to occupy.  Current memory available is
        queried using :func:`psutil.virtual_memory()`
        Default is ``0.8``

    process_size : int, or None, optional
        Maximum process size in bytes of an individual process.  If
        None, processes are assumed to be small enough to all fit in
        memory at once.
    	Default is ``None``

    error_if_zero : bool, optional
    	If True, throw an error if return value would be zero.  Useful
    	for catching case when there is not enough memory for even
    	one process.
        Default is `True`

    """
    if num_processes is None:
        num_processes = psutil.cpu_count(logical=False)
    if num_to_process is None:
        num_to_process = num_processes
    if mem_available is not None:
        max_mem = mem_available
    else:
        mem = psutil.virtual_memory()
        max_mem = mem.available*mem_frac
    if process_size is None:
        max_n = num_processes
    else:
        max_n = int(max_mem / process_size)
    if error_if_zero and max_n == 0:
        raise EnvironmentError(f'Current memory {max_mem/2**20} MiB insufficient for process size {process_size/2**20} MiB')

    return min(num_to_process, num_processes, max_n)

class BigMultiPipe():
    def __init__(self,
                 num_processes=None,
                 mem_available=None,
                 mem_frac=0.8,
                 process_size=None,
                 pre_process_list=None,
                 post_process_list=None,
                 PoolClass=None,
                 outdir=None,
                 create_outdir=False,
                 outname_append='_bmp'):
        self.num_processes = num_processes
        self.mem_available = mem_available
        self.mem_frac = mem_frac
        self.process_size = process_size
        if pre_process_list is None:
            pre_process_list = []
        if post_process_list is None:
            post_process_list = []
        self.pre_process_list = pre_process_list
        self.post_process_list = post_process_list
        if PoolClass is None:
            PoolClass = multiprocessing.Pool
        self.PoolClass = PoolClass
        self.outdir = outdir
        self.create_outdir = create_outdir
        self.outname_append = outname_append

    def pipeline(self, in_names,
                 num_processes=None,
                 mem_available=None,
                 mem_frac=None,
                 process_size=None,
                 PoolClass=None,
                 **kwargs):
        if num_processes is None:
            num_processes = self.num_processes
        if mem_available is None:
            mem_available = self.mem_available
        if mem_frac is None:
            mem_frac = self.mem_frac
        if process_size is None:
            process_size = self.process_size
        if PoolClass is None:
            PoolClass = self.PoolClass
        ncp = num_can_process(len(in_names),
                              num_processes=num_processes,
                              mem_available=mem_available,
                              mem_frac=mem_frac,
                              process_size=process_size)
        wwk = WorkerWithKwargs(self.file_process, **kwargs)
        if ncp == 1:
            retvals = [wwk.worker(i) for i in in_names]
            return retvals
        with PoolClass(processes=ncp) as p:
            retvals = p.map(wwk.worker, in_names)
        return retvals
        
    def file_process(self, in_name, **kwargs):
        data = self.file_reader(in_name, **kwargs)
        data, meta = \
            self.data_process_meta_create(data, in_name=in_name, **kwargs)
        if data is None:
            return (None, meta)
        outname = self.outname_create(in_name, data, meta, **kwargs)
        outname = self.file_writer(data, outname, **kwargs)
        return (outname, meta)

    def file_reader(self, in_name, **kwargs):
        with open(in_name, 'rb') as f:
            data = f.read()
        return data

    def file_writer(self, data, outname, **kwargs):
        with open(outname, 'wb') as f:
            f.write(data)
        return outname

    def data_process_meta_create(self, data, **kwargs):
        (data, kwargs) = self.pre_process(data, **kwargs)
        if data is None:
            return(None, {})
        data = self.data_process(data, **kwargs)
        data, meta = self.meta_create(data, **kwargs)
        return (data, meta)

    def pre_process(self, data,
                    pre_process_list=None,
                    **kwargs):
        if pre_process_list is None:
            pre_process_list = []
        pre_process_list = self.pre_process_list + pre_process_list
        for pp in pre_process_list:
            data, these_kwargs = pp(data, **kwargs)
            kwargs.update(these_kwargs)
            if data is None:
                return (None, kwargs)
            
        return (data, kwargs)

    def data_process(self, data, **kwargs):
        return data

    def meta_create(self, data,
                    post_process_list=None,
                    **kwargs):
        if post_process_list is None:
            post_process_list = self.post_process_list
        meta = {}
        if post_process_list is None:
            post_process_list = []
        for pp in post_process_list:
            data, this_meta = pp(data, meta, **kwargs)
            meta.update(this_meta)
        return (data, meta)

    def outname_create(self, in_name, data, meta,
                       outdir=None,
                       create_outdir=None,
                       outname_append=None,
                       **kwargs):
        if outdir is None:
            outdir = self.outdir
        if create_outdir is None:
            create_outdir = self.create_outdir
        if outname_append is None:
            outname_append = self.outname_append
        
        if not (isinstance(in_name, str)
                and isinstance(outname_append, str)):
            raise ValueError("Not enough information provided to create output filename.  Specify outname or use an input filename and specify a string to append to that output filename to assure input is not overwritten")
        if outdir is None:
            outdir = os.getcwd()
        if create_outdir:
            os.makedirs(outdir, exist_ok=True)
        if not os.path.isdir(outdir):
            raise ValueError(f"outdir {outdir} does not exist.  Create directory or use create_outdir=True")
        bname = os.path.basename(in_name)
        prepend, ext = os.path.splitext(bname)
        outbname = prepend + outname_append + ext
        outname = os.path.join(outdir, outbname)
        return outname

def prune_pout(pout, in_names):
    """
    Removes entries marked for deletion in a BigMultiPipe.pipeline() output

    Parameters
    ----------
    pout : list of tuples (`str` or None, `dict`)
    	Output of a BigMultiPipe.pipeline() run.  The `str` are
        pipeline output filenames, the `dict` is the output metadata.

    in_names : list of `str`
    	Input file names to a BigMultiPipe.pipeline() run.  There will
    	be one pout for each in_name

    Returns
    -------
    (pruned_pout, pruned_in_names) : list of tuples (`str`, `dict`)
        Pruned output with the None output filenames removed in both
        the pout and in_name lists.
    """
    pruned_pout = []
    pruned_in_names = []
    for i in range(len(pout)):
        if pout[i][0] is None:
            # bmp is None
            continue
        pruned_pout.append(pout[i])
        pruned_in_names.append(in_names[i])
    return (pruned_pout, pruned_in_names)


def multi_logging(level, meta, message):
    """Implements logging on a per-process basis in BigMultiPipe
    post-processing routines"""
    # Work directly with the meta dictionary, thus a return value
    # is not needed
    if level in meta:
        meta[level].append(message)
    else:
        meta[level] = [message]

