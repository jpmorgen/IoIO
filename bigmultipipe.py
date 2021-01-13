"""Provides tools for parallel pipeline processing of large data structures

The file bigmultipipe_example.py shows an example of how to develop a
bigmultipipe pipeline starting from code that processes large files
one at a time in a simple for loop.

Discussion of use: The bigmultipipe module provides tools that enable
a flexible, modular approach to constructing data processing pipelines
that optimize computer processing, memory, and disk I/O resources.
The BigMultiPipe base class is subclassed by the user to connect the
file reading, file writing, and data processing methods to the user's
existing processing code.  The BigMultiPipe.pipeline() method runs the
pipeline, maximizing the host computer's processing resources.
Keywords are available to taylor memory and processor use, the most
important of which being process_size, the maximum size in bytes of an
individual process.  Two optional keywords, pre_process_list and
post_process_list can contain lists of functions to be run on the data
before and after the primary processing step.  These keywords enable
additional flexibility in the creation and modification of the
pipeline at object instantiation and/or pipeline runtime.

Discussion of Design: The bigmultipipe module uses a three-stream
approach to address the problem of parallel processing large data
structures.  Stream (1) is the data, which begins and ends on disk,
thus side-stepping the issues of inter-process communication discussed
in the Background Section.  Stream (2) is control.  This stream is
intended to control the primary processing step, but can also control
pre-processing, post processing, file name creation and file writing.
The control stream starts as the keywords that are provided to a
BigMultiPipe object on instantiation.  Using Python's flexible **kwarg
feature, these keywords can be supplemented or overridden when the
BigMultiPipe.pipeline() method is called.  The functions in
pre_process_list can similarly supplement or override these keywords.
Finally, there is stream (3), the output metadata.  Stream (3) is
returned to the caller along with the output filename of each
processed file for use in subsequent processing steps.  Stream (3) can
be used to minimize the number of times the large output data files
are re-read during subsequent processing.   That said, as discussed in
the Background section, the amount of information returned as metadata
should be modest in size.

Background: The parallel pipeline processing of large data structures
is best done in a multithreaded environment, which enables the data to
be easily shared between threads executing in the same process.
Unfortunately, Python's Global Interpreter Lock (GIL) prevents
multiple threads from running at the same time, except in certain
cases, such as I/O wait and some numpy array operations.  Python's
multiprocessing module provides a partial solution to the GIL dilemma.
The multiprocessing module launches multiple independent Python
processes, thus providing true concurrent parallel processing in a way
that does not depend on the underlying code being executed.  The
multiprocessing module also provides tools such as Queue and Pipe
objects that enable communication between these processes.
Unfortunately, these inter-process communication solutions are not
quite as flexible as shared memory between threads in one process
because data must be transfered via pipes between the independent
processes.  This transfer is done using pickle: data are pickled on
one end of the pipe and unpickled on the other.  Depending on the
complexity and size of the object, the pickle/unpickle process can be
very inefficient.  The bigmultipipe module provides a basic framework
for avoiding all of these problems by implementing the three-stream
approach described in the Discussion of Design section.  Interprocess
communication requiring pickle still occurs, however, only filenames
and (hopefully) modest-sized metadata is exchanged in this way.

Statement of scope: This module is best suited for the simple case of
a "straight through" pipeline: one input file to one output file.  For
more complicated pipeline topologies, the MPipe module may be useful:
https://vmlaker.github.io/mpipe/.  For parallel processing of loops
that include certain numpy operations and other optimization tools,
numba may be useful: https://numba.pydata.org/.  Although it has yet
to be tested, bigmultipipe should be mutually compatible with either
or both of these other packages, although the NoDaemonPool version of
multiprocessing.Pool may need to be used if multiple levels of
multiprocessing are being conducted.

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
    """Make 'daemon' attribute always return False, thus enabling
    child processes to run their own processes"""
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class NoDaemonPool(multiprocessing.pool.Pool):
    """Allows child processes to run their own multiple processes"""
    Process = NoDaemonProcess

class WorkerWithKwargs():
    """
    Class to hold static kwargs for use with, e.g., multiprocessing.Pool.map

    Parameters
    ----------
    function : function
        Function called by WorkerWithKwargs.worker() method

    **kwargs : kwargs
        kwargs to be passed to function

    Methods
    -------
    worker(*args)
        Call function(*args, **kwargs)

    Example
    -------
    def add_mult(a, to_add=0, to_mult=1):
        return (a + to_add) * to_mult
    
    wwk = WorkerWithKwargs(add_mult, to_add=3, to_mult=10)
    print(wwk.worker(3))
    60

    is equivalent to

    print(add_mult(3, to_add=3, to_mult=10))
    60

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
        it is less than the maximum possible simultaneous processes.
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
    """Base class for memory- and processing power-optimized pipelines

    Parameters
    ----------
    NOTE: All parameters passed at object instantiation are stored as
    property and used to initialize the identical list of parameters
    to the :func:`BigMultiPipe.pipeline` method.  Any of these
    parameters can be overridden when that method is called.  This
    enables definition of a default pipeline configuration when the
    object is instantiated that can be modified at run-time.

    num_processes, mem_available, mem_frac, process_size, optional
        These parameters tune computer processing and memory resources
        and are used when the :func:`pipeline` method is executed.
        See documentation for :func:`num_can_process` for use, noting
        that the num_to_process argument of that function is
        set to the number of input filenames in :func:`num_can_process`

    outdir : str, None, optional
    	Name of directory into which output files will be written.  If
    	None, current directory in which the Python process is running
    	will be used.
        Default is `None`

    create_outdir : bool, optional
        If True, create outdir and any needed parent directories.
        Does not raise an error if outdir already exists.
        Default is `False`

    outname_append: str, optional
        String to append to outname to avoid risk of input file
        overwrite.  Example input file "test.dat" would become output
        file "test_bmp.dat"
        Default is ``_bmp``

    pre_process_list : list
        List of functions called by :func:`pre_process` before primary
        processing step.  Intended to implement filtering and control
        features of bigmultipipe.  Each function must accept one
        positional parameter, the data, any keyword arguments
        necessary for its internal functioning, and **kwargs, keyword
        parameters not processed by the function.  Example:

        def boost_later(data, boost_target=None, boost_amount=None, **kwargs):

        The return value of each function must be a tuple, of the form
        (data, additional_keywords).  Example:

        return (data, {'need_to_boost_by': boost_amount})

        If data is returned as ``None``, processing of that file
        stops, no output file is written, and None is returned instead
        of an output filename.  See bigmultipipe_example.py for
        example code.

    post_process_list : list
        List of functions called by :func:`post_process` after primary
        processing step.  Indended to enable additional processing
        steps and produce metadata for return to the user.  Each
        function must accept two positional parameters, data and meta
        and any optional **kwargs.  Meta will be of type `dict`.
        Example:

        def later_booster(data, meta, need_to_boost_by=None, **kwargs):

        The return value of each function must be a tuple in the form
        (data, additional_meta).  Where additional_metadata must be of
        type `dict.`  Example:

        return (data, {'average': np.average(data)})

        Alternately, the meta can be modified directly in the function
        and {} returned as the additional_meta.  See
        bigmultipipe_example.py for example code.

    PoolClass : class name or None, optional
        Typcally a subclass of `multiprocessing.Pool.`  The map()
        method of this class implements the multiprocessing feature of
        this module.  If None, `multiprocessing.Pool` is used.
        Default is ``None.``

    kwargs : dict, optional
        Python's **kwargs construct stores additional keyword
        arguments as a dict.  In order to implement the control stream
        discussed in the introduction to this module, this dict is
        captured as property.  When any methods are run, the kwargs
        from the property are copied to a new dict object and combined
        with the **kwargs passed to the method using dict.update().
        This allows the parameters passed to the methods at runtime to
        override the parameters passed to the object at instantiation
        time.  This also provides a mechanism for any function in the
        system to modify the kwargs dict for flexible per-file control
        of the pipeline.

    """
    def __init__(self,
                 num_processes=None,
                 mem_available=None,
                 mem_frac=0.8,
                 process_size=None,
                 outdir=None,
                 create_outdir=False,
                 outname_append='_bmp',
                 pre_process_list=None,
                 post_process_list=None,
                 PoolClass=None,
                 **kwargs):
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
        self.kwargs = kwargs

    def pipeline(self, in_names,
                 num_processes=None,
                 mem_available=None,
                 mem_frac=None,
                 process_size=None,
                 PoolClass=None,
                 **kwargs):
        """Runs pipeline, maximizing processing and memory resources

        Parameters
        ----------
        in_names : list of str
            List of input filenames.  Each file is processed using
            :func:`file_process`

        All other parameters, see documentation for BigMultiPipe

        Returns
        -------
        pout : list of tuples (outname, meta), one tuple or in_name
            Outname is str or None.  If str, it is the name of the
            file to which the processed data were written.  If None,
            the convenience function :func:`prune_pout` can be used to
            remove this tuple from pout and the corresponding in_name
            from the in_names list.  Meta is a dict containing output.

        """
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
        # Allow overriding of self.kwargs by **kwargs
        skwargs = self.kwargs.copy()
        skwargs.update(kwargs)
        kwargs = skwargs
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
        """Process one file in the bigmultipipe system

        This method can be overridden to interface with applications
        where the primary processing routine already reads the input
        data from disk and writes the output data to disk,

        Parameters
        ----------
        in_name: str
            Name of file to process.  Data from the file will be read
            by :func:`file_read` and processed by
            :func:`data_process_meta_create`.  Output filename will be
            created by :func:`outname_create` and data will be written by
            :func:`file_write`

        kwargs see NOTE in BigMultiPipe Parameter section

        Returns
        -------
        (outname, meta) : tuple
            Outname is the name of file to which processed data was
            written.  Meta is the dictionary element of the tuple
            returned by func::`data_process_meta_create`

        """
        # Allow overriding of self.kwargs by **kwargs
        skwargs = self.kwargs.copy()
        skwargs.update(kwargs)
        kwargs = skwargs
        data = self.file_read(in_name, **kwargs)
        if data is None:
            return (None, {})
        data, meta = \
            self.data_process_meta_create(data, in_name=in_name, **kwargs)
        if data is None:
            return (None, meta)
        outname = self.outname_create(in_name, data, meta, **kwargs)
        outname = self.file_write(data, outname, **kwargs)
        return (outname, meta)

    def file_read(self, in_name, **kwargs):
        """Reads data file from disk.  Intended to be overridden by subclass

        Parameters
        ----------
        in_name : str
            Name of file to read

        kwargs see NOTE in BigMultiPipe Parameter section

        Returns
        -------
        data : any type
            Data to be processed
        """
        
        # Allow overriding of self.kwargs by **kwargs
        skwargs = self.kwargs.copy()
        skwargs.update(kwargs)
        kwargs = skwargs
        with open(in_name, 'rb') as f:
            data = f.read()
        return data

    def file_write(self, data, outname, **kwargs):
        """Write data to disk file.  Intended to be overridden by subclass

        Parameters
        ----------
        data : any type
            Data to be processed

        outname : str
            Name of file to write

        kwargs see NOTE in BigMultiPipe Parameter section

        Returns
        -------
        outname : str
            Name of file written
        """
        # Allow overriding of self.kwargs by **kwargs
        skwargs = self.kwargs.copy()
        skwargs.update(kwargs)
        kwargs = skwargs
        with open(outname, 'wb') as f:
            f.write(data)
        return outname

    def data_process_meta_create(self, data, **kwargs):
        """Process data and create metadata

        Parameters
        ----------
        data : any type
            Data to be processed by :func:`data_pre_process`,
            :func:`data_process`, and :func:`data_post_process`

        kwargs see NOTE in BigMultiPipe Parameter section

        Returns
        -------
        (data, meta) : tuple
            Data is the processed data.  Meta is created by
            :func:`data_post_process`

        """
        # Allow overriding of self.kwargs by **kwargs
        skwargs = self.kwargs.copy()
        skwargs.update(kwargs)
        kwargs = skwargs
        (data, kwargs) = self.pre_process(data, **kwargs)
        if data is None:
            return(None, {})
        data = self.data_process(data, **kwargs)
        data, meta = self.data_post_process(data, **kwargs)
        return (data, meta)

    def pre_process(self, data,
                    pre_process_list=None,
                    **kwargs):
        """Conduct pre-processing tasks

        This method can be overridden to permanently insert
        pre-processing tasks in the pipeline for each instantiated
        object and/or the pre_process_list feature can be used for a
        more dynamic approach to inserting pre-processing tasks at
        object instantiation and/or when the pipeline is run

        Parameters
        ----------
        data : any type
            Data to be processed by the functions in pre_process_list

        pre_process_list : list
            See documentation for BigMultiPipe

        kwargs see NOTE in BigMultiPipe Parameter section

        Returns
        -------
        (data, kwargs) : tuple
            Data are the pre-processed data.  Kwargs are the combined
            kwarg outputs from all of the pre_process_list functions.

        """
        # Allow overriding of self.kwargs by **kwargs
        skwargs = self.kwargs.copy()
        skwargs.update(kwargs)
        kwargs = skwargs
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
        """Process the data.  Intended to be overridden in subclass

        Parameters
        ----------
        data : any type
            Data to be processed

        Returns
        -------
        data : any type
            Processed data
        """
        # Allow overriding of self.kwargs by **kwargs
        skwargs = self.kwargs.copy()
        skwargs.update(kwargs)
        kwargs = skwargs
        # Insert call to processing code here
        return data

    def data_post_process(self, data,
                     post_process_list=None,
                     **kwargs):
        """Conduct post-processing tasks, including creation of metadata

        This method can be overridden to permanently insert
        post-processing tasks in the pipeline for each instantiated
        object or the post_process_list feature can be used for a more
        dynamic approach to inserting post-processing tasks at object
        instantiation and/or when the pipeline is run

        Parameters
        ----------
        data : any type
            Data to be processed by the functions in pre_process_list

        post_process_list : list
            See documentation for BigMultiPipe

        kwargs see NOTE in BigMultiPipe Parameter section

        Returns
        -------
        (data, meta) : tuple
            Data are the post-processed data.  Meta are the combined
            meta dicts from all of the post_process_list functions.

        """
        # Allow overriding of self.kwargs by **kwargs
        skwargs = self.kwargs.copy()
        skwargs.update(kwargs)
        kwargs = skwargs
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
        """Create output filename

        Parameters
        ----------
        in_name : str
            Name of input raw data file

        data : any type
            Processed data

        All other parameters, see documentation for BigMultiPipe

        """
        # Allow overriding of self.kwargs by **kwargs
        skwargs = self.kwargs.copy()
        skwargs.update(kwargs)
        kwargs = skwargs
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
    post-processing routines

    Parameters
    ----------
    level : str
        Log message level (e.g., 'debug, info, warn, error')

    meta : dict
        The meta channel of a BigMultiPipe pipeline

    message : str
        Log message

    """
    # Work directly with the meta dictionary, thus a return value
    # is not needed
    if level in meta:
        meta[level].append(message)
    else:
        meta[level] = [message]
