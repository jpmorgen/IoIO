"""Module which enables parallel pipeline processing of CCDData files

"""

from astropy import log
from astropy import units as u
from astropy.io import fits
from astropy.nddata import CCDData, fits_ccddata_reader
import ccdproc as ccdp

from bigmultipipe import BigMultiPipe


def fallback_unit_ccddata_reader(filename, *args, 
                                 unit=None,
                                 fallback_unit=None,
                                 **kwargs):
    if unit is not None:
        return fits_ccddata_reader(filename, *args,
                                   unit=unit, **kwargs)
    # Open the file and read the primary hdr to see if there is a
    # BUNIT there.  Do the open as an memmap to minimize overhead
    # on the second file read in fits_ccddata_reader.  Having the
    # underlying astropy code implement something like
    # hdul_ccddata_reader would save the second file open entirely
    with fits.open(filename, memmap=True) as hdul:
        hdr = hdul[0].header
        bunit = hdr.get('BUNIT')
        if bunit is not None:
            try:
                # fits_ccddata_reader will find BUNIT again.
                # We have to do it without unit=bunit to avoid
                # annoying message
                return fits_ccddata_reader(filename, *args, **kwargs)
            except ValueError:
                # BUNIT may not be valid
                log.warning(f'Potentially invalid BUNIT '
                            'value {bunit} detected in FITS '
                            'header.  Falling back to {fallback_unit}')
        # If we made it here, there is no BUNIT in the header or it is invalid
        return fits_ccddata_reader(filename, *args,
                                   unit=fallback_unit, **kwargs)

class FBUCCDData(CCDData):
    """
    Add ``fallback_unit`` capability to :meth:`CCDData.read() <~astropy.nddata.CCDData.read>`

    Paramters
    ---------
    filename : str
        Name of FITS file

    unit : `~astropy.units.Unit`, optional
        Units of the image data.   If ``None,`` the FITS header
        ``BUNIT`` keyword will be queried to set the unit.  If that
        is not found or an error is raised, `fallback_unit` will be
        used.
        Default is ``None``.

    fallback_unit : `~astropy.units.Unit`, optional
        Units to be used for the image data if `unit` is not provided
        and no valid ``BUNIT`` value is found in the FITS header.
        Default is ``None``.

    kwargs :
        Keywords to pass to :meth:`CCDData.read() <~astropy.nddata.CCDData.read>`

    """
    @classmethod
    def read(cls, filename, *args, 
             fallback_unit=None,
             **kwargs):
        ccd = fallback_unit_ccddata_reader(filename, *args, 
                                           fallback_unit=fallback_unit,
                                           **kwargs)
        return ccd
        # NOTE: to make this work with a more complex object, accept
        # the keyword arguments of that object explicitly into the
        # read method and do something like the code below.  Trying to
        # do a simple return of cls(*args, in_read=filename, **args) This is
        # necessary because of the structure of the underlying astropy
        # code and the way Python handles instantiation of objects in
        # classmethods: an empty object is instantiated with no
        # arguments
        # classmethods does not allow this to be subclassed in a
        # general way.  Therefore, for another object, all of that
        # class's __init__ keywords would need to be accepted to this
        # read method and handled in the following way
        ## Make a vestigial pgd
        #pgd = cls(ccd.data,
        #          unit=ccd.unit,
        #          obj_center=obj_center,
        #          desired_center=desired_center,
        #          quality=quality,
        #          date_obs_key=date_obs_key,
        #          exptime_key=exptime_key,
        #          darktime_key=darktime_key)
        ## Merge in ccd property
        #pgd.__dict__.update(ccd.__dict__)
        #return pgd

def ccddata_read(fname_or_ccd,
                 raw_unit=u.adu,
                 *args, **kwargs):
    """Convenience function to read a FITS file into a CCDData object.

    Catches the case where the raw FITS file does not have a BUNIT
    keyword, which otherwise causes :meth:`CCDData.read` to crash.  In
    this case, :func:`ccddata_read` assigns `ccd` units of
    ``raw_unit``.  Also ads comment to BUNIT "physical units of the
    array values," which is curiously omitted in the astropy fits
    writing system.  Comment is from `official FITS documentation
    <https://heasarc.gsfc.nasa.gov/docs/fcg/standard_dict.html>` where
    BUNIT is in the same family as BZERO and BSCALE

    Parameters
    ----------
    fname_or_ccd : str or `~astropy.nddata.CCDData`
        If str, assumed to be a filename, which is read into a
        `~astropy.nddata.CCDData`.  If `~astropy.nddata.CCDData`,
        return a copy of the CCDData with BUNIT keyword possibly 
        added

    raw_unit : str or `astropy.units.core.UnitBase`
        Physical unit of pixel in case none is specified 
        Default is `astropy.units.adu`

    *args and **kwargs passed to :meth:`CCDData.read
         <astropy.nddata.CCDData.read>` 

    Returns
    -------
    ccd : `~astropy.nddata.CCDData`
        `~astropy.nddata.CCDData` with units set to raw_unit if none
        specified in FITS file 

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

class CCDMultiPipe(BigMultiPipe):
    """Base class for `ccdproc` multiprocessing pipelines

    Parameters
    ----------
    process_size : int
        Maximum process size in bytes of an individual process.  If
        ``None``, calculated from ``naxisN``, ``bitpix`` and
        ``process_expand_factor`` parameters.
        Default is ``None``

    naxis1, naxis2 : int or None
        CCD image size.  If ``None``, read from *first* image
        processed in pipeline.
        Default is ``None``

    bitpix : int
        Maximum bits per pixel during processing.
        Default is 2*64 + 8 = 136, which includes primary and error
        image as double-precision and mask image as byte

    process_expand_factor : float
        Factor to account for the fact that
        :func:`ccdproc.ccd_process` and each of the routines it calls
        make a copy of the data passed to them.  Thus, at any one time
        their can be up to 3 copies of the `~astropy.nddata.CCDData`
        Default is 3.5

    raw_unit : str or `astropy.units.core.UnitBase`
        For reading files: physical unit of pixel in case none is
        specified.
        Default is `astropy.units.adu`

    outname_append : str, optional
        String to append to outname to avoid risk of input file
        overwrite.  Example input file ``test.fits`` would become
        output file ``test_ccdmp.fits``
        Default is ``_ccdmp``

    overwrite : bool
        Set to ``True`` to enable overwriting of output files of the
        same name.
        Default is ``False``

    kwargs : kwargs
        Passed to __init__ method of :class:`bigmultipipe.BigMultiPipe`

    """

    def __init__(self,
                 ccddata_cls=None,
                 process_size=None,
                 naxis1=None,
                 naxis2=None,
                 bitpix=None,
                 process_expand_factor=3.5,
                 outname_append='_ccdmp',
                 overwrite=False,
                 **kwargs):
        self.naxis1 = naxis1
        self.naxis2 = naxis2
        if bitpix is None:
            bitpix = 2*64 + 8
        self.bitpix = bitpix
        self.process_expand_factor = process_expand_factor
        self.process_size = process_size
        self.ccddata_cls = CCDData if ccddata_cls is None else ccddata_cls
        self.overwrite = overwrite
        super().__init__(outname_append=outname_append,
                         **kwargs)

    def pipeline(self, in_names,
                 process_size=None,
                 naxis1=None,
                 naxis2=None,
                 bitpix=None,
                 process_expand_factor=None,
                 **kwargs):
        """Runs pipeline, maximizing processing and memory resources

        Parameters
        ----------
        in_names : `list` of `str`
            List of input filenames.  Each file is processed using
            :func:`file_process`

        All other parameters : see Parameters to :class:`CCDMultiPipe`
        and :class:`bigmultipipe.BigMultiPipe` 

        Returns
        -------
        pout : `list` of tuples ``(outname, meta)``, one `tuple` for each
            ``in_name``.  ``Outname`` is `str` or ``None``.  If `str`,
            it is the name of the file to which the processed data
            were written.  If ``None``, the convenience function
            :func:`prune_pout` can be used to remove this tuple from
            ``pout`` and the corresponding in_name from the in_names list.
            ``Meta`` is a `dict` containing output.

        """
        if process_size is None:
            process_size = self.process_size
        if naxis1 is None:
            naxis1 = self.naxis1
        if naxis2 is None:
            naxis2 = self.naxis2
        if bitpix is None:
            bitpix = self.bitpix
        if process_expand_factor is None:
            process_expand_factor = self.process_expand_factor
        if (process_size is None):
            if naxis1 is None or naxis2 is None:
                ccd = self.file_read(in_names[0], **kwargs)
            if naxis1 is None:
                naxis1 = ccd.meta['NAXIS1']
            if naxis2 is None:
                naxis2 = ccd.meta['NAXIS2']
            process_size = (naxis1 * naxis2
                            * bitpix/8
                            * process_expand_factor)
        return super().pipeline(in_names, process_size=process_size,
                                **kwargs)

    def file_read(self, in_name, **kwargs):
        """Reads FITS file from disk into `~astropy.nddata.CCDData`

        Parameters
        ----------
        in_name : str
            Name of FITS file to read

        kwargs : kwargs
            Passed to :func:`ccddata_read`
            See also: Notes in :class:`bigmultipipe.BigMultiPipe`
            Parameter section 

        Returns
        -------
        data : :class:`~astropy.nddata.CCDData`
            :class:`~astropy.nddata.CCDData` to be processed

        """
        kwargs = self.kwargs_merge(**kwargs)
        data = ccddata_cls.read(in_name, **kwargs)
        return data

    def file_write(self, data, outname, 
                    overwrite=None,
                    **kwargs):
        """Write `~astropy.nddata.CCDData` as FITS file file.

        Parameters
        ----------
        data : `~astropy.nddata.CCDData`
            Processed data

        outname : str
            Name of file to write

        kwargs : kwargs
            Passed to :meth`CCDData.write() <astropy.nddata.CCDData.write>`
            See also: Notes in :class:`bigmultipipe.BigMultiPipe`
            Parameter section 

        Returns
        -------
        outname : str
            Name of file written

        """
        kwargs = self.kwargs_merge(**kwargs)
        if overwrite is None:
            overwrite = self.overwrite
        data.write(outname, overwrite=overwrite)
        return outname
    
    def data_process(self, data, **kwargs):
        """Process data using :func:`ccdproc.ccd_process`
        
        Parameters
        ----------
        data : `~astropy.nddata.CCDData`
            Data to process

        kwargs : kwargs
            Passed to :func:`ccdproc.ccd_process`
            See also: Notes in :class:`bigmultipipe.BigMultiPipe`
            Parameter section 

        """
        kwargs = self.kwargs_merge(**kwargs)
        data = ccdp.ccd_process(data, **kwargs)
        return data

#bname = '/data/io/IoIO/reduced/Calibration/2020-07-07_ccdT_-10.3_bias_combined.fits'
#ccd = CCDData.read(bname)

#fname1 = '/data/Mercury/raw/2020-05-27/Mercury-0005_Na-on.fit'
#ccd = CCDData.read(fname1)
#ccd = FBUCCDData.read(fname1, fallback_unit='adu')
#ccd = FBUCCDData.read(fname1, fallback_unit='aduu')

#ccd = FBUCCDData.read(fname1, unit='electron')
