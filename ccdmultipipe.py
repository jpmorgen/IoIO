"""Module which enables parallel pipeline processing of CCDData files
"""

from astropy import units as u
from astropy.nddata import CCDData
import ccdproc as ccdp

from bigmultipipe import BigMultiPipe

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

class CCDMultiPipe(BigMultiPipe):

    def __init__(self,
                 raw_unit=None,
                 outname_append='_ccdmp',
                 overwrite=False,
                 **kwargs):
        if raw_unit is None:
            raw_unit = u.adu
        self.raw_unit = raw_unit
        self.overwrite = overwrite
        super().__init__(outname_append=outname_append,
                         **kwargs)

    def file_read(self, in_name, **kwargs):
        # Allow overriding of self.kwargs by **kwargs
        skwargs = self.kwargs.copy()
        skwargs.update(kwargs)
        kwargs = skwargs
        data = ccddata_read(in_name, raw_unit=self.raw_unit, **kwargs)
        return data

    def file_write(self, data, outname, 
                    overwrite=None,
                    **kwargs):
        # Allow overriding of self.kwargs by **kwargs
        skwargs = self.kwargs.copy()
        skwargs.update(kwargs)
        kwargs = skwargs
        if overwrite is None:
            overwrite = self.overwrite
        data.write(outname, overwrite=overwrite)
        return outname
    
    def data_process(self, data, **kwargs):
        # Allow overriding of self.kwargs by **kwargs
        skwargs = self.kwargs.copy()
        skwargs.update(kwargs)
        kwargs = skwargs
        data = ccdp.ccd_process(data, **kwargs)
        return data
