"""Defines cor_process, the extension of ccdprocess for the IoIO
coronagraph data processing system"""

import numpy as np

from astropy import log
from astropy import units as u
from astropy.nddata import CCDData, StdDevUncertainty
from astropy.coordinates import (Angle, SkyCoord, AltAz, HADec,
                                 solar_system_ephemeris, get_body)

import ccdproc as ccdp

import IoIO.sx694 as sx694
from IoIO.cordata_base import (IOIO_1_LOCATION, overscan_estimate,
                               CorDataBase, CorDataNDparams)

# These are OBJECT names that come from the early MaxIm autosave
# observations and need to be properly renamed
JUPITER_SYNONYMS = ['IPT', 'Na', 'Na_IPT', 'Na_IPT_R']

# ACK!  I really should have connected and disconnected the telescope
# during the ACP shell-out.  See comments in code where this is used
ACP_TRACK_PAST_MERIDIAN = 45*u.min

def get_filt_name(f, date_obs):
    """Used in standardize_filt_name.  Returns standarized filter name
    for all cases in IoIO dataset.

    Parameters
    ----------
    f : str
        Original filter name

    date_obs : str
        FITS format DATE-OBS keyword representing the date on which
        filter f was recorded into the FITS header

    Returns
    -------
    f : str
        Standardized filter name

    """
    # Dates and documentation from IoIO.notebk
    if date_obs > '2020-03-01':
        # Fri Feb 28 11:43:39 2020 EST  jpmorgen@byted
        # Filters in latest form.  Subsequent renames should hopfully
        # follow similiar conventions (e.g. I for Bessel I-band
        # filter, <primary>_on and <primary>_off)
        # 1 R
        # 2 1.25" SII_on
        # 3 1.25" SII_off
        # 4 Na_off
        # 5 1.25" V
        # 6 1.25" U
        # 7 Na_on
        # 8 1.25" B
        # 9 1.25" R_125
        return f
    if date_obs > '2019-03-31':
        # On 2019-03-31 the 9-position SX "Maxi" falter wheel was
        # installed.  There was considerable confusion during the
        # installation due to incorrectly machined adapters and
        # clearance issues.  The adapters and clearance issues were
        # straightned out, but an unfortunate byproduct of the
        # debugging process was to end up swapping the labeled
        # positions of the U and V and RC and H2O+ filter pairs.  This
        # was fixed on 2020-03-01
        # 
        #
        # Sat Apr 13 21:38:43 2019 EDT  jpmorgen@snipe
        # Documentation of where I thought they were (numbers starting
        # from 1 inserted)
        #  Filter #0 1 (R) offset: 0
        #  Filter #1 2(SII_on) offset: 108.6
        #  Filter #2 3(SII_off) offset: 86.2
        #  Filter #3 4 (Na_off) offset: -220.6
        #  Filter #4 5 (H2O+) offset: -327
        #  Filter #5 6 (RC) offset: 1323.6
        #  Filter #6 7 (Na_on) offset: -242.4
        #  Filter #7 8 (V) offset: 265.8
        #  Filter #8 9 (U) offset: 286.2
        #
        # Tue Feb 25 21:19:11 2020 EST  jpmorgen@byted
        # Documentation of where filters really were from
        # 2019-03-31 -- 2020-02-25
        # 1 R
        # 2 SII_on
        # 3 SII_off
        # 4 Na_off
        # 5 V
        # 6 UV
        # 7 Na_on
        # 8 H2O+
        # 9 RC
        if f == 'H2O+':
            return 'V'
        if f == 'RC':
            return 'U'
        if f == 'V':
            return 'H2O+'
        if f == 'U':
            return 'RC'
        if f == 'UV':
            # Just in case some slipped in with this transient
            return 'U'
        # everything else should be OK
        return f

    # On 20190218, just before the large filter wheel installation, I
    # changed filter naming from verbose to the current convention.
    # The verbose names were Na_continuum_50A_FWHM, Na_5892A_10A_FWHM,
    # Na_5890A_10A_FWHM, [SII]_continuum_40A_FWHM, and
    # [SII]_6731A_10A_FWHM.  The R filter was I think always R, but
    # may have been 'R-band.'  Also had an "open" slot before I got
    # the R filter.  The following code should grab both the old a
    # current naming cases
    if 'R' in f:
        return f
    if 'open' in f:
        return f
    if 'cont' in f or 'off' in f:
        on_off = 'off'
    else:
        # Hopefully the above catches the on/off cases and other filters
        on_off = 'on'
    if 'SII' in f:
        line = 'SII'
    elif 'Na' in f:
        line = 'Na'
    else:
        # We only had 5 slots, so this has covered all the bases
        raise ValueError(f'unknown filter {f}')
    return f'{line}_{on_off}'

def standardize_filt_name(hdr_or_collection):
    """Standardize FILTER keyword across all IoIO data

    Parameters
    ----------
    hdr_or_collection : `~astropy.fits.io.Header` or 
        `~ccdproc.ImageFileCollection'

        input FITS header or ImageFileCollection

    Returns
    -------
    `~astropy.fits.io.Header` or `~ccdproc.ImageFileCollection' with
    FILTER card/column  updated to standard form and OFILTER card (original
    FILTER) added, if appropriate
    """
    if isinstance(hdr_or_collection, ccdp.ImageFileCollection):
        st = hdr_or_collection.summary
        st['ofilter'] = st['filter']
        for row in st:
            row['filter'] = get_filt_name(row['ofilter'],
                                          row['date-obs'])
        return hdr_or_collection

    hdr_in = hdr_or_collection
    if hdr_in.get('ofilter') is not None:
        # We have been here before, so exit quietly
        return hdr_in
    old_filt_name = hdr_in.get('FILTER')
    if old_filt_name is None:
        # Probably a BIAS or DARK
        return hdr_in
    new_filt_name = get_filt_name(old_filt_name, hdr_in['DATE-OBS'])
    if old_filt_name == new_filt_name:
        return hdr_in
    # Only copy if we are going to change the hdr
    hdr = hdr_in.copy()
    hdr['FILTER'] = new_filt_name
    hdr.insert('FILTER',
               ('OFILTER', old_filt_name, 'Original filter name'),
               after=True)
    return hdr

# --> This eventually can move up to the read method
def fix_radecsys_in_hdr(hdr_in):
    """Change the depreciated RADECSYS keyword to RADESYS

    WCS paper I (Greisen, E. W., and Calabretta, M. R., Astronomy &
    Astrophysics, 395, 1061-1075, 2002.) defines RADESYSa instead of
    RADECSYS.  The idea is RADESYS refers to the primary system
    (e.g. FK4, ICRS, etc.) and RADESYS[A-Z] applies to other systems
    in the other keywords that might be in units with other
    distorions, etc.  So moving RADECSYS to RADESYS is forward
    compatible

    """
    radecsys = hdr_in.get('RADECSYS')
    if radecsys is None:
        return hdr_in
    hdr = hdr_in.copy()
    hdr.insert('RADECSYS',
               ('RADESYS', radecsys, 'Equatorial coordinate system'))
    del hdr['RADECSYS']
    return hdr    

def obs_location_to_hdr(hdr_in, location=None):
    """Standardize FITS header keys for observatory location and name
    
    Parameters
    ----------
    location : `astropy.coordinates.EarthLocation`
        Location of observatory to be encoded in FITS header.
        ``location.name`` is used as TELESCOP keyword and
        location.info.meta('longname') as the comment to that keyword
    """
    if location is None:
        return hdr_in
    hdr = hdr_in.copy()
    lon, lat, alt = location.to_geodetic()
    hdr['LAT-OBS'] = (lat.value, 'WGS84 Geodetic latitude (deg)')
    hdr['LONG-OBS'] = (lon.value, 'WGS84 Geodetic longitude (deg)')
    hdr['ALT-OBS'] = (alt.value, 'WGS84 altitude (m)')
    name = location.info.name
    try:
        longname = location.info.meta.get('longname')
    except:
        longname = ''
    if name is not None:
        hdr['TELESCOP'] = (name, longname)
    # delete non-standard keywords
    if hdr.get('OBSERVAT'):
        del hdr['OBSERVAT'] # ACP
    if hdr.get('SITELONG'):
        del hdr['SITELONG'] # MaxIm
    if hdr.get('SITELAT'):
        del hdr['SITELAT'] # MaxIm
    return hdr

def angle_to_major_body(ccd, body_str):
    """Returns angle between pointing direction and solar system major
    body

    Build-in astropy geocentric ephemeris is used for each planet and
    the moon and the ccd.sky_coord is used to construct a similarly
    geocentric pointing direction.  The angle between these two
    directions is returned.  For a more accurate pointing direction
    from the perspective of the observatory, the astroquery.horizons
    module can be used

, so results are not as accurate
    as possible, but more than good enough for rough calculations

    Parameters
    ----------
    ccd : astropy.nddata.CCDData

    body : str
        solar system major body name

    Returns
    -------
    angle : astropy.units.Quantity
        angle between pointing direction and major body

    """
    with solar_system_ephemeris.set('builtin'):
        body_coord = get_body(body_str, ccd.tavg, ccd.obs_location)
    # https://docs.astropy.org/en/stable/coordinates/common_errors.html
    # notes that separation is order dependent, with the calling
    # object's frame (e.g., body_coord) used as the reference frame
    # into which the argument (e.g., ccd.sky_coord) is transformed
    return body_coord.separation(ccd.sky_coord)

def fix_obj_and_coord(ccd_in, **kwargs):
    """Fixes missing/incorrect OBJECT and standardizes coordinate keywords:

    RA, DEC = Read from telescope: nominal center of FOV

    OBJCTRA, OBJCTDEC = RA and DEC of OBJECT being observed, which may
    be offset out of the FOV as discussed in
    `:prop:IoIO.corddata_base.CorDataBase.sky_coord` 

    """
    imagetyp = ccd_in.meta.get('imagetyp') 
    if (imagetyp is not None
        and not 'light' in imagetyp.lower()):
        # bias, dark, etc, don't have object names
        return ccd_in
    ccd = ccd_in.copy()
    
    if ccd.meta.get('objctra') is None:
        # IoIO.py ACP_IPT_Na_R was run from an ACP plan as a shell-out
        # and usually didn't have a MaxIm telescope connection.  This
        # is pretty much the only case where OBJCTRA and OBJCTDEC are
        # not defined since Bob defines these in ACP and MaxIm sets
        # them whenever the telescope is connected (any time it is
        # pointed to the sky!).  Unless I was doing debugging, this
        # shell-out was only called when the telescope was pointed at
        # Jupiter.  Call that close enough.
        ccd.meta['OBJECT'] = 'Jupiter'
        with solar_system_ephemeris.set('builtin'):
            target = get_body('Jupiter', ccd.tavg, ccd.obs_location)
        target_ra_dec = target.to_string(style='hmsdms').split()        
        ccd.meta['OBJCTRA'] = (target_ra_dec[0],
                          '[hms] Target right assention')
        ccd.meta['OBJCTDEC'] = (target_ra_dec[1],
                           '[dms] Target declination')
        ccd.meta['RA'] = (target_ra_dec[0],
                     '[hms] Target right ascension')
        ccd.meta['DEC'] = (target_ra_dec[1],
                      '[dms] Target declination')
        # This ends up making a duplicate when object_to_objctradec is
        # called for real
        #ccd.meta.insert('OBJCTDEC',
        #                ('HIERARCH OBJECT_TO_OBJCTRADEC', True,
        #                 'Assumed ACP shell-out'),
        #                after=True)
        # Reset sky_coord so it reads these keys
        ccd.sky_coord = None
        altaz = ccd.sky_coord.transform_to(
            AltAz(obstime=ccd.tavg, location=ccd.obs_location))
        ccd.meta['OBJCTAZ'] = (altaz.az.value, f'[{altaz.az.unit}]')
        ccd.meta['OBJCTALT'] = (altaz.alt.value,
                                f'[{altaz.alt.unit}]')
        hadec = ccd.sky_coord.transform_to(
            HADec(obstime=ccd.tavg, location=ccd.obs_location))
        hastr = hadec.ha.to_string(unit=u.hour, sep=' ', pad=True)
        ccd.meta['OBJCTHA'] = (hastr, f'[{hadec.ha.unit}]')
        # For east of the merdian, we are guaranteed to be PIERSIDE
        # WEST because ACP did not do early pier flips.  There is a
        # deadband between meridian and ACP_TRACK_PAST_MERIDIAN where
        # we just don't know the answer.  Make sure that we enter
        # something in the PIERSIDE keyword.  This ensures that if it
        # is not set, there is a problem somewhere else in the system!
        if hadec.ha < 0*u.deg:
            ccd.meta['PIERSIDE'] = 'WEST'
        elif hadec.ha > ACP_TRACK_PAST_MERIDIAN:
            ccd.meta['PIERSIDE'] = 'EAST'
        else:
            log.warning(f'Ambiguous PIERSIDE for {hastr}')        
            ccd.meta['PIERSIDE'] = 'UNKNOWN'
    else:
        # Tweak metadata a little: These values are actually read from
        # the telescope and are not the real RA and DEC of the target
        # as implied by the raw metadata
        ccd.meta.comments['OBJCTRA'] = \
            '[hms J2000] Telescope right ascension'
        ccd.meta.comments['OBJCTDEC'] = \
            '[hms J2000] Telescope declination'
        if ccd.meta.get('RA') is None:
            ccd.meta['RA'] = ccd.meta['OBJCTRA']
            ccd.meta['DEC'] = ccd.meta['OBJCTDEC']
        ccd.meta.comments['RA'] = \
            '[hms J2000] Telescope right ascension'
        ccd.meta.comments['DEC'] = \
            '[hms J2000] Telescope declination'
        
    obj = ccd.meta.get('object') 
    if obj is None or obj == '':
        # OBJECT was not always set in the early Jupiter observations.
        # Also took some Mercury observations with MaxIm without
        # specifying object.  Check for all just for the heck of it
        # --> This might be more appropriate in mercury.py
        for body in solar_system_ephemeris.bodies:
            a = angle_to_major_body(ccd, body)
            # This should be accurate enough for our pointing
            if a < 1*u.deg:
                ccd.meta['object'] = body.capitalize()
                break

    if obj in JUPITER_SYNONYMS:
        # I named MaxIm autosave sequences after the filters rather
        # than the object
        ccd.meta['OBJECT'] = 'Jupiter'

    if obj == 'WASP-136b':
        # Fix typo (also in outname_create)
        ccd.meta['OBJECT'] = 'WASP-163b'
    return ccd

# --> I will want some sort of integrator for this to calculate
# airmass at midpoint and handle case where we are close to horizon
def kasten_young_airmass(ccd_in):
    """Record airmass considering curvature of Earth

    Uses formula of F. Kasten and Young, A. T., “Revised optical air
    mass tables and approximation formula”, Applied Optics, vol. 28,
    pp. 4735–4738, 1989 found at
    https://www.pveducation.org/pvcdrom/properties-of-sunlight/air-mass

    """
    if ccd_in.meta.get('oairmass') is not None:
        # We have been here before, so exit quietly
        return ccd_in
    ccd = ccd_in.copy()
    objctalt = ccd.meta.get('objctalt') 
    objctalt = float(objctalt) * u.deg
    zd = 90*u.deg - objctalt
    oairmass = ccd.meta.get('AIRMASS')
    if oairmass is None:
        oairmass = 'Not calculated'
        ccd.meta['AIRMASS'] = oairmass
    ccd.meta.insert('AIRMASS',
               ('OAIRMASS', oairmass, 'Original airmass'),
               after=True)
    denom = np.cos(zd) + 0.50572 * (96.07995 - zd.value)**(-1.6364)
    ccd.meta['AIRMASS'] = (1/denom.value,
                           'Curvature-corrected (Kasten and Young 1989)')
    return(ccd)

def subtract_overscan(ccd, oscan=None, *args, **kwargs):
    """Subtract overscan, estimating it, if necesesary, from image.
    Also subtracts overscan from SATLEVEL keyword

    Note: ccdproc's native subtract_overscan function can't be used
    because it assumes the overscan region is specified by a simple
    rectangle.

    """
    if ccd.meta.get('overscan_value') is not None:
        # We have been here before, so exit quietly
        return ccd
    nccd = ccd.copy()
    if oscan is None:
        oscan = overscan_estimate(ccd, meta=nccd.meta,
                                  *args, **kwargs)
    nccd = nccd.subtract(oscan*u.adu, handle_meta='first_found')
    nccd.meta['HIERARCH OVERSCAN_VALUE'] = (oscan, 'overscan value subtracted (adu)')
    nccd.meta['HIERARCH SUBTRACT_OVERSCAN'] \
        = (True, 'Overscan has been subtracted')
    ## Keep track of our precise saturation level
    #satlevel = nccd.meta.get('satlevel')
    #if satlevel is not None:
    #    satlevel -= oscan
    #    nccd.meta['SATLEVEL'] = satlevel # still in adu
    return nccd

def cor_process(ccd,
                calibration=None,
                auto=False,
                imagetyp=None,
                ccd_meta=True,
                fix_radecsys=True,
                obs_location=True,
                fix_filt_name=True,
                exp_correct=True,
                date_obs_correct=True,
                remove_raw_jd=True,
                correct_obj_and_coord=True,
                correct_obj_offset=True,
                airmass_correct=True,
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
                *args, **kwargs):

    """Perform basic CCD processing/reduction of IoIO ccd data

    The following steps can be included:

    * add CCD metadata (:func:`sx694.metadata`)
    * correct CCD exposure time (:func:`sx694.exp_correct`)
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
    ccd : `~astropy.nddata.CCDData`
        CCDData of image to be processed

    calibration : `~Calibration` or None, optional
        Calibration object to be used to find best bias, dark, and
        flatfield files.  
        Default is ``None``.

    auto : bool
        If True, do reduction automatically based on IMAGETYP
        keyword.  See imagetyp documentation.
        Default is ``False``

    imagetyp : bool, str, or None
        If True, do reduction based on IMAGETYP keyword.  If string,
        use that as IMAGETYP.  Requires calibration object
        bias -> oscan=True
        dark -> oscan=True, master_bias=True
        flat -> oscan=True, master_bias=True, dark_frame=True
        light-> oscan=True, error=True, master_bias=True,
                dark_frame=True, master_flat=True
        Default is ``None``

    ccd_meta : bool
        Add CCD metadata
        Default is ``True``

    fix_radecsys : bool
        Change RADECSYS to RADESYS as per latest FITS standard
        Default is ``True``

    obs_location : bool
        Standardize FITS header keys for observatory location and
        name.  Also deletes inaccurate/non-standard keywords
        Default is ``True``

    fix_filt_name : bool
        Put all filters into namoing convention used starting when the
        9-position SX Maxi filter wheel was installed in late Feb 2020.
        Default is ``True``

    exp_correct : bool
        Correct for exposure time problems
        Default is ``True``

    date_obs_correct : bool

        Correct DATE-OBS keyword for best-estimate shutter time and
        corrected exposure time.  Also removes inaccurate/obsolete
        DATE, TIME-OBS, and UT keywords 

        Default is ``True``

    remove_raw_jd : bool
        Remove *JD* keywords not calculated from DATE-BEG and DATE-AVG
        Default is ``True``

    correct_obj_and_coord : bool
        Correct missing/incorrect OBJECT and coordinate keywords in hdr 
        Default is ``True``

    correct_obj_offset : bool
        Recognizing standard observations are recorded offset from the
        coronagraph, use RAOFF and DECOFF keywords to correct OBJCTRA
        and OBJCTDEC keywords to target position.  RA and DEC keywords
        are set to the previous values of OBJCTRA and OBJCTDEC, which
        is the nominal center of the FOV, subject to telescope
        pointing errors. 
        Default is ``True``

    airmass_correct : bool
        Correct for curvature of earth airmass for very low elevation
        observations
        Default is ``True``

    oscan : number, bool, or None, optional
        Single pedistal value to subtract from image.  If True, oscan
        is estimated using :func:`sx694.overscan_estimate` and subtracted
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

    dark_frame : bool, str, `~atropy.nddata.CCDData` or None, optional
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
        Default is "GAIN" with units `~astropy.units.electron`/`~astropy.units.adu`

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

    Returns
    -------
    ccd : `~astropy.nddata.CCDData`
	Processed image

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

    if ccd.meta.get('cor_processed') is not None:
        return ccd
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

    # Enable autocalibration through imagetyp keyword
    if auto:
        imagetyp = nccd.meta.get('imagetyp')
        if imagetyp is None:
            raise ValueError("CCD metadata contains no IMAGETYP keyword, can't proceed with automatic reduction")

    # Enable imagetyp to select reduction level
    if imagetyp is None:
        pass
    elif 'bias' in imagetyp.lower():
        oscan=True; error=True
    elif 'dark' in imagetyp.lower():
        oscan=True; gain=True; error=True; master_bias=True
    elif 'flat' in imagetyp.lower():
        oscan=True; gain=True; error=True; master_bias=True; dark_frame=True
    elif 'light' in imagetyp.lower():
        oscan=True; gain=True; error=True; master_bias=True; dark_frame=True; master_flat=True; min_value=True
    else:
        raise ValueError(f'Unknown IMAGETYP keyword {imagetyp}')

    # Correct metadata, particularly filter names, before we try calibration

    if ccd_meta:
        # Put in our SX694 camera metadata
        nccd.meta = sx694.metadata(nccd.meta, *args, **kwargs)

    if fix_radecsys:
        nccd.meta = fix_radecsys_in_hdr(nccd.meta)

    if obs_location:
        nccd.meta = obs_location_to_hdr(nccd.meta,
                                        location=IOIO_1_LOCATION)
        # Reset obs_location just in case it was used before
        nccd.obs_location = None

    if fix_filt_name:
        # Fix my indecision about filter names!
        nccd.meta = standardize_filt_name(nccd.meta)

    if exp_correct:
        # Correct exposure time for driver bug
        nccd.meta = sx694.exp_correct(nccd.meta, *args, **kwargs)
        
    if date_obs_correct:
        # DATE-OBS as best estimate shutter open time, add DATE-AVG
        nccd.meta = sx694.date_obs(nccd.meta, *args, **kwargs)

    if remove_raw_jd:
        if nccd.meta.get('JD*'):
            del nccd.meta['JD*']
        # ACP adds these, but based on DATE-OBS, not DATE-AVG
        if nccd.meta.get('HJD-OBS'):
            del nccd.meta['HJD-OBS']
        if nccd.meta.get('BJD-OBS'):
            del nccd.meta['BJD-OBS']

    if correct_obj_and_coord:
        # Add in Jupiter and missing coordinates when ACP shell-out
        # to IoIO.py was used
        nccd = fix_obj_and_coord(nccd)
    
    if correct_obj_offset:
        raoff = nccd.meta.get('RAOFF')
        decoff = nccd.meta.get('DECOFF')
        if raoff is not None or decoff is not None:
            raoff = raoff or 0
            decoff = decoff or 0
            raoff = Angle(raoff*u.arcmin)
            decoff = Angle(decoff*u.arcmin)
            tpos = nccd.sky_coord
            target = SkyCoord(tpos.ra - raoff, tpos.dec - decoff)
            target_ra_dec = target.to_string(style='hmsdms').split()
            tpos_ra_dec = tpos.to_string(style='hmsdms').split()
            nccd.meta['OBJCTRA'] = (
                target_ra_dec[0],
                '[hms J2000] Target nominal right assention')
            nccd.meta['OBJCTDEC'] = (
                target_ra_dec[1],
                '[dms J2000] Target nominal declination')
            nccd.meta['RA'] = (tpos_ra_dec[0],
                               '[hms J2000] Telescope right ascension')
            nccd.meta['DEC'] = (tpos_ra_dec[1],
                                '[dms J2000] Telescope declination')
        
    if airmass_correct:
        # I think this is better at large airmass than what ACP uses,
        # plus it standardizes everything for times I didn't use ACP
        nccd = kasten_young_airmass(nccd)

    # Convert "yes use this calibration" to calibration _filenames_
    # now that we have good metadata
    try:
        if master_bias is True:
            master_bias = calibration.best_bias(nccd)
        if dark_frame is True:
            dark_frame = calibration.best_dark(nccd)
        if master_flat is True:
            master_flat = calibration.best_flat(nccd.meta)

    except Exception as e:
        log.error(f'No calibration available: calibration system problem {e}')
        raise

    # Apply overscan correction unique to the IoIO SX694 CCD.  This
    # uses the string version of master_bias, if available for
    # metadata
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
        master_bias = CorDataBase.read(master_bias)
    else:
        subtract_bias_keyword = None
    if isinstance(dark_frame, str):
        subtract_dark_keyword = \
            {'HIERARCH SUBTRACT_DARK': 'subdark',
             'SUBDARK': 'ccdproc.subtract_dark ccd=<CCDData>, master=DARKFILE',
             'DARKFILE': dark_frame}
        dark_frame = CorDataBase.read(dark_frame)
    else:
        subtract_dark_keyword = None
    if isinstance(master_flat, str):
        flat_correct_keyword = \
            {'HIERARCH FLAT_CORRECT': 'flatcor',
             'FLATCOR': 'ccdproc.flat_correct ccd=<CCDData>, master=FLATFILE',
             'FLATFILE': master_flat}
        master_flat = CorDataNDparams.read(master_flat)
    else:
        flat_correct_keyword = None

    # apply the trim correction
    if isinstance(trim, str):
        nccd = ccdp.trim_image(nccd, fits_section=trim)
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

    if error and readnoise is None:
        # We want to make an error frame but the user has not
        # specified readnoise.  See if we can read from metadata
        readnoise = readnoise_key.value_from(nccd.meta)

    # Create the error frame.  Do this differently than ccdproc for
    # two reasons: (1) bias error should read the readnoise (2) I
    # can't trim my overscan, so there are lots of pixels at the
    # overscan level.  After overscan and bias subtraction, many of
    # them that are probably normal statitical outliers are negative
    # enough to overwhelm the readnoise in the deviation calculation.
    # But I don't want the error estimate on them to be NaN, since the
    # error is really the readnoise.
    if error and imagetyp is not None and 'bias' in imagetyp.lower():
        if gain is None:
            # We don't want to gain-correct, so we need to prepare to
            # convert readnoise (which is in electron) to adu
            gain_for_bias = gain_key.value_from(nccd.meta)
        else:
            # Bias will be gain-corrected to read in electrons
            gain_for_bias = 1*u.electron
        readnoise_array = np.full_like(nccd,
                                       readnoise.value/gain_for_bias.value)
        nccd.uncertainty = StdDevUncertainty(readnoise_array,
                                             unit=nccd.unit,
                                             copy=False)
    else:
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
        try:
            nccd = ccdp.subtract_bias(nccd, master_bias,
                                      add_keyword=subtract_bias_keyword)
        except Exception as e:
            log.warning(f'Problem subtracting bias, probably early data: {e}')
    elif master_bias is None:
        pass
    else:
        raise TypeError(
            'master_bias is not None, fname or a CCDData object.')
    
    # Correct OVERSCAN_MASTER_BIAS keyword, if possible
    hdr = nccd.meta
    osbias = hdr.get('osbias')
    biasfile = hdr.get('biasfile')
    if osbias is None or biasfile is None:
        pass
    elif osbias != biasfile:
        multi_logging('warning', pipe_meta,
                      'OSBIAS and BIASFILE are not the same')
    else:
        del hdr['OSBIAS']
        hdr['OVERSCAN_MASTER_BIAS'] = 'BIASFILE'

    # Subtract the dark frame.  Generally this will just use the
    # default exposure_key we create in our parameters to ccd_process
    if isinstance(dark_frame, CCDData):
        try:
            nccd = ccdp.subtract_dark(nccd, dark_frame,
                                      dark_exposure=dark_exposure,
                                      data_exposure=data_exposure,
                                      exposure_time=exposure_key,
                                      exposure_unit=exposure_unit,
                                      scale=dark_scale,
                                      add_keyword=subtract_dark_keyword)
        except Exception as e:
            log.warning(f'Problem subtracting dark, probably early data: {e}')
            
    elif dark_frame is None:
        pass
    else:
        raise TypeError(
            'dark_frame is not None or a CCDData object.')
    
    if master_flat is None:
        pass
    else:
        if min_value is True:
            min_value = min_value_key.value_from(master_flat.meta)
            flat_correct_keyword['FLATCOR'] += f', min_value={min_value}'
        flat_correct_keyword['FLATCOR'] += f', norm_value={flat_norm_value}'
        try:
            nccd = ccdp.flat_correct(nccd, master_flat,
                                     min_value=min_value,
                                     norm_value=flat_norm_value,
                                     add_keyword=flat_correct_keyword)
        except Exception as e:
            log.warning(f'Problem flatfielding, probably early data: {e}')
            
        for i in range(2):
            for j in range(2):
                ndpar = master_flat.meta.get(f'ndpar{i}{j}')
                if ndpar is None:
                    break
                ndpar_comment = master_flat.meta.comments[f'NDPAR{i}{j}']
                ndpar_comment = 'FLAT ' + ndpar_comment
                nccd.meta[f'FNDPAR{i}{j}'] = (ndpar, ndpar_comment)

    # apply the gain correction only at the end if gain_corrected is False
    if gain is not None and not gain_corrected:
        nccd = ccdp.gain_correct(nccd, gain)

    nccd.meta['HIERARCH COR_PROCESSED'] = (
        True, 'IoIO Coronagraph CCD calibrated')

    return nccd

if __name__ == "__main__":
    log.setLevel('DEBUG')
    #ccd = CorDataBase.read('/data/IoIO/raw/2021-04_Astrometry/Jupiter_near_ND_edge_S1.fit')
    #cdproced = cor_process(ccd)

    ccd = CorDataBase.read('/data/IoIO/raw/20220203/0009P-S001-R001-C001-R_dupe-1.fts')
    ccd = CorDataNDparams.read('/data/IoIO/raw/2021-04_Astrometry/Jupiter_near_ND_edge_S1.fit')

    master_bias = '/data/IoIO/Calibration/2022-02-14_ccdT_-25.3_bias_combined.fits'
    master_dark = '/data/IoIO/Calibration/2022-02-14_ccdT_-25.3_exptime_0.7s_dark_combined.fits'
    master_flat = '/data/IoIO/Calibration/2022-02-10_R_flat.fits' 
    ccdproced = cor_process(ccd,
                            oscan=True,
                            gain=True,
                            error=True,
                            min_value=True,
                            master_bias=master_bias,
                            dark_frame=master_dark,
                            master_flat=master_flat)

    new_ccd = CorDataNDparams(ccdproced)
    print(new_ccd.ND_params)
    print(new_ccd.ND_params-ccdproced.ND_params)
    print(new_ccd.default_ND_params)
    print(new_ccd.default_ND_params-ccdproced.default_ND_params)
    print(new_ccd.ND_params-ccdproced.default_ND_params)
