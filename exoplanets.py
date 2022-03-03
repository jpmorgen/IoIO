#!/usr/bin/python3
"""Reduce files for Elisabeth Adams XRP project"""

import os
import glob

from astropy import log, time, coordinates as coord, units as u

from ccdmultipipe import as_single

from IoIO.utils import reduced_dir, get_dirs_dates

from IoIO.cormultipipe import (RAW_DATA_ROOT, CorMultiPipeBase,
                               nd_filter_mask, mask_nonlin_sat)
from IoIO.calibration import Calibration

EXOPLANET_ROOT = '/data/exoplanets'
GLOB_INCLUDE = ['TOI-*', 'WASP-*', 'GJ*']

# General FITS WCS reference:
# https://fits.gsfc.nasa.gov/fits_wcs.html
# https://heasarc.gsfc.nasa.gov/docs/fcg/standard_dict.html


# https://docs.astropy.org/en/stable/time/index.html
# https://www.aanda.org/articles/aa/pdf/2015/02/aa24653-14.pdf
def barytime(ccd_in, bmp_meta=None, **kwargs):
    """CorMultiPipe post-processing routine to calculate accurate
barycentric dynamic time (TDB) and insert it as the JDBARY FITS key"""
    ccd = ccd_in.copy()

    # Location where photons are detected    
    lon = ccd.meta.get('LONG-OBS') or ccd.meta.get('SITELONG')
    lat = ccd.meta.get('LAT-OBS') or ccd.meta.get('SITELAT')
    alt = ccd.meta.get('ALT-OBS') or 1096.342 * u.m
    # Got IoIO alt from a combination of Google Earth and USGS top
    # maps.  See IoIO.notebk Sun Apr 28 12:50:37 2019 EDT.  My
    # conclusion was that the altitude was compatible with WGS84,
    # though Google Maps apparently uses EGM96 elipsoid for heights
    # and WGS84 for lat-lon.  But EarthLocation does not support
    # EGM96m so use WGS84, which is a best-fit ellipsoid
    location = coord.EarthLocation.from_geodetic(lon, lat, alt, 'WGS84')

    # Local time at observatory
    timesys = ccd.meta.get('TIMESYS') or 'UTC'
    fits_date = ccd.meta.get('DATE-AVG')
    if fits_date is None:
        fits_date = ccd.meta.get('DATE-OBS')
        # This could go on ad-infinitum in general, since there are so
        # many standards to choose from for time.  With MaxIm and ACP,
        # we can stop here
        telescope = ccd.meta.get('TELESCOP')
        if telescope == 'IoIO_1':
            log.warning('Less accurate DATE-OBS keyword being used.  DATE-AVG was somehow not calculated')
        after_key = 'DATE-OBS'
        jdbary_comment = 'Obs. barycentric dynamic time (JD)'
    else:
        after_key = 'MJD-AVG'
        jdbary_comment = 'Obs. midpoint barycentric dynamic time (JD)'
    tm = time.Time(fits_date, format='fits',
                   scale=timesys.lower(), location=location)

    # Fancy conversion to ICRS is likely not done anywhere in IoIO
    # system, so default to FK5 is safe
    radesys = ccd.meta.get('RADESYS') or ccd.meta.get('RADECSYS') or 'FK5'
    # Official plate solutions, if available
    ctype1 = ccd.meta.get('CTYPE1')
    ctype2 = ccd.meta.get('CTYPE2')
    if ctype1 and ctype2 and 'RA' in ctype1 and 'DEC' in ctype2:
        ra = ccd.meta.get('CRVAL1')
        dec = ccd.meta.get('CRVAL2')
        units = (ccd.meta.get('CUNIT1') or u.deg,
                 ccd.meta.get('CUNIT2') or u.deg)
    else:
        # ACP puts RA and DEC in as read from telescope and/or copied
        # from PinPoint CRVAL1, CRVAL2
        # MaxIm puts OBJCTRA and OBJCTDEC in as read from telescope
        ra = ccd.meta.get('RA') or ccd.meta.get('OBJCTRA')
        dec = ccd.meta.get('DEC') or ccd.meta.get('OBJCTDEC')
        # These values are string sexagesimal with RA in hours
        units = (u.hourangle, u.deg)
    direction = coord.SkyCoord(ra, dec, unit=units, frame=radesys.lower())


    # Put it all together
    dt = tm.light_travel_time(direction, kind='barycentric')
    tdb = tm.tdb + dt
    ccd.meta.insert(after_key, 
               ('JDBARY', tdb.jd,
                jdbary_comment),
               after=True)
    return ccd

def expo_calc(vmag):
    # TOI-2109 was nice reference
    vmag = vmag*u.mag(u.ct/u.s)
    toi_mag = 10*u.mag(u.ct/u.s)
    toi_expo = 20*u.s
    dmag = toi_mag - vmag
    return toi_expo * dmag.physical

if __name__ == "__main__":
    log.setLevel('DEBUG')
 
    data_root = RAW_DATA_ROOT
    reduced_root = EXOPLANET_ROOT

    dirs_dates = get_dirs_dates(data_root)


    #### from cormultipipe import RedCorData
    #### 
    #### ccd = RedCorData.read('/data/io/IoIO/raw/20210921/TOI-2109-S004-R021-C002-R.fts')
    #### ccd.meta = obs_location_to_hdr(ccd.meta, location=IOIO_1_LOCATION)
    #### 
    #### #ccd = barytime(ccd)
    #### #print((ccd.meta['JDBARY'] - ccd.meta['BJD-OBS']) * 24*3600)

    ##directory = '/data/io/IoIO/raw/20210823'
    #directory = '/data/io/IoIO/raw/20210921'
    #rd = reduced_dir(directory, create=True)
    #glob_include = ['TOI*']

    dirs, _ = zip(*dirs_dates)
    assert len(dirs) > 0

    c = Calibration(reduce=True)

    cmp = CorMultiPipeBase(auto=True, calibration=c,
                           fits_fixed_ignore=True, outname_ext='.fits', 
                           post_process_list=[barytime,
                                              mask_nonlin_sat,
                                              nd_filter_mask,
                                              as_single])

    for d in dirs:
        flist = []
        for gi in GLOB_INCLUDE:
            flist += glob.glob(os.path.join(d, gi))
        if len(flist) == 0:
            log.debug(f'No exoplant observations in {d}')
            continue
        reddir = reduced_dir(d, reduced_root)
        pout = cmp.pipeline(flist, outdir=reddir,
                            create_outdir=True, overwrite=True)


    #calibration = Calibration(reduce=True)
    #
    #flist = []
    #for gi in glob_include:
    #    flist += glob.glob(os.path.join(directory, gi))
    #
    #cmp = OffCorMultiPipe(auto=True, calibration=calibration,
    #                      fits_fixed_ignore=True, outname_ext='.fits', 
    #                      post_process_list=[barytime, nd_filter_mask, as_single])
    #pout = cmp.pipeline(flist, outdir=rd, overwrite=True, outname_ext='.fits')



    #cmp = OffCorMultiPipe(auto=True, calibration=calibration,
    #                      fits_fixed_ignore=True, outname_ext='.fits', 
    #                      post_process_list=[barytime, nd_filter_mask, as_single])
    #pout = cmp.pipeline(flist[0:3], outdir='/tmp',
    #                    as_scaled_int=True, overwrite=True, outname_ext='.fits')

    #cmp = OffCorMultiPipe(auto=True, calibration=calibration,
    #                   outname_ext='.fits', 
    #                   post_process_list=[nd_filter_mask])
    #pout = cmp.pipeline(flist[0:3], outdir='/tmp/double', overwrite=True, outname_ext='.fits')

    # IMPORTANT: Use the DATE-AVG keyword as your midpoint of the
    # observations, don't calculated it yourself!  As detailed below, there
    # are a lot of instrumental details to get there.  The DATE-AVG keyword
    # name is recommended in the WCS standard, which is why I use it.
    # 
    # DATE-AVG details: DATE-AVG is carefully constructed using calibration
    # measurements detailed in the nearby FITS cards.  The basic issue: the
    # shutter is commanded via the USB bus on a Windows 10 machine, so there
    # is a variable latency in the real shutter open time.  The shutter close
    # time, which is needed for EXPTIME, is an even bigger mess, at least for
    # exposures >0.7s, involving the aformentioned USB latency and an extra
    # ~2s round-trip.  DATE-AVG and EXPTIME are therefore best estimates and
    # their respective uncertainties, DATE-AVG-UNCERTAINTY and
    # EXPTIME-UNCERTAINTY, are noted.  If you want to be really thorough, feel
    # free to use the uncertainties to put error bars on the time axis of your
    # analyses.
    # 
    # Plate scale in arcsec can be calculated using the following keywords
    # which (unless you buy me a fancier camera) are unlikely to change.  
    # 
    # XPIXSZ = 4.5390600000000001 # micron
    # FOCALLEN = 1200.0000000000000 # mm
    # PIXSCALE = 206265 * XPIXSZ/1000 / FOCALLEN
    # 0.78020767575 # arcsec
    # 
    # or astropythonically:
    # 
    # import astropy.units as u
    # from astropy.nddata import CCDData
    # 
    # # Example file
    # ccd = CCDData.read('/data/io/IoIO/reduced/20210820/TOI-2109-S001-R001-C003-R_p.fts')
    # # Be completely general -- not all CCDs have square pixels
    # pixel_size = (ccd.meta['XPIXSZ']**2 + ccd.meta['YPIXSZ'])**0.5
    # focal_length = ccd.meta['FOCALLEN']
    # # Proper FITS header units would make these obsolete
    # pixel_size *= u.micron 
    # focal_length *= u.mm
    # plate_scale = pixel_size / focal_length
    # plate_scale *= u.radian
    # plate_scale = plate_scale.to(u.arcsec)
    # print(f'Plate scale: {plate_scale}')
    # # Proper FITS header units would make this easier
    # ccd.meta['PIXSCALE'] = (plate_scale.value, '[arcsec]')
    # 
    # 
    # Let me know if you have any questions or recommendations to make
