#!/usr/bin/python3
"""Reduce files for Elisabeth Adams XRP project"""

import os
import glob

import numpy as np

import matplotlib.pyplot as plt

from astropy import log
from astropy import units as u
from astropy.coordinates import SkyCoord

from ccdmultipipe import as_single

from IoIO.utils import (reduced_dir, get_dirs_dates, multi_glob,
                        savefig_overwrite, simple_show)
from IoIO.cormultipipe import (RAW_DATA_ROOT, CorMultiPipeBase,
                               nd_filter_mask, mask_nonlin_sat)
from IoIO.calibration import Calibration
from IoIO.standard_star import object_to_objctradec
from IoIO.cor_photometry import CorPhotometry, add_astrometry

EXOPLANET_ROOT = '/data/Exoplanets'
GLOB_INCLUDE = ['TOI-*', 'WASP-*', 'GJ*']
KEYS_TO_SOURCE_TABLE = ['DATE-AVG',
                        ('DATE-AVG-UNCERTAINTY', u.s),
                        ('MJDBARY', u.day),
                        ('EXPTIME', u.s),
                        'AIRMASS']

# General FITS WCS reference:
# https://fits.gsfc.nasa.gov/fits_wcs.html
# https://heasarc.gsfc.nasa.gov/docs/fcg/standard_dict.html


# https://docs.astropy.org/en/stable/time/index.html
# https://www.aanda.org/articles/aa/pdf/2015/02/aa24653-14.pdf
def barytime(ccd_in, bmp_meta=None, **kwargs):
    """CorMultiPipe post-processing routine to calculate the barycentric
    dynamic time (TDB) at the time of observation and record it in the
    ccd metadata in units of MJD as the MJDBARY FITS key.  NOTE: the
    internal astropy ephemerides are used, which are accurate to about
    150 us.  The precision of MJD in the current epoch near
    60000 is ~1 us.  The Starlight Xpress camera, as driven by MaxIm
    DL is able to record shutter times to accuracies of ~100 ms.

    """
    ccd = ccd_in.copy()

    # Fancy conversion to ICRS is likely not done anywhere in IoIO
    # system, so default to FK5 is safe
    radesys = ccd.meta.get('RADESYS') or ccd.meta.get('RADECSYS') or 'FK5'
    # Official plate solutions, if available
    ctype1 = ccd.meta.get('CTYPE1')
    ctype2 = ccd.meta.get('CTYPE2')
    if ctype1 and ctype2 and 'RA' in ctype1 and 'DEC' in ctype2:
        ra = ccd.meta.get('CRVAL1')
        dec = ccd.meta.get('CRVAL2')
        unit = (ccd.meta.get('CUNIT1') or u.deg,
                 ccd.meta.get('CUNIT2') or u.deg)
    else:
        # ACP puts RA and DEC in as read from telescope and/or copied
        # from PinPoint CRVAL1, CRVAL2
        # MaxIm puts OBJCTRA and OBJCTDEC in as read from telescope
        ra = ccd.meta.get('RA') or ccd.meta.get('OBJCTRA')
        dec = ccd.meta.get('DEC') or ccd.meta.get('OBJCTDEC')
        # These values are string sexagesimal with RA in hours
        unit = (u.hourangle, u.deg)
    direction = SkyCoord(ra, dec, unit=unit, frame=radesys.lower())

    # Put it all together
    dt = ccd.tavg.light_travel_time(direction, kind='barycentric')
    tdb = ccd.tavg.tdb + dt
    if ccd.meta.get('MJD-OBS'):
        after_key = 'MJD-OBS'
    else:
        after_key = 'DATE-OBS'    
    ccd.meta.insert(after_key, 
               ('MJDBARY', tdb.mjd,
                'Obs. midpoint barycentric dynamic time (MJD)'),
               after=True)
    return ccd

def obj_plot(ccd, photometry=None, outname=None, show=False, **kwargs):
    # Don't gunk up the source_table ecsv with bounding box
    # stuff, but keep in mind we sorted it, so key off of
    # label
    bbox_table = photometry.source_catalog.to_table(
        ['label', 'bbox_xmin', 'bbox_xmax', 'bbox_ymin', 'bbox_ymax'])
    mask = (photometry.wide_source_table['OBJECT']
            == ccd.meta['OBJECT'])
    label = photometry.wide_source_table[mask]['label']
    bbts = bbox_table[bbox_table['label'] == label]
    expand_bbox = 10
    xmin = bbts['bbox_xmin'][0] - expand_bbox
    xmax = bbts['bbox_xmax'][0] + expand_bbox
    ymin = bbts['bbox_ymin'][0] - expand_bbox
    ymax = bbts['bbox_ymax'][0] + expand_bbox
    source_ccd = ccd[ymin:ymax, xmin:xmax]
    threshold = photometry.threshold[ymin:ymax, xmin:xmax]
    segm = photometry.segm_image.data[ymin:ymax, xmin:xmax]
    #threshold = photometry.threshold_above_back[ymin:ymax, xmin:xmax]

    #https://pythonmatplotlibtips.blogspot.com/2019/07/draw-two-axis-to-one-colorbar.html
    ax = plt.subplot(projection=source_ccd.wcs)
    ims = plt.imshow(source_ccd)
    cbar = plt.colorbar(ims, fraction=0.03, pad=0.11)
    pos = cbar.ax.get_position()
    cax1 = cbar.ax
    cax1.set_aspect('auto')
    cax2 = cax1.twinx()
    ylim = np.asarray(cax1.get_ylim())
    nonlin = source_ccd.meta['NONLIN']
    cax1.set_ylabel(source_ccd.unit)
    cax2.set_ylim(ylim/nonlin*100)
    cax1.yaxis.set_label_position('left')
    cax1.tick_params(labelrotation=90)
    cax2.set_ylabel('% nonlin')

    ax.contour(segm, levels=0, colors='white')
    ax.contour(source_ccd - threshold,
               levels=0, colors='gray')
    ax.set_title(f'{ccd.meta["OBJECT"]} {ccd.meta["DATE-AVG"]}')
    savefig_overwrite(outroot + '.png')
    


class ExoMultiPipe(CorMultiPipeBase):
    def file_write(self, ccd, outname,
                   photometry=None,
                   overwrite=False,
                   **kwargs):
        if photometry is not None:
            photometry.wide_source_table['filename'] = outname
            outroot, _ = os.path.splitext(outname)
            tname = outroot + '.ecsv'
            photometry.wide_source_table.write(
                tname, delimiter=',', overwrite=overwrite)
            gname = outroot + '_gaia.ecsv'
            photometry.source_gaia_join.write(
                gname, delimiter=',', overwrite=overwrite)
            # Don't gunk up the source_table ecsv with bounding box
            # stuff, but keep in mind we sorted it, so key off of
            # label
            bbox_table = photometry.source_catalog.to_table(
                ['label', 'bbox_xmin', 'bbox_xmax', 'bbox_ymin', 'bbox_ymax'])
            mask = (photometry.wide_source_table['OBJECT']
                    == ccd.meta['OBJECT'])
            label = photometry.wide_source_table[mask]['label']
            bbts = bbox_table[bbox_table['label'] == label]
            expand_bbox = 10
            xmin = bbts['bbox_xmin'][0] - expand_bbox
            xmax = bbts['bbox_xmax'][0] + expand_bbox
            ymin = bbts['bbox_ymin'][0] - expand_bbox
            ymax = bbts['bbox_ymax'][0] + expand_bbox
            source_ccd = ccd[ymin:ymax, xmin:xmax]
            threshold = photometry.threshold[ymin:ymax, xmin:xmax]
            segm = photometry.segm_image.data[ymin:ymax, xmin:xmax]
            #threshold = photometry.threshold_above_back[ymin:ymax, xmin:xmax]

            #https://pythonmatplotlibtips.blogspot.com/2019/07/draw-two-axis-to-one-colorbar.html
            ax = plt.subplot(projection=source_ccd.wcs)
            ims = plt.imshow(source_ccd)
            cbar = plt.colorbar(ims, fraction=0.03, pad=0.11)
            pos = cbar.ax.get_position()
            cax1 = cbar.ax
            cax1.set_aspect('auto')
            cax2 = cax1.twinx()
            ylim = np.asarray(cax1.get_ylim())
            nonlin = source_ccd.meta['NONLIN']
            cax1.set_ylabel(source_ccd.unit)
            cax2.set_ylim(ylim/nonlin*100)
            cax1.yaxis.set_label_position('left')
            cax1.tick_params(labelrotation=90)
            cax2.set_ylabel('% nonlin')

            ax.contour(segm, levels=0, colors='white')
            ax.contour(source_ccd - threshold,
                       levels=0, colors='gray')
            ax.set_title(f'{ccd.meta["OBJECT"]} {ccd.meta["DATE-AVG"]}')
            savefig_overwrite(outroot + '.png')
        super().file_write(ccd, outname, overwrite=overwrite, **kwargs)

def expo_calc(vmag):
    # TOI-2109b was nice reference
    expo_correct = 2*u.s
    vmag = vmag*u.mag(u.ct/u.s)
    toi_mag = 10*u.mag(u.ct/u.s)
    toi_expo = 20*u.s + expo_correct
    dmag = toi_mag - vmag
    return toi_expo * dmag.physical - expo_correct


log.setLevel('DEBUG')
# This is eventually going to be something like exo_pipeline
#directory = '/data/IoIO/raw/20210921'
directory = '/data/IoIO/raw/20220319/'
glob_include=GLOB_INCLUDE
calibration=None
photometry=None
keys_to_source_table=KEYS_TO_SOURCE_TABLE
keep_intermediate=False
#keep_intermediate=True
cpulimit=None
outdir_root=EXOPLANET_ROOT
create_outdir=True
kwargs={}

rd = reduced_dir(directory, outdir_root, create=False)
if calibration is None:
    calibration = Calibration(reduce=True)
if photometry is None:
    photometry = CorPhotometry(join_tolerance=5*u.arcsec,
                               cpulimit=cpulimit,
                               keys_to_source_table=keys_to_source_table)
flist = multi_glob(directory, glob_list=glob_include)
cmp = ExoMultiPipe(auto=True,
                   calibration=calibration,
                   photometry=photometry, 
                   fits_fixed_ignore=True, outname_ext='.fits',
                   keep_intermediate=keep_intermediate,
                   post_process_list=[barytime,
                                      mask_nonlin_sat,
                                      nd_filter_mask,
                                      object_to_objctradec,
                                      add_astrometry,
                                      as_single],
                   create_outdir=create_outdir,
                   **kwargs)
pout = cmp.pipeline([flist[0]], outdir='/tmp', overwrite=True)
#pout = cmp.pipeline(flist, outdir=rd, overwrite=True)

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

    ##directory = '/data/IoIO/raw/20210823'
    #directory = '/data/IoIO/raw/20210921'
    #rd = reduced_dir(directory, create=True)
    #glob_include = ['TOI*']

    dirs, _ = zip(*dirs_dates)
    assert len(dirs) > 0

    calibration=None
    photometry=None
    cpulimit=None
    if calibration is None:
        calibration = Calibration(reduce=True)
    if photometry is None:
        photometry = CorPhotometry(cpulimit=cpulimit)

    cmp = CorMultiPipeBase(auto=True, calibration=c,
                           fits_fixed_ignore=True, outname_ext='.fits', 
                           post_process_list=[barytime,
                                              mask_nonlin_sat,
                                              nd_filter_mask,
                                              add_astrometry,
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


#####  from astropy.nddata import CCDData
#####  ccd = CCDData.read('/tmp/WASP-36b-S001-R013-C002-R_p.fits')
#####  source_ccd = ccd
#####  ## https://stackoverflow.com/questions/27151098/draw-colorbar-with-twin-scales
#####  ##https://pythonmatplotlibtips.blogspot.com/2019/07/draw-two-axis-to-one-colorbar.html
#####  #
#####  #fig = plt.figure()
#####  #ax = plt.subplot(projection=source_ccd.wcs)
#####  #ims = plt.imshow(source_ccd)
#####  #cbar = fig.colorbar(ims, fraction=0.046, pad=0.04)
#####  #pos = cbar.ax.get_position()
#####  #cax1 = cbar.ax
#####  #cax1.set_aspect('auto')
#####  #cax2 = cax1.twinx()
#####  #ylim = np.asarray(cax1.get_ylim())
#####  #nonlin = source_ccd.meta['NONLIN']
#####  #
#####  #cax1.yaxis.set_ticks_position('left')
#####  #cax1.yaxis.set_label_position('left')
#####  #cax1.set_ylabel(source_ccd.unit)
#####  #cax2.set_ylim(ylim/nonlin*100)
#####  #cax2.set_ylabel('% nonlin')
#####  #
#####  #
#####  #
#####  #
#####  #
#####  #
#####  ax = plt.subplot(projection=source_ccd.wcs)
#####  ims = plt.imshow(source_ccd)
#####  cbar = plt.colorbar(ims, fraction=0.03, pad=0.11)
#####  pos = cbar.ax.get_position()
#####  cax1 = cbar.ax
#####  cax1.set_aspect('auto')
#####  cax2 = cax1.twinx()
#####  ylim = np.asarray(cax1.get_ylim())
#####  nonlin = source_ccd.meta['NONLIN']
#####  cax1.set_ylabel(source_ccd.unit)
#####  cax2.set_ylim(ylim/nonlin*100)
#####  cax1.yaxis.set_label_position('left')
#####  #cax1.yaxis.set_ticks_position('right')
#####  #cax1.yaxis.set_label_position('right')
#####  cax1.tick_params(labelrotation=45)
#####  cax2.set_ylabel('% nonlin')
#####  #cax2.yaxis.set_ticks_position('left')
#####  #cax2.yaxis.set_label_position('left')
#####  #plt.tight_layout()
#####  
#####  #ax.contour(segm, levels=0, colors='white')
#####  #ax.contour(source_ccd - threshold,
#####  #           levels=0, colors='gray')
#####  ax.set_title(f'{ccd.meta["OBJECT"]} {ccd.meta["DATE-OBS"]}')
#####  plt.show()
