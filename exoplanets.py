#!/usr/bin/python3
"""Reduce files for Elisabeth Adams XRP project"""

import os
import glob
import argparse

import numpy as np

from astropy import log
from astropy import units as u
from astropy.coordinates import SkyCoord
from astroquery.simbad import Simbad

import ccdproc as ccdp

from bigmultipipe import prune_pout
from ccdmultipipe import as_single

from IoIO.utils import (reduced_dir, get_dirs_dates, multi_glob)
from IoIO.cormultipipe import (RAW_DATA_ROOT, CorMultiPipeBase,
                               mask_nonlin_sat, nd_filter_mask)
from IoIO.calibration import Calibration, CalArgparseHandler
from IoIO.photometry import SOLVE_TIMEOUT, JOIN_TOLERANCE
from IoIO.cor_photometry import (KEYS_TO_SOURCE_TABLE, CorPhotometry,
                                 add_astrometry, object_to_objctradec,
                                 CorPhotometryArgparseMixin)

EXOPLANET_ROOT = '/data/Exoplanets'
GLOB_INCLUDE = ['TOI*', 'WASP*', 'KPS*', 'HAT*', 'K2*', 'TrES*',
                'Qatar*', 'GJ*']
# I am not sure the barytime calculations would be valid for solar
# system objects because they are too close.  So we don't put this
# into cor_photometry
KEYS_TO_SOURCE_TABLE = KEYS_TO_SOURCE_TABLE + [('MJDBARY', u.day)]
KEEP_FITS = 3
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

    ctype1 = ccd.meta.get('CTYPE1')
    ctype2 = ccd.meta.get('CTYPE2')
    if ccd.meta.get('OBJECT_TO_OBJCTRADEC'):
        # First choice is to get precise direction to object we are
        # observing
        ra = ccd.meta.get('OBJCTRA')
        dec = ccd.meta.get('OBJCTDEC')
        unit = (u.hourangle, u.deg)
    elif ctype1 and ctype2 and 'RA' in ctype1 and 'DEC' in ctype2:
        # Official plate solutions next preference, if available
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
    # Fancy conversion to ICRS is likely not done anywhere in IoIO
    # system, so default to FK5 is safe
    radesys = ccd.meta.get('RADESYS') or ccd.meta.get('RADECSYS') or 'FK5'
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

class ExoMultiPipe(CorMultiPipeBase):
    def file_write(self, ccd, outname,
                   photometry=None,
                   **kwargs):
        written_name = super().file_write(
            ccd, outname, photometry=photometry, 
            write_local_photometry=True, **kwargs)
        outroot, _ = os.path.splitext(outname)
        try:
            photometry.plot_object(outname=outroot + '.png')
        except Exception as e:
            log.warning(f'Not able to plot object for {outname}: {e}')
        return written_name

def expo_calc(vmag):
    # TOI-2109b was nice reference
    expo_correct = 2*u.s
    vmag = vmag*u.mag(u.ct/u.s)
    toi_mag = 10*u.mag(u.ct/u.s)
    toi_expo = 20*u.s + expo_correct
    dmag = toi_mag - vmag
    return toi_expo * dmag.physical - expo_correct

def exoplanet_pipeline(flist,
                       reduced_directory,
                       calibration=None,
                       photometry=None,
                       keep_intermediate=False,
                       keep_fits=KEEP_FITS,
                       create_outdir=True,
                       **kwargs):
    # Process a list of observations of a single exoplanet
    if calibration is None:
        calibration = Calibration(reduce=True)
    if photometry is None:
        photometry = CorPhotometry(
            precalc=True,
            keys_to_source_table=KEYS_TO_SOURCE_TABLE)
    cmp = ExoMultiPipe(auto=True,
                       calibration=calibration,
                       photometry=photometry, 
                       fits_fixed_ignore=True, outname_ext='.fits',
                       keep_intermediate=keep_intermediate,
                       post_process_list=[mask_nonlin_sat,
                                          nd_filter_mask,
                                          object_to_objctradec,
                                          barytime,
                                          add_astrometry,
                                          as_single],
                       create_outdir=create_outdir,
                       **kwargs)
    #pout = cmp.pipeline([flist[0]], outdir='/tmp', overwrite=True)
    pout = cmp.pipeline(flist, outdir=reduced_directory, overwrite=True)
    pout, flist = prune_pout(pout, flist)
    if len(pout) == 0:
        log.warning(f'No good observations in series {flist[0]}')
    # The way pipeline works, these will be randomized
    outnames, bmp_meta = zip(*pout)
    keep_fits = min(keep_fits, len(outnames))
    for f in outnames[keep_fits:]:
        os.remove(f)    
    return bmp_meta

def exoplanet_directory(directory,
                        glob_include=GLOB_INCLUDE,
                        outdir_root=EXOPLANET_ROOT,
                        **kwargs):
    rd = reduced_dir(directory, outdir_root, create=False)
    # Use multi_glob to get only the exoplanets, but use
    # ccdp.ImageFileCollection to make sure the object is what we
    # think it is.  This avoids having to grep it out of the filename
    all_files = multi_glob(directory, glob_list=glob_include)
    if len(all_files) == 0:
        return
    collection = ccdp.ImageFileCollection(filenames=all_files,
                                          keywords=['object'])
    exoplanets = collection.values('object', unique=True)

    for e in exoplanets:
        # Load our object into the cache so there are no collisions
        # during multiprocessing
        s = Simbad()
        simbad_results = s.query_object(e)
        flist = collection.files_filtered(object=e, include_path=True)
        exoplanet_pipeline(flist, rd, **kwargs)
        # This could combine output ecsvs, do a Sloan catalog search on
        # distinct fields and merge in the results

def exoplanet_tree(raw_data_root=RAW_DATA_ROOT,
                   outdir_root=EXOPLANET_ROOT,
                   glob_include=GLOB_INCLUDE,
                   start=None,
                   stop=None,
                   calibration=None,
                   photometry=None,
                   solve_timeout=None,
                   join_tolerance=JOIN_TOLERANCE,
                   keys_to_source_table=KEYS_TO_SOURCE_TABLE,
                   create_outdir=True,
                   keep_fits=KEEP_FITS,
                   **kwargs):
    dirs_dates = get_dirs_dates(raw_data_root, start=start, stop=stop)
    dirs, _ = zip(*dirs_dates)
    if len(dirs) == 0:
        log.warning('No directories found')
        return
    if calibration is None:
        calibration = Calibration(reduce=True)
    if photometry is None:
        photometry = CorPhotometry(
            precalc=True,
            solve_timeout=solve_timeout,
            join_tolerance=join_tolerance,
            keys_to_source_table=keys_to_source_table)
    for directory in dirs:
        # For now, behave like Calibration.  If the directory is there
        # with files in it, we are done.  Move/delete the directory to
        # trigger recalculation
        rd = reduced_dir(directory, outdir_root, create=False)
        ecsv_list = glob.glob(os.path.join(rd, '*ecsv'))
        if len(ecsv_list) > 0:
            continue
        exoplanet_directory(directory,
                            glob_include=glob_include,
                            outdir_root=outdir_root,
                            calibration=calibration,
                            photometry=photometry,
                            create_outdir=create_outdir,
                            keep_fits=keep_fits)

class ExoArgparseMixin:
    def add_exoplanet_root(self,
                           default=EXOPLANET_ROOT,
                           help=None,
                           **kwargs):
        option = 'exoplanet_root'
        if help is None:
            help = f'exoplanet root (default: {default})'
        self.parser.add_argument('--' + option, 
                                 default=default, help=help, **kwargs)

    def add_start(self, 
                  default=None,
                  help=None,
                  **kwargs):
        option = 'start'
        if help is None:
            help = 'start directory/date (default: earliest)'
        self.parser.add_argument('--' + option, 
                                 default=default, help=help, **kwargs)

    def add_stop(self, 
                 default=None,
                 help=None,
                 **kwargs):
        option = 'stop'
        if help is None:
            help = 'stop directory/date (default: latest)'
        self.parser.add_argument('--' + option, 
                                 default=default, help=help, **kwargs)

    def add_keep_fits(self, 
                 default=KEEP_FITS,
                 help=None,
                 **kwargs):
        option = 'keep_fits'
        if help is None:
            help = (f'keep N reduced FITS files (default: {default})')
        self.parser.add_argument('--' + option, 
                                 default=default, help=help, **kwargs)

class ExoArgparseHandler(ExoArgparseMixin, CorPhotometryArgparseMixin,
                         CalArgparseHandler):
    def add_all(self):
        """Add options used in cmd"""
        self.add_exoplanet_root()
        self.add_start()
        self.add_stop()
        self.add_solve_timeout()
        self.add_join_tolerance()
        self.add_keep_fits()
        super().add_all()

    def cmd(self, args):
        c = CalArgparseHandler.cmd(self, args)
        exoplanet_tree(raw_data_root=args.raw_data_root,
                       outdir_root=args.exoplanet_root,
                       start=args.start,
                       stop=args.stop,
                       calibration=c,
                       join_tolerance=args.join_tolerance,
                       create_outdir=args.create_outdir,
                       keep_fits=args.keep_fits,
                       solve_timeout=args.solve_timeout)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run exoplanet pipeline')
    aph = ExoArgparseHandler(parser)
    aph.add_all()
    args = parser.parse_args()
    aph.cmd(args)
    

    
#log.setLevel('DEBUG')
##directory = '/data/IoIO/raw/20210921'
#directory = '/data/IoIO/raw/20220319/'
##exoplanet_tree()
#exoplanet_directory(directory, ccddata_write=True)



    #pout = exoplanet_pipeline(directory, ccddata_write=False)


    #data_root = RAW_DATA_ROOT
    #reduced_root = EXOPLANET_ROOT
    #
    #dirs_dates = get_dirs_dates(data_root)
    #
    #
    ##### from cormultipipe import RedCorData
    ##### 
    ##### ccd = RedCorData.read('/data/io/IoIO/raw/20210921/TOI-2109-S004-R021-C002-R.fts')
    ##### ccd.meta = obs_location_to_hdr(ccd.meta, location=IOIO_1_LOCATION)
    ##### 
    ##### #ccd = barytime(ccd)
    ##### #print((ccd.meta['JDBARY'] - ccd.meta['BJD-OBS']) * 24*3600)
    #
    ###directory = '/data/IoIO/raw/20210823'
    ##directory = '/data/IoIO/raw/20210921'
    ##rd = reduced_dir(directory, create=True)
    ##glob_include = ['TOI*']
    #
    #dirs, _ = zip(*dirs_dates)
    #assert len(dirs) > 0
    #
    #calibration=None
    #photometry=None
    #cpulimit=None
    #if calibration is None:
    #    calibration = Calibration(reduce=True)
    #if photometry is None:
    #    photometry = CorPhotometry(cpulimit=cpulimit)
    #
    #cmp = CorMultiPipeBase(auto=True, calibration=c,
    #                       fits_fixed_ignore=True, outname_ext='.fits', 
    #                       post_process_list=[barytime,
    #                                          mask_nonlin_sat,
    #                                          nd_filter_mask,
    #                                          add_astrometry,
    #                                          as_single])
    #for d in dirs:
    #    flist = []
    #    for gi in GLOB_INCLUDE:
    #        flist += glob.glob(os.path.join(d, gi))
    #    if len(flist) == 0:
    #        log.debug(f'No exoplant observations in {d}')
    #        continue
    #    reddir = reduced_dir(d, reduced_root)
    #    pout = cmp.pipeline(flist, outdir=reddir,
    #                        create_outdir=True, overwrite=True)


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
