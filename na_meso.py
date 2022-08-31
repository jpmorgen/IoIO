"""Construct model of telluric Na emission. """

import os
import warnings

import numpy as np

import matplotlib.pyplot as plt

from astropy import log
from astropy import units as u
from astropy.nddata import CCDData
from astropy.table import QTable
from astropy.time import Time
from astropy.coordinates import SkyCoord, solar_system_ephemeris, get_body
from astropy.wcs import FITSFixedWarning

from ccdproc import ImageFileCollection

from bigmultipipe import no_outfile, cached_pout, prune_pout

from IoIO.utils import (Lockfile, reduced_dir, get_dirs_dates,
                        multi_glob, closest_in_time,
                        valid_long_exposure, im_med_min_max,
                        add_history, csvname_creator, cached_csv,
                        iter_polyfit, savefig_overwrite)
from IoIO.simple_show import simple_show
from IoIO.cordata_base import CorDataBase
from IoIO.cormultipipe import (IoIO_ROOT, RAW_DATA_ROOT,
                               CorMultiPipeBase, angle_to_major_body,
                               nd_filter_mask, combine_masks,
                               parallel_cached_csvs)
from IoIO.calibration import Calibration, CalArgparseHandler
from IoIO.photometry import (SOLVE_TIMEOUT, JOIN_TOLERANCE,
                             JOIN_TOLERANCE_UNIT)
from IoIO.cor_photometry import CorPhotometry, add_astrometry
 
BASE = 'Na_meso'  
OUTDIR_ROOT = os.path.join(IoIO_ROOT, BASE)
LOCKFILE = '/tmp/na_meso_reduce.lock'
AWAY_FROM_JUPITER = 5*u.deg # nominally 10 is what I used

# For Photometry -- number of boxes for background calculation
N_BACK_BOXES = 20

def sun_angle(ccd_in,
              bmp_meta=None,
              **kwargs):
    """cormultipipe post-processing routine that inserts angle between
    pointing direction and sun"""
    ccd = ccd_in.copy()
    sa = angle_to_major_body(ccd, 'sun')
    bmp_meta['sun_angle'] = sa
    ccd.meta['HIERARCH SUN_ANGLE'] = (sa.value, f'[{sa.unit}]')
    return ccd

def na_meso_process(data,
                    in_name=None,
                    bmp_meta=None,
                    calibration=None,
                    photometry=None,
                    n_back_boxes=N_BACK_BOXES,
                    show=False,
                    off_on_ratio=None,
                    **kwargs):
    """post-processing routine that processes a *pair* of ccd images
    in the order on-band, off-band"""
    if bmp_meta is None:
        bmp_meta = {}
    if off_on_ratio is None and calibration is None:
        calibration = Calibration(reduce=True)
    if photometry is None:
        photometry = CorPhotometry(precalc=True,
                                   n_back_boxes=n_back_boxes,
                                   **kwargs)
    if off_on_ratio is None:
        off_on_ratio, _ = calibration.flat_ratio('Na')
    ccd = data[0]
    raoff0 = ccd.meta.get('RAOFF') or 0
    decoff0 = ccd.meta.get('DECOFF') or 0
    fluxes = []
    # --> If we are going to use this for comets, what we really want
    # --> is just angular distance from Jupiter via the astropy ephemerides
    for ccd in data:
        raoff = ccd.meta.get('RAOFF') or 0
        decoff = ccd.meta.get('DECOFF') or 0
        if raoff != raoff0 or decoff != decoff0:
            log.warning(f'Mismatched RAOFF and DECOFF, skipping {in_name}')
            bmp_meta.clear()
            return None
        if (raoff**2 + decoff**2)**0.5 < 15:
            log.warning(f'Offset RAOFF {raoff} DECOFF {decoff} too small, skipping {in_name}')
            bmp_meta.clear()
            return None
        photometry.ccd = ccd
        exptime = ccd.meta['EXPTIME']*u.s
        flux = photometry.background / exptime
        fluxes.append(flux)

    background = fluxes[0] - fluxes[1]/off_on_ratio
    if show:
        simple_show(background.value)

    # Unfortunately, a lot of data were taken with the filter wheel
    # moving.  This uses the existing bias light/dark patch routine to
    # get uncontaminated part --> consider making this smarter
    best_back, _ = im_med_min_max(background)
    best_back_std = np.std(background)

    # Mesosphere is above the stratosphere, where the density of the
    # atmosphere diminishes to very small values.  So all attenuation
    # has already happened by the time we get up to the mesospheric
    # sodium layer
    # https://en.wikipedia.org/wiki/Atmosphere_of_Earth#/media/File:Comparison_US_standard_atmosphere_1962.svg
    ccd = data[0]
    objctalt = ccd.meta['OBJCTALT']
    objctaz = ccd.meta['OBJCTAZ']
    airmass = ccd.meta.get('AIRMASS')

    tmeta = {'tavg': ccd.tavg,
             'ra': ccd.sky_coord.ra,
             'dec': ccd.sky_coord.dec,
             'raoff': raoff0*u.arcmin,
             'decoff': decoff0*u.arcmin,
             'best_back': best_back,
             'best_back_std': best_back_std,
             'alt': objctalt*u.deg,
             'az': objctaz*u.deg,
             'airmass': airmass}
    _ = sun_angle(data[0], bmp_meta=tmeta, **kwargs)
    
    bmp_meta.update(tmeta)


    # In production, we don't plan to write the file, but prepare the
    # name just in case
    bmp_meta['outname'] = f'Jupiter_raoff_{raoff}_decoff_{decoff}_airmass_{airmass:.2}.fits'
    # Return one image
    data = CCDData(background, meta=ccd.meta, mask=ccd.mask)
    data.meta['OFFBAND'] = in_name[1]
    data.meta['HIERARCH N_BACK_BOXES'] = (n_back_boxes, 'Background grid for photutils.Background2D')
    data.meta['BESTBACK'] = (best_back.value,
                             f'Best background value ({best_back.unit})')
    add_history(data.meta,
                'Subtracted OFFBAND, smoothed over N_BACK_BOXES')

    return data

def na_meso_collection(directory,
                       glob_include='Jupiter*', # --> add comets
                       warning_ignore_list=[],
                       fits_fixed_ignore=False,
                       **kwargs):

    # I think the best thing to do is here make sure we are a valid
    # long exposure, offset from Jupiter, and that the on- and off
    # centers are within some tolerance of each other, since they
    # should nominally be identical.

    flist = multi_glob(directory, glob_include)
    if len(flist) == 0:
        return []
    # Create a collection of valid long Na exposures that are
    # pointed away from Jupiter
    collection = ImageFileCollection(directory, filenames=flist)
    st = collection.summary
    valid = ['Na' in f for f in st['filter']]
    valid = np.logical_and(valid, valid_long_exposure(st))
    # angle_to_major_body(ccd, 'jupiter') would do some good here, but
    # it is designed to work on the ccd CorData level.  So use the
    # innards of that code
    # --> Might be nice to restructure those innards to be available
    # to collections

    # Alternately I could assume Jupiter doesn't move much in one
    # night.  But this isn't too nasty to reproduce
    ras = st['objctra']
    decs = st['objctdec']
    # This is not the exact tavg, but we are just getting close enough
    # to make sure we are not pointed at Juptier
    dateobs_strs = st['date-obs']
    scs = SkyCoord(ras, decs, unit=(u.hourangle, u.deg))
    times = Time(dateobs_strs, format='fits')
    # What I can assume is that a directory has only one observatory
    sample_fname = st[valid]['file'][0]
    sample_fname = os.path.join(directory, sample_fname)
    if fits_fixed_ignore:
        warning_ignore_list.append(FITSFixedWarning)
    with warnings.catch_warnings():
        for w in warning_ignore_list:
            warnings.filterwarnings("ignore", category=w)
        ccd = CorDataBase.read(sample_fname)
    with solar_system_ephemeris.set('builtin'):
        body_coords = get_body('jupiter', times, ccd.obs_location)
    # It is very nice that the code is automatically vectorized
    seps = scs.separation(body_coords)
    valid = np.logical_and(valid, seps > AWAY_FROM_JUPITER)
    if np.all(~valid):
        return []
    if np.any(~valid):
        fbases = st['file'][valid]
        flist = [os.path.join(directory, f) for f in fbases]
    collection = ImageFileCollection(directory, filenames=flist)
    return collection
    
def na_meso_pipeline(directory_or_collection=None,
                     glob_include='Jupiter*', # --> add comets 
                     calibration=None,
                     photometry=None,
                     n_back_boxes=N_BACK_BOXES,
                     num_processes=None,
                     outdir=None,
                     outdir_root=OUTDIR_ROOT,
                     create_outdir=True,
                     **kwargs):

    if isinstance(directory_or_collection, ImageFileCollection):
        # We are passed a collection when running multiple directories
        # in parallel
        directory = directory_or_collection.location
        collection = directory_or_collection
    else:
        directory = directory_or_collection
        collection = na_meso_collection(directory,
                                        glob_include=glob_include,
                                        **kwargs)
    if collection is None:
        return []

    outdir = outdir or reduced_dir(directory, outdir_root, create=False)
    try:
        raoffs = collection.values('raoff', unique=True)
        decoffs = collection.values('decoff', unique=True)
    except Exception as e:
        log.debug(f'Skipping {directory} because of problem with RAOFF/DECOFF: {e}')
        return []
    f_pairs = []
    for raoff in raoffs:
        for decoff in decoffs:
            try:
                subc = collection.filter(raoff=raoff, decoff=decoff)
            except:
                log.debug(f'No match for RAOFF = {raoff} DECOFF = {decoff}')
                continue
            fp = closest_in_time(subc, ('Na_on', 'Na_off'),
                                 valid_long_exposure,
                                 directory=directory)
            f_pairs.extend(fp)
    if len(f_pairs) == 0:
        log.warning(f'No matching set of Na background files found '
                    f'in {directory}')
        return []

    calibration = calibration or Calibration(reduce=True)
    photometry = photometry or CorPhotometry(
        precalc=True,
        n_back_boxes=n_back_boxes,
        **kwargs)
        
    # --> We are going to want add_ephemeris here with a CorPhotometry
    # --> to build up astormetric solutions
    cmp = CorMultiPipeBase(
        auto=True,
        calibration=calibration,
        photometry=photometry,
        fail_if_no_wcs=False,
        create_outdir=create_outdir,
        post_process_list=[nd_filter_mask,
                           combine_masks,
                           na_meso_process,
                           no_outfile],                           
        num_processes=num_processes,
        **kwargs)

    # but get ready to write to reduced directory if necessary
    #pout = cmp.pipeline([f_pairs[0]], outdir=outdir, overwrite=True)
    pout = cmp.pipeline(f_pairs, outdir=outdir, overwrite=True)
    pout, _ = prune_pout(pout, f_pairs)
    return pout

def na_meso_directory(directory_or_collection,
                      read_pout=True,
                      write_pout=True,
                      write_plot=True,
                      outdir=None,
                      outdir_root=OUTDIR_ROOT,
                      create_outdir=True,
                      show=False,
                      **kwargs):

    if isinstance(directory_or_collection, ImageFileCollection):
        # We are passed a collection when running multiple directories
        # in parallel
        directory = directory_or_collection.location
        collection = directory_or_collection
    else:
        directory = directory_or_collection
        collection = na_meso_collection(directory, **kwargs)

    outdir = outdir or reduced_dir(directory, outdir_root, create=False)
    poutname = os.path.join(outdir, BASE + '.pout')
    pout = cached_pout(na_meso_pipeline,
                       poutname=poutname,
                       read_pout=read_pout,
                       write_pout=write_pout,
                       directory_or_collection=collection,
                       outdir=outdir,
                       create_outdir=create_outdir,
                       **kwargs)
    if pout is None or len(pout) == 0:
        return QTable()

    _ , pipe_meta = zip(*pout)
    t = QTable(rows=pipe_meta)
    return t

def na_meso_tree(data_root=RAW_DATA_ROOT,
                 start=None,
                 stop=None,
                 calibration=None,
                 photometry=None,
                 keep_intermediate=None,
                 solve_timeout=SOLVE_TIMEOUT,
                 join_tolerance=JOIN_TOLERANCE*JOIN_TOLERANCE_UNIT,
                 read_csvs=True,
                 write_csvs=True,
                 create_outdir=True,                       
                 show=False,
                 ccddata_cls=None,
                 outdir_root=OUTDIR_ROOT,                 
                 **kwargs):
    dirs_dates = get_dirs_dates(data_root, start=start, stop=stop)
    dirs, _ = zip(*dirs_dates)
    if len(dirs) == 0:
        log.warning('No directories found')
        return
    calibration = calibration or Calibration()
    if photometry is None:
        photometry = CorPhotometry(
            precalc=True,
            solve_timeout=solve_timeout,
            join_tolerance=join_tolerance)

    cached_csv_args = {
        'csvnames': csvname_creator,
        'csv_base': BASE + '.ecsv',
        'write_csvs': write_csvs,
        'calibration': calibration,
        'photometry': photometry,
        'outdir_root': outdir_root,
        'create_outdir': create_outdir}
    cached_csv_args.update(**kwargs)
    summary_table = parallel_cached_csvs(dirs,
                                         code=na_meso_directory,
                                         collector=na_meso_collection,
                                         files_per_process=3,
                                         read_csvs=read_csvs,
                                         **cached_csv_args)
    summary_table.write(os.path.join(outdir_root, BASE + '.ecsv'),
                                     overwrite=True)
    return summary_table



#on_fname = '/data/IoIO/raw/20210617/Jupiter-S007-R001-C001-Na_on.fts'
#off_fname = '/data/IoIO/raw/20210617/Jupiter-S007-R001-C001-Na_off.fts'
#on = CorDataBase.read(on_fname)
#off = CorDataBase.read(off_fname)
#bmp_meta = {'ads': 2}
#ccd = na_meso_process([on, off], in_name=[on_fname, off_fname],
#                      bmp_meta=bmp_meta, show=True)


directory = '/data/IoIO/raw/20210617'

#pout = na_meso_pipeline(directory, fits_fixed_ignore=True)

#t = na_meso_directory(directory, fits_fixed_ignore=True)

#t = na_back_tree(start='2021-06-17', stop='2021-06-17', fits_fixed_ignore=True)

#collection = na_meso_collection(directory)

#t = na_meso_tree(start='2021-06-17', stop='2021-06-17', fits_fixed_ignore=True)

t = na_meso_tree(start='2021-06-01', stop='2021-06-17', fits_fixed_ignore=True)
