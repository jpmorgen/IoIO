"""Construct model of telluric Na emission. """

import numpy as np

import matplotlib.pyplot as plt

from astropy import log
from astropy import units as u
from astropy.nddata import CCDData
from astropy.table import QTable

from ccdproc import ImageFileCollection

from bigmultipipe import no_outfile, cached_pout, prune_pout

from IoIO.utils import (Lockfile, reduced_dir, get_dirs_dates,
                        closest_in_time, valid_long_exposure, im_med_min_max,
                        add_history, cached_csv, iter_polyfit, 
                        savefig_overwrite)
from IoIO.simple_show import simple_show
from IoIO.cordata_base import CorDataBase
from IoIO.cormultipipe import (IoIO_ROOT, RAW_DATA_ROOT,
                               CorMultiPipeBase, angle_to_major_body,
                               nd_filter_mask, combine_masks)
from IoIO.calibration import Calibration, CalArgparseHandler
from IoIO.cor_photometry import CorPhotometry, add_astrometry

NA_MESO_ROOT = os.path.join(IoIO_ROOT, 'Na_meso')
LOCKFILE = '/tmp/na_meso_reduce.lock'

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

    # Add sun angle
    _ = sun_angle(data[0], bmp_meta=bmp_meta, **kwargs)

    tmeta = {'tavg': ccd.tavg,
             'sky_coord': ccd.sky_coord,
             'raoff': raoff0*u.arcmin,
             'decoff': decoff0*u.arcmin,
             'best_back': best_back,
             'best_back_std': best_back_std,
             'alt': objctalt*u.deg,
             'az': objctaz*u.deg,
             'airmass': airmass}
    
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


def na_meso_pipeline(directory=None, # raw directory
                     glob_include='Jupiter*',
                     calibration=None,
                     photometry=None,
                     n_back_boxes=N_BACK_BOXES,
                     num_processes=None,
                     outdir=None,
                     outdir_root=NA_MESO_ROOT,
                     create_outdir=True,
                     **kwargs):

    outdir = outdir or reduced_dir(directory, outdir_root, create=False)
    collection = ImageFileCollection(directory,
                                     glob_include=glob_include)
    if collection is None:
        return []
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

    if calibration is None:
        calibration = Calibration(reduce=True)
    if photometry is None:
        photometry = CorPhotometry(precalc=True,
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
                           na_meso_process],
                           #no_outfile],                           
        num_processes=num_processes,
        **kwargs)

    # but get ready to write to reduced directory if necessary
    #pout = cmp.pipeline([f_pairs[0]], outdir=outdir, overwrite=True)
    pout = cmp.pipeline(f_pairs, outdir=outdir, overwrite=True)
    pout, _ = prune_pout(pout, f_pairs)
    return pout

#on_fname = '/data/IoIO/raw/20210617/Jupiter-S007-R001-C001-Na_on.fts'
#off_fname = '/data/IoIO/raw/20210617/Jupiter-S007-R001-C001-Na_off.fts'
#on = CorDataBase.read(on_fname)
#off = CorDataBase.read(off_fname)
#bmp_meta = {'ads': 2}
#ccd = na_meso_process([on, off], in_name=[on_fname, off_fname],
#                      bmp_meta=bmp_meta, show=True)


directory = '/data/IoIO/raw/20210617'

pout = na_meso_pipeline(directory, fits_fixed_ignore=True)
