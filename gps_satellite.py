#!/usr/bin/python3

import re
import os

from multiprocessing import Pool

import numpy as np

from astropy import log
import astropy.units as u
from astropy.table import QTable, MaskedColumn, vstack
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.stats import biweight_location, mad_std

from bigmultipipe import (WorkerWithKwargs, cached_pout, prune_pout,
                          num_can_process)

from ccdmultipipe import as_single

from IoIO.cordata_base import IOIO_1_LOCATION
from IoIO.ioio_globals import IoIO_ROOT, RAW_DATA_ROOT
from IoIO.utils import (reduced_dir, get_dirs_dates, multi_glob,
                        ccd_center_fov_to_meta)
from IoIO.cormultipipe import (CorMultiPipeBinnedOK, mask_nonlin_sat,
                               nd_filter_mask)
from IoIO.calibration import Calibration
from IoIO.cor_photometry import (CorPhotometry, add_astrometry,
                                 write_photometry)

BASE = 'GPS_Satellites'
OUTDIR_ROOT = os.path.join(IoIO_ROOT, BASE)

GPS_GLOB_LIST = ['Calibration*', 'Autosave*', 'GPS_Satellite*']
MATCH_MAX_SEP = 5*u.arcsec

class GPSMultiPipe(CorMultiPipeBinnedOK):
    def file_write(self, ccd, outname,
                   in_name=None, photometry=None,
                   **kwargs):
        written_name = super().file_write(
            ccd, outname, photometry=photometry, **kwargs)
        write_photometry(in_name=in_name, outname=outname,
                         photometry=photometry,
                         write_wide_source_table=True,
                         **kwargs)
        outroot, _ = os.path.splitext(outname)
        #try:
        #    photometry.plot_object(outname=outroot + '.png')
        #except Exception as e:
        #    log.warning(f'Not able to plot object for {outname}: {e}')
        return written_name

def gps_process(ccd_in, bmp_meta=None, **kwargs):
    if bmp_meta is None:
        bmp_meta = {}
    ccd = ccd_in.copy()
    bmp_meta['tavg'] = ccd.tavg
    ccd = ccd_center_fov_to_meta(ccd, bmp_meta)
    plate_scale = proj_plane_pixel_scales(ccd.wcs)    
    cunit = [u.Unit(cu) for cu in ccd.wcs.world_axis_units]
    plate_scale = plate_scale * cunit
    plate_scale = [ps.to(u.arcsec) / u.pix for ps in plate_scale]
    bmp_meta['plate_scale'] = plate_scale
    return ccd

def gps_pipeline(directory,
                 calibration=None,
                 photometry=None,
                 glob_include=GPS_GLOB_LIST,
                 glob_exclude_list=None,
                 fits_fixed_ignore=True,
                 num_processes=None,
                 outdir=OUTDIR_ROOT,
                 **kwargs):

    calibration = calibration or Calibration(reduce=True)
    photometry = photometry or CorPhotometry(precalc=True, **kwargs)

    fnames = multi_glob(directory, glob_include, glob_exclude_list)

    cmp = GPSMultiPipe(
        auto=True,
        calibration=calibration,
        photometry=photometry,
        fail_if_no_solve=True,
        post_process_list=[mask_nonlin_sat,
                           nd_filter_mask,
                           add_astrometry,
                           gps_process,
                           as_single],
        fits_fixed_ignore=fits_fixed_ignore, 
        num_processes=num_processes,
        date_obs_correct=False, # Make sure to work from commanded shutter time
        **kwargs)
    #pout = cmp.pipeline([fnames[0]], outdir=outdir, overwrite=True)
    pout = cmp.pipeline(fnames, outdir=outdir, overwrite=True)
    pout, _ = prune_pout(pout, fnames)
    if len(pout) == 0:
        log.warning(f'No good observations in series {fnames[0]}')
        return pout
    return pout

def process_one_csv(enum_csvs, csvs=None, gps_t=None, max_sep=MATCH_MAX_SEP):
    # This does one inner loop in the process of comparing each csv to
    # all subsequent csvs
    c0idx, csv0 = enum_csvs
    ct0 = None
    csv0_matched = None
    d2d = None
    for n1, csv1 in enumerate(csvs[c0idx+1:]):
        c1idx = n1 + c0idx + 1
        # Check if original image FOVs are too far apart
        i0_coord = gps_t[c0idx]['center_coord']
        i1_coord = gps_t[c1idx]['center_coord']
        i0_rad = gps_t[c0idx]['fov_rad']
        if i0_coord.separation(i1_coord) > i0_rad:
            continue
        # Read in both tables only if we need to
        # --> The plate scale should be put in at the photometry level
        if ct0 is None:
            ct0 = QTable.read(csv0)
            ct0['plate_scale_ra'] = gps_t[c0idx]['plate_scale'][0]
            ct0['plate_scale_dec'] = gps_t[c0idx]['plate_scale'][1]
        ct1 = QTable.read(csv1)
        ct1['plate_scale_ra'] = gps_t[c1idx]['plate_scale'][0]
        ct1['plate_scale_dec'] = gps_t[c1idx]['plate_scale'][1]
        # Do the match.  The shape of these returns matches t0.  idx
        # is into t1
        idx, d2d, d3d = ct0['coord'].match_to_catalog_sky(ct1['coord'])
        # Prepare to add a masked column to t0, putting the csv1 name
        # in the description
        colmask = d2d > max_sep
        ct0[f'csv_{c1idx}_idx'] = MaskedColumn(data=idx,
                                               mask=colmask,
                                               description=csv1)
        csv0_matched = csv0.replace('.ecsv', '_matched.ecsv')
        ct0.write(csv0_matched, overwrite=True)
    # Return the name of the csv that has all the match indices (and
    # masked values) and the d2d of real matches to provide an
    # estimate of the astrometric accuracy
    if d2d is None:
        return csv0_matched, d2d
    return csv0_matched, d2d[~colmask]


# --> with gps_tree, there should be a start at 2025-01-24, since I
# have no observations before that time
#directory='/data/IoIO/raw/2025-01-24/'
def gps_directory(
        directory='/data/IoIO/raw/2025-03-19/',
        outdir=None,
        outdir_root=OUTDIR_ROOT,
        read_pout=True,
        write_pout=True,
        create_outdir=True,
        glob_include=GPS_GLOB_LIST,
        glob_exclude_list=None,
        max_sep=MATCH_MAX_SEP,
        **kwargs):

    outdir = outdir or reduced_dir(directory, outdir_root, create=False)
    poutname = os.path.join(outdir, BASE + '.pout')
    pout = cached_pout(gps_pipeline,
                       directory=directory,
                       poutname=poutname,
                       read_pout=read_pout,
                       write_pout=write_pout,
                       outdir=outdir,
                       create_outdir=create_outdir,
                       glob_include=glob_include,
                       glob_exclude_list=glob_exclude_list,
                       **kwargs)

    _ , pipe_meta = zip(*pout)
    gps_t = QTable(rows=pipe_meta)

    gps_t.sort('outname')
    csvs = [f.replace('.fits', '.ecsv') for f in gps_t['outname']]

    # Cycle through each csv, matching it each subsequent csv.  This ends
    # up providing all pairwise combinations once recorded through a set
    # of *_matched.ecsv files.  Unwrap the outermost loop with
    # multiprocessing.Pool, since each one is independent.
    loop_num_processes = num_can_process(num_to_process=len(csvs),
                                         num_processes=0.8)
    wwk = WorkerWithKwargs(process_one_csv,
                           csvs=csvs,
                           gps_t=gps_t,
                           max_sep=max_sep)
    with Pool(processes=loop_num_processes) as p:
        matched_csvs_d2d_list = p.map(wwk.worker, enumerate(csvs))

    matched_csvs, d2d_list = zip(*matched_csvs_d2d_list)
    d2ds = None
    aunit = None
    for d2d in d2d_list:
        if d2d is None:
            continue
        # Make sure all units are consistent
        aunit = aunit or d2d.unit
        assert d2d.unit == aunit
        if d2ds is None:
            # or doesn't work easily with arrays
            d2ds = d2d.value
        d2ds = np.append(d2ds, d2d.value)
    d2ds = d2ds*aunit

    # Remove None entries (FOVs too far apart) and put back in order
    matched_csvs = [csv for csv in matched_csvs if csv is not None]
    matched_csvs.sort()

    mt = QTable.read(matched_csvs[0])
    t_trans_cols = mt.colnames
    csv_re = re.compile('csv_.*_idx')
    t_trans_cols = [c for c in t_trans_cols if not csv_re.match(c)]

    # Make a table of all transcient objects
    t_trans = QTable()
    for matched_csv in matched_csvs:
        mt = QTable.read(matched_csv)
        cn = mt.colnames
        # Make a list of indices into mt of all objects that didn't match
        # to other images
        trans_idx = []
        for idx_col in filter(csv_re.match, cn):
            if not hasattr(mt[idx_col], 'mask'):
                # Annoying feature of MaskedColumn with no masks is that
                # it morphs to a Column, at least with older astropy 5.2.1
                continue
            # If we made it here, we have some masked values
            tidx = np.flatnonzero(mt[idx_col].mask)
            trans_idx.extend(tidx.tolist())
        trans_idx = list(set(trans_idx))

        trans_mt = mt[t_trans_cols][trans_idx]
        # with astropy 5.2.1, sky_coord masking is not supported,
        # which is required for vstack.  Replace with RA and DEC while we stack
        trans_mt['RA'] = trans_mt['coord'].ra
        trans_mt['DEC'] = trans_mt['coord'].dec
        tmtcn = trans_mt.colnames
        tmtcn = [c for c in tmtcn if c != 'coord']
        t_trans = vstack([t_trans, trans_mt[tmtcn]],
                         metadata_conflicts='silent')

    # Make translation back into astropy coord
    obstimes = Time(t_trans['DATE-AVG'],
                    location=gps_t['tavg'][0].location)
    t_trans['coord'] = SkyCoord(ra=t_trans['RA'], dec=t_trans['DEC'],
                                obstime=obstimes)
    idx, d2d, d3d = t_trans['coord'].match_to_catalog_sky(t_trans['coord'],
                                                          nthneighbor=2)
    uniq_mask = d2d > max_sep
    t_trans = t_trans[uniq_mask]
    return t_trans, biweight_location(d2ds)

# Use cached_csv in gps_tree instead
#csv_outname = os.path.join(outdir, BASE + '.ecsv')
#t_trans.write(csv_outname, overwrite=True)

log.setLevel('DEBUG')

# This will eventually become gps_tree.  And some of the above may
# become moving_target
# --> put this back in
#t_trans, astrometric_accuracy = gps_directory()


lon = IOIO_1_LOCATION.lon.wrap_at(360*u.deg)
lat = IOIO_1_LOCATION.lat
if lat < 0:
    ns = 'S'
else:
    ns = 'N'

alat = abs(lat)
latstr = f'{alat.dms[0]:.0f} {alat.dms[1]:.0f} {alat.dms[2]:.0f}'
latstr = f'{latstr} {ns}'

alt = IOIO_1_LOCATION.height
altstr = f'{alt.value:.0f}{alt.unit}'

loc = f'Long. {lon.dms[0]:.0f} {lon.dms[1]:.0f} {lon.dms[2]:.0f} E, Lat. {latstr}, Alt. {altstr}, Google Earth'

ades_hdr = f'''COD XXX
COM {loc}
# version=2017
# observatory
! mpcCode XXX
! name Io Input/Output Observatory (IoIO)
# submitter
! name Jeffrey P. Morgenthaler
# observers
! name Jeffrey P. Morgenthaler
# measurers
! name Jeffrey P. Morgenthaler
# telescope
! design Schmidt-Cassegrain
! aperture 0.35
! detector CCD
# fundingSource NSF
# comment
! {loc}
'''

# Now I need to output them in the ADES format

# This will be ades_psv
#print('permID |provID     |trkSub  |artSat      |mode|stn |obsTime                |ra         |dec        |rmsRA|rmsDec|astCat  |mag  |rmsMag|band|photCat |photAp|logSNR|seeing|exp |rmsFit|nStars|notes|remarks')

# print('permID |provID     |trkSub  |artSat      |mode|stn |obsTime                |ra         |dec        |rmsRA|rmsDec|astCat  |mag  |rmsMag|band|photCat |photAp|logSNR|seeing|exp |notes|remarks')
# 
# for row in t_trans:
#     coord= row['coord']
#     obstime = coord.obstime.to_string()
#     obstime = f'{obstime[0:-1]}Z'
#     ra = f'{coord.ra.value:9.6f}'
#     dec = f'{coord.dec.value:+9.6f}'
#     astrometric_accuracy = astrometric_accuracy.to(u.arcsec)
#     aa = astrometric_accuracy.value
#     snr = np.sqrt(row['segment_flux'].value)
#     lsnr = np.log10(snr)
#     # Estimate seeing as 1/2 the source radius
#     seeing = row['equivalent_radius']  * row['plate_scale_dec'] / 2
#     seeing = seeing.value
#     exptime = row['EXPTIME']
#     exptime = exptime.value
# 
#     print(f'       |           |        |            | CCD|XXX |{obstime}|{ra} |{dec} |{aa:.2f} |{aa:.2f}  | TYCHO-2|     |      |    |        |      |{lsnr:.3f} |{seeing:.1f}   |{exptime:4}|     |')


#from cordata import CorData
#ccd= CorData.read('/data/IoIO/GPS_Satellites/2025-03-19/GPS_Satellite-0104_p.fits')


#for c0idx, csv0 in enumerate(csvs):
#    # Cycle through each csv, matching it each subsequent csv.  This
#    # ends up providing all pairwise combinations once
#    ct0 = QTable.read(csv0)
#    for n1, csv1 in enumerate(csvs[c0idx+1:]):
#        ct1 = QTable.read(csv1)
#        c1idx = n1 + c0idx + 1
#        if (gps_t[c0idx]['center_coord'].separation(gps_t[c1idx]['center_coord'])
#            > gps_t[c0idx]['fov_rad']):
#            # FOVs are too far apart
#            continue
#        # Do the match.  The shape of these returns matches t0.  idx
#        # is into t1
#        idx, d2d, d3d = ct0['coord'].match_to_catalog_sky(ct1['coord'])
#        # Prepare to add a masked column to t0, putting the csv1 name
#        # in the description
#        colmask = d2d > max_sep
#        ct0[f'csv_{c1idx}_idx'] = MaskedColumn(data=idx,
#                                               mask=colmask,
#                                               description=csv1)
#        csv0_matched = csv0.replace('.ecsv', '_matched.ecsv')
#        ct0.write(csv0_matched, overwrite=True)


# So this is the set of c0 coords that have matched c1
#c0[sep_constraint]
# This is the index into c0 of the matches.  This is probably what I
# should store on a per c1 basis to then compare to see if they are
# consistent.  Alterntely, I can add a masked column to c0 with these
# results.  The column name would have to be some sort of encoded
# DATE-AVG.  Oh, maybe better yet, the filename of the ecsv
#idx[sep_constraint]

#for obj_coord in c0['coord']:
#    seps = obj_coord.separation(c1['coord'])
#    match = np.argmin(seps)
#    if seps[match] > 5*u.arcsec:
#        continue

 # This will be ccd_center_to_meta

#from cordata import CorData
#from astropy.wcs.utils import pixel_to_skycoord
#ccd = CorData.read('/data/IoIO/GPS_Satellites/2025-03-19/Autosave Image -0058_p.fits')
#
#bmp_meta=None
#if bmp_meta is None:
#    bmp_meta = {}
#
#
#ccd_shape = np.asarray(ccd.shape)
#ccd_center = np.round(ccd_shape/2)
#coord = pixel_to_skycoord(ccd_center[0], ccd_center[1], ccd.wcs)
#ccd.meta['CENTRA'] = (coord.ra.to_string(), 'FOV center RA in deg')
#ccd.meta['CENTDEC'] = (coord.dec.to_string(), 'FOV center DEC in deg')
#bmp_meta['CENTRA'] = ccd.meta['CENTRA']
#bmp_meta['CENTDEC'] = ccd.meta['CENTDEC']

#from utils import ra_to_hms
#
#ra_to_hms(coord.ra)
## This is going to be utils.ra_to_hms
#ra=coord.ra
#ra_hms = f'{ra.hms.h:.0f}h{ra.hms.m:.0f}m{ra.hms.s:0.7f}s'


# from cordata import CorData
# from cor_process import cor_process
# ccd = CorData.read('/data/IoIO/raw/2025-01-24/Calibration-0010.fit')
# calibration = Calibration(reduce=True)
#
# ccd_out = cor_process(ccd, calibration=calibration,
#                       oscan=True,
#                       gain=True,
#                       error=True,
#                       min_value=True)





## This might be something like gps_pipeline (gps_directory is merged
## in here at the moment)
#directory='/data/IoIO/raw/2025-01-24/'
#glob_include=GPS_GLOB_LIST
#glob_exclude_list=None
#outdir=None
#outdir_root=OUTDIR_ROOT
#create_outdir=True
#calibration=None
#photometry=None
#kwargs={}
#
#calibration = calibration or Calibration(reduce=True)
#photometry = photometry or CorPhotometry(precalc=True, **kwargs)
#
#
#fnames = multi_glob(directory, glob_include, glob_exclude_list)
#poutname = os.path.join(outdir, BASE + '.pout')
