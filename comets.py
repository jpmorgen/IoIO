#!/usr/bin/python3

import os

import numpy as np

import matplotlib.pyplot as plt

from astropy import log
from astropy import units as u
from astropy.table import QTable, vstack

from photutils.aperture import CircularAperture

from ccdproc import ImageFileCollection

from bigmultipipe import cached_pout, prune_pout

from ccdmultipipe import as_single

from IoIO.ioio_globals import IoIO_ROOT, RAW_DATA_ROOT
from IoIO.utils import (reduced_dir, get_dirs_dates, multi_glob,
                        dict_to_ccd_meta, cached_csv)
from IoIO.cormultipipe import (CorMultiPipeBase, tavg_to_bmp_meta,
                               mask_nonlin_sat, nd_filter_mask,
                               objctradec_to_obj_center, detflux)
from IoIO.calibration import Calibration
from IoIO.cor_photometry import (CorPhotometry, add_astrometry,
                                 write_photometry)
from IoIO.standard_star import (StandardStar, SSArgparseHandler,
                                extinction_correct, add_zeropoint)
from IoIO.horizons import comet_ephemeris

BASE = 'Comets'
PSISCOPE_ROOT = '/data/PSIScope'
OUTDIR_ROOT = os.path.join(IoIO_ROOT, BASE)
COMET_ROOT = os.path.join(IoIO_ROOT, 'Comets')
# https://www.minorplanetcenter.net/iau/info/PackedDes.html
MPC_GLOB_LIST = ['[CPDX][A-Z][0-9][0-9][A-X][0-9|A-Z][0-9][[0|a-z]*',
                 '[0-9][0-9][0-9][0-9]P-*']
ODD_NAME_ASSOC_LIST = [('CK20F030', 'NEOWISE')]

NEOWISE_GLOB_LIST = ['CK20F030*', 'NEOWISE*']
RAW_COMET_GLOB_LIST = MPC_GLOB_LIST + NEOWISE_GLOB_LIST

SHAPE_MODEL_COMETS = ['0081P', '0103P', '0009P', '0067P']
#COMET_PHOT_BOX = 5
MIN_SEEING = 2.5 # Radius in pixels
# https://photutils.readthedocs.io/en/latest/api/photutils.aperture.CircularAperture.html#photutils.aperture.CircularAperture.do_photometry
SUBPIXELS = 5 # aperture photometry subpixels in subpixel msthod

def comets_in_dir(directory,
                  #glob_include=MPC_GLOB_LIST,
                  glob_include=RAW_COMET_GLOB_LIST,
                  odd_name_assoc_list=ODD_NAME_ASSOC_LIST,
                  glob_exclude_list=None,
                  **kwargs):
    """Returns tuple of:
    . Comet raw name(s) from filenames
    . MPC designations only, doing translations for odd comet observations
    . Total number of observations)
    """
    fnames = multi_glob(directory, glob_include, glob_exclude_list)
    comet_raw_list = []
    comet_mpc_list = []
    for f in fnames:
        # Example raw filenames:
        # /data/IoIO/raw/20190616/0029P-S001-R001-C001-Na_off_dupe-4.fts
        # /data/IoIO/raw/2020-07-15/NEOWISE-0001_Na-on.fit
        bname = os.path.basename(f)
        sb = bname.split('-')
        comet_raw = sb[0]
        comet_raw_list.append(comet_raw)
        for mpc_name, odd_name in odd_name_assoc_list:
            if comet_raw == odd_name:
                comet_mpc_list.append(mpc_name)
            else:
                comet_mpc_list.append(comet_raw)
    ucomet_raw = list(set(comet_raw_list))
    ucomet_mpc = list(set(comet_mpc_list))
    return (ucomet_raw, ucomet_mpc, len(comet_raw_list))

def comet_obs(raw_data_root=RAW_DATA_ROOT,
              include_PSIScope=False,
              start=None,
              stop=None,
              **kwargs):
    """Returns a tuple: (comet_date_list, n_total_obs) characterizing
    the comet observations in a directory tree.  Comet_date_list is a
    list of tuples (comet, date), where comet is the MPC designation
    of a comet observed on a particular date.  n_total_obs is the
    total number of observations of all comets

    """
    dirs_dates = get_dirs_dates(raw_data_root, start=start, stop=stop)
    if include_PSIScope:
        psiscope_dirs_dates = get_dirs_dates(os.path.join(PSISCOPE_ROOT, 'raw'))
        dirs_dates.extend(psiscope_dirs_dates)
    dirs, dates = zip(*dirs_dates)
    if len(dirs) == 0:
        log.warning('No directories found')
        return []
    mpc_list = []
    n_total_obs = 0
    n_nights = 0
    comet_date_list = []
    for d, date in dirs_dates:
        (rawnames, mpcs, n_obs) = comets_in_dir(d, **kwargs)
        n_total_obs += n_obs
        if len(mpcs) > 0:
            n_nights += 1
            mpc_list.extend(mpcs)
        for comet in mpcs:
            comet_date_list.append((comet, date))
    return (comet_date_list, n_total_obs)


class CometMultiPipe(CorMultiPipeBase):
    def file_write(self, ccd, outname,
                   in_name=None, photometry=None,
                   **kwargs):
        written_name = super().file_write(
            ccd, outname, photometry=photometry, **kwargs)
        # --> Need to add proper comet photometry , but currently do
        # that with bmp_meta
        write_photometry(in_name=in_name, outname=outname,
                         photometry=photometry,
                         write_wide_source_table=True,
                         **kwargs)
        outroot, _ = os.path.splitext(outname)
        try:
            photometry.plot_object(outname=outroot + '.png')
        except Exception as e:
            log.warning(f'Not able to plot object for {outname}: {e}')
        return written_name

def comet_phot(ccd_in,
               bmp_meta=None,
               #comet_phot_box=COMET_PHOT_BOX,
               comet_phot_rad=MIN_SEEING,
               comet_phot_subpixels=SUBPIXELS,
               photometry=None,
               **kwargs):
    """cormulipipe post-processing routine to add up counts at obj_center"""
    # The right place to do this is in [cor_]photometry and with a
    # variety of apertures and the Photometry.background for EACH file
    if photometry.wide_source_table is None:
        bmp_meta.clear()
        return None
    ccd = ccd_in.copy()
    mask = (photometry.wide_source_table['OBJECT']
            == ccd.meta['OBJECT'])
    r = photometry.wide_source_table[mask]
    keys = r.colnames
    values = [c for c in r[0]]
    comet_dict = dict(zip(keys, values))
    
    #log.debug(f'comet_phot obj_center: {ccd.obj_center}')
    aper = CircularAperture(ccd.obj_center[::-1], comet_phot_rad)
    phots = []
    # This is OK for one aperture, but if I want many, I'll need to
    # put more columns, like I do in utils.ColnameEncoder
    for method in ['exact', 'center', 'subpixel']:
        apsum, apsum_err= aper.do_photometry(
            ccd.data, ccd.uncertainty.array, ccd.mask,
            method, subpixels=comet_phot_subpixels)
        comet_dict.update(
            {f'comet_phot_{method}': apsum[0]*ccd.unit,
             f'comet_phot_{method}_err': apsum_err[0]*ccd.unit})
    # SkyCoord don't go nicely into FITS headers, which is fine, since
    # we have the information there in other forms.  But neither does
    # it Vstack well right now 
    clean_comet_dict = comet_dict.copy()
    del clean_comet_dict['coord']
    ccd = dict_to_ccd_meta(ccd, clean_comet_dict)
    bmp_meta.update(comet_dict)
    return ccd

def comet_collection(directory,
                     glob_include=MPC_GLOB_LIST,
                     glob_exclude_list=None,
                     **kwargs):
    flist = multi_glob(directory,
                       glob_list=glob_include,
                       glob_exclude_list=glob_exclude_list)
    if len(flist) == 0:
        return ImageFileCollection(directory, glob_exclude='*')
    return ImageFileCollection(directory, filenames=flist)
    
def comet_pipeline(collection=None,
                   calibration=None,
                   photometry=None,
                   standard_star_obj=None,
                   num_processes=None,
                   outdir=None,
                   create_outdir=True,
                   fits_fixed_ignore=True,
                   **kwargs):

    if len(collection.files) == 0:
        return
    calibration = calibration or Calibration(reduce=True)
    photometry = photometry or CorPhotometry(precalc=True, **kwargs)
    standard_star_obj = standard_star_obj or StandardStar(reduce=True)

    # --> Change the order of the pipeline to do add_astrometry later
    # --> so that I can have DOBJ in keys_to_source_table (which will
    # --> need to be put in CorPhotometry at instantiation time)
    cmp = CometMultiPipe(
        auto=True,
        calibration=calibration,
        photometry=photometry,
        standard_star_obj=standard_star_obj,
        create_outdir=create_outdir,
        #fail_if_no_wcs=False, # This hasn't been fully implemented
        post_process_list=[tavg_to_bmp_meta,
                           mask_nonlin_sat,
                           nd_filter_mask,
                           detflux, # get obj plot to read in right units
                           add_astrometry,
                           comet_ephemeris,
                           objctradec_to_obj_center,
                           extinction_correct,
                           add_zeropoint,
                           comet_phot,
                           as_single],
        fits_fixed_ignore=fits_fixed_ignore, 
        num_processes=num_processes,
        **kwargs)
    #pout = cmp.pipeline([collection.files[0]], outdir=outdir, overwrite=True)
    pout = cmp.pipeline(collection.files, outdir=outdir, overwrite=True)
    pout, _ = prune_pout(pout, collection.files)
    return pout

# This is intended to reduce one comet in one directory and put the
# result in outdir, which is of the format OUTDIR_ROOT/<MPC designation>/<date>
def comet_directory(collection,
                    outdir=None,
                    outdir_root=OUTDIR_ROOT,
                    standard_star_obj=None,
                    read_pout=True,
                    write_pout=True,
                    write_plot=True,
                    create_outdir=True,
                    show=False,
                    **kwargs):

    if len(collection.files) == 0:
        return QTable()

    poutname = os.path.join(outdir, BASE + '.pout')
    pout = cached_pout(comet_pipeline,
                       poutname=poutname,
                       read_pout=read_pout,
                       write_pout=write_pout,
                       collection=collection,
                       outdir=outdir,
                       create_outdir=create_outdir,
                       **kwargs)
    if pout is None or len(pout) == 0:
        return QTable()

    _ , pipe_meta = zip(*pout)
    t = QTable(rows=pipe_meta)
    return t

# This is shaping up to be comet_tree
def comet_tree(raw_data_root=RAW_DATA_ROOT,
               outdir_root=OUTDIR_ROOT,
               start=None,
               stop=None,
               mpc_list=['CK20F030'],
               odd_name_assoc_list=ODD_NAME_ASSOC_LIST,
               calibration=None,
               photometry=None,
               read_csvs=True,
               write_csvs=True,
               create_outdir=True,
               **kwargs):

    dirs_dates = get_dirs_dates(raw_data_root, start=start, stop=stop)
    dirs, _ = zip(*dirs_dates)
    if len(dirs) == 0:
        log.warning('No directories found')
        return

    # I don't have a pipeline for PSIScope yet.
    # comets refers to the MPC designation
    raw_comets_dates, _ = comet_obs(include_PSIScope=False)
    comets, dates = zip(*raw_comets_dates)
    uniq_comets = sorted(set(comets))
    ps = os.path.sep
    if mpc_list is None:
        mpc_list = uniq_comets
    for comet in mpc_list:
        # convert back to odd comet name so we capture all of the
        # filenames
        odd_name = [assoc[1] for assoc in odd_name_assoc_list
                    if assoc[0] == comet]
        rawglob = [comet]
        rawglob.extend(odd_name)
        rawglob = [f'{c}*' for c in rawglob]
        summary_table = QTable()
        for directory in dirs:
            (rawname, mpc, _) = comets_in_dir(
                directory, glob_include=rawglob, **kwargs)
            if len(rawname) == 0:
                continue
            rd = reduced_dir(directory,
                             outdir_root + ps + comet,
                             create=False)
            c = comet_collection(directory, glob_include=rawglob)
            comet_csv = os.path.join(rd, f'{BASE}.ecsv')
            t = cached_csv(c,
                           code=comet_directory,
                           csvnames=comet_csv,
                           read_csvs=read_csvs,
                           write_csvs=write_csvs,
                           outdir=rd,
                           create_outdir=create_outdir,
                           calibration=calibration,
                           photometry=photometry,
                           **kwargs)
            if len(t) == 0:
                continue
            # Hack to get around astropy vstack bug with location
            loc = t['tavg'][0].location.copy()
            t['tavg'].location = None
            # Delete coord for now, since vstack is not happy with that
            del t['coord']            
            summary_table = vstack([summary_table, t])
        if len(summary_table) > 0:
            summary_table['tavg'].location = loc
            summary_table.sort('tavg')
            comet_dir, _ = os.path.split(rd)
            summary_table.write(os.path.join(comet_dir, comet + '.ecsv'),
                                overwrite=True)
    # Not making sense to me to concatenate

def plot_comet_obsdates(include_PSIScope=False):
    (raw_comets_dates, n_obs) = comet_obs(include_PSIScope=include_PSIScope)
    comets, dates = zip(*raw_comets_dates)
    uniq_comets = sorted(set(comets))
    uniq_nights = list(set(dates))
    n_comets = len(uniq_comets)
    n_nights = len(uniq_nights)

    print(f'{n_comets} unique comets recorded on {n_nights} nights')
    #print(f'{len(dates)} total observations?')
    print(f'{n_obs} total observations')

    print(uniq_comets)

    #ucomet_idx_list = []
    #for comet in comets:
    #    for ucomet_idx, ucomet in enumerate(uniq_comets):
    #        if comet == ucomet:
    #            ucomet_idx_list.append(ucomet_idx)
    #            break
    #fig = plt.figure(figsize=[8.5, 11/2])
    #plt.plot(dates, ucomet_idx_list, 'k*')
    #fig.autofmt_xdate()
    #plt.show()
    #
    pcomet_idx_list = []
    pcomet_dates_list = []
    lpcomet_idx_list = []
    lpcomet_dates_list = []
    for comet_idx, comet in enumerate(comets):
        for ucomet_idx, ucomet in enumerate(uniq_comets):
            if comet == ucomet:
                if 'CK' in comet:
                    lpcomet_idx_list.append(ucomet_idx)
                    lpcomet_dates_list.append(dates[comet_idx])
                    break
                else:
                    pcomet_idx_list.append(ucomet_idx)
                    pcomet_dates_list.append(dates[comet_idx])
                    break


    #fig = plt.figure(figsize=[8.5, 11/2])
    #fig = plt.figure(figsize=[8.5, 10], tight_layout=True)
    fig = plt.figure(figsize=[8.5, 4], tight_layout=True)
    plt.plot(pcomet_dates_list, pcomet_idx_list, 'r*', label='Periodic Comets')
    plt.plot(lpcomet_dates_list, lpcomet_idx_list, 'k*', label='Long-Period Comets')
    #plt.yticks([])
    #plt.yticks(range(n_comets), uniq_comets)
    uniq_comets_shape_model = [c if c in SHAPE_MODEL_COMETS
                               else ''
                               for c in uniq_comets]
    plt.yticks(range(n_comets), uniq_comets_shape_model)
    #plt.ylabel('Comet (MPC designation)')
    plt.xlabel('Date')
    plt.legend()
    fig.autofmt_xdate()
    plt.show()


log.setLevel('DEBUG')


#t = comet_tree(mpc_list=['CK20F030'])
# --> Need to pick apart ValueError of Ambiguous target name
#t = comet_tree(mpc_list=['0029P'], start='2025-02-01')

#t = comet_tree(mpc_list=['CK23A030'])
#t = comet_tree(mpc_list=['CK23A030'], start='2024-04-30', stop='2024-04-30')
#t = comet_tree(mpc_list=['CK23A030'], start='2024-04-28', stop='2024-05-01')
#t = comet_tree(mpc_list=['CK23A030'], start='2024-01-11', stop='2024-01-11')

# comet_collection('/data/IoIO/raw/20200806/',
#                      glob_include=NEOWISE_GLOB_LIST)
# t = comet_directory(c, outdir='/tmp/NEOWISE')




plot_comet_obsdates(include_PSIScope=False)
#print(comets_in_dir('/data/IoIO/raw/20200806/'))
#print(comets_in_dir('/data/IoIO/raw/2020-07-28'))

#from cordata import CorData
#from cor_process import cor_process

#ccd = CorData.read('/data/IoIO/raw/2020-07-28/NEOWISE-0003_Na_off.fit')
#calibration = Calibration(reduce=True)


#ccd_out = cor_process(ccd, calibration=calibration,
#                      oscan=True,
#                      gain=True,
#                      error=True,
#                      min_value=True)
## That works

#ccd_out = cor_process(ccd, calibration=calibration, auto=True)
## That works

#c = comet_collection('/data/IoIO/raw/2020-07-28/',
#c = comet_collection('/data/IoIO/raw/20200806/',
#                     glob_include=NEOWISE_GLOB_LIST)

# /data/IoIO/raw/20200819/ seems to be last one

#comet_pipeline(c)


#comet_pipeline('/data/IoIO/raw/2020-07-28/')

#comet_pipeline('/data/IoIO/raw/20211017')
#comet_pipeline('/data/IoIO/raw/20211028')

#c = comet_collection('/data/IoIO/raw/20211028')
#c = comet_collection('/data/IoIO/raw/20220112/')


##raw_comet_dates
#
#sorted_comets_dates = sorted(raw_comets_dates,
#                             key=lambda comet_date: comet_date[0])
#comets, dates = zip(*sorted_comets_dates)
#
#for comet in uniq_comets[0:2]:
#    idx = comets.index(comet)
#    print(idx)


#plt.plot(dates[0], comets[0])

# cl = comet_obs(include_PSIScope=False)
# print(f'Number of IoIO comets = {len(cl[0])} Total nights {cl[1]}')
# cl = comet_obs()
# print('Including PSIScope')
# print(len(cl[0]), cl[1])

#pout = comet_pipeline('/data/IoIO/raw/20211028',
#                      fits_fixed_ignore=True)
