#!/usr/bin/python3

"""Reduce IoIO sodium nebula observations"""

import re
import os
import argparse

import numpy as np

import matplotlib.pyplot as plt

from astropy import log
import astropy.units as u
from astropy.table import QTable, vstack

import ccdproc as ccdp

from bigmultipipe import cached_pout

from ccdmultipipe import ccd_meta_to_bmp_meta, as_single

from IoIO.utils import (get_dirs_dates, reduced_dir, dict_to_ccd_meta,
                        multi_glob, sum_ccddata, cached_csv,
                        savefig_overwrite)
from IoIO.cor_process import standardize_filt_name
from IoIO.cormultipipe import (IoIO_ROOT, RAW_DATA_ROOT,
                               calc_obj_to_ND, #crop_ccd,
                               planet_to_object,
                               objctradec_to_obj_center,
                               nd_filter_mask)
from IoIO.calibration import Calibration, CalArgparseHandler
from IoIO.photometry import (SOLVE_TIMEOUT, JOIN_TOLERANCE,
                             JOIN_TOLERANCE_UNIT, rot_to)
from IoIO.cor_photometry import (CorPhotometry,
                                 CorPhotometryArgparseMixin,
                                 mask_galsats)
from IoIO.on_off_pipeline import (TORUS_NA_NEB_GLOB_LIST,
                                  TORUS_NA_NEB_GLOB_EXCLUDE_LIST,
                                  on_off_pipeline)
from IoIO.standard_star import (StandardStar, SSArgparseHandler,
                                extinction_correct, rayleigh_convert)
from IoIO.horizons import galsat_ephemeris
from IoIO.na_back import sun_angle, NaBack, na_meso_sub
from IoIO.torus import (pixel_per_Rj, plot_planet_subim,
                        closest_galsat_to_jupiter)

NA_NEBULA_ROOT = os.path.join(IoIO_ROOT, 'Na_nebula')

def na_apertures(ccd_in, bmp_meta=None, **kwargs):
    # We don't yet rotate the ND filter when we rot_to, so mask it first
    ccd = nd_filter_mask(ccd_in)
    ccd = rot_to(ccd, rot_angle_from_key=['Jupiter_NPole_ang',
                                          'IPT_NPole_ang'])
    # We don't rotate obj_center either, so just put that back with
    # the WCS
    ccd = objctradec_to_obj_center(ccd)
    ccd = mask_galsats(ccd)
    center = ccd.obj_center*u.pixel
    pix_per_Rj = pixel_per_Rj(ccd)
    # --> These may need to be tweaked
    ap_sequence = np.asarray((1, 2, 4, 8, 16, 32, 64, 128)) * u.R_jup
    na_aps = {}
    for ap in ap_sequence:
        b = np.round(center[0] - ap/2 * pix_per_Rj).astype(int)
        t = np.round(center[0] + ap/2 * pix_per_Rj).astype(int)
        # Don't wrap
        b = np.max((b.value, 0))
        t = np.min((t.value, ccd.shape[0]))
        subim = ccd[b:t, :]
        ap_key = f'Na_ap_{ap.value:.0f}_Rj'
        ap_sum, ap_area = sum_ccddata(subim)
        na_aps[f'{ap_key}_sum'] = ap_sum
        na_aps[f'{ap_key}_area'] = ap_area
    # We only  want to mess with our original CCD metadata
    bmp_meta = bmp_meta or {}
    bmp_meta['tavg'] = ccd.tavg
    ccd = ccd_meta_to_bmp_meta(ccd_in, bmp_meta=bmp_meta,
                               ccd_meta_to_bmp_meta_keys=
                               [('Jupiter_PDObsLon', u.deg),
                                ('Jupiter_PDObsLat', u.deg),
                                ('Jupiter_PDSunLon', u.deg)])
    # --> HACK ALERT Should probably have done this in
    # --> na_back to get units in rayleighs at that point
    if ccd.meta.get('MESO'):
        rc = ccd.meta['RAYLEIGH_CONVERSION']
        rc_err = ccd.meta['RAYLEIGH_CONVERSION_ERR']
        meso = ccd.meta['MESO']
        meso_err = ccd.meta['MESO_ERR']
        meso_r = rc * meso
        meso_r_err = meso_r * (rc_err**2 / rc**2 + meso_err**2/meso**2)**0.5
        bmp_meta['MESO'] = meso_r * u.R
        bmp_meta['MESO_ERR'] = meso_r_err * u.R
        
    ccd = dict_to_ccd_meta(ccd, na_aps)
    bmp_meta.update(na_aps)
    return ccd

def na_nebula_plot(t, outdir):
    sum_regexp = re.compile('Na_ap_.*_sum')
    area_regexp = re.compile('Na_ap_.*_area')
    sum_colnames = filter(sum_regexp.match, t.colnames)
    area_colnames = filter(area_regexp.match, t.colnames)

    f = plt.figure(figsize=[11, 8.5])
    date, _ = t['tavg'][0].fits.split('T')
    last_sum = 0
    last_area = 0
    last_ap_bound = 0
    plt.suptitle('Na nebula rectangular aperture surface brightesses')
    ax = plt.subplot()
    for sum_colname, area_colname in zip(sum_colnames, area_colnames):
        cts = t[sum_colname] - last_sum
        area = t[area_colname] - last_area
        sb = cts / area
        ap_bound = sum_colname.split('_')
        ap_bound = int(ap_bound[2])
        av_ap = np.mean((ap_bound, last_ap_bound))
        last_ap_bound = ap_bound
        plt.plot(t['tavg'].datetime, sb, '.', label=f'{av_ap:.0f} Rj')
        #plt.plot(t['tavg'].datetime, cts, '.', label=sum_colname)
        #plt.plot(t['tavg'].datetime, area, '.', label=area_colname)
        last_sum = t[sum_colname]
        last_area = t[area_colname]

    plt.errorbar(t['tavg'].datetime,
                 t['MESO'].value,
                 t['MESO_ERR'].value, fmt='k.', label='Model mesosphere')
    ax.set_ylim(1, 1000)
    #plt.yscale('log')
    plt.ylabel(f'Surf. Bright ({sb.unit})')
    f.autofmt_xdate()
    plt.xlabel(f'UT {date}')
    plt.legend()
    plt.tight_layout()

    savefig_overwrite(os.path.join(outdir, 'Na_nebula_apertures.png'))
    plt.close()


def na_nebula_directory(directory,
                        glob_include=TORUS_NA_NEB_GLOB_LIST,
                        glob_exclude_list=TORUS_NA_NEB_GLOB_EXCLUDE_LIST,
                        outdir=None,
                        outdir_root=NA_NEBULA_ROOT,
                        standard_star_obj=None,
                        na_meso_obj=None,
                        read_pout=True,
                        write_pout=True,
                        create_outdir=True,
                        **kwargs):
     
    flist = multi_glob(directory, glob_include, glob_exclude_list)    
    if len(flist) == 0:
        return []
    outdir = outdir or reduced_dir(directory, outdir_root, create=False)
    poutname = os.path.join(outdir, 'Na_nebula.pout')
    collection = ccdp.ImageFileCollection(directory, filenames=flist)
    # Get rid of off-Jupiter pointings, since they match our
    # glob_include file regexp
    st = collection.summary
    if 'raoff' in st.colnames:
        pointed_at_jupiter = st['raoff'].mask
    else:
        pointed_at_jupiter = np.full(len(flist), True)
    if 'decoff' in st.colnames:
        pointed_at_jupiter = np.logical_and(pointed_at_jupiter,
                                            st['decoff'].mask)
    if np.all(~pointed_at_jupiter):
        return []
    if np.any(~pointed_at_jupiter):
        fbases = st['file'][pointed_at_jupiter]
        flist = [os.path.join(directory, f) for f in fbases]
        collection = ccdp.ImageFileCollection(directory, filenames=flist)
    standard_star_obj = standard_star_obj or StandardStar(reduce=True)
    na_meso_obj = na_meso_obj or NaBack(standard_star_obj=standard_star_obj,
                                        reduce=True)
    pout = cached_pout(on_off_pipeline,
                       poutname=poutname,
                       read_pout=read_pout,
                       write_pout=write_pout,
                       create_outdir=create_outdir,
                       directory=directory,
                       collection=collection,
                       band='Na',
                       standard_star_obj=standard_star_obj,
                       na_meso_obj=na_meso_obj,
                       add_ephemeris=galsat_ephemeris,
                       planet='Jupiter',
                       post_process_list=[calc_obj_to_ND, planet_to_object],
                       plot_planet_rot_from_key=['Jupiter_NPole_ang'],
                       planet_subim_figsize=[6, 4],
                       post_offsub=[sun_angle, na_meso_sub,
                                    extinction_correct, rayleigh_convert,
                                    na_apertures, closest_galsat_to_jupiter,
                                    plot_planet_subim, as_single],
                       outdir_root=outdir_root,
                       **kwargs)
    if pout is None or len(pout) == 0:
        return QTable()

    _ , pipe_meta = zip(*pout)
    t = QTable(rows=pipe_meta)
    na_nebula_plot(t, outdir)
    return t

def na_nebula_tree(raw_data_root=RAW_DATA_ROOT,
                   start=None,
                   stop=None,
                   calibration=None,
                   photometry=None,
                   standard_star_obj=None,
                   na_meso_obj=None,
                   solve_timeout=SOLVE_TIMEOUT,
                   join_tolerance=JOIN_TOLERANCE*JOIN_TOLERANCE_UNIT,
                   read_csvs=True,
                   write_csvs=True,
                   show=False,
                   create_outdir=True,                       
                   outdir_root=NA_NEBULA_ROOT,
                   **kwargs):

    dirs_dates = get_dirs_dates(raw_data_root, start=start, stop=stop)
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
    standard_star_obj = standard_star_obj or StandardStar(reduce=True)
    na_meso_obj = na_meso_obj or NaBack(calibration=calibration,
                                        standard_star_obj=standard_star_obj,
                                        reduce=True)

    summary_table = QTable()
    for directory in dirs:
        rd = reduced_dir(directory, outdir_root, create=False)
        t = cached_csv(na_nebula_directory,
                       csvnames=os.path.join(rd, 'Na_nebula.ecsv'),
                       directory=directory,
                       standard_star_obj=standard_star_obj,
                       read_csvs=read_csvs,
                       write_csvs=write_csvs,
                       outdir=rd,
                       create_outdir=create_outdir,
                       **kwargs)
        # Hack to get around astropy vstack bug with location
        if len(t) == 0:
            continue
        loc = t['tavg'][0].location.copy()
        t['tavg'].location = None
        summary_table = vstack([summary_table, t])
    if len(summary_table) == 0:
        return summary_table

    summary_table['tavg'].location = loc
    na_nebula_plot(summary_table, outdir_root)

    return summary_table

class NaNebulaArgparseHandler(SSArgparseHandler,
                              CorPhotometryArgparseMixin, CalArgparseHandler):
    def add_all(self):
        """Add options used in cmd"""
        self.add_reduced_root(default=NA_NEBULA_ROOT)
        self.add_start()
        self.add_stop()
        self.add_show()
        self.add_read_pout(default=True)
        self.add_write_pout(default=True)        
        self.add_read_csvs(default=True)
        self.add_write_csvs(default=True)
        self.add_solve_timeout()
        self.add_join_tolerance()
        self.add_join_tolerance_unit()
        self.add_keep_intermediate()
        super().add_all()

    def cmd(self, args):
        c, ss = super().cmd(args)
        # Eventually, I am going to want to have this be a proper
        # command-line thing
        na_meso_obj = NaBack(calibration=c,
                             standard_star_obj=ss,
                             reduce=True)
        t = na_nebula_tree(raw_data_root=args.raw_data_root,
                           start=args.start,
                           stop=args.stop,
                           calibration=c,
                           keep_intermediate=args.keep_intermediate,
                           solve_timeout=args.solve_timeout,
                           join_tolerance=(
                               args.join_tolerance*args.join_tolerance_unit),
                           standard_star_obj=ss,
                           na_meso_obj=na_meso_obj,
                           read_pout=args.read_pout,
                           write_pout=args.write_pout,
                           read_csvs=args.read_csvs,
                           write_csvs=args.write_csvs,
                           show=args.show,
                           create_outdir=args.create_outdir,
                           outdir_root=args.reduced_root,
                           num_processes=args.num_processes,
                           mem_frac=args.mem_frac,
                           fits_fixed_ignore=args.fits_fixed_ignore)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run Na nebula reduction')
    aph = NaNebulaArgparseHandler(parser)
    aph.add_all()
    args = parser.parse_args()
    aph.cmd(args)


#log.setLevel('DEBUG')
#
##directory = '/data/IoIO/raw/20210607/'
##directory = '/data/IoIO/raw/20211017/'
#
##directory = '/data/IoIO/raw/2017-05-02'
#directory = '/data/IoIO/raw/2018-05-08/'
#
#t = na_nebula_tree(start='2018-05-08',
#                   stop='2018-05-10',
#                   read_csvs=True,
#                   write_csvs=True,
#                   read_pout=True,
#                   write_pout=True,
#                   fits_fixed_ignore=True)
                   
#calibration=None
#photometry=None
#standard_star_obj=None
#na_meso_obj=None
#solve_timeout=SOLVE_TIMEOUT
#join_tolerance=JOIN_TOLERANCE*JOIN_TOLERANCE_UNIT
#
#outdir_root=NA_NEBULA_ROOT
#fits_fixed_ignore=True
#photometry = (
#    photometry
#    or CorPhotometry(precalc=True,
#                     solve_timeout=solve_timeout,
#                     join_tolerance=join_tolerance))
#calibration = calibration or Calibration(reduce=True)
#standard_star_obj = standard_star_obj or StandardStar(reduce=True)
#na_meso_obj = na_meso_obj or NaBack(calibration=calibration,
#                                    standard_star_obj=standard_star_obj,
#                                    reduce=True)
#outdir_root = outdir_root or os.path.join(IoIO_ROOT, 'Na_nebula')
#t = na_nebula_directory(directory,
#                           calibration=calibration,
#                           photometry=photometry,
#                           standard_star_obj=standard_star_obj,
#                           na_meso_obj=na_meso_obj,                           
#                           solve_timeout=solve_timeout,
#                           join_tolerance=join_tolerance,
#                           outdir_root=outdir_root,
#                           fits_fixed_ignore=fits_fixed_ignore,
#                           read_pout=True,
#                           write_pout=True)


#if pout is None or len(pout) == 0:
#    #return QTable()
#    t = QTable()

#_ , pipe_meta = zip(*pout)
#t = QTable(rows=pipe_meta)

#f = plt.figure(figsize=[11, 8.5])
#date, _ = t['tavg'][0].fits.split('T')
#plt.suptitle('Na nebula aperture surface brightesses')
#
##ax = plt.subplot()
##sb = t['Na_ap_1_Rj_sum'] / t['Na_ap_1_Rj_area']
##plt.plot(t['tavg'].datetime, sb, 'k.', label='1_Rj')
##ax.set_ylim(0, 800)
##plt.show()
#
#cts = t['Na_ap_2_Rj_sum']  - t['Na_ap_1_Rj_sum']
#ax = plt.subplot()
#sb = t['Na_ap_1_Rj_sum'] / t['Na_ap_1_Rj_area']
#plt.plot(t['tavg'].datetime, sb, 'k.', label='1_Rj')
#area = t['Na_ap_2_Rj_area']  - t['Na_ap_1_Rj_area']
#sb = cts / area
#plt.plot(t['tavg'].datetime, sb, 'r.', label='2_Rj')
#ax.set_ylim(0, 800)
#plt.show()

#sum_regexp = re.compile('Na_ap_.*_sum')
#area_regexp = re.compile('Na_ap_.*_area')
#sum_colnames = filter(sum_regexp.match, t.colnames)
#area_colnames = filter(area_regexp.match, t.colnames)
#
#f = plt.figure(figsize=[11, 8.5])
#date, _ = t['tavg'][0].fits.split('T')
#last_sum = 0
#last_area = 0
#last_ap_bound = 0
#plt.suptitle('Na nebula rectangular aperture surface brightesses')
#ax = plt.subplot()
#for sum_colname, area_colname in zip(sum_colnames, area_colnames):
#    cts = t[sum_colname] - last_sum
#    area = t[area_colname] - last_area
#    sb = cts / area
#    ap_bound = sum_colname.split('_')
#    ap_bound = int(ap_bound[2])
#    av_ap = np.mean((ap_bound, last_ap_bound))
#    last_ap_bound = ap_bound
#    plt.plot(t['tavg'].datetime, sb, '.', label=f'{av_ap:.0f} Rj')
#    #plt.plot(t['tavg'].datetime, cts, '.', label=sum_colname)
#    #plt.plot(t['tavg'].datetime, area, '.', label=area_colname)
#    last_sum = t[sum_colname]
#    last_area = t[area_colname]
#
#plt.errorbar(t['tavg'].datetime,
#             t['MESO'].value,
#             t['MESO_ERR'].value, fmt='k.', label='Model mesosphere')
#ax.set_ylim(1, 1000)
##plt.yscale('log')
#plt.ylabel(f'Surf. Bright ({t["MESO"].unit})')
#f.autofmt_xdate()
#plt.xlabel(f'UT {date}')
#plt.legend()
#plt.tight_layout()
#
#plt.show()

#from IoIO.simple_show import simple_show
#from IoIO.cordata import CorData
#ccd = CorData.read('/data/IoIO/Na_nebula/2018-05-08/Na_on-band_002-back-sub.fits')
#ccd.obj_center = (ccd.meta['OBJ_CR1'], ccd.meta['OBJ_CR0'])
#ccd = na_apertures(ccd)

