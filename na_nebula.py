#!/usr/bin/python3

"""Reduce IoIO sodium nebula observations"""

import re
import os
import argparse
import datetime
from cycler import cycler

import numpy as np

from scipy.signal import medfilt

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from astropy import log
import astropy.units as u
from astropy.time import Time
from astropy.table import QTable, unique
from astropy.convolution import Box1DKernel

from ccdproc import ImageFileCollection

from bigmultipipe import cached_pout

from ccdmultipipe import ccd_meta_to_bmp_meta, as_single

from IoIO.utils import (ColnameEncoder, get_dirs_dates, reduced_dir,
                        valid_long_exposure, dict_to_ccd_meta,
                        multi_glob, sum_ccddata, csvname_creator,
                        daily_biweight, daily_convolve,
                        savefig_overwrite, finish_stripchart)
from IoIO.cormultipipe import (IoIO_ROOT, RAW_DATA_ROOT,
                               MAX_NUM_PROCESSES, MAX_CCDDATA_BITPIX,
                               MAX_MEM_FRAC,
                               COR_PROCESS_EXPAND_FACTOR,
                               calc_obj_to_ND, #crop_ccd,
                               planet_to_object,
                               objctradec_to_obj_center,
                               nd_filter_mask, parallel_cached_csvs,
                               pixel_per_Rj, obj_surface_bright)
from IoIO.calibration import Calibration, CalArgparseHandler
from IoIO.photometry import (SOLVE_TIMEOUT, JOIN_TOLERANCE,
                             JOIN_TOLERANCE_UNIT, rot_to)
from IoIO.cor_photometry import (CorPhotometry,
                                 CorPhotometryArgparseMixin,
                                 mask_galsats)
from IoIO.standard_star import (StandardStar, SSArgparseHandler,
                                extinction_correct, rayleigh_convert)
from IoIO.horizons import galsat_ephemeris
from IoIO.na_meso import (NaMeso, NaMesoArgparseHandler, sun_angles,
                          na_meso_meta)
from IoIO.on_off_pipeline import (TORUS_NA_NEB_GLOB_LIST,
                                  TORUS_NA_NEB_GLOB_EXCLUDE_LIST,
                                  on_off_pipeline)
from IoIO.torus import plot_planet_subim, closest_galsat_to_jupiter

BASE = 'Na_nebula'
OUTDIR_ROOT = os.path.join(IoIO_ROOT, BASE)
BOX_NDAYS = 20
MEDFILT_WIDTH = 81

def na_apertures(ccd_in, bmp_meta=None, **kwargs):
    # We don't yet rotate the ND filter when we rot_to, so mask it first
    ccd = nd_filter_mask(ccd_in)
    ccd = rot_to(ccd, rot_angle_from_key=['Jupiter_NPole_ang',
                                          'IPT_NPole_ang'])
    # We don't rotate obj_center either, so just put that back with
    # the WCS
    ccd = objctradec_to_obj_center(ccd)
    # default of 20 doesn't always get the emission.  This might not
    # be enough either
    ccd = mask_galsats(ccd, galsat_mask_side=30*u.pixel)
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
    if bmp_meta is None:
        bmp_meta = {}
    bmp_meta['tavg'] = ccd.tavg

    ccd = ccd_meta_to_bmp_meta(ccd_in, bmp_meta=bmp_meta,
                               ccd_meta_to_bmp_meta_keys=
                               [('Jupiter_PDObsLon', u.deg),
                                ('Jupiter_PDObsLat', u.deg),
                                ('Jupiter_PDSunLon', u.deg)])
        
    ccd = dict_to_ccd_meta(ccd, na_aps)
    bmp_meta.update(na_aps)
    return ccd

def add_annular_apertures(t, encoder, ap_base='Na_ap', subtract_col=None):
    """Add to table t columns for brightnesses for the regions between
    successively larger apertures (e.g. added column n = surface
    brightness for region extending from aperture n and aperture n+1),
    optionally subtracting subtract_col from all results

    """
    sum_regexp = re.compile(ap_base + '_.*_sum')
    area_regexp = re.compile(ap_base + '_.*_area')
    sum_colnames = filter(sum_regexp.match, t.colnames)
    area_colnames = filter(area_regexp.match, t.colnames)

    last_sum = 0
    last_area = 0
    last_ap_bound = 0
    sb_list = []
    ap_list = []
    for sum_colname, area_colname in zip(sum_colnames, area_colnames):
        ap_bound = sum_colname.split('_')
        ap_bound = int(ap_bound[2])
        av_ap = np.mean((ap_bound, last_ap_bound))*u.R_jup
        last_ap_bound = ap_bound
        cts = t[sum_colname] - last_sum
        area = t[area_colname] - last_area
        sb = cts / area
        last_sum = t[sum_colname]
        last_area = t[area_colname]
        colname = encoder.to_colname(t, av_ap)
        #colname = r_to_from_colname(t,
        #                    av_ap,
        #                    colbase='concentric_ap_sb',
        #                    encode=True)
        if subtract_col is not None:
            sb -= t[subtract_col]
        t[colname] = sb
            
        #t[f'concentric_ap_sb_{av_ap:.1f}'] = sb

def na_nebula_plot(t, outdir,
                   tmin=None,
                   min_sb=-200,
                   max_sb=1000, # 400,
                   max_good_sb=np.inf,
                   min_av_ap_dist=0,
                   max_av_ap_dist=np.inf,
                   n_plots=4,
                   tlim=None,
                   show=False):
    t.sort('tavg')
    tlim = (t['tavg'][0].datetime, t['tavg'][-1].datetime)
    outbase = 'Na_nebula_apertures.png'
    sum_regexp = re.compile('Na_ap_.*_sum')
    area_regexp = re.compile('Na_ap_.*_area')
    sum_colnames = filter(sum_regexp.match, t.colnames)
    area_colnames = filter(area_regexp.match, t.colnames)

    last_sum = 0
    last_area = 0
    last_ap_bound = 0
    sb_list = []
    ap_list = []
    for sum_colname, area_colname in zip(sum_colnames, area_colnames):
        ap_bound = sum_colname.split('_')
        ap_bound = int(ap_bound[2])
        av_ap = np.mean((ap_bound, last_ap_bound))
        last_ap_bound = ap_bound
        cts = t[sum_colname] - last_sum
        area = t[area_colname] - last_area
        sb = cts / area
        if av_ap < min_av_ap_dist or av_ap > max_av_ap_dist:
            continue
        ap_list.append(av_ap)
        sb_list.append(sb)
        last_sum = t[sum_colname]
        last_area = t[area_colname]

    if n_plots == 1:
        f = plt.figure(figsize=[8.5, 11/2])
    else:
        f = plt.figure(figsize=[8.5, 11])

    #custom_cycler = (cycler(color=['b', 'g', 'r', 'c', 'm', 'y', 'k']))
    custom_cycler = cycler('color', ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#17becf', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22'])
    plt.rc('axes', prop_cycle=custom_cycler)
                     
    date, _ = t['tavg'][0].fits.split('T')
    plt.suptitle('Na nebula rectangular aperture surface brightesses above & below torus centrifugal plane')

    ax = plt.subplot(n_plots,1,1)
    plt.title(f'{ap_list[-1]:.0f} Rj aperture subtracted')
    for sb, av_ap in zip(sb_list, ap_list):
        # Skip the most distant ap
        if av_ap == ap_list[-1]:
            continue        
        bsub_sb = sb - sb_list[-1]
        mask = bsub_sb < max_good_sb
        plt.plot(t['tavg'][mask].datetime, bsub_sb[mask], '.',
                 label=f'+/- {av_ap:.0f} Rj')
        med_sb = medfilt(bsub_sb[mask], MEDFILT_WIDTH)
        plt.plot(t['tavg'][mask].datetime, med_sb, '-',
                 label=f'+/- {av_ap:.0f} Rj medfilt')
        
    ax.set_xlim(tlim)
    ax.set_ylim(0, max_sb/2)
    plt.ylabel(f'Na Surf. Bright ({sb.unit})')
    #plt.legend()

    f.autofmt_xdate()
    if n_plots == 1:
        finish_stripchart(outdir, outbase, show=show)
        return
    
    ax = plt.subplot(n_plots,1,2)
    plt.title('Best Guess Telluric Na subtracted')
    for sb, av_ap in zip(sb_list, ap_list):
        bsub_sb = sb-t['meso_or_model']
        plt.plot(t['tavg'].datetime, bsub_sb,
                 '.', label=f'+/- {av_ap:.0f} Rj')
        mask = bsub_sb < max_good_sb
        med_sb = medfilt(bsub_sb[mask], MEDFILT_WIDTH)
        plt.plot(t['tavg'][mask].datetime, med_sb, '-',
                 label=f'+/- {av_ap:.0f} Rj medfilt')
    meso = t['meso_or_model']
    meso_err = t['meso_or_model_err']
    plt.errorbar(t['tavg'].datetime, meso.value, meso_err.value,
                 fmt='.', color=mcolors.to_rgb('grey'), alpha=0.2,
                 label='Best guess Telluric Na')
    ax.set_xlim(tlim)
    ax.set_ylim(min_sb, max_sb)
    #plt.yscale('log')
    plt.ylabel(f'Surf. Bright ({sb.unit})')
    plt.legend()

    f.autofmt_xdate()
    if n_plots == 2:
        finish_stripchart(outdir, outbase, show=show)
        return

    ax = plt.subplot(n_plots,1,3)
    plt.title('Telluric Na empirical model subtracted')
    for sb, av_ap in zip(sb_list, ap_list):
        bsub_sb = sb - t['model_meso']
        plt.plot(t['tavg'].datetime, bsub_sb,
                 '.', label=f'+/- {av_ap:.0f} Rj')
        mask = bsub_sb < max_good_sb
        med_sb = medfilt(bsub_sb[mask], MEDFILT_WIDTH)
        plt.plot(t['tavg'][mask].datetime, med_sb, '-',
                 label=f'+/- {av_ap:.0f} Rj medfilt')
    meso = t['model_meso']
    meso_err = t['model_meso_err']
    plt.errorbar(t['tavg'].datetime, meso.value, meso_err.value,
                 fmt='.', color=mcolors.to_rgb('grey'), alpha=0.2,
                 label='Telluric Na model')
    ax.set_xlim(tlim)
    ax.set_ylim(min_sb, max_sb)
    #plt.yscale('log')
    plt.ylabel(f'Surf. Bright ({sb.unit})')
    #plt.legend()

    f.autofmt_xdate()
    if n_plots == 3:
        finish_stripchart(outdir, outbase, show=show)
        return

    ax = plt.subplot(n_plots,1,4)
    plt.title('Measured Telluric Na subtracted')
    for sb, av_ap in zip(sb_list, ap_list):
        bsub_sb = sb-t['measured_meso']
        plt.plot(t['tavg'].datetime, bsub_sb,
                 '.', label=f'+/- {av_ap:.0f} Rj')
        mask = bsub_sb < max_good_sb
        med_sb = medfilt(bsub_sb[mask], MEDFILT_WIDTH)
        plt.plot(t['tavg'][mask].datetime, med_sb, '-',
                 label=f'+/- {av_ap:.0f} Rj medfilt')
    meso = t['measured_meso']
    meso_err = t['measured_meso_err']
    plt.errorbar(t['tavg'].datetime, meso.value, meso_err.value,
                 fmt='.', color=mcolors.to_rgb('grey'), alpha=0.2,
                 label='Measured Telluric Na')
    ax.set_xlim(tlim)
    ax.set_ylim(min_sb, max_sb)
    #plt.yscale('log')
    plt.ylabel(f'Surf. Bright ({sb.unit})')
    #plt.legend()

    f.autofmt_xdate()
    if n_plots == 4:
        finish_stripchart(outdir, outbase, show=show)
        return

    obj_sb = t['obj_surf_bright']
    obj_sb_err = t['obj_surf_bright_err']
    ax = plt.subplot(n_plots,1,5)
    plt.title('Jupiter attenuated surface brightness')
    plt.errorbar(t['tavg'].datetime, obj_sb.value, obj_sb_err.value,
                 fmt='k.')#, label='Telluric Na odel')
    ax.set_xlim(tlim)
    ax.set_ylim(0, 200000)
    plt.ylabel(f'Atten. Surf. Bright ({obj_sb.unit})')
    
    f.autofmt_xdate()
    if n_plots == 5:
        finish_stripchart(outdir, outbase, show=show)
        return

    ax = plt.subplot(n_plots,1,6)
    plt.plot(t['tavg'].datetime, t['extinction_correction_value'], '.')
    plt.title('Extinction correction factor')
    ax.set_ylim(1, 5)
    plt.ylabel(f'Extinction correction')

    f.autofmt_xdate()
    if n_plots == 6:
        finish_stripchart(outdir, outbase, show=show)
        return

    ax = plt.subplot(n_plots,1,7)
    plt.plot(t['tavg'].datetime, sb_list[-1]/meso, '.',
             label=f'{ap_list[-1]:.0f} Rj')
    plt.axhline(1)
    plt.yscale('log')
    ax.set_xlim(tlim)
    ax.set_ylim(0.03, 30)
    plt.ylabel(f'{av_ap:.0f} Rj / telluric')
    plt.legend()

    f.autofmt_xdate()
    if n_plots == 7:
        finish_stripchart(outdir, outbase, show=show)
        return

    plt.xlabel(f'UT')# {date}')

    finish_stripchart(outdir, outbase, show=show)

def na_nebula_collection(directory,
                         glob_include=TORUS_NA_NEB_GLOB_LIST,
                         glob_exclude_list=TORUS_NA_NEB_GLOB_EXCLUDE_LIST,
                         **kwargs):
    flist = multi_glob(directory, glob_include, glob_exclude_list)
    if len(flist) == 0:
        return ImageFileCollection(directory, glob_exclude='*')
    # Create a collection of valid long Na exposures that are
    # pointed at Jupiter (not offset for mesospheric foreground
    # observations)
    collection = ImageFileCollection(directory, filenames=flist)
    st = collection.summary
    valid = ['Na' in f for f in st['filter']]
    valid = np.logical_and(valid, valid_long_exposure(st))
    if 'raoff' in st.colnames:
        valid = np.logical_and(valid, st['raoff'].mask)
    if 'decoff' in st.colnames:
        valid = np.logical_and(valid, st['decoff'].mask)
    if np.all(~valid):
        return ImageFileCollection(directory, glob_exclude='*')
    if np.any(~valid):
        fbases = st['file'][valid]
        flist = [os.path.join(directory, f) for f in fbases]
    return ImageFileCollection(directory, filenames=flist)

def na_nebula_directory(directory_or_collection,
                        glob_include=TORUS_NA_NEB_GLOB_LIST,
                        glob_exclude_list=TORUS_NA_NEB_GLOB_EXCLUDE_LIST,
                        outdir=None,
                        outdir_root=OUTDIR_ROOT,
                        standard_star_obj=None,
                        na_meso_obj=None,
                        read_pout=True,
                        write_pout=True,
                        create_outdir=True,
                        **kwargs):

    if isinstance(directory_or_collection, ImageFileCollection):
        # We are passed a collection when running multiple directories
        # in parallel
        directory = directory_or_collection.location
        collection = directory_or_collection
    else:
        directory = directory_or_collection
        collection = na_nebula_collection(directory, **kwargs)

    if len(collection.files) == 0:
        return QTable()

    outdir = outdir or reduced_dir(directory, outdir_root, create=False)
    poutname = os.path.join(outdir, BASE + '.pout')
    standard_star_obj = standard_star_obj or StandardStar(reduce=True)
    na_meso_obj = na_meso_obj or NaMeso()
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
                       post_offsub=[sun_angles, na_meso_meta,
                                    extinction_correct, rayleigh_convert,
                                    obj_surface_bright, na_apertures,
                                    closest_galsat_to_jupiter,
                                    plot_planet_subim, as_single],
                       outdir=outdir,
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
                   outdir_root=OUTDIR_ROOT,
                   **kwargs):

    dirs_dates = get_dirs_dates(raw_data_root, start=start, stop=stop)
    dirs, _ = zip(*dirs_dates)
    if len(dirs) == 0:
        log.warning('No directories found')
        return []
    calibration = calibration or Calibration()
    if photometry is None:
        photometry = CorPhotometry(
            precalc=True,
            solve_timeout=solve_timeout,
            join_tolerance=join_tolerance)
    standard_star_obj = standard_star_obj or StandardStar(reduce=True)
    na_meso_obj = na_meso_obj or NaMeso()
    cached_csv_args = {
        'csvnames': csvname_creator,
        'csv_base': BASE + '.ecsv',
        'write_csvs': write_csvs,
        'calibration': calibration,
        'photometry': photometry,
        'standard_star_obj': standard_star_obj,
        'na_meso_obj': na_meso_obj,
        'outdir_root': outdir_root,
        'create_outdir': create_outdir}
    cached_csv_args.update(**kwargs)
    # Not sure why I am not getting a full house with files_per_process=2
    summary_table = parallel_cached_csvs(dirs,
                                         code=na_nebula_directory,
                                         collector=na_nebula_collection,
                                         files_per_process=3,
                                         read_csvs=read_csvs,
                                         **cached_csv_args)
    summary_table.write(os.path.join(outdir_root, BASE + '.ecsv'),
                        overwrite=True)
    ijd = summary_table['tavg'].jd.astype(int)
   
    na_nebula_plot(summary_table, outdir_root, show=show)
    return summary_table

class NaNebulaArgparseHandler(NaMesoArgparseHandler, SSArgparseHandler,
                              CorPhotometryArgparseMixin, CalArgparseHandler):
    def add_all(self):
        """Add options used in cmd"""
        self.add_reduced_root(default=OUTDIR_ROOT)
        self.add_start()
        self.add_stop()
        #self.add_show()
        #self.add_read_pout(default=True)
        #self.add_write_pout(default=True)        
        #self.add_read_csvs(default=True)
        #self.add_write_csvs(default=True)
        #self.add_solve_timeout()
        #self.add_join_tolerance()
        #self.add_join_tolerance_unit()
        #self.add_keep_intermediate()
        super().add_all()

    def cmd(self, args):
        c, ss, m = super().cmd(args)
        t = na_nebula_tree(raw_data_root=args.raw_data_root,
                           start=args.start,
                           stop=args.stop,
                           calibration=c,
                           keep_intermediate=args.keep_intermediate,
                           solve_timeout=args.solve_timeout,
                           join_tolerance=(
                               args.join_tolerance
                               *u.Unit(args.join_tolerance_unit)),
                           standard_star_obj=ss,
                           na_meso_obj=m,
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
#outdir_root=OUTDIR_ROOT
#fits_fixed_ignore=True
#photometry = (
#    photometry
#    or CorPhotometry(precalc=True,
#                     solve_timeout=solve_timeout,
#                     join_tolerance=join_tolerance))
#calibration = calibration or Calibration(reduce=True)
#standard_star_obj = standard_star_obj or StandardStar(reduce=True)
#na_meso_obj = na_meso_obj or NaMeso(calibration=calibration,
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

#na_nebula_tree(read_csvs=False)
summary_table = QTable.read('/data/IoIO/Na_nebula/Na_nebula.ecsv')
#na_nebula_plot(t, '/tmp', max_good_sb=1000*u.R, show=True, n_plots=4)
#na_nebula_plot(t, '/tmp', show=True, n_plots=3, min_av_ap_dist=10, max_sb=600)


# Add annular apertures to summary_table
encoder = ColnameEncoder('annular_sb')
add_annular_apertures(summary_table, encoder, subtract_col='meso_or_model')

# Add daily biweight locations to summary_table
summary_table['ijd'] = summary_table['tavg'].jd.astype(int)
sb_colnames = filter(encoder.col_regexp.match, summary_table.colnames)
for sb_col in sb_colnames:
    av_ap = encoder.from_colname(summary_table, sb_col)
    daily_biweight(summary_table,
                   day_col='ijd',
                   data_col=sb_col,
                   biweight_col='biweight_' + sb_col,
                   std_col='std_' + sb_col)
    print(av_ap)

# Create a new table, one row per day with biweight & mad_std columns
day_table = unique(summary_table, keys='ijd')
bwt_std_colnames = filter(encoder.supplemented_regexp.match, day_table.colnames)
bwt_std_colnames = list(bwt_std_colnames)
day_table_colnames = ['ijd'] + bwt_std_colnames
day_table = QTable(day_table[day_table_colnames])
print(day_table_colnames)

# Boxcar median each biweight column
first_day = np.min(day_table['ijd'])
last_day = np.max(day_table['ijd'])
all_days = np.arange(first_day, last_day+1)
bwt_col_regexp = re.compile('biweight_' + encoder.colbase + '_.*')
bwt_colnames = list(filter(bwt_col_regexp.match, day_table.colnames))
for bwt_col in bwt_colnames:
    day_table = daily_convolve(day_table,
                               'ijd',
                               bwt_col,
                               'boxfilt_' + bwt_col,
                               Box1DKernel(BOX_NDAYS),
                               all_days=all_days)
day_table['itdatetime'] = Time(day_table['ijd'], format='jd').datetime
box_col_regexp = re.compile('boxfilt_biweight_' + encoder.colbase + '_.*')
box_colnames = list(filter(box_col_regexp.match, day_table.colnames))
f = plt.figure()
ax = plt.subplot()
for box_col in box_colnames:
    print(box_col)
    av_ap = encoder.from_colname(day_table, box_col)
    plt.plot(day_table['itdatetime'], day_table[box_col],
             label=f'{av_ap}')
plt.xlabel('date')
plt.ylabel(f'Surf. bright {day_table[box_col].unit}')
plt.legend()
f.autofmt_xdate()
plt.show()

