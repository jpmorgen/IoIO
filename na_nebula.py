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
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator

from astropy import log
import astropy.units as u
from astropy.time import Time
from astropy.table import QTable, unique, vstack
from astropy.convolution import Box1DKernel

from ccdproc import ImageFileCollection

from bigmultipipe import cached_pout

from ccdmultipipe import ccd_meta_to_bmp_meta, as_single

from IoIO.ioio_globals import IoIO_ROOT, RAW_DATA_ROOT
from IoIO.cordata_base import IOIO_1_UTC_OFFSET
from IoIO.utils import (ColnameEncoder, get_dirs_dates, reduced_dir,
                        nan_biweight, nan_mad, valid_long_exposure,
                        dict_to_ccd_meta, multi_glob, sum_ccddata,
                        csvname_creator, add_itime_col,
                        add_daily_biweights, contiguous_sections,
                        linspace_day_table, add_medfilt_columns,
                        daily_convolve, savefig_overwrite,
                        finish_stripchart, pixel_per_Rj,
                        plot_planet_subim, plot_column)
from IoIO.cormultipipe import (MAX_NUM_PROCESSES, MAX_CCDDATA_BITPIX,
                               MAX_MEM_FRAC,
                               COR_PROCESS_EXPAND_FACTOR,
                               tavg_to_bmp_meta, calc_obj_to_ND, #crop_ccd,
                               planet_to_object,
                               objctradec_to_obj_center,
                               nd_filter_mask, parallel_cached_csvs,
                               obj_surface_bright)
from IoIO.calibration import Calibration, CalArgparseHandler
from IoIO.cor_boltwood import CorBoltwood, weather_to_bmp_meta
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
from IoIO.torus import closest_galsat_to_jupiter, add_mask_col, add_medfilt
from IoIO.juno import JunoTimes, PJAXFormatter

BASE = 'Na_nebula'
NA_NEBULA_ROOT = os.path.join(IoIO_ROOT, BASE)
BOX_NDAYS = 20 # GOING OBSOLETE
MEDFILT_WIDTH = 81 # GOING OBSOLETE
# Filter applied to 24 Rj aperture --> To be robust, this/code below
# should be generalized
# Fri Mar 01 12:24:18 2024 EST  jpmorgen@snipe
# Increased from 35*u.R to check for outburst
MAX_24_Rj_SB = 40*u.R

def draw_na_apertures(ax, ap_sequence=None,
                      min_av_ap_dist=4*u.R_jup, # selects SB boxes
                      max_av_ap_dist=128*u.R_jup,
                      **kwargs):
    # ap_sequence are the concentric
    aps = [ap/2 for ap in ap_sequence
           if min_av_ap_dist < ap and ap < max_av_ap_dist]
    for ap in aps:
        ax.axhline(ap.value)
        ax.axhline(-ap.value)

def na_apertures(ccd_in, bmp_meta=None,
                 plot_planet_rot_from_key=None, # Capture here so no multiple
                 in_name=None,
                 outname_append=None,
                 **kwargs):
    # --> This would be *much* faster
    # --> https://astropy-regions.readthedocs.io/en/latest/api/regions.RectanglePixelRegion.html

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

    sum_encoder = ColnameEncoder('Na_sum', formatter='.0f')
    area_encoder = ColnameEncoder('Na_area', formatter='.0f')
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
        na_aps[sum_encoder.to_colname(ap)] = ap_sum
        na_aps[area_encoder.to_colname(ap)] = ap_area
    # We only  want to mess with our original CCD metadata
    if bmp_meta is None:
        bmp_meta = {}

    # Experimenting with plotting in rotated frame.

    # Note that this has to be done with the rotated ccd, which is
    # dereferenced, below
    in_name = os.path.basename(ccd_in.meta['RAWFNAME'])
    in_name, in_ext = os.path.splitext(in_name)
    in_name = f'{in_name}_apertures{in_ext}'
    plot_planet_subim(ccd,
                      plot_planet_rot_from_key=None,
                      plot_planet_overlay=draw_na_apertures,
                      ap_sequence=ap_sequence,
                      in_name=in_name,
                      outname_append='',
                      **kwargs)

    # Modify the input ccddata's metadata
    ccd = ccd_meta_to_bmp_meta(ccd_in, bmp_meta=bmp_meta,
                               ccd_meta_to_bmp_meta_keys=
                               [('Jupiter_PDObsLon', u.deg),
                                ('Jupiter_PDObsLat', u.deg),
                                ('Jupiter_PDSunLon', u.deg)])
        
    ccd = dict_to_ccd_meta(ccd, na_aps)

    bmp_meta.update(na_aps)
    return ccd

# --> This needs to be moved up into na_apertures
def add_annular_apertures(t):
    """Add to table t columns for brightnesses for the regions between
    successively larger apertures (e.g. added column n = surface
    brightness for region extending from aperture n and aperture n+1)

    """
    sum_encoder = ColnameEncoder('Na_sum', formatter='.0f')
    area_encoder = ColnameEncoder('Na_area', formatter='.0f')
    sum_colnames = sum_encoder.colbase_list(t.colnames)
    area_colnames = area_encoder.colbase_list(t.colnames)

    sb_encoder = ColnameEncoder('annular_sb', formatter='.1f')

    last_sum = 0
    last_area = 0
    last_ap_bound = 0
    ap_list = []
    for sum_colname, area_colname in zip(sum_colnames, area_colnames):
        # Although we are calculating aperture SBs using the
        # difference between concentric apertures extending above and
        # below the equatorial plane, once we have the rectangular
        # aperture SBs, we call them by the distance from the plane
        ap_bound = sum_encoder.from_colname(sum_colname) / 2
        ap_bound = ap_bound.value
        av_ap = np.mean((ap_bound, last_ap_bound))*u.R_jup
        last_ap_bound = ap_bound
        # These are not affected by what we call the aperture distance
        cts = t[sum_colname] - last_sum
        area = t[area_colname] - last_area
        sb = cts / area
        last_sum = t[sum_colname]
        last_area = t[area_colname]
        colname = sb_encoder.to_colname(av_ap)
        t[colname] = sb
        ap_list.append(av_ap)

def subtract_col_from(t,
                      encoder=None,
                      subtract_colname=None,
                      subtracted_prefix=None):
    for col in encoder.colbase_list(t.colnames):
        t[f'{subtracted_prefix}_{col}'] = t[col] - t[subtract_colname]

def boxcar_medians(
        summary_table,
        include_col_encoder=None, # includes with colbase_middle_list [might just make list]
        median_col_encoder=None):

    # Create a new table, one row per day with biweight & mad_std
    # columns This assumes that it is OK if any other prepends come
    # along for the ride.  And yes, it does exclude the original
    day_table = unique(summary_table, keys='ijdlt')
    include_colnames = include_col_encoder.colbase_middle_list(
        day_table.colnames)
    day_table_colnames = ['ijdlt'] + list(include_colnames)
    day_table = QTable(day_table[day_table_colnames])
    #print(day_table_colnames)

    # Boxcar median each biweight column
    first_day = np.min(day_table['ijdlt'])
    last_day = np.max(day_table['ijdlt'])
    all_days = np.arange(first_day, last_day+1)
    med_colnames = median_col_encoder.colbase_list(day_table.colnames)
    for med_col in med_colnames:
        #print(med_col)
        day_table = daily_convolve(day_table,
                                   'ijdlt',
                                   med_col,
                                   'boxfilt_' + med_col,
                                   Box1DKernel(BOX_NDAYS),
                                   all_days=all_days)

    #day_table['itdatetime'] = Time(day_table['ijd'], format='jd').datetime

    #box_colbase_regexp = re.compile('boxfilt_biweight_' + encoder.colbase + '_.*')
    #box_colnames = list(filter(box_colbase_regexp.match, day_table.colnames))
    #f = plt.figure()
    #ax = plt.subplot()
    #for box_col in box_colnames:
    #    print(box_col)
    #    av_ap = encoder.from_colname(box_col)
    #    plt.plot(day_table['itdatetime'], day_table[box_col],
    #             label=f'{av_ap}')
    #plt.xlabel('date')
    #plt.ylabel(f'Surf. bright {day_table[box_col].unit}')
    #plt.legend()
    #f.autofmt_xdate()
    #if show:
    #    plt.show()
    #plt.close()

    return day_table

def add_sb_diffs(summary_table_in,
                 show=True):
    summary_table = summary_table_in.copy()
    encoder = ColnameEncoder('annular_sb', formatter='.1f')
    diff_encoder = ColnameEncoder('annular_sb_diff', formatter='.1f')
    add_annular_apertures(summary_table)
    sb_colnames = filter(encoder.colbase_regexp.match, summary_table.colnames)
    prev_col = None
    #av_aps = []
    #sb_diffs = []
    for sb_col in sb_colnames:
        col_val = summary_table[sb_col]
        av_ap = encoder.from_colname(sb_col)
        if prev_col is None:
            prev_col = col_val
            prev_av_ap = av_ap
            continue
        diff = prev_col - col_val
        av_ap = (prev_av_ap + av_ap)/2
        summary_table[diff_encoder.to_colname(av_ap)] = diff
        prev_col = col_val
        prev_av_ap = av_ap

    f = plt.figure()
    ax = plt.subplot()
    custom_cycler = cycler('color', ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#17becf', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22'])
    plt.rc('axes', prop_cycle=custom_cycler)

    diff_colnames = list(filter(diff_encoder.colbase_regexp.match,
                                summary_table.colnames))
    summary_table['datetime'] = summary_table['tavg'].datetime
    for diff_col in diff_colnames[3:]:
        av_ap = diff_encoder.from_colname(diff_col)
        plt.plot(summary_table['datetime'], summary_table[diff_col], '.',
                 label=f'{av_ap}')
    plt.xlabel('Date')
    plt.ylabel(f'Surf. bright difference {summary_table[diff_col].unit}')
    plt.legend()

    f.autofmt_xdate()
    if show:
        plt.show()
    plt.close()

    return summary_table

# --> This is becoming obsolete
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
    add_annular_apertures(t)
    sb_encoder = ColnameEncoder('annular_sb', formatter='.1f')
    largest_ap = sb_encoder.largest_colbase(t.colnames)
    subtract_col_from(t, sb_encoder, largest_ap, 'largest_sub')
    largest_sub_encoder = ColnameEncoder('largest_sub', formatter='.1f')

    ap_list = sb_encoder.colbase_list(t.colnames)
    
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
    plt.title(f'{sb_encoder.from_colname(largest_ap):.0f} aperture subtracted')
    for ap in ap_list:
        # Skip the most distant ap
        if ap == largest_ap:
            continue        
        bsub_sb = t[ap] - t[largest_ap]
        mask = bsub_sb < max_good_sb
        plt.plot(t['tavg'][mask].datetime, bsub_sb[mask], '.',
                 label=f'+/- {ap:.0f} Rj')
        med_sb = medfilt(bsub_sb[mask], MEDFILT_WIDTH)
        plt.plot(t['tavg'][mask].datetime, med_sb, '-',
                 label=f'+/- {ap:.0f} Rj medfilt')
        
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

def create_na_nebula_day_table(
        t_na_nebula_in=None,
        max_time_gap=15*u.day,
        max_gap_fraction=0.3, # beware this may leave some NANs if dt = 1*u.day
        medfilt_width=10*u.day,
        medfilt_mode='mirror',
        utc_offset=IOIO_1_UTC_OFFSET,
        dt=3*u.day):

    if t_na_nebula_in is None:
        t_na_nebula_in = os.path.join(NA_NEBULA_ROOT, BASE+'_cleaned.ecsv')
    if isinstance(t_na_nebula_in, str):
        t_na_nebula = QTable.read(t_na_nebula_in)
    else:
        t_na_nebula = t_na_nebula_in.copy()

    max_jd_gap = max_time_gap.to(u.day).value
    jd_dt = dt.to(u.day).value

    # Fill original table with values we need for day_table
    add_itime_col(t_na_nebula, time_col='tavg', itime_col='ijdlt',
                  utc_offset=utc_offset, dt=jd_dt)
    t_na_nebula['jd'] = t_na_nebula['tavg'].jd

    largest_sub_encoder = ColnameEncoder('largest_sub', formatter='.1f')
    sb_colnames = largest_sub_encoder.colbase_list(t_na_nebula.colnames)
    biweight_cols = ['jd'] + sb_colnames
    added_cols =add_daily_biweights(t_na_nebula, day_col='ijdlt',
                                    colnames=biweight_cols)

    # Create day_table
    day_table_cols = ['ijdlt'] + added_cols
    day_table = unique(t_na_nebula[day_table_cols], keys='ijdlt')
    cdts = contiguous_sections(day_table, 'ijdlt', max_jd_gap)


    icdts = linspace_day_table(cdts, algorithm='interpolate_replace_nans',
                               max_gap_fraction=max_gap_fraction,
                               time_col_offset=utc_offset,
                               dt=dt)
    width = medfilt_width/dt
    width = width.decompose().value
    width = np.round(width).astype(int)
    add_medfilt_columns(icdts, colnames=added_cols,
                        medfilt_width=width,
                        mode=medfilt_mode)

    # --> I might want to do other operations in here and maybe even
    # --> have a function I pass in to do that

    # Put it all back to together again 
    na_nebula_day_table = QTable()
    for icdt in icdts:
        na_nebula_day_table = vstack([na_nebula_day_table, icdt])

    # Give us a time column back
    jds = na_nebula_day_table['biweight_jd']
    loc = t_na_nebula['tavg'][0].location.copy()
    tavgs = Time(jds, format='jd', location=loc)
    tavgs.format = 'fits'
    na_nebula_day_table['tavg'] = tavgs

    return na_nebula_day_table

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
                        outdir_root=NA_NEBULA_ROOT,
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
                       post_process_list=[tavg_to_bmp_meta,
                                          weather_to_bmp_meta,
                                          calc_obj_to_ND,
                                          planet_to_object],
                       plot_planet_rot_from_key=['Jupiter_NPole_ang'],
                       planet_subim_figsize=[5, 4],
                       planet_subim_dx=45*u.R_jup,
                       planet_subim_dy=40*u.R_jup, 
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
    add_mask_col(t)
    #na_nebula_plot(t, outdir)
    return t


def plot_na_nebula_surf_brights(
        t_na_nebula=None,
        na_nebula_day_table=None,
        min_av_ap_dist=6*u.R_jup, # selects SB boxes
        max_av_ap_dist=25*u.R_jup,
        na_sb_lim=None,
        fig=None,
        ax=None,
        tlim=None):

    fig = fig or plt.figure()
    ax = ax or fig.add_subplot()

    if t_na_nebula is None:
        t_na_nebula = os.path.join(NA_NEBULA_ROOT, BASE+'_cleaned.ecsv')
    if isinstance(t_na_nebula, str):
        t_na_nebula = QTable.read(t_na_nebula)
    if na_nebula_day_table is None:
        na_nebula_day_table = os.path.join(NA_NEBULA_ROOT,
                                           BASE+'_day_table.ecsv')
    if isinstance(na_nebula_day_table, str):
        na_nebula_day_table = QTable.read(na_nebula_day_table)

    custom_cycler = cycler('color', ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#17becf', '#8c564b', '#e377c2', '#7f7f7f'])
    plt.rc('axes', prop_cycle=custom_cycler)

    err_handles = []
    line_handles = []

    # Day table
    biweight_encoder = ColnameEncoder('biweight', formatter='.1f')
    biweight_cols = biweight_encoder.colbase_list(na_nebula_day_table.colnames)
    std_encoder = ColnameEncoder('std', formatter='.1f')
    std_cols = std_encoder.colbase_list(na_nebula_day_table.colnames)
    for bwt_col, std_col in zip(biweight_cols, std_cols):
        if '_jd' in bwt_col:
            continue
        av_ap = biweight_encoder.from_colname(bwt_col)
        if av_ap < min_av_ap_dist or av_ap > max_av_ap_dist:
            continue
        medfilt_colname = f'medfilt_{bwt_col}'
        h = plot_column(na_nebula_day_table,
                        colname=bwt_col,
                        err_colname=std_col,
                        fmt='.',
                        label=f'{av_ap.value} R$_\mathrm{{J}}$ ',
                        alpha=0.25,
                        fig=fig, ax=ax)
        err_handles.append(h)
        h = plot_column(na_nebula_day_table,
                        colname=medfilt_colname,
                        label=f'{av_ap.value} R$_\mathrm{{J}}$ medfilt',
                        linewidth=2,
                        fig=fig, ax=ax)
        line_handles.append(h)

    ax.set_xlim(tlim)
    ax.set_ylim(na_sb_lim)
    ax.set_ylabel(f'Na Neb. Surf. Bright ({na_nebula_day_table[bwt_col].unit})')
    ax.legend(ncol=2, handles=err_handles+line_handles)

 # --> This is becoming obsolete
def plot_nightly_medians(table_or_fname,
                         fig=None,
                         ax=None,
                         tlim=None,
                         min_sb=0, # For plot ylim
                         max_sb=300,
                         min_av_ap_dist=6*u.R_jup, # selects SB boxes
                         max_av_ap_dist=25*u.R_jup,
                         show=False,
                         fig_close=False,
                         medfilt_width=21,
                         max_night_gap=15,
                         **kwargs):
    if isinstance(table_or_fname, str):
        t = QTable.read(table_or_fname)
    else:
        t = table_or_fname
    if fig is None:
        fig = plt.figure()
    if ax is None:
        ax = fig.add_subplot()

    custom_cycler = cycler('color', ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#17becf', '#8c564b', '#e377c2', '#7f7f7f'])
    plt.rc('axes', prop_cycle=custom_cycler)
        
    day_table = unique(t, keys='ijdlt')
    day_table.sort('ijdlt')
    deltas = day_table['ijdlt'][1:] - day_table['ijdlt'][0:-1]
    gap_idx = np.flatnonzero(deltas > max_night_gap)
    # Plot our points at 07:00 UT, which is midnight localtime
    utm = day_table['ijdlt'] + 7/24
    # Convert back to Julian day --> this may be wrong, but we are
    # refactoring anyway
    jd = utm - 0.5
    day_table['itdatetime'] = Time(jd, format='jd').datetime
    biweight_encoder = ColnameEncoder('biweight', formatter='.1f')
    biweight_cols = biweight_encoder.colbase_list(day_table.colnames)
    std_encoder = ColnameEncoder('std', formatter='.1f')
    std_cols = std_encoder.colbase_list(day_table.colnames)
    point_handles = []
    medfilt_handles = []
    for bwt_col, std_col in zip(biweight_cols, std_cols):
        av_ap = biweight_encoder.from_colname(bwt_col)
        if av_ap < min_av_ap_dist or av_ap > max_av_ap_dist:
            continue
        h = ax.errorbar(day_table['itdatetime'], day_table[bwt_col], 
                         day_table[std_col], fmt='.', 
                         label=f'{av_ap.value} R$_\mathrm{{J}}$', alpha=0.25)
        point_handles.append(h)        
        add_medfilt(day_table, bwt_col, medfilt_width=medfilt_width)
        # Quick-and-dirty gap work.  Could do this in the day_table,
        # but I would need to muck with all of the columns
        times = day_table['itdatetime']
        vals = day_table[bwt_col+'_medfilt']
        times = np.append(times, times[gap_idx] + datetime.timedelta(days=1))
        vals = np.append(vals, (np.nan, ) * len(gap_idx))
        sort_idx = np.argsort(times)
        h = ax.plot(times[sort_idx], vals[sort_idx], '-',
                     label=f'{av_ap.value} R$_\mathrm{{J}}$ medfilt',
                     linewidth=2)
        medfilt_handles.append(h[0])
        bwt = nan_biweight(day_table[bwt_col])
        mad = nan_mad(day_table[bwt_col])
        log.info(f'{bwt_col} Biweight +/- MAD = {bwt} +/- {mad}')
    handles = point_handles
    handles.extend(medfilt_handles)

    ax.set_xlabel('Date')
    ax.set_ylabel(f'Na Neb. Surf. Bright ({t[bwt_col].unit})')
    #plt.title('Na nebula -- nightly medians')
    if tlim is None:
        tlim = ax.get_xlim()
    ax.set_xlim(tlim)
    ax.set_ylim(min_sb, max_sb)
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.legend(ncol=2, handles=handles)
    fig.autofmt_xdate()

    jts = JunoTimes()
    secax = ax.secondary_xaxis('top',
                                functions=(jts.plt_date2pj, jts.pj2plt_date))
    secax.xaxis.set_minor_locator(MultipleLocator(1))
    secax.set_xlabel('PJ')    
    ax.format_coord = PJAXFormatter(times[sort_idx], vals[sort_idx])
    
    if show:
        plt.show()
    if fig_close:
        plt.close()

    # --> day_table might also be useful to return, but it has a
    # non-serializable datetime object in it.  I think astropy times
    # plot properly these days, so this could be ammended
    return t

def plot_obj_surf_bright(table_or_fname,
                         fig=None,
                         ax=None,
                         tlim=None,
                         sb_lim=(0,200000),
                         show=False,
                         fig_close=False):
    if isinstance(table_or_fname, str):
        t = QTable.read(table_or_fname)
    else:
        t = table_or_fname
    if fig is None:
        fig = plt.figure()
    if ax is None:
        ax = plt.subplot()

    obj_sb = t['obj_surf_bright']
    obj_sb_err = t['obj_surf_bright_err']
    plt.title('Jupiter attenuated surface brightness')
    plt.errorbar(t['tavg'].datetime, obj_sb.value, obj_sb_err.value,
                 fmt='k.')
    plt.ylabel(f'Atten. Surf. Bright ({obj_sb.unit})')

    if tlim is None:
        tlim = ax.get_xlim()
    ax.set_xlim(tlim)
    ax.set_ylim(sb_lim)
    plt.legend()
    fig.autofmt_xdate()
    if show:
        plt.show()
    if fig_close:
        plt.close()

def na_nebula_tree(raw_data_root=RAW_DATA_ROOT,
                   start=None,
                   stop=None,
                   calibration=None,
                   cor_boltwood=None,
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
        return []
    calibration = calibration or Calibration()
    cor_boltwood = cor_boltwood or CorBoltwood(precalc=True)
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
        'cor_boltwood': cor_boltwood,
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

    # --> This should probably be at the directory level  
    # Supplement summary_table with columns needed for plots
    add_annular_apertures(summary_table)
    # --> could do some multi-aperture, Na_meso etc., plots here
    
    sb_encoder = ColnameEncoder('annular_sb', formatter='.1f')
    largest_ap = sb_encoder.largest_colbase(summary_table.colnames)
    subtract_col_from(summary_table, sb_encoder, largest_ap, 'largest_sub')
    largest_sub_encoder = ColnameEncoder('largest_sub', formatter='.1f')
    ls_cols = largest_sub_encoder.colbase_list(summary_table.colnames)

    #print('len(summary_table): ', len(summary_table))

    # Mask bad measurements
    # This is pretty effective and may be useful for eventually making
    # a movie, as long as I capture the appropriate filenames in summary_table
    # --> I might want to add this to the mask column or have a special one
    mask = None
    for ls_col in ls_cols:
        if mask is None:
            mask = summary_table[ls_col] >= 0
            continue
        mask = np.logical_and(mask, summary_table[ls_col] >= 0)
    mask = np.logical_and(mask, summary_table[ls_cols[-2]] < MAX_24_Rj_SB)
    clean_t = summary_table[mask]

    # Mask rows marked bad for other reasons
    clean_t = clean_t[~clean_t['mask']]
    #print('len(clean_t): ', len(clean_t))

    clean_t.sort('tavg')
    #add_itime_col(clean_t, time_col='tavg', itime_col='ijdlt',
    #              utc_offset=IOIO_1_UTC_OFFSET, dt=1*u.day)
    #sb_colnames = largest_sub_encoder.colbase_list(clean_t.colnames)
    #add_daily_biweights(clean_t, day_col='ijdlt', colnames=sb_colnames)
    clean_t.write(os.path.join(outdir_root, BASE + '_cleaned.ecsv'),
                  overwrite=True)
    torus_day_table = create_na_nebula_day_table(clean_t)
    torus_day_table.write(os.path.join(outdir_root, BASE + '_day_table.ecsv'),
                          overwrite=True)
  
    return summary_table

class NaNebulaArgparseHandler(NaMesoArgparseHandler, SSArgparseHandler,
                              CorPhotometryArgparseMixin, CalArgparseHandler):
    def add_all(self):
        """Add options used in cmd"""
        self.add_reduced_root(default=NA_NEBULA_ROOT)
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
                           cor_boltwood=CorBoltwood(precalc=True),
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
                           fits_fixed_ignore=args.fits_fixed_ignore,
                           fits_verify_ignore=args.fits_verify_ignore)

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
##directory = '/data/IoIO/raw/2018-05-08/'
#directory = '/data/IoIO/raw/20221224/'
#
##t = na_nebula_tree(start='2018-05-08',
##                   stop='2018-05-10',
##                   read_csvs=True,
##                   write_csvs=True,
##                   read_pout=True,
##                   write_pout=True,
##                   fits_fixed_ignore=True)
#                  
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
##na_meso_obj = na_meso_obj or NaMeso(calibration=calibration,
##                                    standard_star_obj=standard_star_obj,
##                                    reduce=True)
#na_meso_obj = na_meso_obj or NaMeso()
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
#

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
#summary_table = QTable.read('/data/IoIO/Na_nebula/Na_nebula.ecsv')
#na_nebula_plot(t, '/tmp', max_good_sb=1000*u.R, show=True, n_plots=4)
#na_nebula_plot(t, '/tmp', show=True, n_plots=3, min_av_ap_dist=10, max_sb=600)

#summary_table = add_annular_boxcar_medians(summary_table)
#add_annular_boxcar_medians(summary_table, subtract_col='meso_or_model')

#add_sb_diffs(summary_table)

# show=True
# min_av_ap_dist=0
# min_av_ap_dist=12*u.R_jup
# max_av_ap_dist=np.inf
# max_av_ap_dist=50*u.R_jup
# 
# t = summary_table
# add_annular_apertures(t)
# sb_encoder = ColnameEncoder('annular_sb', formatter='.1f')
# largest_ap = sb_encoder.largest_colbase(t.colnames)
# subtract_col_from(t, sb_encoder, largest_ap, 'largest_sub')
# largest_sub_encoder = ColnameEncoder('largest_sub', formatter='.1f')
# 
# 
# # largest aperture subtracted, all points
# f = plt.figure()
# ax = plt.subplot()
# ls_cols = largest_sub_encoder.colbase_list(t.colnames)
# for ls_col in ls_cols:
#     av_ap = largest_sub_encoder.from_colname(ls_col)
#     plt.plot(t['tavg'].datetime, t[ls_col], '.',
#              label=f'{av_ap}')
# plt.xlabel('date')
# plt.ylabel(f'Surf. bright {t[ls_col].unit}')
# plt.legend()
# f.autofmt_xdate()
# if show:
#     plt.show()
# plt.close()
# 
# mask = None
# for ls_col in ls_cols:
#     if mask is None:
#         mask = t[ls_col] >= 0
#         continue
#     mask = np.logical_and(mask, t[ls_col] >= 0)
# 
# mask = np.logical_and(mask, t[ls_cols[-2]] < MAX_24_Rj_SB)
#     
# # largest aperture subtracted, masked
# f = plt.figure()
# ax = plt.subplot()
# for ls_col in ls_cols:
#     av_ap = largest_sub_encoder.from_colname(ls_col)
#     if av_ap < min_av_ap_dist or av_ap > max_av_ap_dist:
#         continue
#     plt.plot(t['tavg'][mask].datetime, t[ls_col][mask], '.',
#              label=f'{av_ap}')
# plt.xlabel('date')
# plt.ylabel(f'Surf. bright {t[ls_col].unit}')
# plt.legend()
# f.autofmt_xdate()
# if show:
#     plt.show()
# plt.close()
# 
# clean_t = t[mask]
# 
# add_daily_biweights(clean_t, encoder=largest_sub_encoder)
# 
# biweight_encoder = ColnameEncoder('biweight', formatter='.1f')
# biweight_cols = biweight_encoder.colbase_list(clean_t.colnames)
# 
# day_table = boxcar_medians(clean_t,
#                            include_col_encoder=largest_sub_encoder,
#                            median_col_encoder=biweight_encoder)
# f = plt.figure()
# ax = plt.subplot()
# for bwt_col in biweight_cols:
#     av_ap = largest_sub_encoder.from_colname(bwt_col)
#     if av_ap < min_av_ap_dist or av_ap > max_av_ap_dist:
#         continue
#     plt.plot(day_table['itdatetime'], day_table[bwt_col], '.',
#              label=f'{av_ap}')
# plt.xlabel('date')
# plt.ylabel(f'Surf. bright ({t[ls_col].unit})')
# plt.title('Na nebula -- nightly medians')
# plt.legend()
# f.autofmt_xdate()
# if show:
#     plt.show()
# plt.close()
# 
# # Boxcar medians -- didn't really turn out all that well
# boxcar_encoder = ColnameEncoder('boxfilt_biweight', formatter='.1f')
# box_colnames = boxcar_encoder.colbase_list(day_table.colnames)
# f = plt.figure()
# ax = plt.subplot()
# for box_col in box_colnames:
#     av_ap = boxcar_encoder.from_colname(box_col)
#     if av_ap < min_av_ap_dist or av_ap > max_av_ap_dist:
#         continue
#     plt.plot(day_table['itdatetime'], day_table[box_col], '.',
#              label=f'{av_ap}')
# plt.xlabel('date')
# plt.ylabel(f'Surf. bright {day_table[box_col].unit}')
# plt.legend()
# f.autofmt_xdate()
# if show:
#     plt.show()
# plt.close()

#plot_nightly_medians('/data/IoIO/Na_nebula/Na_nebula_cleaned.ecsv')
#plot_obj_surf_bright('/data/IoIO/Na_nebula/Na_nebula_cleaned.ecsv', show=True)

