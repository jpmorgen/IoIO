#!/usr/bin/python3

"""Reduce IoIO Io plamsa torus observations"""

import os
import argparse
from datetime import date

import numpy as np

from scipy.signal import medfilt
from scipy.ndimage import median_filter

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as patches
from matplotlib.ticker import MultipleLocator
import matplotlib.transforms as transforms

from astropy import log
import astropy.units as u
from astropy.time import Time
from astropy.table import QTable, unique, vstack
from astropy.convolution import Gaussian1DKernel
from astropy.coordinates import Angle, SkyCoord
from astropy.modeling import models, fitting
from astropy.timeseries import TimeSeries, LombScargle

from bigmultipipe import cached_pout, outname_creator

from ccdmultipipe import ccd_meta_to_bmp_meta, as_single

from IoIO.utils import (dict_to_ccd_meta, nan_biweight, nan_mad,
                        nan_median_filter, get_dirs_dates,
                        reduced_dir, cached_csv, savefig_overwrite,
                        finish_stripchart, multi_glob, pixel_per_Rj,
                        plot_planet_subim, plot_columns,
                        interpolate_replace_nans_columns,
                        add_itime_col, add_daily_biweights,
                        contiguous_sections, linspace_day_table,
                        add_medfilt_columns)
from IoIO.simple_show import simple_show
from IoIO.cordata_base import SMALL_FILT_CROP
from IoIO.cordata import CorData
from IoIO.cormultipipe import (IoIO_ROOT, RAW_DATA_ROOT,
                               tavg_to_bmp_meta, calc_obj_to_ND, crop_ccd,
                               planet_to_object, mean_image,
                               objctradec_to_obj_center, obj_surface_bright)
from IoIO.calibration import Calibration, CalArgparseHandler
from IoIO.cor_boltwood import CorBoltwood, weather_to_bmp_meta
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
from IoIO.horizons import GALSATS, galsat_ephemeris
from IoIO.juno import JunoTimes, PJAXFormatter, juno_pj_axis

TORUS_ROOT = os.path.join(IoIO_ROOT, 'Torus')
MAX_ROOT = os.path.join(IoIO_ROOT, 'for_Max')

# --> from Barbosa & Kivelson (1983)
# Corotation electric field = 150 mV/m

# https://lasp.colorado.edu/home/mop/files/2015/02/CoOrd_systems7.pdf
# Has sysIII of intersection of mag and equatorial plains at 290.8.
# That means tilt is toward 200.8, which is my basic recollection
# --> FIX THESE to values in paper
CENTRIFUGAL_EQUATOR_AMPLITUDE = 6.8*u.deg # --> 6.3*u.deg
JUPITER_MAG_SYSIII = 290.8*u.deg # --> 286.61*u.deg
MAX_N_MASK = 10
RIGHT_INNER = 4.75 * u.R_jup
RIGHT_OUTER = 6.75 * u.R_jup
LEFT_OUTER = -6.75 * u.R_jup
LEFT_INNER = -4.75 * u.R_jup
# Defines a buffer zone inside of the *_INNER *_OUTER ansa box in
# which the ansa peak can wander during the fit
DR_BOUNDS_BUFFER = 0.25 * u.R_jup
# Width wander.  Upper limit at 0.4 was a little too tight when there
# is a big slope.
# '/data/IoIO/Torus/20221109/SII_on-band_006-back-sub.fits' seems to
# want 0.8 to kick-start the process
#ANSA_WIDTH_BOUNDS = (0.1*u.R_jup, 0.5*u.R_jup)
ANSA_WIDTH_BOUNDS = (0.1*u.R_jup, 0.8*u.R_jup)

# How much above and below the centripetal plane to extract from our image.
# There is a conflict between too much open torus and not allowing for
# distortion in the IPT due to the magnetic field. +/- 1.75 is too
# broad.  +/- 1.25 is about right
ANSA_DY = 1.25 * u.R_jup

IO_ORBIT_R = 421700 * u.km 
IO_ORBIT_R = IO_ORBIT_R.to(u.R_jup)

SYSIII = 9*u.hr+55*u.min+29.71*u.s # --> Want reference other than Boulder PDF
SYSIV_LOW = 10.1*u.hr
SYSIV_HIGH = 10.35*u.hr
IO_ORBIT = 42.45930686*u.hr
EUROPA_ORBIT = 3.551181*u.day
EUROPA_ORBIT = EUROPA_ORBIT.to(u.hr)
GANYMEDE_ORBIT = 7.15455296*u.day
GANYMEDE_ORBIT = GANYMEDE_ORBIT.to(u.hr)
CALLISTO_ORBIT = 16.6890184*u.day
CALLISTO_ORBIT = CALLISTO_ORBIT.to(u.hr)


#def IPT_NPole_ang(ccd, **kwargs):
#    sysIII = ccd.meta['Jupiter_PDObsLon']
#    ccd.meta['HIERARCH IPT_NPole_ang'] = (
#        CENTRIFUGAL_EQUATOR_AMPLITUDE * np.sin(sysIII - JUPITER_MAG_SYSIII),
#        '[degree]')
#    return ccd

def bad_ansa(side):
    # numpy.ma don't work with astropy units, 
    # astropy.utils.masked.Masked doesn't work with pickle
    #masked = Masked(np.NAN, mask=True)
    masked = np.NAN
    return {f'ansa_{side}_r_peak': masked*u.R_jup,
            f'ansa_{side}_r_peak_err': masked*u.R_jup,
            f'ansa_{side}_r_stddev': masked*u.R_jup,
            f'ansa_{side}_r_stddev_err': masked*u.R_jup,
            f'ansa_{side}_r_amplitude': masked*u.R,
            f'ansa_{side}_r_amplitude_err': masked*u.R,
            f'ansa_{side}_y_peak': masked*u.R_jup,
            f'ansa_{side}_y_peak_err': masked*u.R_jup,
            f'ansa_{side}_y_stddev': masked*u.R_jup,
            f'ansa_{side}_y_stddev_err': masked*u.R_jup,
            f'ansa_{side}_surf_bright': masked*u.R,
            f'ansa_{side}_surf_bright_err': masked*u.R,
            f'ansa_{side}_cont': masked*u.R,
            f'ansa_{side}_cont_err': masked*u.R,
            f'ansa_{side}_slope': masked*u.R/u.jupiterRad,
            f'ansa_{side}_slope_err': masked*u.R/u.jupiterRad}

def ansa_parameters(ccd,
                    side=None,
                    ansa_max_n_mask=MAX_N_MASK,
                    ansa_right_inner=RIGHT_INNER,
                    ansa_right_outer=RIGHT_OUTER,
                    ansa_left_outer=LEFT_OUTER,
                    ansa_left_inner=LEFT_INNER,
                    ansa_dr_bounds_buffer=DR_BOUNDS_BUFFER,
                    ansa_width_bounds=ANSA_WIDTH_BOUNDS,
                    ansa_dy=ANSA_DY,
                    rprof_ax=None,
                    vprof_ax=None,
                    show=False,
                    in_name=None,
                    outname=None,                    
                    **kwargs):
    center = ccd.obj_center*u.pixel
    # Work in Rj for our model
    pix_per_Rj = pixel_per_Rj(ccd)

    # Start with a basic fix on the ansa r position
    if side == 'left':
        left = center[1] + ansa_left_outer*pix_per_Rj
        right = center[1] + ansa_left_inner*pix_per_Rj
        left = int(round(left.value))
        right = int(round(right.value))
        side_mult = -1
        mean_bounds = (ansa_left_outer + ansa_dr_bounds_buffer,
                       ansa_left_inner - ansa_dr_bounds_buffer)
    elif side == 'right':
        left = center[1] + ansa_right_inner*pix_per_Rj
        right = center[1] + ansa_right_outer*pix_per_Rj
        left = int(round(left.value))
        right = int(round(right.value))
        side_mult = +1
        mean_bounds = (ansa_right_inner + ansa_dr_bounds_buffer,
                       ansa_right_outer - ansa_dr_bounds_buffer)
    else:
        raise ValueError(f'ansa_side must be right or left: {side}')    
        
    cold_mean_bounds = np.asarray((5.0, 5.5)) * u.R_jup

    top = center[0] + ansa_dy * pix_per_Rj
    bottom = center[0] - ansa_dy * pix_per_Rj
    top = int(round(top.value))
    bottom = int(round(bottom.value))

    ansa = ccd[bottom:top, left:right]

    #if show:
    #    simple_show(ansa)

    mask_check = ansa.mask
    if np.sum(mask_check) > ansa_max_n_mask:
        return bad_ansa(side)

    # Calculate profile along centripetal equator, keeping it in units
    # of rayleighs
    # --> I don't properly handle masked pixels here!  I should use
    # ansa.sum method, but only after upgrade.  I should also make
    # sure to poke holes in the r_Rj axis
    r_pix = np.arange(ansa.shape[1]) * u.pixel
    r_Rj = (r_pix + left*u.pixel - center[1]) / pix_per_Rj
    r_prof = np.sum(ansa, 0) * ansa.unit
    r_prof /= ansa.shape[1]
    if not np.all(np.isfinite(r_prof)):
        # This may be a stop-gap measure that apparently didn't work
        return bad_ansa(side)

    # --> Work in progress
    #plot_left  = left  - 2*u.R_jup
    #plot_right = right + 2*u.R_jup
    #plot_r_ccd = ccd[bottom:top, plot_left:plot_right]
    #
    #full_r_pix = np.arange(ccd.shape[1]) * u.pixel
    #full_r_Rj = full_r_pix / pix_per_Rj
    #full_r_prof = np.sum(ccd, 0) * ansa.unit
    #full_r_prof /= ansa.shape[1]

    # Fit is definitly better
    #r_ansa = (np.argmax(r_prof)*u.pixel
    #          + left*u.pixel - center[1]) / pix_per_Rj
    #dRj = 0.5

    # Only relative values don't matter here since we are using as a weight
    if ccd.uncertainty.uncertainty_type == 'std':
        r_dev = np.sum(ansa.uncertainty.array**2, 0)
    else:
        r_dev = np.sum(ansa.uncertainty.array, 0)

    # Cold torus is a bridge too far, though it works sometimes
                    #+ models.Gaussian1D(mean=side_mult*(5.25),
                    #                    stddev=0.1,
                    #                    amplitude=amplitude/10,
                    #                    bounds={'mean': side_mult*cold_mean_bounds,
                    #                            'stddev': (0.05, 0.15),
                    #                            'amplitude': (0, amplitude)})

    try:
        amplitude = np.max(r_prof) - np.min(r_prof)
    except Exception as e:
        log.error(f'RAWFNAME of problem: {ccd.meta["RAWFNAME"]} {e}')
        return bad_ansa(side)
    
    # This is the fit object for both horizontal and vertical
    fit = fitting.LevMarLSQFitter(calc_uncertainties=True)
    #fit = fitting.LMLSQFitter(calc_uncertainties=True)
    # Change from min to max for c0, since intercept is on that side of things
    # np.max(r_prof)
    r_model_init = (
        models.Gaussian1D(mean=side_mult*IO_ORBIT_R,
                          stddev=0.3*u.R_jup,
                          amplitude=amplitude,
                          bounds={'mean': mean_bounds,
                                  'stddev': ansa_width_bounds})
        + models.Polynomial1D(1, c0=np.max(r_prof)))
    # The weights need to read in the same unit as the thing being
    # fitted for the std to come back with sensible units.
    try:
        r_fit = fit(r_model_init, r_Rj, r_prof,
                    weights=r_dev**-0.5, maxiter=500)
    except Exception as e:
        log.error(f'RAWFNAME of problem: {ccd.meta["RAWFNAME"]} {e}')
        return bad_ansa(side)

    r_ansa = r_fit.mean_0.quantity
    dRj = r_fit.stddev_0.quantity

    # #print(r_fit.fit_deriv(r_Rj))
    # print(r_fit)
    # print(f'r = {r_ansa} +/- {r_fit.mean_0.std}')
    # print(f'dRj = {dRj} +/- {r_fit.stddev_0.std}')
    # print(f'r_amp = {r_fit.amplitude_0.quantity} +/- {r_fit.amplitude_0.std}')

    date_obs, _ = ccd.meta['DATE-OBS'].split('T') 
    rprof_ax.plot(r_Rj, r_prof)
    rprof_ax.plot(r_Rj, r_fit(r_Rj))
    rprof_ax.set_xlabel(r'Jovicentric radial distance (R$\mathrm{_J}$)')
    rprof_ax.set_ylabel(f'Surface brightness ({r_prof.unit})')
    rprof_ax.axvline(r_ansa.value, color='k')
    trans = transforms.blended_transform_factory(
        rprof_ax.transData, rprof_ax.transAxes)
    rprof_ax.text(r_ansa.value-0.13, 0.08,
                  f'{r_ansa.value:4.2f}' + r' R$\mathrm{_J}$', rotation=90,
                  transform=trans)
    rprof_ax.text(r_ansa.value+dRj.value-0.13, 0.08,
                  r'$\sigma$ = ' + f'{dRj.value:3.2f} ' + r'R$\mathrm{_J}$',
                  rotation=90,
                  transform=trans)
    rprof_ax.margins(x=0, y=0)

    if r_fit.mean_0.std is None:
        # --> Consider incorporating this into the overall output,
        # possibly being more forgiving in letting things go through
        # and/or check for messages in an intelligent way
        #print(fit.fit_info['message'])
        return bad_ansa(side)

    rprof_ax.axvline((r_ansa-dRj).value, linestyle='--', color='k')
    rprof_ax.axvline((r_ansa+dRj).value, linestyle='--', color='k')

    # Refine our left and right based on the above fit
    narrow_left = center[1] + (r_ansa - dRj) * pix_per_Rj
    narrow_right = center[1] + (r_ansa + dRj) * pix_per_Rj
    narrow_left = int(round(narrow_left.value))
    narrow_right = int(round(narrow_right.value))

    narrow_ansa = ccd[bottom:top, narrow_left:narrow_right]
    #simple_show(narrow_ansa)

    # Calculate profile perpendicular to centripetal equator, keeping
    # it in units of rayleighs
    y_pix = np.arange(narrow_ansa.shape[0]) * u.pixel
    y_Rj = (y_pix + bottom*u.pixel - center[0]) / pix_per_Rj
    y_prof = np.sum(narrow_ansa, 1) * ansa.unit
    y_prof /= narrow_ansa.shape[0]
    if ccd.uncertainty.uncertainty_type == 'std':
        y_dev = np.sum(ansa.uncertainty.array**2, 1)
    else:
        y_dev = np.sum(ansa.uncertainty.array, 1)
    amplitude = np.max(y_prof) - np.min(y_prof)
    y_model_init = (models.Gaussian1D(mean=0,
                                      stddev=0.5*u.R_jup,
                                      amplitude=amplitude)
                    + models.Polynomial1D(0, c0=np.min(y_prof)))
    y_fit = fit(y_model_init, y_Rj, y_prof, weights=y_dev**-0.5)

    vprof_ax.plot(y_prof, y_Rj)
    vprof_ax.plot(y_fit(y_Rj), y_Rj)
    vprof_ax.set_ylabel(r'R$\mathrm{_J}$', labelpad=0)
    vprof_ax.set_xlabel(y_prof.unit)
    vprof_ax.margins(x=0, y=0)

    if y_fit.mean_0.std is None:
        return bad_ansa(side)

    y_pos = y_fit.mean_0.quantity
    y_pos_std = y_fit.mean_0.std or np.NAN
    dy_pos = y_fit.stddev_0.quantity
    dy_pos_std = y_fit.stddev_0.std or np.NAN
    y_amplitude = y_fit.amplitude_0.quantity
    y_amplitude_std = y_fit.amplitude_0.std or np.NAN
    # r_ansa et al. are defined above
    r_ansa_std = r_fit.mean_0.std or np.NAN
    dRj_std = r_fit.stddev_0.std or np.NAN
    r_amp = r_fit.amplitude_0.quantity
    r_amp_std = r_fit.amplitude_0.std or np.NAN
    cont = r_fit.c0_1.quantity
    cont_std = r_fit.c0_1.std or np.NAN
    slope = r_fit.c1_1.quantity
    slope_std = r_fit.c1_1.std or np.NAN


    # Units of amplitude are already in rayleigh.  When we do the
    # integral, we technically should multiply the amplitude by the
    # peak stddev.  But that gives us an intensity meausrement.  To
    # convert that to sb, we divide again by the stddev.
    sb = (2 * np.pi)**-0.5 * y_amplitude
    sb_err = (2 * np.pi)**-0.5 * y_amplitude_std

    # Update bad_ansa when changing these
    return {f'ansa_{side}_r_peak': r_ansa,
            f'ansa_{side}_r_peak_err': r_ansa_std * r_ansa.unit,
            f'ansa_{side}_r_stddev': dRj,
            f'ansa_{side}_r_stddev_err': dRj_std * dRj.unit,
            f'ansa_{side}_r_amplitude': r_amp,
            f'ansa_{side}_r_amplitude_err': r_amp_std * r_amp.unit,
            f'ansa_{side}_y_peak': y_pos,
            f'ansa_{side}_y_peak_err': y_pos_std * y_pos.unit,
            f'ansa_{side}_y_stddev': dy_pos,
            f'ansa_{side}_y_stddev_err': dy_pos_std * dy_pos.unit,
            f'ansa_{side}_surf_bright': sb,
            f'ansa_{side}_surf_bright_err': sb_err * sb.unit,
            f'ansa_{side}_cont': cont,
            f'ansa_{side}_cont_err': cont_std * cont.unit,
            f'ansa_{side}_slope': slope,
            f'ansa_{side}_slope_err': slope_std * slope.unit}

def draw_ansa_boxes(ax,
                    right_inner=RIGHT_INNER,
                    right_outer=RIGHT_OUTER,
                    left_outer=LEFT_OUTER,
                    left_inner=LEFT_INNER,
                    ansa_dy=ANSA_DY,
                    **kwargs):
    left = patches.Rectangle((left_outer.value, -ansa_dy.value),
                             (left_inner-left_outer).value,
                             ansa_dy.value*2,
                             linewidth=1,
                             edgecolor='w',
                             facecolor='none')
    right = patches.Rectangle((right_outer.value, -ansa_dy.value),
                             (right_inner-right_outer).value,
                             ansa_dy.value*2,
                             linewidth=1,
                             edgecolor='w',
                             facecolor='none')
    ax.add_patch(left)
    ax.add_patch(right)

def characterize_ansas(ccd_in, bmp_meta=None, galsat_mask_side=None, 
                       plot_planet_rot_from_key=None,
                       in_name=None,
                       outname_append=None,
                       **kwargs):
    # MAKE SURE THIS IS CALLED BEFORE ANY ROTATIONS The underlying
    # reproject stuff in rot_to is a little unstable when it comes to
    # multiple rotations.  When autorotate is used, N is not always
    # up.  If it is not used, the CCD image can get shoved off to one
    # side in a weird way.
    try:
        rccd = rot_to(ccd_in, rot_angle_from_key=['Jupiter_NPole_ang',
                                                  'IPT_NPole_ang'])
    except Exception as e:
        # This helped me spot that I had OBJECT set improperly, but
        # might as well leave it in
        log.error(f'RAWFNAME of problem: {ccd_in.meta["RAWFNAME"]}')
        raise e
    rccd = objctradec_to_obj_center(rccd)
    rccd = mask_galsats(rccd, galsat_mask_side=galsat_mask_side)
    # We only  want to mess with our original CCD metadata
    ccd = ccd_in.copy()
    if bmp_meta is None:
        bmp_meta = {}
    ccd = ccd_meta_to_bmp_meta(ccd, bmp_meta=bmp_meta,
                               ccd_meta_to_bmp_meta_keys=
                               [('Jupiter_PDObsLon', u.deg),
                                ('Jupiter_PDObsLat', u.deg),
                                ('Jupiter_PDSunLon', u.deg)])

    # Prepare to create a multi-panel plot.  I play a little
    # fast-and-lose with plot_planet_subim, since it does the figure
    # writing, I create the figure and subplots here, fill all other
    # subplots and lastly fill the image subim subplot and write to
    # disk
    fig = plt.figure(figsize=[7.25, 4.8], tight_layout=True)
    gs = fig.add_gridspec(2, 4, width_ratios=(0.15, 0.35, 0.35, 0.15))
    vprof_axes = (fig.add_subplot(gs[0,0]), fig.add_subplot(gs[0,3]))
    im_ax = fig.add_subplot(gs[0,1:3])
    rprof_axes = (fig.add_subplot(gs[1,0:2]), fig.add_subplot(gs[1,2:]))
    fig.subplots_adjust(wspace=0, hspace=0)    
    vprof_axes[0].sharey(vprof_axes[1])
    rprof_axes[0].sharey(rprof_axes[1])

    for side, rprof_ax, vprof_ax in zip(('left', 'right'), rprof_axes, vprof_axes):
        ansa_meta = ansa_parameters(rccd, side=side,
                                    rprof_ax=rprof_ax,
                                    vprof_ax=vprof_ax,
                                    **kwargs)
        ccd = dict_to_ccd_meta(ccd, ansa_meta)
        bmp_meta.update(ansa_meta)

    max_vprof = np.max((vprof_axes[0].get_xlim(), vprof_axes[1].get_xlim()))
    min_vprof = np.min((vprof_axes[0].get_xlim(), vprof_axes[1].get_xlim()))
    for ax in vprof_axes:
        ax.set_xlim((min_vprof, max_vprof))
    vprof_axes[0].invert_xaxis()
    
    in_name = os.path.basename(ccd_in.meta['RAWFNAME'])
    in_name, in_ext = os.path.splitext(in_name)
    in_name = f'{in_name}_reduction{in_ext}'
    plot_planet_subim(rccd,
                      fig=fig,
                      ax=im_ax,
                      plot_planet_rot_from_key=None,
                      plot_planet_overlay=draw_ansa_boxes,
                      in_name=in_name,
                      outname_append='',
                      **kwargs)
    return ccd

def closest_galsat_to_jupiter(ccd_in, bmp_meta=None, **kwargs):
    if bmp_meta is None:
        bmp_meta = {}
    ccd = ccd_in.copy()
    galsats = list(GALSATS.keys())
    g = galsats[0]
    ra = ccd.meta[f'{g}_RA']
    dec = ccd.meta[f'{g}_DEC']
    jup_sc = SkyCoord(ra, dec, unit=u.deg)
    galsats = galsats[1:]
    min_ang = 90*u.deg
    for g in galsats:
        ra = ccd.meta[f'{g}_RA']
        dec = ccd.meta[f'{g}_DEC']
        sc = SkyCoord(ra, dec, unit=u.deg)
        if sc.separation(jup_sc) < min_ang:
            min_ang = sc.separation(jup_sc)
            closest_galsat = g
            Rj_arcsec = ccd.meta['Jupiter_ang_width'] * u.arcsec / 2
            closest_Rj = min_ang.to(u.arcsec) / Rj_arcsec * u.R_jup
            ccd.meta['HIERARCH CLOSEST_GALSAT'] = closest_galsat
            ccd.meta['HIERARCH CLOSEST_GALSAT_DIST'] = (
                closest_Rj.value, closest_Rj.unit)
            bmp_meta['closest_galsat'] = closest_Rj
    return ccd        

def add_mask_col(t, d_on_off_max=5, obj_to_ND_max=30):
    t['mask'] = False
    # These first few were masked when I added the columns
    # after-the-fact in early 2023
    if 'd_on_off' in t.colnames and not np.ma.is_masked(t['d_on_off']):        
        t['mask'] = np.logical_or(t['mask'], t['d_on_off'] > d_on_off_max)
    if 'obj_to_ND' in t.colnames and not np.ma.is_masked(t['obj_to_ND']):
        t['mask'] = np.logical_or(t['mask'], t['obj_to_ND'] > obj_to_ND_max)
    if 'mean_image' in t.colnames and not np.ma.is_masked(t['mean_image']):
        # This is specific to the torus data
        t['mask'] = np.logical_or(t['mask'], t['mean_image'] < 35*u.R)
        t['mask'] = np.logical_or(t['mask'], t['mean_image'] > 200*u.R)
    if 'ansa_right_surf_bright_err' in t.colnames:
        # This is specific to the torus data
        t['mask'] = np.logical_or(t['mask'],
                                  t['ansa_right_surf_bright_err'] > 10*u.R)
        t['mask'] = np.logical_or(t['mask'],
                                  t['ansa_left_surf_bright_err'] > 10*u.R)

def create_torus_day_table(
        t_torus_in=None,
        max_time_gap=15*u.day,
        max_gap_fraction=0.3, # beware this may leave some NANs if dt = 1*u.day
        jd_medfilt_width=10,
        jd_medfilt_mode='mirror',
        ut_offset=-7*u.hr, # For IoIO1 --> This should be a constant
        dt=3*u.day):
    """Returns a table of torus data sampled at regular intervals
    within contiguous sets of data.  Adds median filtered columns in
    each segment of continous data

    Parameters
    ----------
    t_torus_in : Table
        Input t_torus data table.  If none, read from disk

    max_time_gap : TimeDelta
        Gaps larger than maximum gap trigger the formation of another
        segment of data

    jd_medfilt_width : float
        Median filter in JD to apply to data within each contiguous
        segment
        Default is `10'

    jd_medfilt_mode : str
        Median filtering mode

    ut_offset : TimeDelta
        UT offset of observatory

    dt : TimeDelta
        Desired timestep of output "day" table.  3*u.day gives good
        performance for surface brightness

    """

    if t_torus_in is None:
        t_torus_in = os.path.join(IoIO_ROOT, 'Torus', 'Torus.ecsv')
    if isinstance(t_torus_in, str):
        t_torus = QTable.read(t_torus_in)
    else:
        t_torus = t_torus_in.copy()

    max_jd_gap = max_time_gap.to(u.day).value
    jd_dt = dt.to(u.day).value

    # Fill original table with values we need for day_table
    add_itime_col(t_torus, time_col='tavg', itime_col='ijdlt',
                  ut_offset=ut_offset, dt=jd_dt)
    t_torus['jd'] = t_torus['tavg'].jd
    biweight_cols = ['jd', 'obj_surf_bright',
                     'ansa_left_surf_bright', 'ansa_right_surf_bright',
                     'ansa_left_r_peak', 'ansa_right_r_peak']
    if 'epsilon' in t_torus.colnames:
        biweight_cols.extend(['epsilon', 'epsilon_err'])
    added_cols = add_daily_biweights(t_torus,
                                     day_col='ijdlt',
                                     colnames=biweight_cols)
    # Create day_table
    day_table_cols = ['ijdlt'] + added_cols
    day_table = unique(t_torus[day_table_cols], keys='ijdlt')
    cdts = contiguous_sections(day_table, 'ijdlt', max_jd_gap)


    icdts = linspace_day_table(cdts, algorithm='interpolate_replace_nans',
                               max_gap_fraction=max_gap_fraction,
                               time_col_offset=ut_offset,
                               dt=dt)
    add_medfilt_columns(icdts, colnames=added_cols,
                        medfilt_width=jd_medfilt_width,
                        mode=jd_medfilt_mode)

    # --> I might want to do other operations in here and maybe even
    # --> have a function I pass in to do that
    
    # Put it all back to together again 
    torus_day_table = QTable()
    for icdt in icdts:
        torus_day_table = vstack([torus_day_table, icdt])

    # Give us a time column back
    jds = torus_day_table['biweight_jd']
    loc = t_torus['tavg'][0].location.copy()
    tavgs = Time(jds, format='jd', location=loc)
    tavgs.format = 'fits'
    torus_day_table['tavg'] = tavgs

    return torus_day_table

# --> A better way to do this might be piecewise.
def add_medfilt(t, colname, mask_col='mask', medfilt_width=21, mode='mirror'):
    """TABLE MUST BE SORTED FIRST"""
    
    if len(t) < medfilt_width/2:
        return
    if mask_col in t.colnames:
        bad_mask = t[mask_col]
    else:
        bad_mask = False
    # mirror might be a little more physical than reflect, though
    # edges are just hard, period
    meds = nan_median_filter(t[colname], mask=bad_mask,
                             size=medfilt_width, mode=mode)
    #bad_mask = np.logical_or(bad_mask, np.isnan(t[colname]))
    #vals = t[colname][~bad_mask]
    #meds = medfilt(vals, medfilt_width)
    #meds = median_filter(vals, size=medfilt_width, mode='reflect')
    #t[f'{colname}_medfilt'][~bad_mask] = meds
    t[f'{colname}_medfilt'] = meds
                  
def plot_axhlines(ax, hlines=None, **kwargs):
    if hlines is None:
        return
    for hline in hlines:
        ax.axhline(hline)

def plot_axvlines(ax, vlines=None, **kwargs):
    if vlines is None:
        return
    for vline in vlines:
        if isinstance(vline, tuple):
            vline, color = vline
        else:
            color = None
        vline = date.fromisoformat(vline)
        ax.axvline(vline, color=color)

def plot_column_vals(t,
                     colnames,
                     scale=None,
                     fmts=None,
                     labels=None,
                     ylabel=None,
                     medfilt_width=21,
                     medfilt_colname=None, # medfilt col to plot
                     medfilt_collabel=None,
                     fig=None,
                     ax=None,
                     top_axis=False,
                     ylim=None,
                     tlim=None,
                     legend=True,
                     show=False,
                     max_night_gap=20,
                     **kwargs): # These are passed to plot_[hv]lines

    if fig is None:
        fig = plt.figure()
    if ax is None:
        ax = fig.add_subplot()
    if scale is None:
        scale = np.full(len(colnames), 1)

    t = t.copy()
    t.sort('tavg')

    biweights = []
    mads = []
    for ic, colname in enumerate(colnames):
        add_medfilt(t, colname, medfilt_width=medfilt_width)
        biweights.append(nan_biweight(scale[ic] * t[colname]))
        mads.append(nan_mad(t[colname]))
        log.info(f'{colname} * {scale[ic]} Biweight +/- MAD = {biweights[-1]} +/- {mads[-1]}')

    datetimes = t['tavg'].datetime
    if len(t) > 40 and medfilt_colname:
        # Plot the median filter of the largest signal
        ic = colnames.index(medfilt_colname)
        if scale[ic] == 1:
            scale_str = ''
        else:
            scale_str = f'{scale[ic]} * '
        alpha = 0.1
        p_med = ax.plot(datetimes,
                        scale[ic] * t[f'{medfilt_colname}_medfilt'],
                        'k*', markersize=6,
                        label=scale_str + medfilt_collabel)
    else:
        alpha = 0.5
        p_med = None

    handles = []
    for ic, colname in enumerate(colnames):
        if scale[ic] == 1:
            scale_str = ''
        else:
            scale_str = f'{scale[ic]} * '
        # --> This only works for masked array.  I should slow down
        # --> and create val and val_err with the appropriate typing
        # --> logic
        h = ax.errorbar(datetimes,
                        scale[ic] * t[colname].filled(np.NAN),
                        abs(scale[ic]) * t[f'{colname}_err'].filled(np.NAN),
                        fmt=fmts[ic], alpha=alpha,
                        label=scale_str + labels[ic])
        handles.append(h)
    if p_med:
        # This gets the legend in the correct order
        handles.extend(p_med)
    if tlim is None:
        tlim = ax.get_xlim()
    ax.set_xlabel('Date')
    ax.set_ylabel(ylabel)
    if legend:
        ax.legend(handles=handles)

    ax.set_xlim(tlim)
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.set_ylim(ylim)
    fig.autofmt_xdate()
    ax.format_coord = PJAXFormatter(datetimes, t[colnames[0]])

    plot_axhlines(ax, **kwargs)
    plot_axvlines(ax, **kwargs)

    if top_axis:
        jts = JunoTimes()
        secax = ax.secondary_xaxis('top',
                                   functions=(jts.plt_date2pj, jts.pj2plt_date))
        secax.xaxis.set_minor_locator(MultipleLocator(1))
        secax.set_xlabel('PJ')

    return t

def plot_ansa_brights(t, **kwargs):

    t = plot_column_vals(t, colnames=['ansa_left_surf_bright',
                                      'ansa_right_surf_bright'],
                         fmts=['b.', 'r.'],
                         labels=['Dawn', 'Dusk'],
                         ylabel=f'IPT Ribbon Surf. Bright '\
                         f'({t["ansa_left_surf_bright"].unit})',
                         medfilt_colname='ansa_right_surf_bright',
                         medfilt_collabel='Dusk medfilt',
                         **kwargs)
    return t
                     

def plot_torus_surf_bright(t_torus, torus_day_table,
                           fig=None,
                           ax=None):

    if fig is None:
        fig = plt.figure()
    if ax is None:
        ax = fig.add_subplot()
    plot_handles = plot_columns(
        torus_day_table,
        colnames=['biweight_ansa_left_surf_bright',
                  'biweight_ansa_right_surf_bright',
                  'biweight_ansa_left_surf_bright_medfilt',
                  'biweight_ansa_right_surf_bright_medfilt'],
        fmts=['b.', 'r.', 'b', 'r'],
        labels=['Dawn daily interp',
                'Dusk daily interp', 'Dawn medfilt', 'Dusk medfilt'],
        fig=fig,
        ax=ax)
    errorbar_handles = plot_columns(
        t_torus,
        colnames=['ansa_left_surf_bright', 'ansa_right_surf_bright'],
        fmts=['b.', 'r.'],
        labels=['Dawn', 'Dusk'],
        alphas=0.1,
        fig=fig,
        ax=ax)

    # Get the legend to read in the order I prefer
    handles = errorbar_handles
    handles.extend(plot_handles)
    ylabel=f'IPT Ribbon Surf. Bright '\
        f'({torus_day_table["biweight_ansa_left_surf_bright"].unit})'
    ax.set_ylabel(ylabel)
    ax.set_ylim()
    ax.legend(ncol=2, handles=handles)
    juno_pj_axis(ax)

def add_epsilon_cols(t,
                     outbase='',
                     prefix='',
                     postfix='',
                     err_prefix='',
                     err_postfix='',
                     out_prefix=None,
                     out_err_postfix=None):
    """Add epsilon column

    """
    if out_prefix is None:
        out_prefix = prefix
    if out_err_postfix is None:
        out_err_postfix = err_postfix
    right_col = f'{prefix}ansa_right_r_peak{postfix}'
    left_col = f'{prefix}ansa_left_r_peak{postfix}'
    r_peak = t[right_col]
    l_peak = t[left_col]
    av_peak = (np.abs(r_peak) + np.abs(l_peak)) / 2
    epsilon = -(r_peak + l_peak) / av_peak
    t[f'{out_prefix}{outbase}{postfix}'] = epsilon
    left_err = t[f'{err_prefix}ansa_left_r_peak{err_postfix}']
    right_err = t[f'{err_prefix}ansa_right_r_peak{err_postfix}']
    denom_var = left_err**2 + right_err**2
    num_var = denom_var / 2
    epsilon_err = epsilon * ((denom_var / (r_peak + l_peak)**2)
                             + (num_var / av_peak**2))**0.5
    epsilon_err = np.abs(epsilon_err)
    t[f'{out_prefix}{outbase}{postfix}{out_err_postfix}'] = epsilon_err

def add_epsilons(t, ansa_medfilt_width=21, epsilon_medfilt_width=11):
    # Table must be sorted first
    add_medfilt(t, 'ansa_right_r_peak', medfilt_width=ansa_medfilt_width)
    add_medfilt(t, 'ansa_left_r_peak', medfilt_width=ansa_medfilt_width)

    r_peak = t['ansa_right_r_peak']
    l_peak = t['ansa_left_r_peak']
    av_peak = (np.abs(r_peak) + np.abs(l_peak)) / 2
    # Current values for epsilon are messed up because peaks are not
    # coming in at the right places presumably due to the simple
    # Gaussian modeling.  The offset should be to the left (east,
    # dawn) such that Io dips inside the IPT on that side.  To prepare
    # for a sensible answer, cast my r_peak in left-handed coordinate
    # system so epsilon is positive if l_peak is larger
    epsilon = -(r_peak + l_peak) / av_peak
    denom_var = t['ansa_left_r_peak_err']**2 + t['ansa_right_r_peak_err']**2
    num_var = denom_var / 2
    epsilon_err = epsilon * ((denom_var / (r_peak + l_peak)**2)
                             + (num_var / av_peak**2))**0.5
    t['epsilon'] = epsilon
    t['epsilon_err'] = epsilon_err
    add_medfilt(t, 'epsilon', medfilt_width=epsilon_medfilt_width)
    kernel = Gaussian1DKernel(10)
    interpolate_replace_nans_columns(t, ['ansa_right_r_peak_medfilt',
                                         'ansa_left_r_peak_medfilt'],
                                     kernel)
    r_med_peak = t['ansa_right_r_peak_medfilt_interp']
    l_med_peak = t['ansa_left_r_peak_medfilt_interp']
    av_med_peak = (np.abs(r_med_peak) + np.abs(l_med_peak)) / 2
    t['medfilt_interp_epsilon'] = -(r_med_peak + l_med_peak) / av_med_peak

def plot_epsilons(t,
                  fig=None,
                  ax=None,
                  medfilt_width=21,
                  min_eps=-0.015,
                  max_eps=0.06,
                  tlim=None,
                  legend=True,
                  **kwargs):
    if fig is None:
        fig = plt.figure()
    if ax is None:
        ax = plt.subplot()

    t = t.copy()
    add_medfilt(t, 'ansa_right_r_peak', medfilt_width=medfilt_width)
    add_medfilt(t, 'ansa_left_r_peak', medfilt_width=medfilt_width)

    r_peak = t['ansa_right_r_peak']
    l_peak = t['ansa_left_r_peak']
    av_peak = (np.abs(r_peak) + np.abs(l_peak)) / 2
    # Current values for epsilon are messed up because peaks are not
    # coming in at the right places presumably due to the simple
    # Gaussian modeling.  The offset should be to the left (east,
    # dawn) such that Io dips inside the IPT on that side.  To prepare
    # for a sensible answer, cast my r_peak in left-handed coordinate
    # system so epsilon is positive if l_peak is larger
    epsilon = -(r_peak + l_peak) / av_peak
    denom_var = t['ansa_left_r_peak_err']**2 + t['ansa_right_r_peak_err']**2
    num_var = denom_var / 2
    epsilon_err = epsilon * ((denom_var / (r_peak + l_peak)**2)
                             + (num_var / av_peak**2))**0.5
    epsilon = epsilon.filled(np.NAN)
    epsilon_err = np.abs(epsilon_err.filled(np.NAN))
    epsilon_biweight = nan_biweight(epsilon)
    epsilon_mad = nan_mad(epsilon)

    bad_mask = np.isnan(epsilon)
    good_epsilons = epsilon[~bad_mask]

    if len(good_epsilons) > 20:
        alpha = 0.2
        # --> Medfilt is not the best option here, median_filter is,
        # but I am changing this implementation
        med_epsilon = medfilt(good_epsilons, 11)
        p_med = ax.plot(t['tavg'][~bad_mask].datetime, med_epsilon,
                         'k*', markersize=6, label='Medfilt')
        # These don't overlap anywhere 
        #r_med_peak = t['ansa_right_r_peak_medfilt']
        #l_med_peak = -t['ansa_left_r_peak_medfilt']
        #av_med_peak = (np.abs(r_med_peak) + np.abs(l_med_peak)) / 2
        #medfilt_epsilon = -(r_med_peak + l_med_peak) / av_med_peak
        #plt.plot(t['tavg'].datetime, medfilt_epsilon,
        #         'k*', markersize=6, label='Epsilon from medfilts')
        kernel = Gaussian1DKernel(10)
        interpolate_replace_nans_columns(t, ['ansa_right_r_peak_medfilt',
                                             'ansa_left_r_peak_medfilt'],
                                         kernel)
        r_med_peak = t['ansa_right_r_peak_medfilt_interp']
        l_med_peak = t['ansa_left_r_peak_medfilt_interp']
        av_med_peak = (np.abs(r_med_peak) + np.abs(l_med_peak)) / 2
        t['medfilt_interp_epsilon'] = -(r_med_peak + l_med_peak) / av_med_peak  
        p_interps = ax.plot(t['tavg'].datetime, t['medfilt_interp_epsilon'],
                             'c*', markersize=3, label='From interpolations')
    else:
        alpha = 0.5
        p_med = None
       
    p_eps = ax.errorbar(t['tavg'].datetime,
                        epsilon,
                        epsilon_err, fmt='k.', alpha=alpha,
                        label='Epsilon')

    handles = [p_eps]
    if p_med:
        handles.extend([p_med[0], p_interps[0]])
    ax.set_ylabel(r'Sky plane $|\vec\epsilon|$ (dawnward)')
    if tlim is None:
        tlim = (None, None)
    ax.hlines(np.asarray((epsilon_biweight,
                          epsilon_biweight - epsilon_mad,
                          epsilon_biweight + epsilon_mad)),
              *tlim,
              linestyles=('-', '--', '--'),
              label=f'{epsilon_biweight:.3f} +/- {epsilon_mad:.3f}')
    ax.axhline(0.025, color='y', label='Nominal 0.025')
    if legend:
        ax.legend(handles=handles)
    #plt.title('Epsilon')
    ax.set_xlim(tlim)
    ax.set_ylim(min_eps, max_eps)
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    fig.autofmt_xdate()
    ax.format_coord = PJAXFormatter(t['tavg'].datetime, epsilon)

    plot_axhlines(ax, **kwargs)
    plot_axvlines(ax, **kwargs)

    #jts = JunoTimes()
    #secax = ax.secondary_xaxis('top',
    #                            functions=(jts.plt_date2pj, jts.pj2plt_date))
    #secax.xaxis.set_minor_locator(MultipleLocator(1))
    #secax.set_xlabel('PJ')

    return t

def plot_ansa_pos(t,
                  fig=None,
                  ax=None,
                  tlim=None,
                  show=False,
                  medfilt_width=21,
                  legend=True,
                  **kwargs):
    if fig is None:
        fig = plt.figure()
    if ax is None:
        ax = plt.subplot()

    t = t.copy()
    add_medfilt(t, 'ansa_right_r_peak', medfilt_width=medfilt_width)
    add_medfilt(t, 'ansa_left_r_peak', medfilt_width=medfilt_width)
    # Make it clear we are plotting the perturbation from Io's orbital
    # position on the right and left sides of Jupiter.  We will make t
    # eastward (negative of these), when plotting
    rights = t['ansa_right_r_peak'].filled(np.NAN) - IO_ORBIT_R
    lefts = t['ansa_left_r_peak'].filled(np.NAN) - (-IO_ORBIT_R)
    rights_err = t['ansa_right_r_peak_err'].filled(np.NAN)
    lefts_err = t['ansa_left_r_peak_err'].filled(np.NAN)
    right_bad_mask = np.logical_or(np.isnan(rights), np.isnan(rights_err))
    left_bad_mask = np.logical_or(np.isnan(lefts), np.isnan(lefts_err))
    rights = rights[~right_bad_mask]
    rights_err = rights_err[~right_bad_mask]
    lefts = lefts[~left_bad_mask]
    lefts_err = lefts_err[~left_bad_mask]

    medfilt_handles = []
    if len(rights) and len(lefts) > 40:
        alpha = 0.2
        h = ax.plot(t['tavg'].datetime,
                     -(t['ansa_left_r_peak_medfilt'] - (-IO_ORBIT_R)),
                     'k*', markersize=12, label='Dawn medfilt')
        medfilt_handles.append(h[0])
        h = ax.plot(t['tavg'].datetime,
                     -(t['ansa_right_r_peak_medfilt'] - IO_ORBIT_R),
                     'k+', markersize=12, label='Dusk medfilt')
        medfilt_handles.append(h[0])
    else:
        alpha = 0.5
    point_handles = []
    h = ax.errorbar(t['tavg'][~left_bad_mask].datetime,
                     -lefts, lefts_err, fmt='b.',
                     label='Dawn', alpha=alpha)
    point_handles.append(h)
    h = ax.errorbar(t['tavg'][~right_bad_mask].datetime,
                     -rights, rights_err, fmt='r.',
                     label='Dusk', alpha=alpha)
    point_handles.append(h)
    handles = point_handles
    handles.extend(medfilt_handles)

    #plt.plot(t['tavg'].datetime,
    #         np.abs(t['ansa_right_r_peak']) + t['ansa_right_r_stddev'],
    #         'g^')
    ax.set_ylabel(r'Dawnward ribbon shift from Io orbit (R$_\mathrm{J}$)')
    ax.axhline(0, color='y', label='Io orbit')
    if legend:
        ax.legend(handles=handles, ncol=2)
    ax.set_xlim(tlim)
    ax.set_ylim(-0.3, 0.4)
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    fig.autofmt_xdate()
    ax.format_coord = PJAXFormatter(t['tavg'].datetime,
                                    -(t['ansa_left_r_peak_medfilt'] +
                                      IO_ORBIT_R))
    plot_axhlines(ax, **kwargs)
    plot_axvlines(ax, **kwargs)

    #jts = JunoTimes()
    #secax = ax.secondary_xaxis('top',
    #                            functions=(jts.plt_date2pj, jts.pj2plt_date))
    #secax.xaxis.set_minor_locator(MultipleLocator(1))
    #secax.set_xlabel('PJ')    

    return t

def plot_ansa_r_amplitudes(t, **kwargs):
    plot_column_vals(t, colnames=['ansa_left_r_amplitude',
                                  'ansa_right_r_amplitude'],
                     fmts=['b.', 'r.'],
                     labels=['Dawn', 'Dusk'],
                     ylabel='r Gauss amplitude ' \
                     f'({t["ansa_left_r_amplitude"].unit})',
                     medfilt_colname='ansa_right_r_amplitude',
                     medfilt_collabel='Dusk medfilt',                    
                     **kwargs)

def plot_ansa_r_stddevs(t, **kwargs):
    # Too mixed in with large error bars
    plot_column_vals(t, colnames=['ansa_left_r_stddev',
                                  'ansa_right_r_stddev'],
                     fmts=['b.', 'r.'],
                     labels=['Dawn', 'Dusk'],
                     ylabel='r Gauss stddev ' \
                     f'({t["ansa_left_r_stddev"].unit})',
                     medfilt_colname='ansa_right_r_stddev',
                     medfilt_collabel='Dusk medfilt',
                     **kwargs)

def plot_dawn_r_stddev(t, **kwargs):
    # Dawn and dusk modulations at the 0.15 Rj level, but they are not
    # convincingly correlated with peak positions.  They are corelated
    # with surface brightness & r amplitude in some cases, possibly
    # via the bright = broad correlation that Nick has noted
    plot_column_vals(t, colnames=['ansa_left_r_stddev'],
                     fmts=['b.'],
                     labels=['Dawn'],
                     ylabel='r Gauss stddev ' \
                     f'({t["ansa_left_r_stddev"].unit})',
                     medfilt_colname='ansa_left_r_stddev',
                     medfilt_collabel='Dawn medfilt',
                     **kwargs)
def plot_dusk_r_stddev(t, **kwargs):
    plot_column_vals(t, colnames=['ansa_right_r_stddev'],
                     fmts=['r.'],
                     labels=['Dusk'],
                     ylabel='r Gauss stddev ' \
                     f'({t["ansa_right_r_stddev"].unit})',
                     medfilt_colname='ansa_right_r_stddev',
                     medfilt_collabel='Dusk medfilt',
                     **kwargs)

def plot_ansa_y_peaks(t, **kwargs):
    # This shows flat trend
    plot_column_vals(t, colnames=['ansa_left_y_peak',
                                  'ansa_right_y_peak'],
                     fmts=['b.', 'r.'],
                     labels=['Dawn', 'Dusk'],
                     ylabel='y peak pos' \
                     f'({t["ansa_left_y_peak"].unit})',
                     medfilt_colname=None,
                     **kwargs)

def plot_dawn_y_stddev(t, **kwargs):
    plot_column_vals(t, colnames=['ansa_left_y_stddev'],
                     fmts=['b.'],
                     labels=['Dawn'],
                     ylabel='y Gauss stddev' \
                     f'({t["ansa_left_y_stddev"].unit})',
                     medfilt_colname='ansa_left_y_stddev',
                     medfilt_collabel='Dawn medfilt',
                     **kwargs)
def plot_dusk_y_stddev(t, **kwargs):
    plot_column_vals(t, colnames=['ansa_right_y_stddev'],
                     fmts=['r.'],
                     labels=['Dusk'],
                     ylabel='y Gauss stddev' \
                     f'({t["ansa_right_y_stddev"].unit})',
                     medfilt_colname='ansa_right_y_stddev',
                     medfilt_collabel='Dusk medfilt',
                     **kwargs)

def plot_dawn_cont(t, **kwargs):
    plot_column_vals(t, colnames=['ansa_left_cont'],
                     fmts=['b.'],
                     labels=['Dawn'],
                     ylabel='r continuum' \
                     f'({t["ansa_left_cont"].unit})',
                     medfilt_colname='ansa_left_cont',
                     medfilt_collabel='Dawn medfilt',
                     **kwargs)
def plot_dusk_cont(t, **kwargs):
    plot_column_vals(t, colnames=['ansa_right_cont'],
                     fmts=['r.'],
                     labels=['Dusk'],
                     ylabel='r continuum' \
                     f'({t["ansa_right_cont"].unit})',
                     medfilt_colname='ansa_right_cont',
                     medfilt_collabel='Dusk medfilt',
                     **kwargs)

def plot_dawn_slope(t, **kwargs):
    plot_column_vals(t, colnames=['ansa_left_slope'],
                     fmts=['b.'],
                     labels=['Dawn'],
                     ylabel='r slope' \
                     f'({t["ansa_left_slope"].unit})',
                     medfilt_colname='ansa_left_slope',
                     medfilt_collabel='Dawn medfilt',
                     **kwargs)
def plot_dusk_slope(t, **kwargs):
    plot_column_vals(t, colnames=['ansa_right_slope'],
                     scale=[-1],
                     fmts=['r.'],
                     labels=['Dusk'],
                     ylabel='r slope' \
                     f'({t["ansa_right_slope"].unit})',
                     medfilt_colname='ansa_right_slope',
                     medfilt_collabel='Dusk medfilt',
                     **kwargs)

#--> This is obsolete
def torus_stripchart(t, outdir,
                     figsize=None,
                     nplots=6,
                     tlim=None,
                     show=False):
    """Intended for nightly plots"""
    if nplots == 2:
            figsize = [11, 11]
    else:
        figsize = [21, 18]
    outbase = 'Characterize_Ansas.png'

    # Hints from https://matplotlib.org/stable/gallery/subplots_axes_and_figures/subplots_demo.html
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(nplots, hspace=0)
    axs = gs.subplots(sharex=True)
    plot_ansa_brights(t,
                      fig=fig, ax=axs[1],
                      tlim=tlim)
    if nplots > 1:
        plot_epsilons(t, fig=fig, ax=axs[2],
                      tlim=tlim)
    if nplots > 2:
        plot_ansa_pos(t, fig=fig, ax=axs[3],
                      tlim=tlim)
    if nplots > 3:
        axs[4].plot(t['tavg'].datetime, t['closest_galsat'], 'k.')
        axs[4].set_xlim(tlim)
        axs[4].set_ylabel(r'Closest galsat (R$_\mathrm{J}$)')
        axs[4].set_xlabel(f'UT {date}')
    if nplots > 4:
        dawn_sysIII = Angle(t['Jupiter_PDObsLon'] + 90*u.deg)
        dusk_sysIII = Angle(t['Jupiter_PDObsLon'] - 90*u.deg)
        dawn_sysIII = dawn_sysIII.wrap_at(360*u.deg)
        dusk_sysIII = dusk_sysIII.wrap_at(360*u.deg)

        axs[5].errorbar(dawn_sysIII.filled(np.NAN),
                     t['ansa_left_surf_bright'].filled(np.NAN),
                     t['ansa_left_surf_bright_err'].filled(np.NAN), fmt='b.',
                     label='Dawn')
        axs[5].errorbar(dusk_sysIII.filled(np.NAN),
                     t['ansa_right_surf_bright'].filled(np.NAN),
                     t['ansa_right_surf_bright_err'].filled(np.NAN), fmt='r.',
                     label='Dusk')
        axs[5].set_ylabel(f'Surf. Bright ({t["ansa_left_surf_bright"].unit})')
        axs[5].set_xlabel(r'Ansa $\lambda{\mathrm{III}}$')
        axs[5].set_xticks(np.arange(0,360,45))
        axs[5].minorticks_on()
        axs[5].set_xlim(0, 360)
        axs[5].legend()

    finish_stripchart(outdir, outbase, show=show)
    

def torus_directory(directory,
                    outdir=None,
                    outdir_root=TORUS_ROOT,
                    standard_star_obj=None,
                    read_pout=True,
                    write_pout=True,
                    create_outdir=True,
                    **kwargs):
    standard_star_obj = standard_star_obj or StandardStar(reduce=True)
    outdir = outdir or reduced_dir(directory, outdir_root, create=False)
    poutname = os.path.join(outdir, 'Torus.pout')
    pout = cached_pout(on_off_pipeline,
                       poutname=poutname,
                       read_pout=read_pout,
                       write_pout=write_pout,
                       create_outdir=create_outdir,
                       directory=directory,
                       glob_include=TORUS_NA_NEB_GLOB_LIST,
                       glob_exclude_list=TORUS_NA_NEB_GLOB_EXCLUDE_LIST,
                       band='SII',
                       standard_star_obj=standard_star_obj,
                       add_ephemeris=galsat_ephemeris,
                       planet='Jupiter',
                       crop_ccd_coord=SMALL_FILT_CROP,
                       post_process_list=[tavg_to_bmp_meta,
                                          weather_to_bmp_meta,
                                          calc_obj_to_ND, crop_ccd,
                                          planet_to_object],
                       plot_planet_rot_from_key=['Jupiter_NPole_ang'],
                       planet_subim_dx=10*u.R_jup,
                       planet_subim_dy=5*u.R_jup,                       
                       post_offsub=[extinction_correct, rayleigh_convert,
                                    obj_surface_bright, mean_image,
                                    characterize_ansas,
                                    closest_galsat_to_jupiter,
                                    plot_planet_subim, as_single],
                       outdir=outdir,
                       **kwargs)
    if pout is None or len(pout) == 0:
        return QTable()

    _ , pipe_meta = zip(*pout)
    t = QTable(rows=pipe_meta)
    add_mask_col(t)
    # --> Fix plotting
    #torus_stripchart(t, outdir)
    return t

def torus_tree(raw_data_root=RAW_DATA_ROOT,
               start=None,
               stop=None,
               calibration=None,
               cor_boltwood=None,
               photometry=None,
               keep_intermediate=None,
               solve_timeout=SOLVE_TIMEOUT,
               join_tolerance=JOIN_TOLERANCE*JOIN_TOLERANCE_UNIT,
               standard_star_obj=None,
               read_csvs=True,
               write_csvs=True,
               show=False,
               create_outdir=True,                       
               outdir_root=TORUS_ROOT,
               **kwargs):

    dirs_dates = get_dirs_dates(raw_data_root, start=start, stop=stop)
    dirs, _ = zip(*dirs_dates)
    if len(dirs) == 0:
        log.warning('No directories found')
        return
    calibration = calibration or Calibration()
    cor_boltwood = cor_boltwood or CorBoltwood(precalc=True)
    if photometry is None:
        photometry = CorPhotometry(
            precalc=True,
            solve_timeout=solve_timeout,
            join_tolerance=join_tolerance)
    standard_star_obj = standard_star_obj or StandardStar(reduce=True)

    summary_table = QTable()
    for directory in dirs:
        rd = reduced_dir(directory, outdir_root, create=False)
        t = cached_csv(directory,
                       code=torus_directory,
                       csvnames=os.path.join(rd, 'Characterize_Ansas.ecsv'),
                       calibration=calibration,
                       cor_boltwood=cor_boltwood,
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
    summary_table.sort('tavg')
    summary_table.write(os.path.join(outdir_root, 'Torus.ecsv'),
                                     overwrite=True)
    #torus_stripchart(summary_table, outdir_root, show=show)

    return summary_table

def export_for_max(directory,
                   start=None, # not used yet
                   stop=None,
                   max_root=MAX_ROOT):
    # Develop export_for_max
    flist = multi_glob(directory, '*-back-sub.fits')
    if flist is None:
        return
    outdir = reduced_dir(directory, max_root, create=True)
    flist.sort()
    for fname in flist:
        outbase = os.path.basename(fname)
        outfname, ext = os.path.splitext(outbase)
        outfname = os.path.join(outdir, outfname + '_NPole_up_small' + ext)
        log.info(outfname)
        ccd = CorData.read(fname)
        ccd = rot_to(ccd, rot_angle_from_key='Jupiter_NPole_ang')
        ccd.mask = None
        ccd.uncertainty = None
        ccd = as_single(ccd)
        ccd.write(outfname, overwrite=True)

class SecAxFormatter(object):
    """Provides second Y axis value interactive pyplot window"""
    def __init__(self, frequency, power):
        self.frequency = frequency
        self.power = power
    def __call__(self, frequency, power):
        frequency *= self.frequency.unit
        return \
            f'freq: {frequency:0.4f} ' \
            f'power: {power:0.2f} ' \
            f'period: {1/frequency:0.2f} ' \
            f'period: {(1/frequency).to(u.day):0.2f}'
        #return f'{frequency} {power} {1/frequency}'

#def one_over(x):
#    """Vectorized 1/x, treating x==0 manually"""
#    x = np.array(x).astype(float)
#    near_zero = np.isclose(x, 0)
#    x[near_zero] = np.inf
#    x[~near_zero] = 1 / x[~near_zero]
#    return x
#
#inverse = one_over
#
#fig, ax = plt.subplots()
#ax.plot(frequency, power)
#ax.set_xlim((min(frequency.value), max(frequency.value)))
#ax.format_coord = CCDImageFormatter()
##ax.set_xscale('log')
##ax.loglog(frequency.value, power)
#ax.set_xlabel('Frequency ' + frequency.unit.to_string())
##ax.set_xlim((1/1000, .2))
#plt.axvline(1/SYSIII.value, color='orange')
#plt.axvline(1/SYSIV_LOW.value, color='green')
#plt.axvline(1/SYSIV_HIGH.value, color='green', linestyle='--')
#secax = ax.secondary_xaxis('top', functions=(one_over, inverse))
##secax.set_xscale('log')
#secax.set_xlabel('Period ' + (1/frequency).unit.to_string())
##secax.xaxis.set_minor_locator(AutoMinorLocator())
#plt.show()

def one_over(x):
    """Vectorized 1/x, treating x==0 manually"""
    x = np.array(x).astype(float)
    near_zero = np.isclose(x, 0)
    x[near_zero] = np.inf
    x[~near_zero] = 1 / x[~near_zero]
    return x

def periodogram(start, stop,
                autopower=False,
                min_period=9.5*u.hr,
                max_period=11*u.hr,
                plot_type='period',
                side=None,
                torus_table='/data/IoIO/Torus/Torus.ecsv',
                false_alarm_level=0.001,
                nfreq=25000,
                plotdir=None):

    t = QTable.read(torus_table)
    t.sort('tavg')

    if side == 'west':
        side = 'right'
    elif side == 'dusk':
        side = 'right'
    elif side == 'east':
        side = 'left'
    elif side == 'dawn':
        side = 'left'
    if side == 'right':
        side_str = 'Dusk'
    elif side == 'left':
        side_str = 'Dawn'
    else:
        raise ValueError(f'Invalid side: {side}')
    #Io_freqs = np.asarray((0, 0.5, 1, 3/2, 1.75, 2))
    Io_freqs = np.asarray((0, 1, 3/2, 1.75))
    Io_freqs = Io_freqs / IO_ORBIT
    #Europa_freqs = np.asarray((0, 0.5, 1, 3/2, 1.75, 2))
    Europa_freqs = np.asarray((0, 1, 3/2))
    Europa_freqs = Europa_freqs / EUROPA_ORBIT
    #Ganymede_freqs = np.asarray((0, 0.5, 1, 3/2, 1.75, 2))
    Ganymede_freqs = np.asarray((0, 1, 3/2))
    Ganymede_freqs = Ganymede_freqs / GANYMEDE_ORBIT
    #Callisto_freqs = np.asarray((0, 0.5, 1, 3/2, 1.75, 2))
    Callisto_freqs = np.asarray((0, 0.5, 1, 3/2))
    #Callisto_freqs = np.asarray((0,))
    Callisto_freqs = Callisto_freqs / CALLISTO_ORBIT
    n_day_samples = 24*u.hr/min_period
    day_freqs =  np.arange(n_day_samples)
    day_freqs = day_freqs / (24*u.hr)
    day_freqs[0] = 0
    n_freqs = (len(Io_freqs)
               * len(Europa_freqs)
               * len(Ganymede_freqs)
               * len(Callisto_freqs)
               * len(day_freqs))
    assert Io_freqs.unit == day_freqs.unit
    obs_freqs = np.zeros(n_freqs) * Io_freqs.unit
    i = 0
    obs_freq = 0
    ## Trying some dispassionate way to set limit
    #moon_freq = 200
    #obs_freq = 0
    #for day in day_freqs:
    #    obs_freq = day
    #    for Io in Io_freqs:
    #        if obs_freqs[i-1] + obs_freq + Io < moon_freq * Io:
    #            obs_freq += Io
    #        for Europa in Europa_freqs:
    #            if obs_freqs[i-1] + obs_freq + Europa < moon_freq * Europa:
    #                obs_freq += Europa
    #            for Ganymede in Ganymede_freqs:
    #                if obs_freqs[i-1] + obs_freq + Ganymede < moon_freq * Ganymede:
    #                    obs_freq += Ganymede
    #                for Callisto in Callisto_freqs:
    #                    if obs_freqs[i-1] + obs_freq + Callisto < moon_freq * Callisto:
    #                        obs_freq += Callisto
    #                    obs_freqs[i] = obs_freq
    #                    i += 1
    for day in day_freqs:
        for Io in Io_freqs:
            if (Io > 1 / IO_ORBIT
                and obs_freq > 0.900/u.hr):
                Io = 0
            for Europa in Europa_freqs:
                #if Europa > 0 and obs_freq > 0.076/u.hr:
                #    Interference at 10.25hr
                #    continue
                if (Europa > 1 / EUROPA_ORBIT
                    and obs_freq > 0.0195/u.hr):
                    Europa = 0
                for Ganymede in Ganymede_freqs:
                    if (Ganymede > 1 / GANYMEDE_ORBIT
                        and obs_freq > 0.0195/u.hr):
                        #and obs_freq > 0.040/u.hr):
                        #if Ganymede > 0 and obs_freq > 0.032/u.hr:
                        Ganymede = 0
                    for Callisto in Callisto_freqs:
                        if (Callisto > 1 / CALLISTO_ORBIT
                            and obs_freq > 0.0195/u.hr
                            or Callisto > 0 / CALLISTO_ORBIT
                            and obs_freq > 0.0923/u.hr):
                            Callisto = 0
                        obs_freq = Io + Europa + Ganymede + Callisto + day
                        obs_freqs[i] = obs_freq
                        i += 1
                            # --> I need to make this is adjusted for 1lt to Jupiter I might need
                            # --> to phase by sysIII to the western ansa from the perspective of
                            # --> the sun

    mask = np.logical_and(start < t['tavg'], t['tavg'] < stop)
    mask = np.logical_and(mask, np.isfinite(t[f'ansa_{side}_surf_bright']))
    t = t[mask]
    #plt.plot(t['tavg'].jd, t['ansa_right_surf_bright'])
    #plt.show()
    t_in_hr = t['tavg'] - t['tavg'][0]
    t_in_hr = t_in_hr.to(u.hr)
    #t_in_hr = t_in_hr.to(u.day)
    #plt.plot(t_in_hr, t['ansa_right_surf_bright'])
    #plt.show()

    # --> the problem of double-peaks is autopower. I want to specify
    # --> power with my own frequency grid
    # https://docs.astropy.org/en/stable/timeseries/lombscargle.html
    # details


    if autopower:
        ls = LombScargle(t_in_hr,
                         t[f'ansa_{side}_surf_bright'],
                         t[f'ansa_{side}_surf_bright_err'])
        frequency, power = ls.autopower()
    else:
        # This just saves computation time, which is negligible.  Aliasing
        # is the same (or non-exisitant), at least in the sysIII/IV area
        frequency = np.linspace(1/max_period, 1/min_period, nfreq)
        ls = LombScargle(t_in_hr,
                         t[f'ansa_{side}_surf_bright'],
                         t[f'ansa_{side}_surf_bright_err'],
                         t[f'ansa_{side}_surf_bright_err'])
        power = ls.power(frequency, method='slow')

    fig,ax = plt.subplots(figsize=[22, 17])
    start_str, _ = start.fits.split('T')
    stop_str, _ = stop.fits.split('T')
    ax.set_title(f'{side_str} {start_str} -- {stop_str}')
    # Awkward because of how Quntities are handled
    #Io_alias_freqs = np.asarray(( 0.75, 3/5, 1/2))
    #Io_alias_freqs = Io_alias_freqs/IO_ORBIT + 1/(12*u.hr)
        
    if plot_type == 'period':
        ax.plot(1/frequency, power)
        ax.set_xlim((max_period.value, min_period.value))
        #ax.set_xlim((min_period.value, max_period.value))
        ax.set_xlabel('Period ' + (1/frequency).unit.to_string())
        sysIII_plot = SYSIII.value
        sysIV_low = SYSIV_LOW.value
        sysIV_high = SYSIV_HIGH.value
        obs_freqs_to_plot = 1/obs_freqs.value
        top_ax_label = 'Frequency ' + frequency.unit.to_string()
    if plot_type == 'frequency':
        ax.plot(frequency, power)
        plt.xlim((1/max_period.value, 1/min_period.value))
        ax.set_xlabel('Frequency ' + frequency.unit.to_string())
        sysIII_plot = 1/SYSIII.value
        sysIV_low = 1/SYSIV_LOW.value
        sysIV_high = 1/SYSIV_HIGH.value
        obs_freqs_to_plot = obs_freqs.value
        top_ax_label = 'Period ' + (1/frequency).unit.to_string()

    ax.set_ylabel('Normalized power')
    ax.axvline(sysIII_plot, color='orange')
    ax.axvline(sysIV_low, color='green')
    ax.axvline(sysIV_high, color='green', linestyle='--')
    ax.axhline(ls.false_alarm_level(false_alarm_level),
                                    color='red')
    
    for obs_freq in obs_freqs_to_plot:
        ax.axvline(obs_freq, color='cyan', linestyle='-')
    plt.legend(['Periodogram',
                r'$\mathrm{\lambda_{III}}$',
                r'$\mathrm{\lambda_{IV}}$ low',
                r'$\mathrm{\lambda_{IV}}$ high',
                f'{false_alarm_level} false alarm level',
                'Moons'])
    ax.format_coord = SecAxFormatter(frequency, power)

    secax = ax.secondary_xaxis('top', functions=(one_over, one_over))
    secax.set_xlabel(top_ax_label)
    plt.minorticks_on()
    # This doesn't work for some reason
    #secax.tick_params(axis='x', which='minor')

    min_period_str = min_period.to_string()
    min_period_str = min_period_str.replace(' ', '_')
    max_period_str = max_period.to_string()
    max_period_str = max_period_str.replace(' ', '_')
    if plotdir:
        plotname = f'periodogram_{plot_type}_{side_str}_{start_str}--{stop_str}_{min_period_str}-{max_period_str}.png'
        plotname = os.path.join(plotdir, plotname)
        savefig_overwrite(plotname)
    else:
        plt.show()
    plt.close()

def phase_plot(start, stop, fold_period=SYSIII,
               side=None,
               plotdir=None):
    torus_table='/data/IoIO/Torus/Torus.ecsv'
    t = QTable.read(torus_table)
    t.sort('tavg')
    if side == 'west':
        side = 'right'
    elif side == 'dusk':
        side = 'right'
    elif side == 'east':
        side = 'left'
    elif side == 'dawn':
        side = 'left'
    if side == 'right':
        side_str = 'Dusk'
    elif side == 'left':
        side_str = 'Dawn'
    else:
        raise ValueError(f'Invalid side: {side}')

    t = t[~t['mask']]
    mask = np.logical_and(start < t['tavg'], t['tavg'] < stop)
    mask = np.logical_and(mask, np.isfinite(t[f'ansa_{side}_surf_bright']))
    t = t[mask]

    add_medfilt(t, 'ansa_right_surf_bright',
                medfilt_width=21)
    add_medfilt(t, 'ansa_left_surf_bright',
                medfilt_width=21)

    ts = TimeSeries(time=t['tavg'])
    ts[f'{side}_ansa_surf_bright'] = t[f'ansa_{side}_surf_bright']

    ts_folded = ts.fold(period=fold_period)
    ts_folded.sort('time')

    ### This wasn't nearly as good as the median filtering
    ##kernel = Gaussian1DKernel(stddev=50)
    ##ts_folded['west_ansa_surf_bright_gauss'] = convolve(
    ##    ts_folded['west_ansa_surf_bright'], kernel)

    ts_folded[f'{side}_ansa_surf_bright']

    add_medfilt(ts_folded, f'{side}_ansa_surf_bright',
                medfilt_width=51)

    plt.plot(ts_folded.time.jd*24.,
             ts_folded[f'{side}_ansa_surf_bright'], 'k.')
    plt.plot(ts_folded.time.jd*24.,
             ts_folded[f'{side}_ansa_surf_bright_medfilt'], 'r-')
    #plt.plot(ts_folded.time.jd*24.,
    #         ts_folded['west_ansa_surf_bright_gauss'], 'c-')

    plt.ylim((0, 250))
    # This masks potentially spurious signals at some periods
    #plt.plot(ts_folded.time.jd*24.,
    #         ts_folded['west_ansa_normed'], 'k.')
    #plt.ylim((0.25, 2))
    plt.xlabel('Hour')
    plt.ylabel(f'{side_str} Ansa Surf. Bright. '
               f'({ts_folded[f"{side}_ansa_surf_bright"].unit})')
    start_str, _ = start.fits.split('T')
    stop_str, _ = stop.fits.split('T')
    fold_str = f'{fold_period:3.3f}'
    plt.title(f'{side_str} {start_str} -- {stop_str} Folded at {fold_str}')
    fold_str = fold_str.replace(' ', '_')
    if plotdir:
        plotname = f'phase_plot_{side_str}_{start_str}--{stop_str}_{fold_str}.png'
        plotname = os.path.join(plotdir, plotname)
        savefig_overwrite(plotname)
    else:
        plt.show()
    plt.close()
    
class TorusArgparseHandler(SSArgparseHandler,
                           CorPhotometryArgparseMixin, CalArgparseHandler):
    def add_utall(self):
        """Add options used in cmd"""
        self.add_reduced_root(default=TORUS_ROOT)
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
        t = torus_tree(raw_data_root=args.raw_data_root,
                       start=args.start,
                       stop=args.stop,
                       calibration=c,
                       cor_boltwood=CorBoltwood(precalc=True),
                       keep_intermediate=args.keep_intermediate,
                       solve_timeout=args.solve_timeout,
                       join_tolerance=(
                           args.join_tolerance*args.join_tolerance_unit),
                       standard_star_obj=ss,
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
        description='Run torus reduction')
    aph = TorusArgparseHandler(parser)
    aph.add_all()
    args = parser.parse_args()
    aph.cmd(args)


# Crumy night (hopefully rare in this time!)
#directory = '/data/IoIO/raw/20210607/'
# Hmm.  Bad in a similar way
#directory = '/data/IoIO/raw/20210610/'

# OK
#directory = '/data/IoIO/raw/2017-05-02'
## Good with ANSA_WIDTH_BOUNDS = (0.1*u.R_jup, 0.5*u.R_jup)
#directory = '/data/IoIO/raw/2018-05-08/'
#directory = '/data/IoIO/raw/20221224/'

# Good directory = '/data/IoIO/raw/20221224/'
#
##outdir_root=TORUS_ROOT
#outdir_root = '/data/IoIO/NO_BACKUP/Torus_testing/'
#
#t = torus_directory(directory,
#                    outdir_root=outdir_root)

##rd = reduced_dir(directory, outdir_root, create=True)

                    
# t = cached_csv(torus_directory,
#                csvnames=os.path.join(rd, 'Characterize_Ansas.ecsv'),
#                directory=directory,
#                standard_star_obj=None,
#                read_csvs=False,
#                write_csvs=True,
#                read_pout=False,
#                write_pout=True,
#                fits_fixed_ignore=True,
#                outdir=rd,
#                create_outdir=True)

#_ , pipe_meta = zip(*pout)
#t = QTable(rows=pipe_meta)


# from IoIO.cordata import CorData
# ccd = CorData.read('/data/IoIO/Torus/2018-05-08/SII_on-band_026-back-sub.fits')
# #ccd = CorData.read('/data/IoIO/Torus/20230129/SII_on-band_001-back-sub.fits')
# ###ccd = CorData.read('/data/IoIO/Torus/2018-05-08/SII_on-band_017-back-sub.fits')
# ##ccd.obj_center = np.asarray((ccd.meta['obj_cr1'], ccd.meta['obj_cr0']))
# ###                                      
# bmp_meta = {}
# ccd = characterize_ansas(ccd, bmp_meta=bmp_meta, show=True)

#plot_planet_subim(ccd, outname='/tmp/test.fits')

#plt.plot(t['tavg'].datetime, t['ansa_left_r_peak'])

#t = QTable.read('/data/IoIO/Torus/Torus.ecsv')
#plot_ansa_brights(t)
#plt.show()

# plot_obj_surf_bright('/data/IoIO/Torus/Torus.ecsv', show=True)
