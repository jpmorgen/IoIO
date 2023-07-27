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
from matplotlib.ticker import MultipleLocator

from astropy import log
import astropy.units as u
from astropy.table import QTable, vstack
from astropy.convolution import Gaussian1DKernel, interpolate_replace_nans
from astropy.coordinates import Angle, SkyCoord
from astropy.modeling import models, fitting
from astropy.timeseries import TimeSeries, LombScargle

from bigmultipipe import cached_pout, outname_creator

from ccdmultipipe import ccd_meta_to_bmp_meta, as_single

from IoIO.utils import (dict_to_ccd_meta, nan_biweight, nan_mad,
                        get_dirs_dates, reduced_dir, cached_csv,
                        savefig_overwrite, finish_stripchart,
                        multi_glob, pixel_per_Rj, plot_planet_subim)
from IoIO.simple_show import simple_show
from IoIO.cordata_base import SMALL_FILT_CROP
from IoIO.cordata import CorData
from IoIO.cormultipipe import (IoIO_ROOT, RAW_DATA_ROOT,
                               tavg_to_bmp_meta, calc_obj_to_ND, crop_ccd,
                               planet_to_object, mean_image,
                               objctradec_to_obj_center, obj_surface_bright)
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
from IoIO.horizons import GALSATS, galsat_ephemeris
from IoIO.juno import JunoTimes, PJAXFormatter

TORUS_ROOT = os.path.join(IoIO_ROOT, 'Torus')
MAX_ROOT = os.path.join(IoIO_ROOT, 'for_Max')

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
# is a big slope
ANSA_WIDTH_BOUNDS = (0.1*u.R_jup, 0.5*u.R_jup)

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

    if show:
        simple_show(ansa)

    mask_check = ansa.mask
    if np.sum(mask_check) > ansa_max_n_mask:
        return bad_ansa(side)

    # Calculate profile along centripetal equator, keeping it in units
    # of rayleighs
    r_pix = np.arange(ansa.shape[1]) * u.pixel
    r_Rj = (r_pix + left*u.pixel - center[1]) / pix_per_Rj
    r_prof = np.sum(ansa, 0) * ansa.unit
    r_prof /= ansa.shape[1]

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
    r_model_init = (
        models.Gaussian1D(mean=side_mult*IO_ORBIT_R,
                          stddev=0.3*u.R_jup,
                          amplitude=amplitude,
                          bounds={'mean': mean_bounds,
                                  'stddev': ansa_width_bounds})
        + models.Polynomial1D(1, c0=np.min(r_prof)))
    # The weights need to read in the same unit as the thing being
    # fitted for the std to come back with sensible units.
    r_fit = fit(r_model_init, r_Rj, r_prof, weights=r_dev**-0.5)
    r_ansa = r_fit.mean_0.quantity
    dRj = r_fit.stddev_0.quantity

    # #print(r_fit.fit_deriv(r_Rj))
    # print(r_fit)
    # print(r_fit.mean_0.std)
    # print(r_ansa)
    # print(dRj)
    # print(r_amp)

    if show:
        f = plt.figure(figsize=[5, 5])
        date_obs, _ = ccd.meta['DATE-OBS'].split('T') 
        #outname = outname_creator(in_name, outname=outname, **kwargs)
        #plt.title(f'{date_obs} {os.path.basename(outname)}')
        plt.plot(r_Rj, r_prof)
        plt.plot(r_Rj, r_fit(r_Rj))
        plt.xlabel(r'Jovicentric radial distance (R$\mathrm{_J}$)')
        plt.ylabel(f'Surface brightness ({r_prof.unit})')
        plt.tight_layout()
        plt.show()

    if r_fit.mean_0.std is None:
        return bad_ansa(side)

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

    if show:
        plt.plot(y_Rj, y_prof)
        plt.plot(y_Rj, y_fit(y_Rj))
        plt.xlabel(y_Rj.unit)
        plt.ylabel(y_prof.unit)
        plt.show()

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

def characterize_ansas(ccd_in, bmp_meta=None, galsat_mask_side=None, 
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
    for side in ['left', 'right']:
        ansa_meta = ansa_parameters(rccd, side=side, **kwargs)
        ccd = dict_to_ccd_meta(ccd, ansa_meta)
        bmp_meta.update(ansa_meta)
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

def nan_median_filter(data, mask=True, **kwargs):
    ndata = data.copy()
    mask = np.logical_and(mask, ~np.isnan(data))
    meds = median_filter(data[mask], **kwargs)
    if isinstance(ndata, u.Quantity):
        meds *= ndata.unit
    ndata[mask] = meds
    return ndata
    
def add_medfilt(t, colname, mask_col='mask', medfilt_width=21):
    """TABLE MUST BE SORTED FIRST"""
    
    if len(t) < medfilt_width/2:
        return
    if mask_col in t.colnames:
        bad_mask = t[mask_col]
    else:
        bad_mask = False
    meds = nan_median_filter(t[colname], mask=~bad_mask,
                             size=medfilt_width, mode='reflect')
    #bad_mask = np.logical_or(bad_mask, np.isnan(t[colname]))
    #vals = t[colname][~bad_mask]
    #meds = medfilt(vals, medfilt_width)
    #meds = median_filter(vals, size=medfilt_width, mode='reflect')
    #t[f'{colname}_medfilt'][~bad_mask] = meds
    t[f'{colname}_medfilt'] = meds

def add_interpolated(t, colname, kernel):
    # --> This is a bug in the making, since I am not handling the masked values properly 
    if isinstance(t[colname], u.Quantity):
        vals = t[colname].value
        unit = t[colname].unit
    else:
        vals = t[colname]
        unit = 1
    if isinstance(vals, np.ma.MaskedArray):
        # This makes masked entries into NANs, but not for astropy columns
        # https://stackoverflow.com/questions/56213393/replace-masked-with-nan-in-numpy-masked-array        
        vals = vals.filled(np.NAN)

    # --> HACK ALERT! I need ot do this properly within the astropy ecosystem
    vals = np.asarray(vals)
    vals = interpolate_replace_nans(vals, kernel, boundary='extend')
    t[f'{colname}_interp'] = vals*unit
    #t[f'{colname}_interp'] = interpolate_replace_nans(t[colname], kernel, boundary='extend')
                  
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
                     show=False,
                     max_night_gap=20,
                     **kwargs): # These are passed to plot_[hv]lines

    if fig is None:
        fig = plt.figure()
    if ax is None:
        ax = plt.subplot()
    if scale is None:
        scale = np.full(len(colnames), 1)

    t = t[~t['mask']]    
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
        h = ax.errorbar(datetimes,
                        scale[ic] * t[colname].value,
                        scale[ic] * t[f'{colname}_err'].value,
                        fmt=fmts[ic], alpha=alpha,
                        label=scale_str + labels[ic])
        handles.append(h)
    if p_med:
        # This gets the legend in the correct order
        handles.extend(p_med)
    if tlim is None:
        tlim = ax.get_xlim()
    ax.set_xlabel('date')
    ax.set_ylabel(ylabel)
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

def plot_ansa_brights(t, **kwargs):
    plot_column_vals(t, colnames=['ansa_left_surf_bright',
                                  'ansa_right_surf_bright'],
                     fmts=['b.', 'r.'],
                     labels=['Dawn', 'Dusk'],
                     ylabel=f'IPT Ansa Surf. Bright '\
                     f'({t["ansa_left_surf_bright"].unit})',
                     medfilt_colname='ansa_right_surf_bright',
                     medfilt_collabel='Dusk medfilt',
                     **kwargs)
                     
#def plot_ansa_brights(t,
#                      fig=None,
#                      ax=None,
#                      min_sb=0,
#                      max_sb=225,
#                      tlim=None,
#                      show=False,
#                      max_night_gap=20):
#    if fig is None:
#        fig = plt.figure()
#    if ax is None:
#        ax = plt.subplot()
#
#    t = t[~t['mask']]    
#    t.sort('tavg')
#
#    # --> vstack Doesn't work with masked columns
#    ## Insert NANs in the table at times > max_night_gap so that plotted
#    ## of line of median filter has gaps
#    ## Hack to get around astropy vstack bug with location
#    #loc = t['tavg'][0].location.copy()
#    #t['tavg'].location = None
#    #deltas = t['tavg'][1:] - t['tavg'][0:-1]
#    #gap_idx = np.flatnonzero(deltas > max_night_gap)
#    #gap_times = t['tavg'][gap_idx] + TimeDelta(1, format='jd')
#    #gap_t = t[0:len(gap_idx)]
#    #gap_t['tavg'] = gap_times
#    #t = vstack([t, gap_t])
#    #t.sort('tavg')
#    #t['tavg'].location = loc
#
#    add_ansa_surf_bright_medfilt(t)
#    right_sb_biweight = nan_biweight(t['ansa_right_surf_bright'])
#    right_sb_mad = nan_mad(t['ansa_right_surf_bright'])
#    left_sb_biweight = nan_biweight(t['ansa_left_surf_bright'])
#    left_sb_mad = nan_mad(t['ansa_left_surf_bright'])
#
#    mean_surf_bright = np.mean((left_sb_biweight.value,
#                                right_sb_biweight.value))
#    max_sb_mad = np.max((left_sb_mad.value, right_sb_mad.value))
#    ylim_surf_bright = (mean_surf_bright - 5*max_sb_mad,
#                        mean_surf_bright + 5*max_sb_mad)
#    if not np.isfinite(ylim_surf_bright[0]):
#        ylim_surf_bright = None
#    datetimes = t['tavg'].datetime
#    if len(t) > 40:
#        alpha = 0.1
#        p_med = ax.plot(datetimes,
#                         t['ansa_right_surf_bright_medfilt'],
#                         'k*', markersize=6, label='Dusk medfilt')
#        #plt.plot(t['tavg'].datetime, t['ansa_left_surf_bright_medfilt'],
#        #         'k+', markersize=6, label='Dawn medfilt')
#        ## Add gaps as NANs so plot with line makes gaps
#        #
#        ## --> This doesn't work because there are too many NANs from
#        ## the original median filtering
#        #times = t['tavg'].datetime
#        #deltas = times[1:] - times[0:-1]
#        ##print(len(deltas))
#        #one_day = datetime.timedelta(days=1)
#        #gap_idx = np.flatnonzero(deltas > max_night_gap*one_day)
#        ##print(len(gap_idx))
#        ##print(times[gap_idx])
#        ##print(times[gap_idx] + one_day)
#        #times = np.append(times, times[gap_idx] + one_day)
#        #vals = np.append(t['ansa_right_surf_bright_medfilt'],
#        #                 (np.NAN, ) * len(gap_idx))
#        #sort_idx = np.argsort(times)
#        ##print(sort_idx)
#        ##p_med = plt.plot(times[sort_idx], vals[sort_idx],
#        ##                 'k-', linewidth=2, label='Dusk medfilt')
#        #p_med = plt.plot(times[sort_idx], vals[sort_idx],
#        #                 'k*', markersize=6, label='Dusk medfilt')
#    else:
#        alpha = 0.5
#        p_med = None
#
#    p_left = ax.errorbar(datetimes,
#                          t['ansa_left_surf_bright'].value,
#                          t['ansa_left_surf_bright_err'].value,
#                          fmt='b.', alpha=alpha,
#                          label='Dawn')
#    p_right = ax.errorbar(datetimes,
#                           t['ansa_right_surf_bright'].value,
#                           t['ansa_right_surf_bright_err'].value,
#                           fmt='r.', alpha=alpha,
#                           label='Dusk')
#    handles = [p_left, p_right]
#    if p_med:
#        handles.extend(p_med)
#    if tlim is None:
#        tlim = ax.get_xlim()
#    #plt.title(r'Torus Ansa Brightnesses in [SII] 6731 $\mathrm{\AA}$')
#    ax.set_xlabel('date')
#    ax.set_ylabel(f'IPT Ansa Surf. Bright ({t["ansa_left_surf_bright"].unit})')
#    ax.legend(handles=handles)
#
#    ax.set_xlim(tlim)
#    ax.xaxis.set_minor_locator(mdates.MonthLocator())
#    # #ax.set_ylim(ylim_surf_bright)
#    ax.set_ylim(min_sb, max_sb)
#    fig.autofmt_xdate()
#    ax.format_coord = PJAXFormatter(datetimes,
#                                    t['ansa_right_surf_bright'])
#
#    #jts = JunoTimes()
#    #secax = ax.secondary_xaxis('top',
#    #                           functions=(jts.plt_date2pj, jts.pj2plt_date))
#    #secax.tick_params(tick1On=False)
#    #secax.tick_params(tick2On=False)
#    #secax.tick_params(label1On=False)
#    #secax.tick_params(label2On=False)
#    #
#    #secax.xaxis.set_minor_locator(MultipleLocator(1))
#    #secax.set_xlabel('PJ')

def plot_epsilons(t,
                  fig=None,
                  ax=None,
                  medfilt_width=21,
                  min_eps=-0.015,
                  max_eps=0.06,
                  tlim=None,
                  show=False,
                  **kwargs): # These are passed to plot_[hv]lines
    if fig is None:
        fig = plt.figure()
    if ax is None:
        ax = plt.subplot()

    t = t[~t['mask']]    
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
    epsilon_biweight = nan_biweight(epsilon)
    epsilon_mad = nan_mad(epsilon)

    bad_mask = np.isnan(epsilon)
    good_epsilons = epsilon[~bad_mask]

    if len(good_epsilons) > 20:
        alpha = 0.2
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
        add_interpolated(t, 'ansa_right_r_peak_medfilt', kernel)
        add_interpolated(t, 'ansa_left_r_peak_medfilt', kernel)
        r_med_peak = t['ansa_right_r_peak_medfilt_interp']
        l_med_peak = t['ansa_left_r_peak_medfilt_interp']
        av_med_peak = (np.abs(r_med_peak) + np.abs(l_med_peak)) / 2
        medfilt_epsilon = -(r_med_peak + l_med_peak) / av_med_peak
        p_interps = ax.plot(t['tavg'].datetime, medfilt_epsilon,
                             'c*', markersize=3, label='From interpolations')
    else:
        alpha = 0.5
        p_med = None
       
    p_eps = ax.errorbar(t['tavg'].datetime,
                        epsilon.value,
                        epsilon_err.value, fmt='k.', alpha=alpha,
                        label='Epsilon')

    handles = [p_eps]
    if p_med:
        handles.extend([p_med[0], p_interps[0]])
    ax.set_ylabel(r'Sky plane $|\vec\epsilon|$ (dawnward)')
    ax.hlines(np.asarray((epsilon_biweight,
                           epsilon_biweight - epsilon_mad,
                           epsilon_biweight + epsilon_mad)),
               *tlim,
               linestyles=('-', '--', '--'),
               label=f'{epsilon_biweight:.3f} +/- {epsilon_mad:.3f}')
    ax.axhline(0.025, color='y', label='Nominal 0.025')
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

def plot_ansa_pos(t,
                  fig=None,
                  ax=None,
                  tlim=None,
                  show=False,
                  medfilt_width=21,
                  **kwargs): # These are passed to plot_[hv]lines):
    if fig is None:
        fig = plt.figure()
    if ax is None:
        ax = plt.subplot()

    t = t[~t['mask']]    
    add_medfilt(t, 'ansa_right_r_peak', medfilt_width=medfilt_width)
    add_medfilt(t, 'ansa_left_r_peak', medfilt_width=medfilt_width)
    # Make it clear we are plotting the perturbation from Io's orbital
    # position on the right and left sides of Jupiter.  We will make t
    # eastward (negative of these), when plotting
    rights = t['ansa_right_r_peak'] - IO_ORBIT_R
    lefts = t['ansa_left_r_peak'] - (-IO_ORBIT_R)
    right_bad_mask = np.isnan(rights)
    rights = rights[~right_bad_mask]
    left_bad_mask = np.isnan(lefts)
    lefts = lefts[~left_bad_mask]

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
                     -lefts.value,
                     t['ansa_right_r_peak_err'][~left_bad_mask].value, fmt='b.',
                     label='Dawn', alpha=alpha)
    point_handles.append(h)
    h = ax.errorbar(t['tavg'][~right_bad_mask].datetime,
                     -rights.value,
                     t['ansa_left_r_peak_err'][~right_bad_mask].value, fmt='r.',
                     label='Dusk', alpha=alpha)
    point_handles.append(h)
    handles = point_handles
    handles.extend(medfilt_handles)

    #plt.plot(t['tavg'].datetime,
    #         np.abs(t['ansa_right_r_peak']) + t['ansa_right_r_stddev'],
    #         'g^')
    ax.set_ylabel(r'Dawnward ansa shift from Io orbit (R$_\mathrm{J}$)')
    ax.axhline(0, color='y', label='Io orbit')
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

# --> This is becoming obsolete
def torus_stripchart(table_or_fname, outdir,
                     fig=None,
                     n_plots=5,
                     min_sb=0,
                     max_sb=250,
                     tlim=None,
                     show=False):
    if isinstance(table_or_fname, str):
        t = QTable.read(table_or_fname)
    else:
        t = table_or_fname
    if fig is None:
        if n_plots == 1:
            f = plt.figure(figsize=[8.5, 11/2])
        else:
            f = plt.figure(figsize=[8.5, 11])
    else:
        f = fig

    outbase = 'Characterize_Ansas.png'
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
    right_r_peak_biweight = nan_biweight(t['ansa_right_r_peak'])
    right_r_peak_mad = nan_mad(t['ansa_right_r_peak'])
    left_r_peak_biweight = nan_biweight(t['ansa_left_r_peak'])
    left_r_peak_mad = nan_mad(t['ansa_left_r_peak'])
    right_sb_biweight = nan_biweight(t['ansa_right_surf_bright'])
    right_sb_mad = nan_mad(t['ansa_right_surf_bright'])
    left_sb_biweight = nan_biweight(t['ansa_left_surf_bright'])
    left_sb_mad = nan_mad(t['ansa_left_surf_bright'])
    epsilon_biweight = nan_biweight(epsilon)
    epsilon_mad = nan_mad(epsilon)

    # This was a not-so-successful bid at getting something like
    # f.autofmt_xdate() running but with my own definitions of things
    #time_expand = timedelta(minutes=5)
    #tlim = (np.min(t['tavg'].datetime) - time_expand,
    #        np.max(t['tavg'].datetime) + time_expand)
    #tlim = (np.min(t['tavg'].datetime),
    #        np.max(t['tavg'].datetime))
    mean_surf_bright = np.mean((left_sb_biweight.value,
                                right_sb_biweight.value))
    max_sb_mad = np.max((left_sb_mad.value, right_sb_mad.value))
    ylim_surf_bright = (mean_surf_bright - 5*max_sb_mad,
                        mean_surf_bright + 5*max_sb_mad)
    if not np.isfinite(ylim_surf_bright[0]):
        ylim_surf_bright = None

    date, _ = t['tavg'][0].fits.split('T')
    plt.suptitle('Torus Ansa Characteristics')

    # Get E = positive to left sign correct now that I am not thinking in
    # literal image left and right terms

    ax = plt.subplot(n_plots, 1, 1)
    t.sort('tavg')

    # --> Might need an if to plot only when I want to
    #bad_mask = np.isnan(t['ansa_left_surf_bright'])
    #lefts = t['ansa_left_surf_bright'][~bad_mask]
    #med_left = medfilt(lefts, 21)
    #plt.plot(t['tavg'][~bad_mask].datetime, med_left,
    #         'k-', linewidth=3, label='Dawn medfilt')
    bad_mask = np.isnan(t['ansa_right_surf_bright'])
    rights = t['ansa_right_surf_bright'][~bad_mask]
    if len(rights) > 40:
        alpha = 0.1
        med_right = medfilt(rights, 21)
        plt.plot(t['tavg'][~bad_mask].datetime, med_right,
                 'k-', linewidth=3, label='Dusk medfilt')
    else:
        alpha = 0.5

    plt.errorbar(t['tavg'].datetime,
                 t['ansa_left_surf_bright'].value,
                 t['ansa_left_surf_bright_err'].value, fmt='b.', alpha=alpha,
                 label='Dawn')
    plt.errorbar(t['tavg'].datetime,
                 t['ansa_right_surf_bright'].value,
                 t['ansa_right_surf_bright_err'].value, fmt='r.', alpha=alpha,
                 label='Dusk')
    if tlim is None:
        tlim = ax.get_xlim()
    plt.ylabel(f'Ansa Av. Surf. Bright ({t["ansa_left_surf_bright"].unit})')
    #plt.hlines(np.asarray((left_sb_biweight.value,
    #                       left_sb_biweight.value - left_sb_mad.value,
    #                       left_sb_biweight.value + left_sb_mad.value)),
    #           *tlim,
    #           colors='r',
    #           linestyles=('-', '--', '--'),
    #           label=(f'Dawn {left_sb_biweight:.0f} '
    #                  f'+/- {left_sb_mad:.0f}'))
    #plt.hlines(np.asarray((right_sb_biweight.value,
    #                       right_sb_biweight.value - right_sb_mad.value,
    #                       right_sb_biweight.value + right_sb_mad.value)),
    #           *tlim,
    #           linestyles=('-', '--', '--'),
    #           colors='g',
    #           label=(f'Dusk {right_sb_biweight:.0f} '
    #                  f'+/- {right_sb_mad:.0f}'))
    ax.legend()
    ax.set_xlim(tlim)
    #ax.set_ylim(ylim_surf_bright)
    ax.set_ylim(min_sb, max_sb)
    f.autofmt_xdate()

    if n_plots == 1:
        finish_stripchart(outdir, outbase, show=show)
        return
        
    ax = plt.subplot(n_plots, 1, 2)

    bad_mask = np.isnan(epsilon)
    good_epsilons = epsilon[~bad_mask]

    if len(good_epsilons) > 40:
        alpha = 0.1
        med_epsilon = medfilt(good_epsilons, 21)
        plt.plot(t['tavg'][~bad_mask].datetime, med_epsilon,
                 'r-', linewidth=3, label='Epsilon medfilt')

    else:
        alpha = 0.5
        
    plt.errorbar(t['tavg'].datetime,
                 epsilon.value,
                 epsilon_err.value, fmt='k.', alpha=alpha,
                 label='Epsilon')

    ax.set_ylim(-0.05, 0.08)
    plt.ylabel(r'Sky plane $|\vec\epsilon|$')
    plt.hlines(np.asarray((epsilon_biweight,
                           epsilon_biweight - epsilon_mad,
                           epsilon_biweight + epsilon_mad)),
               *tlim,
               linestyles=('-', '--', '--'),
               label=f'{epsilon_biweight:.3f} +/- {epsilon_mad:.3f}')
    plt.axhline(0.025, color='y', label='Nominal 0.025')
    ax.legend()
    ax.set_xlim(tlim)

    if n_plots == 2:
        finish_stripchart(outdir, outbase, show=show)
        return

    ax = plt.subplot(n_plots, 1, 3)
    plt.errorbar(t['tavg'].datetime,
                 np.abs(t['ansa_left_r_peak'].value),
                 t['ansa_left_r_peak_err'].value, fmt='b.',
                 label='Dawn')
    #plt.plot(t['tavg'].datetime,
    #         np.abs(t['ansa_left_r_peak']) + t['ansa_left_r_stddev'],
    #         'r^')
    plt.errorbar(t['tavg'].datetime,
                 np.abs(t['ansa_right_r_peak'].value),
                 t['ansa_right_r_peak_err'].value, fmt='r.',
                 label='Dusk')
    #plt.plot(t['tavg'].datetime,
    #         np.abs(t['ansa_right_r_peak']) + t['ansa_right_r_stddev'],
    #         'g^')
    plt.ylabel(r'Ansa position (R$_\mathrm{J}$)')
    plt.axhline(IO_ORBIT_R.value, color='y', label='Io orbit')
    ax.legend()
    ax.set_xlim(tlim)
    ax.set_ylim(5.5, 6.0)

    if n_plots == 3:
        finish_stripchart(outdir, outbase, show=show)
        return
        
    ax = plt.subplot(n_plots, 1, 4)
    plt.plot(t['tavg'].datetime, t['closest_galsat'], 'k.')
    ax.set_xlim(tlim)
    plt.ylabel(r'Closest galsat (R$_\mathrm{J}$)')

    # import matplotlib.dates as mdates
    #plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    #--> still would like to get this nicer, but for now do it this
    #way for the summary plot
    #f.autofmt_xdate()
    plt.xlabel(f'UT {date}')

    dawn_sysIII = Angle(t['Jupiter_PDObsLon'] + 90*u.deg)
    dusk_sysIII = Angle(t['Jupiter_PDObsLon'] - 90*u.deg)
    dawn_sysIII = dawn_sysIII.wrap_at(360*u.deg)
    dusk_sysIII = dusk_sysIII.wrap_at(360*u.deg)

    if n_plots == 4:
        finish_stripchart(outdir, outbase, show=show)
        return

    ax = plt.subplot(n_plots, 1, 5)
    plt.errorbar(dawn_sysIII.value,
                 t['ansa_left_surf_bright'].value,
                 t['ansa_left_surf_bright_err'].value, fmt='b.',
                 label='Dawn')
    plt.errorbar(dusk_sysIII.value,
                 t['ansa_right_surf_bright'].value,
                 t['ansa_right_surf_bright_err'].value, fmt='r.',
                 label='Dusk')
    plt.ylabel(f'Surf. Bright ({t["ansa_left_surf_bright"].unit})')
    plt.xlabel(r'Ansa $\lambda{\mathrm{III}}$')
    plt.xticks(np.arange(0,360,45))
    plt.minorticks_on()
    ax.set_xlim(0, 360)
    ax.set_ylim(ylim_surf_bright)
    ax.legend()
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
    torus_stripchart(t, outdir)
    return t

def torus_tree(raw_data_root=RAW_DATA_ROOT,
               start=None,
               stop=None,
               calibration=None,
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
    summary_table.write(os.path.join(outdir_root, 'Torus.ecsv'),
                                     overwrite=True)
    torus_stripchart(summary_table, outdir_root, n_plots=3, show=show)

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
    def add_all(self):
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
                       fits_fixed_ignore=args.fits_fixed_ignore)

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
