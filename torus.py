#!/usr/bin/python3

"""Reduce IoIO Io plamsa torus observations"""

import os
from datetime import timedelta

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.dates as mdates

from astropy import log
import astropy.units as u
#from astropy.utils.masked import Masked
from astropy.table import QTable
from astropy.coordinates import SkyCoord
from astropy.modeling import models, fitting

from bigmultipipe import cached_pout, outname_creator

from ccdmultipipe import ccd_meta_to_bmp_meta, as_single

from IoIO.utils import (reduced_dir, nan_biweight, nan_mad,
                        cached_csv, savefig_overwrite)
from IoIO.simple_show import simple_show
from IoIO.cordata_base import SMALL_FILT_CROP
from IoIO.cormultipipe import (IoIO_ROOT, objctradec_to_obj_center,
                               crop_ccd)
from IoIO.calibration import Calibration
from IoIO.photometry import SOLVE_TIMEOUT, JOIN_TOLERANCE, rot_to
from IoIO.cor_photometry import CorPhotometry
from IoIO.on_off_pipeline import (TORUS_NA_NEB_GLOB_LIST,
                                  TORUS_NA_NEB_GLOB_EXCLUDE_LIST,
                                  on_off_pipeline)
from IoIO.standard_star import (StandardStar, extinction_correct,
                                rayleigh_convert)
from IoIO.horizons import GALSATS, galsat_ephemeris

TORUS_ROOT = os.path.join(IoIO_ROOT, 'Torus')

# https://lasp.colorado.edu/home/mop/files/2015/02/CoOrd_systems7.pdf
# Has sysIII of intersection of mag and equatorial plains at 290.8.
# That means tilt is toward 200.8, which is my basic recollection
CENTRIFUGAL_EQUATOR_AMPLITUDE = 6.8*u.deg
JUPITER_MAG_SYSIII = 290.8*u.deg
GALSAT_MASK_SIDE = 20 * u.pixel
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


def mask_galsats(ccd_in, galsat_mask_side=GALSAT_MASK_SIDE, **kwargs):
    assert galsat_mask_side.unit == u.pixel
    ccd = ccd_in.copy()
    galsat_mask = np.zeros_like(ccd.mask)
    galsats = list(GALSATS.keys())
    galsats = galsats[1:]
    for g in galsats:
        ra = ccd.meta[f'{g}_RA']
        dec = ccd.meta[f'{g}_DEC']
        sc = SkyCoord(ra, dec, unit=u.deg)
        try:
            pix = ccd.wcs.world_to_array_index(sc)
            hs = int(round(galsat_mask_side.value/2))
            bot = pix[0] - hs
            top = pix[0] + hs
            left = pix[1] - hs
            right = pix[1] + hs
            galsat_mask[bot:top, left:right] = True
        except Exception as e:
            # The wcs method may fail or the array index might fail.
            # Not sure yet would the Exception would be.  Either way,
            # we just want to skip it
            log.debug(f'Problem masking galsat: {e}')
            continue
        ccd.mask = np.ma.mask_or(ccd.mask, galsat_mask)
    return ccd

def IPT_NPole_ang(ccd, **kwargs):
    sysIII = ccd.meta['Jupiter_PDObsLon']
    ccd.meta['HIERARCH IPT_NPole_ang'] = (
        CENTRIFUGAL_EQUATOR_AMPLITUDE * np.sin(sysIII - JUPITER_MAG_SYSIII),
        '[degree]')
    return ccd

def pixel_per_Rj(ccd):
    Rj_arcsec = ccd.meta['Jupiter_ang_width'] * u.arcsec / 2
    cdelt = ccd.wcs.proj_plane_pixel_scales() # tuple
    lcdelt = [c.value for c in cdelt]
    pixscale = np.mean(lcdelt) * cdelt[0].unit
    pixscale = pixscale.to(u.arcsec) / u.pixel
    return Rj_arcsec / pixscale / u.R_jup 
   
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
            f'ansa_{side}_surf_bright_err': masked*u.R}

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
                    **kwargs):
    if side not in ['right', 'left']:
        raise ValueError(f'ansa_side must be right or left: {side}')    
    center = ccd.obj_center*u.pixel
    # Work in Rj for our model
    pix_per_Rj = pixel_per_Rj(ccd)
    fit = fitting.LevMarLSQFitter(calc_uncertainties=True)

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

    amplitude = np.max(r_prof) - np.min(r_prof)
    r_model_init = (
        models.Gaussian1D(mean=side_mult*5.9*u.R_jup,
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
    r_amp = r_fit.amplitude_0.quantity

    if show:
        plt.plot(r_Rj, r_prof)
        plt.plot(r_Rj, r_fit(r_Rj))
        plt.ylabel(r_prof.unit)
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
    dy_pos = r_fit.stddev_0.quantity
    y_amplitude_std = y_fit.amplitude_0.std or np.NAN
    r_ansa_std = r_fit.mean_0.std or np.NAN
    dRj_std = r_fit.stddev_0.std or np.NAN
    r_amp_std = r_fit.amplitude_0.std or np.NAN
    y_pos_std = y_fit.mean_0.std or np.NAN
    dy_pos_std = r_fit.stddev_0.std or np.NAN
    
    # Units of amplitude are already in rayleigh.  When we do the
    # integral, we technically should multiply the amplitude by the
    # peak stddev.  But that gives us an intensity meausrement.  To
    # convert that to sb, we divide again by the stddev.
    sb = (2 * np.pi)**-0.5 * y_fit.amplitude_0.quantity
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
            f'ansa_{side}_surf_bright_err': sb_err * sb.unit}

def characterize_ansas(ccd_in, bmp_meta=None,
                       **kwargs):
    # MAKE SURE THIS IS CALLED BEFORE ANY ROTATIONS The underlying
    # reproject stuff in rot_to is a little unstable when it comes to
    # multiple rotations.  When autorotate is used, N is not always
    # up.  If it is not used, the CCD image can get shoved off to one
    # side in a weird way.
    rccd = rot_to(ccd_in, rot_angle_from_key=['Jupiter_NPole_ang',
                                              'IPT_NPole_ang'])
    rccd = objctradec_to_obj_center(rccd)
    rccd = mask_galsats(rccd)
    # We only  want to mess with our original CCD metadata
    ccd = ccd_in.copy()
    bmp_meta = bmp_meta or {}
    bmp_meta['tavg'] = ccd.tavg
    ccd = ccd_meta_to_bmp_meta(ccd, bmp_meta=bmp_meta,
                               ccd_meta_to_bmp_meta_keys=
                               [('Jupiter_PDObsLon', u.deg),
                                ('Jupiter_PDObsLat', u.deg),
                                ('Jupiter_PDSunLon', u.deg)])
    for side in ['left', 'right']:
        ansa_meta = ansa_parameters(rccd, side=side, **kwargs)
        bmp_meta.update(ansa_meta)
        for k in ansa_meta.keys():
            #if np.ma.is_masked(ansa_meta[k]):
            if np.isnan(ansa_meta[k]):
                ccd.meta[f'HIERARCH {k}'] = 'NAN'
            elif isinstance(ansa_meta[k], u.Quantity):
                ccd.meta[f'HIERARCH {k}'] = (ansa_meta[k].value,
                                             f'[{ansa_meta[k].unit}]')
            else:
                ccd.meta[f'HIERARCH {k}'] = ansa_meta[k]
    return ccd

def plot_planet_subim(ccd_in,
                      plot_planet_rot_from_key=None,
                      pix_per_planet_radius=pixel_per_Rj,
                      in_name=None,
                      outname=None,
                      planet_subim_axis_label=r'R$\mathrm{_j}$',
                      planet_subim_dx=10*u.R_jup,
                      planet_subim_dy=5*u.R_jup,
                      planet_subim_vmin=30,
                      planet_subim_vmax=5000,
                      planet_subim_figsize=[5, 2.5],
                      plot_planet_cmap='gist_heat',
                      **kwargs):
    in_name = os.path.basename(ccd_in.meta['RAWFNAME'])
    # This will pick up outdir, if specified
    outname = outname_creator(in_name, outname=outname, **kwargs)
    if outname is None:
        raise ValueError('in_name or outname must be specified')
    ccd = rot_to(ccd_in, rot_angle_from_key=plot_planet_rot_from_key)
    pix_per_Rp = pix_per_planet_radius(ccd)
    center = ccd.wcs.world_to_pixel(ccd.sky_coord)*u.pixel
    l = np.round(center[0] - planet_subim_dx * pix_per_Rp).astype(int)
    r = np.round(center[0] + planet_subim_dx * pix_per_Rp).astype(int)
    b = np.round(center[1] - planet_subim_dy * pix_per_Rp).astype(int)
    t = np.round(center[1] + planet_subim_dy * pix_per_Rp).astype(int)
    subim = ccd[b.value:t.value, l.value:r.value]
    nr, nc = subim.shape
    subim.data[subim.data < planet_subim_vmin] = planet_subim_vmin
    x = (np.arange(nc)*u.pixel - (center[0] - l)) / pix_per_Rp
    y = (np.arange(nr)*u.pixel - (center[1] - b)) / pix_per_Rp
    X, Y = np.meshgrid(x.value, y.value)
    f = plt.figure(figsize=planet_subim_figsize)
    plt.pcolormesh(X, Y, subim,
                   norm=LogNorm(vmin=planet_subim_vmin,
                                vmax=planet_subim_vmax),
                   cmap=plot_planet_cmap)
    Rp_name = planet_subim_dx.unit.long_names[0]
    plt.ylabel(planet_subim_axis_label)
    plt.xlabel(planet_subim_axis_label)
    plt.axis('scaled')
    cbar = plt.colorbar()
    cbar.ax.set_xlabel(ccd.unit.to_string())
    date_obs, _ = ccd.meta['DATE-OBS'].split('T')
    plt.title(f'{date_obs} {os.path.basename(outname)}')
    plt.tight_layout()
    outroot, _ = os.path.splitext(outname)
    d = os.path.dirname(outname)
    os.makedirs(d, exist_ok=True)
    savefig_overwrite(outroot + '.png')
    plt.close()
    return ccd_in

def torus_directory(directory,
                    outdir=None,
                    outdir_root=TORUS_ROOT,
                    standard_star_obj=None,
                    read_pout=True,
                    write_pout=True,
                    **kwargs):
    standard_star_obj = standard_star_obj or StandardStar(reduce=True)
    #standard_star_obj = standard_star_obj or StandardStar(stop='2022-01-01',
    #                                                      reduce=True)
    outdir = outdir or reduced_dir(directory, outdir_root, create=False)
    poutname = os.path.join(outdir, 'Torus.pout')
    pout = cached_pout(on_off_pipeline,
                       poutname=poutname,
                       read_pout=read_pout,
                       write_pout=write_pout,
                       directory=directory,
                       glob_include=TORUS_NA_NEB_GLOB_LIST,
                       glob_exclude_list=TORUS_NA_NEB_GLOB_EXCLUDE_LIST,
                       band='SII',
                       standard_star_obj=standard_star_obj,
                       add_ephemeris=galsat_ephemeris,
                       crop_ccd_coord=SMALL_FILT_CROP,
                       pre_offsub=crop_ccd,
                       plot_planet_rot_from_key=['Jupiter_NPole_ang'],
                       post_offsub=[extinction_correct, rayleigh_convert,
                                    characterize_ansas,
                                    plot_planet_subim, as_single],
                       outdir=outdir,
                       **kwargs)

    _ , pipe_meta = zip(*pout)
    t = QTable(rows=pipe_meta)

    time_expand = timedelta(minutes=5)
    tlim = (np.min(t['tavg'].datetime) - time_expand,
            np.max(t['tavg'].datetime) + time_expand)
    r_peak = t['ansa_right_r_peak']
    l_peak = t['ansa_left_r_peak']
    av_peak = (np.abs(r_peak) + np.abs(l_peak)) / 2
    epsilon = (r_peak + l_peak) / av_peak
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

    f = plt.figure(figsize=[8.5, 11])
    date, _ = t['tavg'][0].fits.split('T')
    plt.suptitle('Torus Ansa Characteristics')

    # Get E = positive to left sign correct now that I am not thinking in
    # literal image left and right terms

    ax = plt.subplot(5, 1, 1)
    plt.errorbar(t['tavg'].datetime,
                 -t['ansa_left_r_peak'].value,
                 t['ansa_left_r_peak_err'].value, fmt='k.')
    plt.ylabel(r'East ansa (R$_\mathrm{J}$)')
    plt.axhline(IO_ORBIT_R.value, color='r', label='Io orbit')
    ax.legend()
    ax.set_xlim(tlim)
    ax.set_ylim(5.5, 6.0)

    ax = plt.subplot(5, 1, 2)
    plt.errorbar(t['tavg'].datetime,
                 t['ansa_left_surf_bright'].value,
                 t['ansa_left_surf_bright_err'].value, fmt='k.')
    plt.ylabel(f'East SB ({t["ansa_left_surf_bright"].unit})')
    plt.hlines(np.asarray((left_sb_biweight.value,
                           left_sb_biweight.value - left_sb_mad.value,
                           left_sb_biweight.value + left_sb_mad.value)),
               *tlim,
               linestyles=('-', '--', '--'),
               label=(f'{left_sb_biweight:.0f} '
                      f'+/- {left_sb_mad:.0f}'))
    ax.legend()
    ax.set_xlim(tlim)

    ax = plt.subplot(5, 1, 3)
    plt.errorbar(t['tavg'].datetime,
                 t['ansa_right_surf_bright'].value,
                 t['ansa_right_surf_bright_err'].value, fmt='k.')
    plt.ylabel(f'West SB ({t["ansa_right_surf_bright"].unit})')
    plt.hlines(np.asarray((right_sb_biweight.value,
                           right_sb_biweight.value - right_sb_mad.value,
                           right_sb_biweight.value + right_sb_mad.value)),
               *tlim,
               linestyles=('-', '--', '--'),
               label=(f'{right_sb_biweight:.0f} '
                      f'+/- {right_sb_mad:.0f}'))
    ax.legend()
    ax.set_xlim(tlim)

    ax = plt.subplot(5, 1, 4)
    plt.errorbar(t['tavg'].datetime,
                 -t['ansa_right_r_peak'].value,
                 t['ansa_right_r_peak_err'].value, fmt='k.')
    plt.ylabel(r'West ansa (R$_\mathrm{J}$)')
    plt.axhline(-IO_ORBIT_R.value, color='r', label='Io orbit')
    ax.legend()
    ax.set_xlim(tlim)
    ax.set_ylim(-6.0, -5.5)

    ax = plt.subplot(5, 1, 5)
    plt.errorbar(t['tavg'].datetime,
                 epsilon.value,
                 epsilon_err.value, fmt='k.')
    ax.set_ylim(-0.010, 0.050)
    plt.ylabel(r'Sky plane $|\vec\epsilon|$')
    plt.hlines(np.asarray((epsilon_biweight,
                           epsilon_biweight - epsilon_mad,
                           epsilon_biweight + epsilon_mad)),
               *tlim,
               linestyles=('-', '--', '--'),
               label=f'{epsilon_biweight:.3f} +/- {epsilon_mad:.3f}')
    plt.axhline(0.025, color='r', label='Nominal 0.025')
    ax.legend()
    ax.set_xlim(tlim)

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    f.autofmt_xdate()
    plt.xlabel(f'UT {date}')
    plt.tight_layout()

    savefig_overwrite(os.path.join(outdir, 'Characterize_Ansas.png'))

    return t

log.setLevel('DEBUG')

# Crumy night (hopefully rare in this time!)
#directory = '/data/IoIO/raw/20210607/'
# Hmm.  Bad in a similar way
#directory = '/data/IoIO/raw/20210610/'

# OK
#directory = '/data/IoIO/raw/2017-05-02'
## Good with ANSA_WIDTH_BOUNDS = (0.1*u.R_jup, 0.5*u.R_jup)
directory = '/data/IoIO/raw/2018-05-08/'

outdir_root=TORUS_ROOT
rd = reduced_dir(directory, outdir_root, create=False)
t = cached_csv(torus_directory,
               csvnames=os.path.join(rd, 'Characterize_Ansas.ecsv'),
               directory=directory,
               standard_star_obj=None,
               read_csvs=False,
               write_csvs=True,
               read_pout=False,
               write_pout=True,
               fits_fixed_ignore=True,
               outdir=rd,
               create_outdir=True)
#t = torus_directory(directory,
#                    standard_star_obj=None,
#                    read_pout=True,
#                    write_pout=True,
#                    fits_fixed_ignore=True,
#                    create_outdir=True)
                    
#_ , pipe_meta = zip(*pout)
#t = QTable(rows=pipe_meta)


#from IoIO.cordata import CorData
#ccd = CorData.read('/data/IoIO/Torus/2018-05-08/SII_on-band_024-back-sub.fits')
##ccd = CorData.read('/data/IoIO/Torus/2018-05-08/SII_on-band_017-back-sub.fits')
#ccd.obj_center = np.asarray((ccd.meta['obj_cr1'], ccd.meta['obj_cr0']))
##                                      
#bmp_meta = {}
#ccd = characterize_ansas(ccd, bmp_meta=bmp_meta, show=True)

#plot_planet_subim(ccd, outname='/tmp/test.fits')

#plt.plot(t['tavg'].datetime, t['ansa_left_r_peak'])
