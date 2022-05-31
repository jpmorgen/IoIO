#!/usr/bin/python3

"""Reduce IoIO Io plamsa torus observations"""

import os

import numpy as np

from astropy import log
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.modeling import models, fitting

from ccdmultipipe import as_single

from IoIO.simple_show import simple_show
from IoIO.cordata_base import SMALL_FILT_CROP
from IoIO.cormultipipe import IoIO_ROOT, objctradec_to_obj_center
from IoIO.calibration import Calibration
from IoIO.photometry import SOLVE_TIMEOUT, JOIN_TOLERANCE, rot_to
from IoIO.cor_photometry import CorPhotometry
from IoIO.on_off_pipeline import TORUS_NA_NEB_GLOB_LIST, on_off_pipeline
from IoIO.standard_star import (StandardStar, extinction_correct,
                                rayleigh_convert)
from IoIO.horizons import GALSATS, galsat_ephemeris

TORUS_ROOT = os.path.join(IoIO_ROOT, 'Torus')

# https://lasp.colorado.edu/home/mop/files/2015/02/CoOrd_systems7.pdf
# Has sysIII of intersection of mag and equatorial plains at 290.8.
# That means tilt is toward 200.8, which is my basic recollection
CENTRIFUGAL_EQUATOR_AMPLITUDE = 6.8*u.deg
JUPITER_MAG_SYSIII = 290.8*u.deg
GALSAT_MASK_SIDE = 20 # pixels
MAX_N_MASK = 10
RIGHT_INNER = 4.75
RIGHT_OUTER = 6.75
LEFT_OUTER = -6.75
LEFT_INNER = -4.75

def mask_galsats(ccd_in, galsat_mask_side=GALSAT_MASK_SIDE, **kwargs):
    ccd = ccd_in.copy()
    galsat_mask = np.zeros_like(ccd.mask)
    galsats = list(GALSATS.keys())
    galsats = galsats[1:]
    for g in galsats:
        ra = ccd.meta[f'{g}_RA']
        dec = ccd.meta[f'{g}_DEC']
        sc = SkyCoord(ra, dec, unit=u.deg)
        pix = ccd.wcs.world_to_array_index(sc)
        hs = int(round(galsat_mask_side/2))
        bot = pix[0] - hs
        top = pix[0] + hs
        left = pix[1] - hs
        right = pix[1] + hs
        galsat_mask[bot:top, left:right] = True
        ccd.mask = np.ma.mask_or(ccd.mask, galsat_mask)
    return ccd

def IPT_NPole_ang(ccd, **kwargs):
    sysIII = ccd.meta['Jupiter_PDObsLon']
    ccd.meta['HIERARCH IPT_NPole_ang'] = (
        CENTRIFUGAL_EQUATOR_AMPLITUDE * np.sin(sysIII - JUPITER_MAG_SYSIII))
    return ccd

def Rj_to_pixel(ccd):
    Rj = ccd.meta['Jupiter_ang_width'] * u.arcsec / 2
    cdelt = ccd.wcs.proj_plane_pixel_scales()
    vcdelt = [c.value for c in cdelt]
    pixscale = np.mean(vcdelt) * cdelt[0].unit
    pixscale = pixscale.to(u.arcsec)
    Rj_pix = Rj / pixscale
    return Rj_pix.value
   
def bad_ansa(ccd, bmp_meta=None, ansa_side=None, **kwargs):
    side = ansa_side
    if side not in ['right', 'left']:
        raise ValueError(f'ansa_side must be right or left: {side}')    
    bmp_meta = bmp_meta or {}
    ansa_meta = {f'ansa_{side}_r_peak': np.NAN,
                 f'ansa_{side}_r_peak_err': np.NAN,
                 f'ansa_{side}_r_stddev': np.NAN,
                 f'ansa_{side}_r_stddev_err': np.NAN,
                 f'ansa_{side}_r_amplitude': np.NAN,
                 f'ansa_{side}_r_amplitude_err': np.NAN,
                 f'ansa_{side}_y_peak': np.NAN,
                 f'ansa_{side}_y_peak_err': np.NAN,
                 f'ansa_{side}_y_stddev': np.NAN,
                 f'ansa_{side}_y_stddev_err': np.NAN,
                 f'ansa_{side}_surf_bright': np.NAN,
                 f'ansa_{side}_surf_bright_err': np.NAN}
    for k in ansa_meta.keys():
        ccd.meta[f'HIERARCH {k}'] = 'NAN'
    bmp_meta.update(ansa_meta)
    return ccd

def ansa_parameters(ccd_in,
                    bmp_meta=None,
                    ansa_side=None,
                    ansa_max_n_mask=MAX_N_MASK,
                    ansa_right_inner=RIGHT_INNER,
                    ansa_right_outer=RIGHT_OUTER,
                    ansa_left_outer=LEFT_OUTER,
                    ansa_left_inner=LEFT_INNER,
                    **kwargs):
    side = ansa_side
    if side not in ['right', 'left']:
        raise ValueError(f'ansa_side must be right or left: {side}')    
    ccd = ccd_in.copy()
    bmp_meta = bmp_meta or {}
    center = ccd.obj_center
    # Work in Rj for our model
    Rj_pix = Rj_to_pixel(ccd)
    fit = fitting.LevMarLSQFitter(calc_uncertainties=True)

    # Start with a basic fix on the ansa r position
    if side == 'right':
        left = int(round(center[1] + ansa_right_inner*Rj_pix))
        right = int(round(center[1] + ansa_right_outer*Rj_pix))
        side_mult = +1
        mean_bounds = (ansa_right_inner+0.25, ansa_right_outer-0.25)
    elif side == 'left':
        left = int(round(center[1] + ansa_left_outer*Rj_pix))
        right = int(round(center[1] + ansa_left_inner*Rj_pix))
        side_mult = -1
        mean_bounds = (ansa_left_outer-0.25, ansa_left_inner+0.25)
    cold_mean_bounds = np.asarray((5.0, 5.5))

    # There is a conflict between too much open torus and not allowing for
    # distortion in the IPT due to the magnetic field. +/- 1.75 is too
    # broad
    top = int(round(center[0] + 1.25*Rj_pix))
    bottom = int(round(center[0] - 1.25*Rj_pix))

    mask_check = ccd.mask[bottom:top, left:right]
    #print(f'mask_check {np.sum(mask_check)}')
    if np.sum(mask_check) > ansa_max_n_mask:
        return bad_ansa(ccd, bmp_meta=bmp_meta, ansa_side=side)

    ansa = ccd[bottom:top, left:right]
    #simple_show(ansa)

    # Calculate profile along centripetal equator, keeping it in units
    # of rayleighs
    r_pix = np.arange(ansa.shape[1])
    r_Rj = (r_pix + left - center[1]) / Rj_pix
    r_prof = [np.sum(ansa.data[:, x]) for x in r_pix]
    r_prof = np.asarray(r_prof)
    r_prof /= ansa.shape[1]

    r_ansa = (np.argmax(r_prof) + left - center[1]) / Rj_pix
    dRj = 0.5

    # Only relative values don't matter here since we are using as a weight
    if ccd.uncertainty.uncertainty_type == 'std':
        r_dev = [np.sum(ansa.uncertainty.array[:, x])**2 for x in r_pix]
    else:
        r_dev = [np.sum(ansa.uncertainty.array[:, x]) for x in r_pix]
    r_dev = np.asarray(r_dev)

    # Cold torus is a bridge too far, though it works sometimes
                    #+ models.Gaussian1D(mean=side_mult*(5.25),
                    #                    stddev=0.1,
                    #                    amplitude=amplitude/10,
                    #                    bounds={'mean': side_mult*cold_mean_bounds,
                    #                            'stddev': (0.05, 0.15),
                    #                            'amplitude': (0, amplitude)})

    amplitude = np.max(r_prof) - np.min(r_prof)
    r_model_init = (models.Gaussian1D(mean=side_mult*5.9,
                                      stddev=0.3,
                                      amplitude=amplitude,
                                      bounds={'mean': mean_bounds,
                                              'stddev': (0.1, 0.4)})
                    + models.Polynomial1D(1, c0=np.min(r_prof)))
    r_fit = fit(r_model_init, r_Rj, r_prof, weights=r_dev**-0.5)
    r_ansa = r_fit.mean_0.value
    dRj = r_fit.stddev_0.value

    #plt.plot(r_Rj, r_prof)
    #plt.plot(r_Rj, r_fit(r_Rj))
    #plt.show()

    #print(f'Ansa position = {r_ansa} +/- {dRj} Rj')

    # Refine our left and right based on the above fit
    narrow_left = int(round(center[1] + (r_ansa - dRj) * Rj_pix))
    narrow_right = int(round(center[1] + (r_ansa + dRj) * Rj_pix))

    narrow_ansa = ccd[bottom:top, narrow_left:narrow_right]
    #simple_show(narrow_ansa)

    # Calculate profile perpendicular to centripetal equator, keeping
    # it in units of rayleighs
    y_pix = np.arange(narrow_ansa.shape[0])
    y_Rj = (y_pix + bottom - center[0]) / Rj_pix
    y_prof = [np.sum(narrow_ansa.data[y, :]) for y in y_pix]
    y_prof = np.asarray(y_prof)
    y_prof /= narrow_ansa.shape[0]
    if ccd.uncertainty.uncertainty_type == 'std':
        y_dev = [np.sum(narrow_ansa.uncertainty.array[y, :])**2 for y in y_pix]
    else:
        y_dev = [np.sum(narrow_ansa.uncertainty.array[y, :]) for y in y_pix]
    y_dev = np.asarray(y_dev)
    amplitude = np.max(y_prof) - np.min(y_prof)
    y_model_init = (models.Gaussian1D(mean=0,
                                      stddev=0.5,
                                      amplitude=amplitude)
                    + models.Polynomial1D(0, c0=np.min(y_prof)))
    y_fit = fit(y_model_init, y_Rj, y_prof, weights=y_dev**-0.5)

    #plt.plot(y_Rj, y_prof)
    #plt.plot(y_Rj, y_fit(y_Rj))
    #plt.show()

    # Units of amplitude are in rayleigh, as per adjustments above,
    # but stddev is in Rj
    pix_stddev = y_fit.stddev_0 * Rj_pix
    pix_stddev_err = y_fit.stddev_0.std * Rj_pix
    sb = (2 * np.pi)**-0.5 * y_fit.amplitude_0 * pix_stddev
    sb_err = ((y_fit.amplitude_0.std / y_fit.amplitude_0)**2
                + (pix_stddev_err**2 / pix_stddev)**2)**0.5
    sb_err *= sb

    #print(f'surface brightness = {sb} +/ {sb_err}')

    # Update bad_ansa when changing these
    ansa_meta = {f'ansa_{side}_r_peak': r_ansa,
                 f'ansa_{side}_r_peak_err': r_fit.mean_0.std,
                 f'ansa_{side}_r_stddev': r_fit.stddev_0.value,
                 f'ansa_{side}_r_stddev_err': r_fit.stddev_0.std,
                 f'ansa_{side}_r_amplitude': r_fit.amplitude_0.value,
                 f'ansa_{side}_r_amplitude_err': r_fit.amplitude_0.std,
                 f'ansa_{side}_y_peak': y_fit.mean_0.value,
                 f'ansa_{side}_y_peak_err': y_fit.mean_0.std,
                 f'ansa_{side}_y_stddev': r_fit.stddev_0.value,
                 f'ansa_{side}_y_stddev_err': r_fit.stddev_0.std,
                 f'ansa_{side}_surf_bright': sb,
                 f'ansa_{side}_surf_bright_err': sb_err}
    for k in ansa_meta.keys():
        ccd.meta[f'HIERARCH {k}'] = ansa_meta[k]
    bmp_meta.update(ansa_meta)
    return ccd

def characterize_ansas(ccd_in, bmp_meta=None,
                       characterize_ansa_crop=SMALL_FILT_CROP,
                       **kwargs):
    # MAKE SURE THIS IS CALLED BEFORE ANY ROTATIONS
    # The underlying reproject stuff in rot_to is a little unstable
    # when it comes to multiple rotations.  When autorotate is used,
    # you have to keep track of individual past rotations (not
    # expected, since celestial N should be the constant reference).
    # When autorotate is not used, rotation come out as expected, but
    # crpix is all weird.  
    cac = np.asarray(characterize_ansa_crop)
    ccd = ccd_in[cac[0,0]:cac[1,0], cac[0,1]:cac[1,1]]
    ccd = ccd_in.copy()
    # MASK GALSATS

    # This behavior is a little unexpected and inconsistent given how
    # I wrote rot_to.  I expect rot_to to start with celestial N time
    # I do a rotation, so I have to rotate by all rotations to get to
    # the desired rotation.  But switching to using shapley calculates
    # incremental rotation, which actually simplifies this logic
    rfk = ccd.meta.get('ROT_FROM_KEYS')
    

    # Make sure we are rotated in the correct direction.  Since rot_to
    # always reorients and rotates relative to celestial N, we just
    # have to make sure all our keys are already in ROT_FROM_KEYS.
    ansa_rots = ['Jupiter_NPole_ang', 'IPT_NPole_ang']
    rfk = ccd.meta.get('ROT_FROM_KEYS')
    if rfk is None:
        rot_angle_from_key = ansa_rots
    else:
        # Check to make sure we have exactly the keys we want, no more
        rfk = rfk.split()
        check_rots = [k in ansa_rots for k in rfk]
        if (len(check_rots) != len(ansa_rots)
            or len(rfk) > len(ansa_rots)):
            rot_angle_from_key = ansa_rots
        else:
            rot_angle_from_key = []
    if len(rot_angle_from_key) > 0:
        print('before corrective rotation')
        ccd.write('/tmp/before.fits', overwrite=True)
        simple_show(ccd)
        print(f'rot_angle_from_key {rot_angle_from_key}')
        ccd = rot_to(ccd, rot_angle_from_key=rot_angle_from_key)
        ccd.write('/tmp/after.fits', overwrite=True)
        simple_show(ccd)
    bmp_meta = bmp_meta or {}
    bmp_meta['tavg'] = ccd.tavg
    for side in ['left', 'right']:
        ccd = ansa_parameters(ccd, bmp_meta=bmp_meta,
                              ansa_side=side, **kwargs)
    return ccd

calibration=None
photometry=None
standard_star_obj=None
solve_timeout=None
join_tolerance=JOIN_TOLERANCE
outdir_root=TORUS_ROOT
fits_fixed_ignore=True

log.setLevel('DEBUG')

#directory = '/data/IoIO/raw/20210607/'

#directory = '/data/IoIO/raw/2017-05-02'
directory = '/data/IoIO/raw/2018-05-08/'

standard_star_obj = standard_star_obj or StandardStar(reduce=True)

##pout = on_off_pipeline(directory,
##                       glob_include=TORUS_NA_NEB_GLOB_LIST,
##                       band='SII',
##                       standard_star_obj=standard_star_obj,
##                       add_ephemeris=galsat_ephemeris,
##                       rot_angle_from_key='Jupiter_NPole_ang',
##                       post_offsub=[extinction_correct, rayleigh_convert,
##                                     rot_to],
##                       outdir_root=outdir_root,
##                       fits_fixed_ignore=fits_fixed_ignore)

#out = on_off_pipeline(directory,
#                      glob_include=TORUS_NA_NEB_GLOB_LIST,
#                      band='SII',
#                      standard_star_obj=standard_star_obj,
#                      add_ephemeris=galsat_ephemeris,
#                      rot_angle_from_key=['Jupiter_NPole_ang',
#                                          'IPT_NPole_ang'],
#                      post_offsub=[extinction_correct, rayleigh_convert,
#                                   rot_to, objctradec_to_obj_center,
#                                   mask_galsats, characterize_ansas,
#                                   as_single],
#                      outdir_root=outdir_root,
#                      fits_fixed_ignore=fits_fixed_ignore)

pout = on_off_pipeline(directory,
                       glob_include=TORUS_NA_NEB_GLOB_LIST,
                       band='SII',
                       standard_star_obj=standard_star_obj,
                       add_ephemeris=galsat_ephemeris,
                       rot_angle_from_key=['Jupiter_NPole_ang'],
                       post_offsub=[extinction_correct, rayleigh_convert,
                                    rot_to, objctradec_to_obj_center,
                                    mask_galsats, characterize_ansas,
                                    as_single],
                       outdir_root=outdir_root,
                       fits_fixed_ignore=fits_fixed_ignore)

