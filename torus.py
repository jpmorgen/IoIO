#!/usr/bin/python3

"""Reduce IoIO Io plamsa torus observations"""

import os

import numpy as np

from astropy import log
import astropy.units as u

from IoIO.cormultipipe import IoIO_ROOT
from IoIO.calibration import Calibration
from IoIO.photometry import SOLVE_TIMEOUT, JOIN_TOLERANCE, rot_to
from IoIO.cor_photometry import CorPhotometry
from IoIO.on_off_pipeline import TORUS_NA_NEB_GLOB_LIST, on_off_pipeline
from IoIO.standard_star import (StandardStar, extinction_correct,
                                rayleigh_convert)
from IoIO.horizons import galsat_ephemeris

TORUS_ROOT = os.path.join(IoIO_ROOT, 'Torus')

# https://lasp.colorado.edu/home/mop/files/2015/02/CoOrd_systems7.pdf
# Has sysIII of intersection of mag and equatorial plains at 290.8.
# That means tilt is toward 200.8, which is my basic recollection
CENTRIFUGAL_EQUATOR_AMPLITUDE = 6.8*u.deg
JUPITER_MAG_SYSIII = 290.8*u.deg

def IPT_NPole_ang(ccd, **kwargs):
    sysIII = ccd.meta['Jupiter_PDObsLon']
    ccd.meta['HIERARCH IPT_NPole_ang'] = (
        CENTRIFUGAL_EQUATOR_AMPLITUDE * np.sin(sysIII - JUPITER_MAG_SYSIII))
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

#pout = on_off_pipeline(directory,
#                       glob_include=TORUS_NA_NEB_GLOB_LIST,
#                       band='SII',
#                       standard_star_obj=standard_star_obj,
#                       add_ephemeris=galsat_ephemeris,
#                       rot_angle_from_key='Jupiter_NPole_ang',
#                       post_backsub=[extinction_correct, rayleigh_convert,
#                                     rot_to],
#                       outdir_root=outdir_root,
#                       fits_fixed_ignore=fits_fixed_ignore)

pout = on_off_pipeline(directory,
                       glob_include=TORUS_NA_NEB_GLOB_LIST,
                       band='SII',
                       standard_star_obj=standard_star_obj,
                       add_ephemeris=galsat_ephemeris,
                       rot_angle_from_key=['Jupiter_NPole_ang',
                                           'IPT_NPole_ang'],
                       post_backsub=[extinction_correct, rayleigh_convert,
                                     rot_to],
                       outdir_root=outdir_root,
                       fits_fixed_ignore=fits_fixed_ignore)


