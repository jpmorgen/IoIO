#!/usr/bin/python3

"""Reduce IoIO sodium nebula observations"""

import os

import numpy as np

from astropy import log

import ccdproc as ccdp

from IoIO.utils import multi_glob
from IoIO.cormultipipe import IoIO_ROOT
from IoIO.calibration import Calibration
from IoIO.photometry import SOLVE_TIMEOUT, JOIN_TOLERANCE, rot_to
from IoIO.cor_photometry import CorPhotometry
from IoIO.on_off_pipeline import TORUS_NA_NEB_GLOB_LIST, on_off_pipeline
from IoIO.standard_star import (StandardStar, extinction_correct,
                                rayleigh_convert)
from IoIO.horizons import galsat_ephemeris
from IoIO.na_back import sun_angle, NaBack, na_meso_sub

NA_NEBULA_ROOT = os.path.join(IoIO_ROOT, 'Na_nebula')

def na_nebula_directory(directory,
                        calibration=None,
                        photometry=None,
                        standard_star_obj=None,
                        na_meso_obj=None,
                        solve_timeout=None,
                        join_tolerance=JOIN_TOLERANCE,
                        outdir_root=NA_NEBULA_ROOT,
                        fits_fixed_ignore=True,
                        **kwargs):
                        
    flist = multi_glob(directory, TORUS_NA_NEB_GLOB_LIST)
    if len(flist) == 0:
        return []
    # Get rid of off-Jupiter pointings
    collection = ccdp.ImageFileCollection(directory, filenames=flist)
    st = collection.summary
    if 'raoff' in st.colnames or 'decoff' in st.colnames:
        good_mask = np.logical_and(st['raoff'].mask, st['decoff'].mask)
        fbases = st['file'][good_mask]
        nflist = [os.path.join(directory, f) for f in fbases]
        if len(nflist) == 0:
            return []
        collection = ccdp.ImageFileCollection(directory, filenames=nflist)
    na_meso_obj = na_meso_obj or NaBack(reduce=True)
    standard_star_obj = standard_star_obj or StandardStar(reduce=True)
    pout = on_off_pipeline(directory,
                           collection=collection,
                           band='Na',
                           na_meso_obj=na_meso_obj,
                           standard_star_obj=standard_star_obj,
                           add_ephemeris=galsat_ephemeris,
                           rot_angle_from_key='Jupiter_NPole_ang',
                           post_offsub=[sun_angle, na_meso_sub,
                                         extinction_correct, rayleigh_convert,
                                         rot_to],
                           outdir_root=outdir_root,
                           fits_fixed_ignore=fits_fixed_ignore,
                           **kwargs)

log.setLevel('DEBUG')

#directory = '/data/IoIO/raw/20210607/'
directory = '/data/IoIO/raw/20211017/'

#directory = '/data/IoIO/raw/2017-05-02'
#directory = '/data/IoIO/raw/2018-05-08/'
calibration=None
photometry=None
standard_star_obj=None
na_meso_obj=None
solve_timeout=None
join_tolerance=JOIN_TOLERANCE
outdir_root=NA_NEBULA_ROOT
fits_fixed_ignore=False#True
photometry = (
    photometry
    or CorPhotometry(precalc=True,
                     solve_timeout=solve_timeout,
                     join_tolerance=join_tolerance))
calibration = calibration or Calibration(reduce=True)
na_meso_obj = na_meso_obj or NaBack(reduce=True)
standard_star_obj = standard_star_obj or StandardStar(reduce=True)
outdir_root = outdir_root or os.path.join(IoIO_ROOT, 'Na_nebula')
na_nebula_directory(directory,
                    calibration=calibration,
                    photometry=photometry,
                    solve_timeout=solve_timeout,
                    join_tolerance=join_tolerance,
                    outdir_root=outdir_root,
                    fits_fixed_ignore=fits_fixed_ignore)
