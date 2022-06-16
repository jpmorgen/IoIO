#!/usr/bin/python3

"""Reduce IoIO sodium nebula observations"""

import os

import numpy as np

from astropy import log

import ccdproc as ccdp

from ccdmultipipe import as_single

from IoIO.utils import multi_glob, reduced_dir
from IoIO.cor_process import standardize_filt_name
from IoIO.cormultipipe import (IoIO_ROOT, RAW_DATA_ROOT,
                               calc_obj_to_ND, #crop_ccd,
                               planet_to_object,
                               objctradec_to_obj_center)
from IoIO.calibration import Calibration
from IoIO.photometry import (SOLVE_TIMEOUT, JOIN_TOLERANCE,
                             JOIN_TOLERANCE_UNIT, rot_to)
from IoIO.cor_photometry import CorPhotometry
from IoIO.on_off_pipeline import (TORUS_NA_NEB_GLOB_LIST,
                                  TORUS_NA_NEB_GLOB_EXCLUDE_LIST,
                                  on_off_pipeline)
from IoIO.standard_star import (StandardStar, extinction_correct,
                                rayleigh_convert)
from IoIO.horizons import galsat_ephemeris
from IoIO.na_back import sun_angle, NaBack, na_meso_sub
from IoIO.torus import pixel_per_Rj

NA_NEBULA_ROOT = os.path.join(IoIO_ROOT, 'Na_nebula')

def na_nebula_directory(directory,
                        glob_include=TORUS_NA_NEB_GLOB_LIST,
                        glob_exclude_list=TORUS_NA_NEB_GLOB_EXCLUDE_LIST,
                        outdir=None,
                        outdir_root=NA_NEBULA_ROOT,
                        na_meso_obj=None,
                        standard_star_obj=None,
                        fits_fixed_ignore=True,
                        **kwargs):
    standard_star_obj = standard_star_obj or StandardStar(reduce=True)
    # I will want this in the cached_pout somehow
    flist = multi_glob(directory, glob_include, glob_exclude_list)    
    if len(flist) == 0:
        return []
    outdir = outdir or reduced_dir(directory, outdir_root, create=False)
    poutname = os.path.join(outdir, 'Torus.pout')
    collection = ccdp.ImageFileCollection(directory, filenames=flist)
    standardize_filt_name(collection)
    # Get rid of off-Jupiter pointings
    st = collection.summary
    if 'raoff' in st.colnames or 'decoff' in st.colnames:
        good_mask = np.logical_and(st['raoff'].mask, st['decoff'].mask)
        fbases = st['file'][good_mask]
        nflist = [os.path.join(directory, f) for f in fbases]
        if len(nflist) == 0:
            return []
        collection = ccdp.ImageFileCollection(directory, filenames=nflist)
    na_meso_obj = na_meso_obj or NaBack(reduce=True)
    pout = on_off_pipeline(directory,
                           collection=collection,
                           band='Na',
                           na_meso_obj=na_meso_obj,
                           add_ephemeris=galsat_ephemeris,
                           planet='Jupiter',
                           rot_angle_from_key=['Jupiter_NPole_ang',
                                               'IPT_NPole_ang'],
                           standard_star_obj=standard_star_obj,
                           post_process_list=[calc_obj_to_ND, planet_to_object],
                           post_offsub=[sun_angle, na_meso_sub,
                                        extinction_correct, rayleigh_convert,
                                        rot_to, as_single],
                           outdir_root=outdir_root,
                           fits_fixed_ignore=fits_fixed_ignore,
                           **kwargs)

log.setLevel('DEBUG')

#directory = '/data/IoIO/raw/20210607/'
#directory = '/data/IoIO/raw/20211017/'

#directory = '/data/IoIO/raw/2017-05-02'
directory = '/data/IoIO/raw/2018-05-08/'
calibration=None
photometry=None
standard_star_obj=None
na_meso_obj=None
solve_timeout=SOLVE_TIMEOUT
join_tolerance=JOIN_TOLERANCE*JOIN_TOLERANCE_UNIT

outdir_root=NA_NEBULA_ROOT
fits_fixed_ignore=True
photometry = (
    photometry
    or CorPhotometry(precalc=True,
                     solve_timeout=solve_timeout,
                     join_tolerance=join_tolerance))
calibration = calibration or Calibration(reduce=True)
na_meso_obj = na_meso_obj or NaBack(reduce=True)
standard_star_obj = standard_star_obj or StandardStar(reduce=True,
                                                      stop='2022-06-13')
outdir_root = outdir_root or os.path.join(IoIO_ROOT, 'Na_nebula')
#na_nebula_directory(directory,
#                    calibration=calibration,
#                    photometry=photometry,
#                    standard_star_obj=standard_star_obj,                    
#                    solve_timeout=solve_timeout,
#                    join_tolerance=join_tolerance,
#                    outdir_root=outdir_root,
#                    fits_fixed_ignore=fits_fixed_ignore)

from IoIO.simple_show import simple_show
from IoIO.cordata import CorData
ccd = CorData.read('/data/IoIO/Na_nebula/2018-05-08/Na_on-band_001-back-sub.fits')

# Start of na_apertures
