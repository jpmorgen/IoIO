#!/usr/bin/python3
"""Reduce Mercury observations"""

import ccdproc as ccdp

from ccdmultipipe import as_single

from IoIO.cormultipipe import detflux, objctradec_to_obj_center
from IoIO.standard_star import (StandardStar, extinction_correct,
                                rayleigh_convert)
from IoIO.horizons import obj_ephemeris
from IoIO.on_off_pipeline import on_off_pipeline

MERCURY_ROOT = '/data/Mercury'

standard_star_obj = StandardStar(reduce=True)

# --> Might want to have a pre-backsub routine that adds Mercury as
# --> OBJECT, since that got messed up on occation -- was set to a
# --> focus star and I forgot to go explicitly go to Mercury

#directory = '/data/IoIO/raw/2018-05-08/'
directory = '/data/IoIO/raw/2021-10-28/'

# --> Eventually it might be nice to reduce all of the images
# separately, somehow rejecting the ones that have too much saturation
# and then revisiting closest_in_time

collection = ccdp.ImageFileCollection(
    directory, glob_include='Mercury*_Na*',
    glob_exclude='*_moving_to*')

pout = on_off_pipeline(directory,
                       collection=collection,
                       band='Na',
                       horizons_id='199',
                       horizons_id_type='majorbody',
                       obj_ephm_prefix='Mercury',
                       standard_star_obj=standard_star_obj,
                       add_ephemeris=obj_ephemeris,
                       post_backsub=[extinction_correct,
                                     rayleigh_convert, as_single],
                       outdir_root=MERCURY_ROOT,
                       fits_fixed_ignore=True,
                       solve_timeout=2)

