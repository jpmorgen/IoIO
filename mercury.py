#!/usr/bin/python3
"""Reduce Mercury observations"""

import astropy.units as u

import ccdproc as ccdp

from ccdmultipipe import as_single

from IoIO.cormultipipe import (CorMultiPipeBase, detflux,
                               nd_filter_mask)
from IoIO.standard_star import (StandardStar, extinction_correct,
                                rayleigh_convert)
from IoIO.photometry import (SOLVE_TIMEOUT, JOIN_TOLERANCE,
                             rot_to, flip_along)
from IoIO.cor_photometry import CorPhotometry, write_photometry
from IoIO.horizons import obj_ephemeris
from IoIO.on_off_pipeline import on_off_pipeline

MERCURY_ROOT = '/data/Mercury'
OFF_ND_EDGE_EXPAND = (-10, +5)

class MercuryMultiPipe(CorMultiPipeBase):
    def file_write(self, ccd, outname,
                   in_name=None, photometry=None,
                   **kwargs):
        written_name = super().file_write(
            ccd, outname, photometry=photometry, **kwargs)
        # f_pairs has been reversed to photometry is from on-band
        write_photometry(in_name=in_name[1], outname=outname,
                         photometry=photometry,
                         write_wide_source_table=True,
                         **kwargs)
        outroot, _ = os.path.splitext(outname)
        return written_name

def backsub(ccd_in, bmp_meta=None,
            back_off_boxes=20,
            off_nd_edge_expand=0, **kwargs):
    ccd = ccd_in.copy()
    photometry = CorPhotometry(ccd, n_back_boxes=back_off_boxes)
    back = ccd.copy()
    back.mask = None
    back = nd_filter_mask(back, nd_edge_expand=off_nd_edge_expand)
    back.data = photometry.background.value
    back.uncertainty.array = photometry.back_rms.value    
    back.data[back.mask] = 0
    back.uncertainty.array[back.mask] = 0
    back.mask = None
    ccd = ccd.subtract(back, handle_meta='first_found')
    return ccd

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

#pout = on_off_pipeline(directory,
#                       collection=collection,
#                       band='Na',
#                       PipeObj=MercuryMultiPipe,
#                       smooth_off=True,
#                       horizons_id='199',
#                       horizons_id_type='majorbody',
#                       obj_ephm_prefix='Mercury',
#                       standard_star_obj=standard_star_obj,
#                       add_ephemeris=obj_ephemeris,
#                       off_nd_edge_expand=OFF_ND_EDGE_EXPAND,
#                       post_backsub=[backsub, extinction_correct,
#                                     rayleigh_convert, as_single],
#                       outdir_root=MERCURY_ROOT,
#                       fits_fixed_ignore=True)#,
#                       #solve_timeout=2)

## Checks out
#pout = on_off_pipeline(directory,
#                       collection=collection,
#                       band='Na',
#                       PipeObj=MercuryMultiPipe,
#                       smooth_off=True,
#                       horizons_id='199',
#                       horizons_id_type='majorbody',
#                       obj_ephm_prefix='Mercury',
#                       standard_star_obj=standard_star_obj,
#                       add_ephemeris=obj_ephemeris,
#                       rot_angle_from_key='TARGET_NPole_ang',
#                       #flip_along_axis=1,
#                       #flip_angle_from_key='TARGET_sunTargetPA',
#                       off_nd_edge_expand=OFF_ND_EDGE_EXPAND,
#                       post_backsub=[backsub, extinction_correct, rot_to,
#                                     as_single],
#                       outdir_root=MERCURY_ROOT,
#                       fits_fixed_ignore=True,
#                       solve_timeout=2)

pout = on_off_pipeline(directory,
                       collection=collection,
                       band='Na',
                       PipeObj=MercuryMultiPipe,
                       smooth_off=True,
                       horizons_id='199',
                       horizons_id_type='majorbody',
                       obj_ephm_prefix='Mercury',
                       standard_star_obj=standard_star_obj,
                       add_ephemeris=obj_ephemeris,
                       rot_angle_from_key='TARGET_NPole_ang',
                       flip_along_axis=1,
                       flip_angle_from_key='TARGET_sunTargetPA',
                       off_nd_edge_expand=OFF_ND_EDGE_EXPAND,
                       post_backsub=[backsub, extinction_correct,
                                     rayleigh_convert, rot_to,
                                     flip_along, as_single],
                       outdir_root=MERCURY_ROOT,
                       fits_fixed_ignore=True,
                       solve_timeout=2)

## Checks out
#pout = on_off_pipeline(directory,
#                       collection=collection,
#                       band='Na',
#                       PipeObj=MercuryMultiPipe,
#                       smooth_off=True,
#                       horizons_id='199',
#                       horizons_id_type='majorbody',
#                       obj_ephm_prefix='Mercury',
#                       standard_star_obj=standard_star_obj,
#                       add_ephemeris=obj_ephemeris,
#                       rot_to_angle=270*u.deg,
#                       rot_angle_from_key='TARGET_sunTargetPA',
#                       off_nd_edge_expand=OFF_ND_EDGE_EXPAND,
#                       post_backsub=[backsub, extinction_correct,
#                                     rayleigh_convert, rot_to,
#                                     as_single],
#                       outdir_root=MERCURY_ROOT,
#                       fits_fixed_ignore=True,
#                       solve_timeout=2)

#pout = on_off_pipeline(directory,
#                       collection=collection,
#                       band='Na',
#                       PipeObj=MercuryMultiPipe,
#                       smooth_off=True,
#                       horizons_id='199',
#                       horizons_id_type='majorbody',
#                       obj_ephm_prefix='Mercury',
#                       standard_star_obj=standard_star_obj,
#                       add_ephemeris=obj_ephemeris,
#                       rot_to_angle=270*u.deg,
#                       rot_angle_from_key='TARGET_sunTargetPA',
#                       flip_along_axis=0,
#                       flip_angle_from_key='TARGET_NPole_ang',
#                       off_nd_edge_expand=OFF_ND_EDGE_EXPAND,
#                       post_backsub=[backsub, extinction_correct,
#                                     rayleigh_convert, rot_to,
#                                     flip_along, as_single],
#                       outdir_root=MERCURY_ROOT,
#                       fits_fixed_ignore=True,
#                       solve_timeout=2)
