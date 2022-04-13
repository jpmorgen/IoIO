#!/usr/bin/python3

"""Reduce IoIO sodium nebula observations"""

import numpy as np

from scipy.ndimage import shift

from matplotlib.colors import LogNorm

from astropy import log
import astropy.units as u

import ccdproc as ccdp

from bigmultipipe import assure_list, outname_creator

from IoIO.utils import (reduced_dir, multi_glob, closest_in_time,
                        valid_long_exposure, add_history)
from IoIO.simple_show import simple_show
from IoIO.cor_process import standardize_filt_name
from IoIO.calibration import Calibration
from IoIO.cordata import CorData
from IoIO.cormultipipe import (IoIO_ROOT, RAW_DATA_ROOT, CorMultiPipeBase,
                               mask_nonlin_sat, combine_masks,
                               detflux, nd_filter_mask,
                               objctradec_to_obj_center)
from IoIO.photometry import SOLVE_TIMEOUT, JOIN_TOLERANCE, is_flux
from IoIO.cor_photometry import CorPhotometry, add_astrometry
from IoIO.standard_star import (StandardStar, extinction_correct,
                                rayleigh_convert)
from IoIO.na_back import sun_angle, NaBack, na_meso_sub
from IoIO.horizons import galsat_ephemeris

# Use these to pick out on- and off-band filter explicitly, since we
# play with the order of reduction to reduce the on-band images last
# so their metadata and Photometry object stick around the longest.
ON_REGEXP = '_on'
OFF_REGEXP = '_off'

# Early on Na
# Na-*

# Early-on both SII and Na
# IPT-*
# Na_IPT-*
# Na_IPT_R-*
# 
# With IoIO.py shell-out:
# Na_*
# SII_*
# R_*
# U_*
# V_*
# B_*
# 
# Hopefully I will eventually have this.  Note, na_back protects against
# Jupiter sneaking into the background observations by making sure RAOFF
# and/or DECOFF are present.  --> I'll need to do something similar here
# Jupiter*

# So for Na nebula or torus, this boils down to --> But eventually I
# am going to want a separate function that creates the collection,
# making sure Jupiter doesn't have RAOFF and DECOFF
TORUS_NA_NEB_GLOB_LIST = ['IPT*',
                          'Na*', 
                          'SII*', 
                          'Jupiter*']

def off_band_subtract(ccd_in,
                      in_name=None,
                      bmp_meta=None,
                      off_on_ratio=None,
                      calibration=None,
                      show=False,
                      outdir=None, # These get set by the pipeline
                      create_outdir=None,
                      outname_append=None,
                      **kwargs):
    """cormultipipe post-processing routine that subtracts off-band image
    from on-band

    """
    if bmp_meta is None:
        bmp_meta = {}
    if off_on_ratio is None and calibration is None:
        calibration = Calibration(reduce=True)
    on_ccd_fname, off_ccd_fname = [
        (ccd, fname)
        for oo_regexp in [ON_REGEXP, OFF_REGEXP]
        for ccd, fname in zip(ccd_in, in_name)
        if oo_regexp in ccd.meta['FILTER']]
    on, on_fname = on_ccd_fname
    off, off_fname = off_ccd_fname
    filt = on.meta['FILTER']
    band, _ = filt.split('_')
    if off_on_ratio is None:
        off_on_ratio, _ = calibration.flat_ratio(band)

    shift_off = on.obj_center - off.obj_center
    d_on_off = np.linalg.norm(shift_off)
    off = ccdp.transform_image(off, shift, shift=shift_off)
    if d_on_off > 5:
            log.warning('On- and off-band image centers are > 5 pixels apart')

    assert is_flux(on.unit) and is_flux(off.unit)
    off = off.divide(off_on_ratio,
                     handle_meta='first_found')
    on = on.subtract(off, handle_meta='first_found')
    on.meta['OFF_BAND'] = off_fname
    on.meta['OFF_SCL'] = (1/off_on_ratio, 'scale factor applied to OFF_BAND')
    on.meta['HIERARCH SHIFT_OFF_X'] = (shift_off[1], '[pix] x dist on - off')
    on.meta['HIERARCH SHIFT_OFF_Y'] = (shift_off[0], '[pix] y dist on - off')
    on.meta['HIERARCH SHIFT_OFF_D'] = (d_on_off, '[pix] dist on - off')
    add_history(on.meta, 'Scaled and shifted OFF_BAND by OFF_SCL and SHIFT_OFF')
    outname = outname_creator(on_fname, outdir=outdir,
                              create_outdir=create_outdir,
                              outname_append=outname_append)
    tmeta = {'band': band,
             'off_on_ratio': off_on_ratio,
             'shift_off': shift_off,
             'outname': outname}
    bmp_meta.update(tmeta)
    return on
            
def on_off_pipeline(directory=None, # raw day directory
                    bmp=None, # BigMultiPipe object
                    band=None,
                    collection=None,
                    glob_include=None, # Used if collection is None
                    calibration=None,
                    photometry=None,
                    add_ephemeris=None,
                    pre_backsub=None,
                    post_backsub=None,
                    num_processes=None,
                    outdir=None,
                    outdir_root=None,
                    create_outdir=True,
                    **kwargs):

    if collection is not None and glob_include is not None:
        raise ValueError('specify either collection of glob_include to make sure the correct files are selected')
    if calibration is None:
        calibration = Calibration(reduce=True)
    if photometry is None:
        photometry = CorPhotometry(precalc=True)
    add_ephemeris = assure_list(add_ephemeris)
    pre_backsub = assure_list(pre_backsub)
    post_backsub = assure_list(post_backsub)
    post_process_list = [combine_masks,
                         mask_nonlin_sat,
                         *add_ephemeris,
                         add_astrometry,
                         *pre_backsub,
                         off_band_subtract,
                         *post_backsub]
    if outdir is None and outdir_root is None:
        # Right now, dump Na and SII into separate top-level reduction
        # directories
        outdir_root = os.path.join(IoIO_ROOT, band)
        outdir = outdir or reduced_dir(directory, outdir_root, create=False)
    if collection is None:
        flist = multi_glob(directory, glob_include)
        collection = ccdp.ImageFileCollection(directory, filenames=flist)
    standardize_filt_name(collection)
    f_pairs = closest_in_time(collection, (f'{band}_on', f'{band}_off'),
                              valid_long_exposure,
                              directory=directory)
    # I want to reduce the on-band images last so their metadata and
    # Photometry objects stick around for subsequent processing.  This
    # will make it hard to get the information for the off-band
    # filters in case I want to use that for calibration purposes
    # too.  Cross that bridge when I come to it.  Note that list
    # method reverse does not work with lists of strings, presumably
    # because strings are invariants and reverse() does it work in place
    f_pairs = [list(reversed(p)) for p in f_pairs]
    if len(f_pairs) == 0:
        log.warning(f'No matching set of on-off {band} files found '
                    f'in {directory}')
        return []
 
    cmp = CorMultiPipeBase(
        ccddata_cls=CorData,
        auto=True,
        calibration=calibration,
        photometry=photometry,
        outname_append='-back-sub',
        outname_ext='.fits', 
        create_outdir=create_outdir,
        post_process_list=post_process_list,
        num_processes=num_processes,
        **kwargs)
    pout = cmp.pipeline([f_pairs[0]], outdir=outdir, overwrite=True)
    return pout
    #pout = cmp.pipeline(f_pairs, outdir=outdir, overwrite=True)
    pout, _ = prune_pout(pout, f_pairs)
    return pout

log.setLevel('DEBUG')

#### from cor_process import cor_process
#### c = Calibration(reduce=True)
#### fname = '/data/IoIO/raw/2018-05-08/Na_off-band_011.fits'
#### rccd = CorData.read(fname)
#### ccd = cor_process(rccd, calibration=c, auto=True)


#from IoIO.cormultipipe import full_frame
#ccd = CorData.read('/data/IoIO/raw/2017-05-02/IPT-0011_on-band.fit')
#print(full_frame(ccd))

### from cor_process import cor_process
### c = Calibration(reduce=True)
### fname = '/data/IoIO/raw/2021-05-08/Mercury-0006_Na-on.fit'
### rccd = CorData.read(fname)
### ccd = cor_process(rccd, calibration=c, auto=True)
### ccd = obj_ephemeris(ccd, horizons_id=199)
### print(ccd.meta)


##################################
# This would be na_neb_directory or torus_directory

##directory = '/data/IoIO/raw/2017-05-02'
directory = '/data/IoIO/raw/2018-05-08/'
calibration=None
photometry=None
solve_timeout=None
join_tolerance=JOIN_TOLERANCE

if calibration is None:
    calibration = Calibration(reduce=True)
if photometry is None:
    photometry = CorPhotometry(precalc=True,
        solve_timeout=solve_timeout,# These belong go at the calling level
        join_tolerance=join_tolerance)

#pout = on_off_pipeline(directory, band='Na',
#                       fits_fixed_ignore=True)

na_meso_obj = NaBack(reduce=True)
standard_star_obj = StandardStar(reduce=True, write_summary_plots=True)

# --> here is where I make a multiglob flist, put it into a collection
# and discard frames where RAOFF and DECOFF are present
pout = on_off_pipeline(directory,
                       glob_include=TORUS_NA_NEB_GLOB_LIST,
                       band='Na',
                       na_meso_obj=na_meso_obj,
                       standard_star_obj=standard_star_obj,
                       add_ephemeris=galsat_ephemeris,
                       pre_backsub=[objctradec_to_obj_center, detflux],
                       post_backsub=[sun_angle, na_meso_sub,
                                     extinction_correct, rayleigh_convert],
                       fits_fixed_ignore=True,
                       process_expand_factor=15) # --> Not sure if this is accurate

#pout = on_off_pipeline(directory,
#                       glob_include=TORUS_NA_NEB_GLOB_LIST,
#                       band='SII',
#                       standard_star_obj=standard_star_obj,
#                       add_ephemeris=galsat_ephemeris,
#                       pre_backsub=[objctradec_to_obj_center, detflux],
#                       post_backsub=[extinction_correct, rayleigh_convert],
#                       fits_fixed_ignore=True,
#                       process_expand_factor=15) # --> Not sure if this is accurate
#


#pout = on_off_pipeline(directory, band='SII',
#                       add_ephemeris=galsat_ephemeris,
#                       pre_backsub=[objctradec_to_obj_center, detflux],
#                       fits_fixed_ignore=True)



#from IoIO.cormultipipe import full_frame
#ccd = CorData.read('/data/IoIO/Na/2018-05-08/Na_on-band_010-back-sub.fits')
#simple_show(ccd, norm=LogNorm())


#on = CorData.read(p[0])
#off = CorData.read(p[1])
#off_band_subtract([on, off], in_name=p, band='Na')

#from IoIO.cordata import CorData
#from IoIO.cor_process import cor_process
#from IoIO.calibration import Calibration
#
#fname = '/data/IoIO/raw/2018-05-08/Na_on-band_010.fits'
## Bad focus
##fname = '/data/IoIO/raw/20220312/HAT-P-36b-S002-R010-C002-R.fts'
##fname = '/data/IoIO/raw/20220308/HAT-P-36b-S002-R010-C002-R.fts'
#rccd = CorData.read(fname)
#c = Calibration(reduce=True)
#ccd = cor_process(rccd, calibration=c, auto=True, gain=True)
#ccd = ccd.divide(ccd.meta['EXPTIME']*u.s, handle_meta='first_found')
#ecccd = extinction_correct(ccd, standard_star_obj=standard_star_obj)
#rccd = to_rayleigh(ecccd, standard_star_obj=standard_star_obj)

#mecdata = standard_star_obj.extinction_correct(
#    u.Magnitude(ccd.data*ccd.unit),
#    ccd.meta['AIRMASS'],
#    ccd.meta['FILTER'])
#mecerr = standard_star_obj.extinction_correct(
#    u.Magnitude(ccd.uncertainty.array*ccd.unit),
#    ccd.meta['AIRMASS'],
#    ccd.meta['FILTER'])
#mecdata = mecdata.physical
#mecerr = mecerr.physical
#mccd = ccd.copy()
#mccd.data = mecdata.value
#mccd.uncertainty = StdDevUncertainty(mecerr.value)



#from cordata import CorData
#ccd = CorData.read('/data/IoIO/raw/2018-05-08/Na_on-band_011.fits')

#bmp_meta = {}
#ccd = galsat_ephemeris(ccd, bmp_meta=bmp_meta)

#import matplotlib.pyplot as plt
#    
#def simple_show(im, **kwargs):
#    fig, ax = plt.subplots()
#    ax.imshow(im, origin='lower',
#              cmap=plt.cm.gray,
#              filternorm=0, interpolation='none',
#              **kwargs)
#    ax.format_coord = CCDImageFormatter(im.data)
#    plt.show()