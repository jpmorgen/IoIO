#!/usr/bin/python3

"""Reduce IoIO sodium nebula observations"""

import numpy as np

from scipy.ndimage import shift

from matplotlib.colors import LogNorm

from astropy import log
import astropy.units as u
from astropy.table import MaskedColumn
from astropy.coordinates import Angle, SkyCoord
from astroquery.jplhorizons import Horizons

from ccdproc import ImageFileCollection, transform_image

from bigmultipipe import assure_list, outname_creator#, cached_pout, prune_pout

from IoIO.utils import (reduced_dir, multi_glob, closest_in_time,
                        valid_long_exposure, add_history,
                        location_to_dict)
from IoIO.simple_show import simple_show
from IoIO.cor_process import IOIO_1_LOCATION, standardize_filt_name
from IoIO.calibration import Calibration
from IoIO.cordata import CorData
from IoIO.cormultipipe import (IoIO_ROOT, RAW_DATA_ROOT,
                               CorMultiPipeBase, mask_nonlin_sat,
                               combine_masks, detflux, nd_filter_mask)
from IoIO.photometry import is_flux
from IoIO.cor_photometry import CorPhotometry, add_astrometry
from IoIO.standard_star import (StandardStar, extinction_correct,
                                rayleigh_convert)
from IoIO.na_back import sun_angle, NaBack, na_meso_sub

# Use these to pick out on- and off-band filter explicitly, since we
# play with the order of reduction to reduce the on-band images last
# so their metadata and Photometry object stick around the longest.
ON_REGEXP = '_on'
OFF_REGEXP = '_off'

# Early on Na
# Na-*

# Early on both SII and Na
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
# and/or DECOFF are present.  --> I'll need to do somethign similar here
# Jupiter*

# So for Na nebula or torus, this boils down to
NA_SII_GLOB_LIST = ['IPT*',
                    'Na*', 
                    'SII*', 
                    'Jupiter*']

# https://ssd.jpl.nasa.gov/horizons/manual.html

# HORIZONS associations
GALSATS = {
    'Jupiter': 599,
    'Io': 501,
    'Europa': 502,
    'Ganymede': 503,
    'Callisto': 504}

OBS_COL_NUMS = \
    '1, 3, 6, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26, 27'

OBS_COL_TO_META = ['RA', 'DEC', 'r', 'r_rate', 'delta', 'delta_rate',
                   'PDObsLon', 'PDObsLat', 'PDSunLon', 'SubSol_ang',
                   'SubSol_dist', 'sysIII', 'phi']
GALSAT_COL_NUMS = '14, 15'
KEYS_TO_SOURCE_TABLE = ['DATE-AVG',
                        'FILTER',
                        'AIRMASS']


#GALSAT_COL_TO_META = ['PDObsLon', 'PDObsLat', 'PDSunLon', 'SubSol_ang']

###class OnOffCorMultiPipe(CorMultiPipeBase):
###    def outname_create(self, in_names,
###                       **kwargs):
###        """Create outname from first file in in_names list.  Further
###        modification to the outname can be done using outname_append
###
###        """
###        outname = super().outname_create(in_names[0], **kwargs)
###        return outname
    
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
    off = transform_image(off, shift, shift=shift_off)
    if d_on_off > 5:
            log.warning('On- and off-band image centers are > 5 pixels apart')

    assert is_flux(on.unit) and is_flux(off.unit)
    off = off.divide(off_on_ratio*u.dimensionless_unscaled,
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
            
def galsat_ephemeris(ccd_in,
                     bmp_meta=None,
                     in_name=None,
                     obs_loc=None,
                     obs_col_to_meta=None,
                     **kwargs):
    """cormultipipe post-processing routine to add Jupiter and Galilean
    satellite metadata to ccd.meta and bmp_meta

    """
    if isinstance(ccd_in, list):
        if isinstance(in_name, list):
            return [galsat_ephemeris(ccd, bmp_meta=bmp_meta, 
                                     in_name=fname, obs_loc=obs_loc,
                                     obs_col_to_meta=obs_col_to_meta,
                                     **kwargs)
                    for ccd, fname in zip(ccd_in, in_name)]
        else:
            return [galsat_ephemeris(ccd, bmp_meta=bmp_meta, 
                                     in_name=in_name, obs_loc=obs_loc,
                                     obs_col_to_meta=obs_col_to_meta,
                                     **kwargs)
                for ccd in ccd_in]
    if ccd_in.meta['OBJECT'] != 'Jupiter':
        return ccd_in
    ccd = ccd_in.copy()
    if obs_loc is None:
        obs_loc = IOIO_1_LOCATION
    if obs_col_to_meta is None:
        obs_col_to_meta = OBS_COL_TO_META
    obs_name = obs_loc.info.name

    # Get our ephemerides from the perspective of the observatory
    # id_type='majorbody' required because minor bodies (e.g. asteroids)
    # are the default
    obs_eph = None
    for galsat in GALSATS:
        h = Horizons(id=GALSATS[galsat],
                     epochs=ccd.tavg.jd,
                     location=location_to_dict(obs_loc),
                     id_type='majorbody')
        e = h.ephemerides(quantities=OBS_COL_NUMS)
        if obs_eph is None:
            obs_eph = e
        else:        
            obs_eph.add_row(e[0])

    # Repeat from the perspective of each satellite so we can get proper
    # Jovian sysIII and heliocentric phi.  Note, because of light travel
    # time of ~30 min to Earth, the system looks different when observed
    # at Earth than when the light left each object in the system.  We
    # want sysIII and phi when the light left each object.
    gs_eph = None
    for galsat in GALSATS:
        if galsat == 'Jupiter':
            continue
        mask = obs_eph['targetname'] == f'{galsat} ({GALSATS[galsat]})'
        lt = obs_eph['lighttime'][mask]
        epoch = ccd.tavg.jd*u.day - lt
        h = Horizons(id=GALSATS['Jupiter'],
                     epochs=epoch.value,
                     location=f'500@{GALSATS[galsat]}',
                     id_type='majorbody')
        e = h.ephemerides(quantities=GALSAT_COL_NUMS)
        if gs_eph is None:
            gs_eph = e
        else:
            gs_eph.add_row(e[0])
    # Note that PDSonLon is given in sysIII, so we use that and the moon's
    # sysIII to calculate phi relative to Jupiter's sub-solar point.  Add
    # 180 deg, since Jovian orbital phase ref point is midnight.  NOTE:
    # QTables don't play Quantity by default.  If you combine columns the
    # unit is blown away.  If you combine as Quantity, as I prefer to do,
    # the Column-ness is blown away
    sysIIIs = gs_eph['PDObsLon'].quantity
    phis = gs_eph['PDSunLon'].quantity - sysIIIs + 180*u.deg
    phis = MaskedColumn(phis)
    # Prepare to add columns to obs_eph.  Jupiter row is masked
    sysIIIs = gs_eph['PDObsLon'].insert(0, np.NAN)
    phis = phis.insert(0, np.NAN)
    sysIIIs.mask[0] = True
    phis.mask[0] = True

    obs_eph.add_columns([sysIIIs, phis], names=['sysIII', 'phi'])

    for galsat in GALSATS:
        mask = obs_eph['targetname'] == f'{galsat} ({GALSATS[galsat]})'
        te = obs_eph[mask]
        if galsat == 'Jupiter':
            # Put Jupiter's precise coordinates into header
            ra = Angle(te['RA'].quantity)
            dec = Angle(te['DEC'].quantity)
            ccd.meta['OBJCTRA'] = (ra[0].to_string(unit=u.hour),
                                   'OBJECT RA from HORIZONS')
            ccd.meta['OBJCTDEC'] = (dec[0].to_string(unit=u.deg),
                                    'OBJECT DEC from HORIZONS')
            
        for col in obs_col_to_meta:
            if te[col].mask[0]:
                continue
            ccd.meta[f'HIERARCH {galsat}_{col}'] = \
                (te[col][0], f'[{te[col].unit.to_string()}]')


    # I am going to leave this out for now because I can get it from
    # ccd.meta.  I may want to distill it to a simple dictionary, but
    # that is what the metadata are....
    #bmp_meta['obs_eph'] = obs_eph
    return ccd

def objctradec_to_obj_center(ccd_in, bmp_meta=None, **kwargs):
    """Sets obj_center from OBJCTRA and OBJCTDEC keywords

    OBJCTRA and OBJCTDEC are assumed to be set from an astrometry
    calculation and thus center_quality is set to 10

    """
    if isinstance(ccd_in, list):
        return [objctradec_to_obj_center(ccd, bmp_meta=bmp_meta,
                                         **kwargs)
                for ccd in ccd_in]
    if ccd_in.wcs is None:
        return ccd_in
    objctra = ccd_in.meta.get('OBJCTRA')
    objctdec = ccd_in.meta.get('OBJCTDEC')
    if objctra is None and objctdec is None:
        return ccd_in        
    ccd = ccd_in.copy()
    bmp_meta = bmp_meta or {}
    cent = SkyCoord(objctra, objctdec, unit=(u.hour, u.deg))
    x, y = ccd.wcs.world_to_pixel(cent)
    ocenter = ccd.obj_center
    oquality = ccd.center_quality
    ccd.obj_center = (y, x)
    ccd.center_quality = 10
    dcent = ccd.obj_center - ocenter
    norm_dcent = np.linalg.norm(dcent)
    ccd.meta['OOBJ_CR0'] = (ocenter[1], 'Old OBJ_CR0')
    ccd.meta['OOBJ_CR1'] = (ocenter[0], 'Old OBJ_CR1')
    ccd.meta['HIERARCH OCENTER_QUALITY'] = (
        oquality, 'Old center quality')
    ccd.meta['DOBJ_CR0'] = (dcent[1], 'Delta original to current OBJ_CR0')
    ccd.meta['DOBJ_CR1'] = (dcent[0], 'Delta original to current OBJ_CR1')
    ccd.meta['DOBJ_CRN'] = (norm_dcent, 'norm of DOBJ_CR0 and DOBJ_CR1')
    add_history(ccd.meta, 'Used OBJCTRA and OBJCTDEC keywords to calculate obj_cent')
    bmp_meta['dobj_center'] = dcent
    return ccd
    
def on_off_pipeline(directory=None, # raw day directory
                    bmp=None, # BigMultiPipe object
                    band=None,
                    glob_include=NA_SII_GLOB_LIST,
                    calibration=None,
                    photometry=None,
                    add_ephemeris=None,
                    pre_backsub=None,
                    post_backsub=None,
                    num_processes=None,
                    outdir=None,
                    outdir_root=None,
                    create_outdir=True,
                    cpulimit=60, # astrometry.net solve-field
                    keys_to_source_table=KEYS_TO_SOURCE_TABLE,
                    **kwargs):

    if calibration is None:
        calibration = Calibration(reduce=True)
    if photometry is None:
        photometry = CorPhotometry(join_tolerance=5*u.arcsec,
                                   cpulimit=cpulimit,
                                   keys_to_source_table=keys_to_source_table)
    add_ephemeris = assure_list(add_ephemeris)
    pre_backsub = assure_list(pre_backsub)
    post_backsub = assure_list(post_backsub)
    post_process_list = [combine_masks,
                         mask_nonlin_sat,
                         detflux,
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
    flist = multi_glob(directory, glob_include)
    collection = ImageFileCollection(directory, filenames=flist)
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
        process_expand_factor=15, # --> Not sure if this is accurate
        **kwargs)
    pout = cmp.pipeline([f_pairs[0]], outdir=outdir, overwrite=True)
    return pout
#pout = cmp.pipeline(f_pairs, outdir=outdir, overwrite=True)
    pout, _ = prune_pout(pout, f_pairs)
    return pout

#def na_nebula_pipeline(directory,
#                       glob_include=NA_SII_GLOB_LIST,
#                       calibration=None,
#                       photometry=None,
#                       outdir=None,
#                       outdir_root=None,
#                       create_outdir=True,
#                       cpulimit=60, # astrometry.net solve-field
#                       **kwargs):
#    if calibration is None:
#        calibration = Calibration(reduce=True)
#    if photometry is None:
#        photometry = CorPhotometry(cpulimit=cpulimit)
#    cmp = CorMultiPipeBase(
#        ccddata_cls=CorData,
#        auto=True,
#        calibration=calibration,
#        photometry=photometry,
#        outname_append='-back-sub',
#        outname_ext='.fits', 
#        create_outdir=create_outdir,
#        post_process_list=[combine_masks,
#                           mask_nonlin_sat,
#                           detflux,
#                           galsat_ephemeris,
#                           add_astrometry,
#                           off_band_subtract],
#        process_expand_factor=15, # --> Not sure if this is accurate
#        **kwargs)
    
log.setLevel('DEBUG')




#from IoIO.cormultipipe import full_frame
#ccd = CorData.read('/data/IoIO/raw/2017-05-02/IPT-0011_on-band.fit')
#print(full_frame(ccd))


##directory = '/data/IoIO/raw/2017-05-02'
directory = '/data/IoIO/raw/2018-05-08/'
#pout = on_off_pipeline(directory, band='Na',
#                       fits_fixed_ignore=True)

na_meso_obj = NaBack(reduce=True)
standard_star_obj = StandardStar(reduce=True, write_summary_plots=True)

# --> Ack! exctinction correcting doesn't work on images because

pout = on_off_pipeline(directory, band='Na',
                       na_meso_obj=na_meso_obj,
                       standard_star_obj=standard_star_obj,
                       add_ephemeris=galsat_ephemeris,
                       pre_backsub=objctradec_to_obj_center,
                       post_backsub=[sun_angle, na_meso_sub,
                                     extinction_correct, rayleigh_convert],
                       fits_fixed_ignore=True)

#pout = on_off_pipeline(directory, band='SII',
#                       add_ephemeris=galsat_ephemeris,
#                       pre_backsub=objctradec_to_obj_center,
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
