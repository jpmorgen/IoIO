#!/usr/bin/python3

"""Reduce IoIO sodium nebula observations"""

import numpy as np

from scipy.ndimage import shift

from matplotlib.colors import LogNorm

from astropy import log
import astropy.units as u
from astropy.table import MaskedColumn
from astropy.coordinates import solar_system_ephemeris, get_body
from astroquery.jplhorizons import Horizons

from ccdproc import ImageFileCollection, transform_image

from bigmultipipe import cached_pout, prune_pout

from IoIO.utils import (is_flux, reduced_dir, multi_glob,
                        closest_in_time, valid_long_exposure,
                        add_history, simple_show,
                        best_fits_time, angle_to_major_body,
                        location_to_dict)
from IoIO.cor_process import (IOIO_1_LOCATION, standardize_filt_name,
                              obs_location_from_hdr)
from IoIO.calibration import Calibration
from IoIO.cordata import CorData
from IoIO.cormultipipe import (IoIO_ROOT, RAW_DATA_ROOT,
                               CorMultiPipeBase, mask_nonlin_sat,
                               combine_masks, detflux, nd_filter_mask)
from cor_photometry import CorPhotometry

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
#GALSAT_COL_TO_META = ['PDObsLon', 'PDObsLat', 'PDSunLon', 'SubSol_ang']

class OnOffCorMultiPipe(CorMultiPipeBase):
    def outname_create(self, in_names,
                       **kwargs):
        """Create outname from first file in in_names list.  Further
        modification to the outname can be done using outname_append

        """
        outname = super().outname_create(in_names[0], **kwargs)
        return outname
    
def off_band_subtract(data,
                      in_name=None,
                      bmp_meta=None,
                      off_on_ratio=None,
                      calibration=None,
                      show=False,
                      **kwargs):
    """cormultipipe post-processing routine that subtracts off-band image
    from on-band

    """
    if bmp_meta is None:
        bmp_meta = {}
    if off_on_ratio is None and calibration is None:
        calibration = Calibration(reduce=True)
    filt = (data[0]).meta['FILTER']
    band, _ = filt.split('_')
    if off_on_ratio is None:
        off_on_ratio, _ = calibration.flat_ratio(band)
        
    on = data[0]
    off = data[1]
    shift_off = on.obj_center - off.obj_center
    d_on_off = np.linalg.norm(shift_off)
    off = transform_image(off, shift, shift=shift_off)
    if d_on_off > 5:
            log.warning('On- and off-band image centers are > 5 pixels apart')

    assert is_flux(on.unit) and is_flux(off.unit)
    off = off.divide(off_on_ratio*u.dimensionless_unscaled,
                     handle_meta='first_found')
    on = on.subtract(off, handle_meta='first_found')
    on.meta['OFF_BAND'] = in_name[1]
    on.meta['OFF_SCL'] = (1/off_on_ratio, 'scale factor applied to OFF_BAND')
    on.meta['HIERARCH SHIFT_OFF_X'] = (shift_off[0], '[pix] x dist on - off')
    on.meta['HIERARCH SHIFT_OFF_Y'] = (shift_off[1], '[pix] y dist on - off')
    on.meta['HIERARCH SHIFT_OFF_D'] = (d_on_off, '[pix] dist on - off')
    add_history(on.meta, 'Scaled and shifted OFF_BAND by OFF_SCL and SHIFT_OFF')
    tmeta = {'band': band,
             'off_on_ratio': off_on_ratio,
             'shift_off': shift_off}
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
                                     in_name=name, obs_loc=obs_loc,
                                     obs_col_to_meta=obs_col_to_meta,
                                     **kwargs)
                    for ccd, name in zip(ccd_in, in_name)]
        else:
            return [galsat_ephemeris(ccd, bmp_meta=bmp_meta, 
                                     in_name=in_name, obs_loc=obs_loc,
                                     obs_col_to_meta=obs_col_to_meta,
                                     **kwargs)
                for ccd in ccd_in]
    ccd = ccd_in.copy()
    if obs_loc is None:
        obs_loc = IOIO_1_LOCATION
    if obs_col_to_meta is None:
        obs_col_to_meta = OBS_COL_TO_META
    obs_name = obs_loc.info.name
    tm = best_fits_time(ccd.meta)

    # Get our ephemerides from the perspective of the observatory
    # id_type='majorbody' required because minor bodies (e.g. asteroids)
    # are the default
    obs_eph = None
    for galsat in GALSATS:
        h = Horizons(id=GALSATS[galsat],
                     epochs=tm.jd,
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
        epoch = tm.jd*u.day - lt
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

def add_astrometry(ccd_in, bmp_meta=None, photometry=None,
                   in_name=None, outdir=None, create_outdir=None,
                   cpulimit=None, **kwargs):
    if isinstance(ccd_in, list):
        if isinstance(in_name, list):
            return [add_astrometry(ccd, bmp_meta=bmp_meta, 
                                   photometry=photometry,
                                   in_name=name, outdir=outdir,
                                   create_outdir=create_outdir,
                                   cpulimit=cpulimit, **kwargs)
                    for ccd, name in zip(ccd_in, in_name)]
        else:
            return [add_astrometry(ccd, bmp_meta=bmp_meta, 
                                   photometry=photometry,
                                   in_name=in_name, outdir=outdir,
                                   create_outdir=create_outdir,
                                   cpulimit=cpulimit, **kwargs)
                    for ccd in ccd_in]            
    ccd = ccd_in.copy()
    bmp_meta = bmp_meta or {}
    photometry = photometry or CorPhotometry()
    photometry.ccd = ccd
    photometry.cpulimit = photometry.cpulimit or cpulimit
    if outdir is not None:
        bname = os.path.basename(in_name)
        outname = os.path.join(outdir, bname)
        if create_outdir:
            os.makedirs(outdir, exist_ok=True)
    else:
        # This is safe because the astormetry stuff does not 
        outname = in_name
    wcs = photometry.astrometry(outname=outname)
    # I am currently not putting the wcs into the metadata because I
    # don't need it -- it is available as ccd.wcs or realtively easily
    # extracted from disk like I do in Photometry.astrometry.  I am
    # also not putting the SourceTable into the metadata, because it
    # is still hanging around in the Photometry object.
    return ccd    

def on_off_pipeline(directory=None, # raw day directory
                    band=None,
                    glob_include=NA_SII_GLOB_LIST,
                    calibration=None,
                    photometry=None,
                    num_processes=None,
                    outdir=None,
                    outdir_root=None,
                    create_outdir=True,
                    cpulimit=60, # astrometry.net solve-field
                    **kwargs):

    if outdir is None and outdir_root is None:
        # Right now, dump Na and SII into separate top-level reduction
        # directories
        outdir_root = os.path.join(IoIO_ROOT, band)
        outdir = outdir or reduced_dir(directory, outdir_root, create=False)
    flist = multi_glob(directory, glob_include)
    collection = ImageFileCollection(directory, filenames=flist)
    standardize_filt_name(collection)
    if calibration is None:
        calibration = Calibration(reduce=True)
    if photometry is None:
        photometry = CorPhotometry(cpulimit=cpulimit)
    f_pairs = closest_in_time(collection, (f'{band}_on', f'{band}_off'),
                              valid_long_exposure,
                              directory=directory)
    if len(f_pairs) == 0:
        log.warning(f'No matching set of on-off {band} files found '
                    f'in {directory}')
        return []

    # I want to make this a callable that I octopus stomach in from
    # wherever it needs to be called from
    cmp = OnOffCorMultiPipe(
        ccddata_cls=CorData,
        auto=True,
        calibration=calibration,
        photometry=photometry,
        outname_append='-back-sub',
        outname_ext='.fits', 
        create_outdir=create_outdir,
        post_process_list=[combine_masks,
                           mask_nonlin_sat,
                           detflux,
                           galsat_ephemeris,
                           add_astrometry,
                           off_band_subtract],
        num_processes=num_processes,
        process_expand_factor=15,
        **kwargs)
    pout = cmp.pipeline([f_pairs[0]], outdir=outdir, overwrite=True)
    return pout
#pout = cmp.pipeline(f_pairs, outdir=outdir, overwrite=True)
    pout, _ = prune_pout(pout, f_pairs)
    return pout


log.setLevel('DEBUG')


#from IoIO.cormultipipe import full_frame
#ccd = CorData.read('/data/IoIO/raw/2017-05-02/IPT-0011_on-band.fit')
#print(full_frame(ccd))


##directory = '/data/IoIO/raw/2017-05-02'
directory = '/data/IoIO/raw/2018-05-08/'
pout = on_off_pipeline(directory, band='Na',
                       fits_fixed_ignore=True)

#from IoIO.cormultipipe import full_frame
#ccd = CorData.read('/data/IoIO/Na/2018-05-08/Na_on-band_010-back-sub.fits')
#simple_show(ccd, norm=LogNorm())


#on = CorData.read(p[0])
#off = CorData.read(p[1])
#off_band_subtract([on, off], in_name=p, band='Na')


#from cordata import CorData
#ccd = CorData.read('/data/IoIO/raw/2018-05-08/Na_on-band_011.fits')
#obs_loc = None
#obs_col_to_meta = None
#
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
