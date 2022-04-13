"""Provide JPL HORIZONS functionality for IoIO data"""

import numpy as np

from astropy import log
import astropy.units as u
from astropy.coordinates import Angle
from astropy.table import MaskedColumn

from astroquery.jplhorizons import Horizons

from IoIO.utils import location_to_dict
from IoIO.cor_process import IOIO_1_LOCATION

# https://ssd.jpl.nasa.gov/horizons/manual.html

# HORIZONS associations
GALSATS = {
    'Jupiter': 599,
    'Io': 501,
    'Europa': 502,
    'Ganymede': 503,
    'Callisto': 504}

# --> In a horizons module, could potentially have the default columns
# for all of t he different object types and handle those nicely into
# the ccd.meta
OBS_COL_NUMS = \
    '1, 3, 6, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26, 27'
GALSAT_COL_NUMS = '14, 15'
OBS_COL_TO_META = ['RA', 'DEC', 'r', 'r_rate', 'delta', 'delta_rate',
                   'PDObsLon', 'PDObsLat', 'PDSunLon', 'SubSol_ang',
                   'SubSol_dist']
GALSAT_OBS_COL_TO_META = OBS_COL_TO_META + ['sysIII', 'phi']


#GALSAT_COL_TO_META = ['PDObsLon', 'PDObsLat', 'PDSunLon', 'SubSol_ang']

##################################################################
# This stuff could go in a separate horizons or ephemeris module and
# could include comet and asteroid.  Would want to hack apart the
# targetname returned by HORIZONS to get a better prepend.  Note that
# this is just for a single object in the FOV.  More complicated
# systems need a custom setup, like galsat_ephemeris
##################################################################
def obj_ephemeris(ccd_in,
                  horizons_id=None,
                  bmp_meta=None,
                  in_name=None,
                  obs_loc=None,
                  obs_col_to_meta=None,
                  **kwargs):
    """cormultipipe post-processing routine to add generic object
    ephemeris metadata to ccd.meta and bmp_meta

    """
    if isinstance(ccd_in, list):
        if isinstance(in_name, list):
            return [obj_ephemeris(ccd, horizons_id=horizons_id,
                                  bmp_meta=bmp_meta, 
                                  in_name=fname, obs_loc=obs_loc,
                                  obs_col_to_meta=obs_col_to_meta,
                                  **kwargs)
                    for ccd, fname in zip(ccd_in, in_name)]
        else:
            return [obj_ephemeris(ccd, bmp_meta=bmp_meta, 
                                  horizons_id=horizons_id,
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
    h = Horizons(id=horizons_id,
                 epochs=ccd.tavg.jd,
                 location=location_to_dict(obs_loc),
                 id_type='majorbody')
    e = h.ephemerides(quantities=OBS_COL_NUMS)
    ra = Angle(e['RA'].quantity)
    dec = Angle(e['DEC'].quantity)
    ccd.meta['OBJCTRA'] = (ra[0].to_string(unit=u.hour),
                           'OBJECT RA from HORIZONS')
    ccd.meta['OBJCTDEC'] = (dec[0].to_string(unit=u.deg),
                            'OBJECT DEC from HORIZONS')
    target = e['targetname'][0]
    for col in obs_col_to_meta:
        if e[col].mask[0]:
            continue
        ccd.meta[f'HIERARCH {target}_{col}'] = \
            (e[col][0], f'[{e[col].unit}]')
    return ccd

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
        obs_col_to_meta = GALSAT_OBS_COL_TO_META
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
                (te[col][0], f'[{te[col].unit}]')


    # I am going to leave this out for now because I can get it from
    # ccd.meta.  I may want to distill it to a simple dictionary, but
    # that is what the metadata are....
    #bmp_meta['obs_eph'] = obs_eph
    return ccd

