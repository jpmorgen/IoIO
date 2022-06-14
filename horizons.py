"""Provide JPL HORIZONS functionality for IoIO data"""

from random import randint
from time import sleep

import numpy as np

from astropy import log
import astropy.units as u
from astropy.coordinates import Angle
from astropy.table import MaskedColumn

from astroquery.jplhorizons import HorizonsClass
from astroquery.utils import async_to_sync

from IoIO.cor_process import IOIO_1_LOCATION

# Wait time to try HORIZONS query again when it is rate limiting.
HORIZONS_WAIT_MIN = 10
HORIZONS_WAIT_MAX = 100

# https://ssd.jpl.nasa.gov/horizons/manual.html
# HORIZONS associations
GALSATS = {
    'Jupiter': 599,
    'Io': 501,
    'Europa': 502,
    'Ganymede': 503,
    'Callisto': 504}

OBJ_COL_NUMS = \
    '1, 3, 9, 10, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 27'

OBJ_COL_TO_META = ['RA', 'DEC', 'RA_rate', 'DEC_rate', 'V',
                   'surfbright', 'illumination', 'ang_width',
                   'PDObsLon', 'PDObsLat', 'PDSunLon', 'PDSunLat',
                   'SubSol_ang', 'SubSol_dist', 'NPole_ang',
                   'NPole_dist', 'EclLon', 'EclLat', 'r', 'r_rate',
                   'delta', 'delta_rate', 'lighttime', 'vel_sun',
                   'vel_obs', 'elong', 'elongFlag', 'alpha',
                   'sunTargetPA', 'velocityPA']


GALSAT_FROM_OBS_COL_NUMS = \
    '1, 3, 6, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26, 27'
GALSAT_COL_NUMS = '14, 15'
GALSAT_OBS_COL_TO_META = ['RA', 'DEC', 'ang_width', 'r', 'r_rate',
                          'delta', 'delta_rate', 'PDObsLon',
                          'PDObsLat', 'PDSunLon', 'SubSol_ang',
                          'SubSol_dist', 'NPole_ang', 'NPole_dist',
                          'sysIII', 'phi']

# https://lasp.colorado.edu/home/mop/files/2015/02/CoOrd_systems7.pdf
# sysIII is confusing because it is a left-handed coordinate system.
# JUNO_JMAG_VIP4 defines a right-handed coordinates system with X-axis
# defined by the intersection of the magnetic and geographic equators.
# When viewed the observer is situated at the sysIII of that X-axis,
# the magnetic field is tilted CW at its maximum value because the
# tilt of the magnetic field is toward sysIII 200.8 AND sysIII is
# left-handed.  The tilt of the field is 9.5 deg, but the centrifugal
# equator of the IPT is less than that [woudl like a reference]
JUNO_JMAG_VIP4 = 290.8*u.deg
CENTRIFUGAL_EQUATOR_AMPLITUDE = 6.8*u.deg

@async_to_sync
class RateLimitedHorizonsClass(HorizonsClass):
    """Define a class AND method to handle HORIZONS returning a server
    error when the rate of queries is too large.  Waits a random time
    and tries again.

    """
    # Tried to override the ephemerides and ephemerides_async methods
    # and that never worked because of all of the nested decorators:
    # the try at this level was never triggered properly.  Making a
    # separate method that calls the properly decorated top-level
    # ephemerides method seems to be the only way I can get it to work
    def rate_limited_ephemerides(self, *args, **kwargs):
        try:
            return self.ephemerides(*args, **kwargs)
        except ValueError as e:
            if not 'There was an unexpected problem with server. Please wait a minute or so and try again.' in str(e):
                raise e
            log.warning(f'JPL throttling connection')
            sleep(randint(HORIZONS_WAIT_MIN, HORIZONS_WAIT_MAX))
            return self.rate_limited_ephemerides(*args, **kwargs)
RateLimitedHorizons = RateLimitedHorizonsClass()

def location_to_dict(loc):
    """Useful for JPL horizons"""
    return {'lon': loc.lon.value,
            'lat': loc.lat.value,
            'elevation': loc.height.to(u.km).value}

def table_row_to_ccd_meta(ccd_in,
                          t=None,
                          col_list=None,
                          prefix='',
                          row=0):
    ccd = ccd_in.copy()
    for col in col_list:
        if t[col].mask[row]:
            continue
        if len(prefix) == 0:
            prefix_str = ''
        else:
            prefix_str = f'{prefix}_'
        ccd.meta[f'HIERARCH {prefix_str}{col}'] = \
            (t[col][row], f'[{t[col].unit}]')
    return ccd

def obj_ephemeris(ccd_in,
                  in_name=None,
                  bmp_meta=None,
                  horizons_id=None,
                  horizons_id_type=None,
                  obs_loc=None,
                  quantities=None,
                  obj_ephm_prefix='TARGET',
                  obs_col_to_meta=None, # Observer table columns
                  **kwargs):
    """cormultipipe post-processing routine to add generic object
    ephemeris metadata to ccd.meta and bmp_meta

    """
    quantities = quantities or OBJ_COL_NUMS
    if isinstance(ccd_in, list):
        if isinstance(in_name, list):
            return [obj_ephemeris(ccd,
                                  in_name=fname,
                                  bmp_meta=bmp_meta,
                                  horizons_id=horizons_id,
                                  horizons_id_type=horizons_id_type,
                                  obs_loc=obs_loc,
                                  quantities=quantities,
                                  obj_ephm_prefix=obj_ephm_prefix,
                                  obs_col_to_meta=obs_col_to_meta,
                                  **kwargs)
                    for ccd, fname in zip(ccd_in, in_name)]
        else:
            return [obj_ephemeris(ccd,
                                  in_name=in_name,
                                  bmp_meta=bmp_meta,
                                  horizons_id=horizons_id,
                                  horizons_id_type=horizons_id_type,
                                  obs_loc=obs_loc,
                                  quantities=quantities,
                                  obj_ephm_prefix=obj_ephm_prefix,
                                  obs_col_to_meta=obs_col_to_meta,
                                  **kwargs)
                for ccd in ccd_in]
    ccd = ccd_in.copy()
    obs_loc = obs_loc or IOIO_1_LOCATION
    obs_col_to_meta = obs_col_to_meta or OBJ_COL_TO_META
    obs_name = obs_loc.info.name
    h = RateLimitedHorizons(id=horizons_id,
                    epochs=ccd.tavg.jd,
                    location=location_to_dict(obs_loc),
                    id_type=horizons_id_type)
    e = h.rate_limited_ephemerides(quantities=quantities)
    ra = Angle(e['RA'].quantity)
    dec = Angle(e['DEC'].quantity)
    ccd.meta['OBJCTRA'] = (ra[0].to_string(unit=u.hour),
                           'OBJECT RA from HORIZONS')
    ccd.meta['OBJCTDEC'] = (dec[0].to_string(unit=u.deg),
                            'OBJECT DEC from HORIZONS')
    ccd.meta.insert('OBJCTDEC',
                    ('HIERARCH OBJECT_TO_OBJCTRADEC', True,
                     'OBJCT* point to OBJECT'),
                    after=True)
    # Refresh sky_coord!
    ccd.sky_coord = None
    ccd.meta['HIERARCH HORIZONS_TARGET'] = e['targetname'][0]
    ccd = table_row_to_ccd_meta(ccd, e, obs_col_to_meta,
                                prefix=obj_ephm_prefix, row=0)
    #for col in obs_col_to_meta:
    #    if e[col].mask[0]:
    #        continue
    #    ccd.meta[f'HIERARCH OBJECT_{col}'] = \
    #        (e[col][0], f'[{e[col].unit}]')
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
    object = ccd_in.meta['OBJECT']
    if object != 'Jupiter':
        log.warning(f'Called galsat_ephemeris on OBJECT {object}')
        return ccd_in
    ccd = ccd_in.copy()
    obs_loc = obs_loc or IOIO_1_LOCATION
    obs_col_to_meta = obs_col_to_meta or GALSAT_OBS_COL_TO_META
    obs_name = obs_loc.info.name

    # Get our ephemerides from the perspective of the observatory
    # id_type='majorbody' required because minor bodies (e.g. asteroids)
    # are the default
    obs_eph = None
    for galsat in GALSATS:
        h = RateLimitedHorizons(id=GALSATS[galsat],
                        epochs=ccd.tavg.jd,
                        location=location_to_dict(obs_loc),
                        id_type='majorbody')
        e = h.rate_limited_ephemerides(quantities=GALSAT_FROM_OBS_COL_NUMS)
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
        h = RateLimitedHorizons(id=GALSATS['Jupiter'],
                        epochs=epoch.value,
                        location=f'500@{GALSATS[galsat]}',
                        id_type='majorbody')
        e = h.rate_limited_ephemerides(quantities=GALSAT_COL_NUMS)
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

    # Put all this into ccd.meta
    for galsat in GALSATS:
        mask = obs_eph['targetname'] == f'{galsat} ({GALSATS[galsat]})'
        te = obs_eph[mask]
        for col in obs_col_to_meta:
            if te[col].mask[0]:
                continue
            ccd.meta[f'HIERARCH {galsat}_{col}'] = \
                (te[col][0], f'[{te[col].unit}]')

        if galsat == 'Jupiter':
            # Put Jupiter's precise coordinates into header
            ra = Angle(te['RA'].quantity)
            dec = Angle(te['DEC'].quantity)
            ccd.meta['OBJCTRA'] = (ra[0].to_string(unit=u.hour),
                                   'OBJECT RA from HORIZONS')
            ccd.meta['OBJCTDEC'] = (dec[0].to_string(unit=u.deg),
                                    'OBJECT DEC from HORIZONS')
            if ccd.meta.get('OBJECT_TO_OBJCTRADEC'):
                # For some reason astropy doesn't catch the duplicate,
                # which is from ACP shell-out repair
                del ccd.meta['OBJECT_TO_OBJCTRADEC']
            ccd.meta.insert('OBJCTDEC',
                            ('HIERARCH OBJECT_TO_OBJCTRADEC', True,
                             'OBJCT* point to OBJECT'),
                            after=True)
            # Refresh sky_coord!
            ccd.sky_coord = None
            # Put IPT centrifugal equator into ccd.meta.
            # JUNO_JMAG_VIP4 X-axis effectively defines the *descending*
            # node of IPT tilt because it is left handed, though this
            # is cast in terms of the
            sysIII = te['PDObsLon'].quantity
            IPT_NPole_ang = -(CENTRIFUGAL_EQUATOR_AMPLITUDE
                             * np.cos(sysIII[0] - JUNO_JMAG_VIP4))
            ccd.meta['HIERARCH IPT_NPole_ang'] = (
                IPT_NPole_ang.value, f'[{IPT_NPole_ang.unit}]')

    # I am going to leave this out for now because I can get it from
    # ccd.meta.  I may want to distill it to a simple dictionary, but
    # that is what the metadata are....
    #bmp_meta['obs_eph'] = obs_eph
    return ccd

#from IoIO.cordata import CorData
#from IoIO.cor_process import cor_process
#from IoIO.calibration import Calibration
#
#fname = '/data/IoIO/raw/2021-10-28/Mercury-0001_Na_on.fit'
#rccd = CorData.read(fname)
#c = Calibration(reduce=True)
#ccd = cor_process(rccd, calibration=c, auto=True)
#
#ccd = obj_ephemeris(ccd,
#                    horizons_id=199,
#                    horizons_id_type='majorbody',
#                    quantities=OBJ_COL_NUMS,
#                    obs_col_to_meta=OBJ_COL_TO_META)

#ccd = obj_ephemeris(ccd,
#                    horizons_id='Ceres',
#                    id_type='asteroid_name',
#                    quantities=MERCURY_COL_NUMS,
#                    obs_col_to_meta=OBS_COL_TO_META)
#
                    
