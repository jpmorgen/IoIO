"""Module interfacing IoIO reduction/analysis with Juno mission stuff"""

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

import astropy.units as u
from astropy.time import Time
from astroquery.jplhorizons import HorizonsClass

from IoIO.horizons import OBJ_COL_NUMS

MJD_MATPLOTLIB0 = Time(plt.rcParams['date.epoch'], format='fits')
MJD_MATPLOTLIB0 = MJD_MATPLOTLIB0.mjd
PRE_POST_JUNO_PJ = 100*u.yr.to(u.day)

# This is from https://lasp.colorado.edu/mop/files/2024/01/EM2-spreadsheet.png and some painful hand OCR work
# See /data/IoIO/analysis/Juno Perijoves/
# https://lasp.colorado.edu/mop/files/2022/04/EMorbit-1.png
# https://lasp.colorado.edu/mop/files/2025/08/35-74.png
# https://lasp.colorado.edu/mop/files/2025/08/75-110.png
JUNO_PERIJOVES = [
    '2016-07-05T02:47:32', 
    '2016-08-27T12:50:44', 
    '2016-10-19T18:10:54', 
    '2016-12-11T17:03:41', 
    '2017-02-02T12:57:09', 
    '2017-03-27T08:51:52', 
    '2017-05-19T06:00:47', 
    '2017-07-11T01:54:42', 
    '2017-09-01T21:48:50', 
    '2017-10-24T17:42:31', 
    '2017-12-16T17:56:59', 
    '2018-02-07T13:51:30', 
    '2018-04-01T09:45:43', 
    '2018-05-24T05:39:51', 
    '2018-07-16T05:17:22', 
    '2018-09-07T01:11:41', 
    '2018-10-29T21:06:00', 
    '2018-12-21T16:59:48', 
    '2019-02-12T17:34:31', 
    '2019-04-06T12:14:22', 
    '2019-05-29T08:08:18', 
    '2019-07-21T04:02:43', 
    '2019-09-12T03:40:44', 
    '2019-11-03T22:18:14', 
    '2019-12-26T17:36:13', 
    '2020-02-17T17:51:55', 
    '2020-04-10T13:47:40', 
    '2020-06-02T10:20:03', 
    '2020-07-25T06:15:27', 
    '2020-09-16T02:10:52', 
    '2020-11-08T01:49:42', 
    '2020-12-30T21:45:45', 
    '2021-02-21T17:40:34', 
    '2021-04-15T23:32:25', 
    '2021-06-08T07:46:01', 
    '2021-07-21T08:15:05.123', 
    '2021-09-02T22:42:51.807', 
    '2021-10-16T17:13:32.189', 
    '2021-11-29T14:13:29.633', 
    '2022-01-12T10:33:01.817', 
    '2022-02-25T01:59:56.638', 
    '2022-04-09T15:49:17.759', 
    '2022-05-23T02:15:53.911', 
    '2022-07-05T09:17:22.233', 
    '2022-08-17T14:45:39.432', 
    '2022-09-29T17:11:55.543', 
    '2022-11-06T21:38:36.500', 
    '2022-12-15T03:23:22.583', 
    '2023-01-22T05:44:18.955', 
    '2023-03-01T05:53:21.552', 
    '2023-04-08T08:13:34.385', 
    '2023-05-16T07:22:44.593', 
    '2023-06-23T06:55:08.478', 
    '2023-07-31T09:05:43.729', 
    '2023-09-07T11:58:01.653', 
    '2023-10-15T10:52:49.914', 
    '2023-11-22T12:17:18.437', 
    '2023-12-30T12:36:20.761', 
    '2024-02-03T21:47:29.385', 
    '2024-03-07T15:42:20.208', 
    '2024-04-09T08:48:18.595', 
    '2024-05-12T03:38:38.633', 
    '2024-06-13T19:53:40.879', 
    '2024-07-16T14:33:58.043', 
    '2024-08-18T06:57:08.249', 
    '2024-09-20T02:28:52.041', 
    '2024-10-22T18:11:01.570', 
    '2024-11-24T13:05:27.701', 
    '2024-12-27T05:22:29.622', 
    '2025-01-28T23:05:12.061', 
    '2025-03-02T16:04:32.345', 
    '2025-04-04T09:30:54.803', 
    '2025-05-07T03:01:32.636', 
    '2025-06-08T20:30:46.975', 
    '2025-07-11T13:40:03.142',
    '2025-08-13T07:04:54.435', 
    '2025-09-14T23:52:03.861', 
    '2025-10-17T16:04:41.336', 
    '2025-11-19T08:38:15.734', 
    '2025-12-22T01:15:03.254', 
    '2026-01-23T17:18:32.509', 
    '2026-02-25T09:34:12.327', 
    '2026-03-30T01:09:04.634', 
    '2026-05-01T16:09:43.612', 
    '2026-06-03T07:24:40.696', 
    '2026-07-05T22:01:47.798', 
    '2026-08-07T14:17:41.171', 
    '2026-09-09T05:18:03.069', 
    '2026-10-11T21:26:32.158', 
    '2026-11-13T12:45:34.660', 
    '2026-12-16T04:12:58.090', 
    '2027-01-17T19:33:47.240', 
    '2027-02-19T10:45:12.506', 
    '2027-03-24T01:59:39.498', 
    '2027-04-25T16:52:18.098', 
    '2027-05-28T08:50:09.205', 
    '2027-06-30T00:26:13.588', 
    '2027-08-01T17:07:49.676', 
    '2027-09-03T09:12:05.985', 
    '2027-10-06T01:57:18.115', 
    '2027-11-07T18:28:25.962', 
    '2027-12-10T11:11:06.940', 
    '2028-01-12T03:40:48.756', 
    '2028-02-13T19:55:57.986', 
    '2028-03-17T12:00:52.747', 
    '2028-04-19T04:21:25.037', 
    '2028-05-21T20:16:12.245', 
    '2028-06-23T12:24:43.851', 
    '2028-07-26T03:46:14.687', 
    '2028-08-27T19:28:03.694', 
    '2028-09-29T11:00:15.805'] 

JUNO_APOJOVES = [
    '2016-07-31T19:46:02',
    '2016-09-23T03:44:48',
    '2016-11-15T05:36:45',
    '2017-01-07T03:11:30',
    '2017-02-28T22:55:48',
    '2017-04-22T19:14:57',
    '2017-06-14T15:58:35',
    '2017-08-06T11:44:04',
    '2017-09-28T07:51:01',
    '2017-11-20T05:57:23',
    '2018-01-12T03:52:42',
    '2018-03-05T23:55:41',
    '2018-04-27T19:36:40',
    '2018-06-19T17:30:40',
    '2018-08-11T15:18:43',
    '2018-10-03T10:58:52',
    '2018-11-25T07:01:26',
    '2019-01-17T05:19:21',
    '2019-03-11T02:48:11',
    '2019-05-02T22:18:47',
    '2019-06-24T18:01:57',
    '2019-08-16T16:01:52',
    '2019-10-08T12:52:15',
    '2019-11-30T07:39:10',
    '2020-01-22T05:44:55',
    '2020-03-15T03:44:40',
    '2020-05-07T00:16:41',
    '2020-06-28T20:24:51',
    '2020-08-20T16:08:49',
    '2020-10-12T14:05:43',
    '2020-12-04T11:37:23',
    '2021-01-26T07:36:06',
    '2021-03-20T08:39:35',
    '2021-05-12T15:29:09',
    '2021-06-29T20:04:19',
    '2021-08-12T03:18:50.976',
    '2021-09-24T19:58:31.774',
    '2021-11-07T15:42:36.613',
    '2021-12-21T12:20:15.666',
    '2022-02-03T06:18:38.017',
    '2022-03-18T21:05:10.551',
    '2022-05-01T09:02:37.335',
    '2022-06-13T17:58:29.297',
    '2022-07-27T00:00:05.970',
    '2022-09-08T04:08:21.444',
    '2022-10-18T19:29:15.746',
    '2022-11-26T00:29:20.153',
    '2023-01-03T04:35:02.848',
    '2023-02-10T05:47:47.751',
    '2023-03-20T07:09:41.279',
    '2023-04-27T07:44:47.669',
    '2023-06-04T07:15:00.057',
    '2023-07-12T07:58:49.849',
    '2023-08-19T10:31:08.860',
    '2023-09-26T11:22:03.934',
    '2023-11-03T11:40:35.052',
    '2023-12-11T12:20:08.333',
    '2024-01-17T05:08:41.014',
    '2024-02-20T06:47:07.550',
    '2024-03-24T00:20:02.626',
    '2024-04-25T18:10:28.207',
    '2024-05-28T11:50:30.194',
    '2024-06-30T05:09:43.349',
    '2024-08-01T22:49:01.597',
    '2024-09-03T16:41:00.076',
    '2024-10-06T10:22:06.612',
    '2024-11-08T03:36:58.966',
    '2024-12-10T21:12:42.022',
    '2025-01-12T14:13:21.549',
    '2025-02-14T07:35:18.310',
    '2025-03-19T00:48:06.736',
    '2025-04-20T18:16:59.022',
    '2025-05-23T11:44:03.069',
    '2025-06-25T05:06:16.193',
    '2025-07-27T22:18:55.019',
    '2025-08-29 15:31:14.947',
    '2025-10-01 07:53:58.979',
    '2025-11-03 00:23:12.773',
    '2025-12-05 16:53:52.704',
    '2026-01-07 09:18:31.929',
    '2026-02-09 01:25:35.159',
    '2026-03-13 17:20:43.511',
    '2026-04-15 08:39:04.739',
    '2026-05-17 23:44:44.525',
    '2026-06-19 14:44:39.079',
    '2026-07-22 06:06:57.304',
    '2026-08-23 21:48:23.180',
    '2026-09-25 13:18:51.743',
    '2026-10-28 05:05:19.572',
    '2026-11-29 20:27:32.326',
    '2027-01-01 11:52:13.267',
    '2027-02-03 03:08:49.335',
    '2027-03-07 18:19:47.839',
    '2027-04-09 09:25:52.512',
    '2027-05-12 00:48:58.383',
    '2027-06-13 16:39:05.001',
    '2027-07-16 08:45:43.624',
    '2027-08-18 01:10:01.205',
    '2027-09-19 17:33:47.709',
    '2027-10-22 10:12:42.252',
    '2027-11-24 02:50:01.951',
    '2027-12-26 19:25:48.380',
    '2028-01-28 11:48:17.906',
    '2028-03-01 03:57:42.865',
    '2028-04-02 20:10:54.842',
    '2028-05-05 12:18:35.113',
    '2028-06-07 04:20:14.558',
    '2028-07-09 20:04:56.630',
    '2028-08-11 11:36:19.333',
    '2028-09-13 03:13:29.133']

# It is a bridge too far to calculate ephemeris from Juno to the IoIO
# location:
#loc = location_to_dict(IOIO_1_LOCATION)
#loc['body'] = 399
#h = HorizonsClass(id=loc, epochs=pj_ts.mjd,
#                  location='500@-61')
#e = h.ephemerides(quantities=OBJ_COL_NUMS)
#
# But:
# re = 1*u.Rearth
# re = re.to(u.km)
# re / (3E5*u.km/u.s)
# <Quantity 0.02126033 s>
# so we only have a fuzz of 20 ms on the accuracy of the light travel
# time, which is better than our ~200ms DATE-OBS-UNCERTAINTY

def scet2body(t, spacecraft=-61, body=399, tformat='fits'):
    """Convert Spacecraft Event Time (SCET) to UT on body

    Parameters
    ----------
    t : [array of] Quantity, ~astropy.time.Time, or [list of] string
        SCET of events to convert to body time.  If string, must be
        FITS time format yyyy-mm-ddThh:mm:ss[.000].  If Quantity, JD or
        MJD are assumed

    spacecraft : int
        JPL spacecraft code
        Default : -61 for Juno

    body : int
        JPL body center code.  Topocentric coordinates are not allowed
        Default : 399 for Earth

    tformat : str
        Time format of return value(s)
        Default : 'fits'

    Returns
    -------
    t : ~astropy.time.Time
        Body center time(s) corresponding to SCET(s)

    """
    if isinstance(t, list) or isinstance(t, str):
        t = Time(t, format='fits')
    if isinstance(t, Time):
        t = t.mjd
    if not isinstance(t, u.Quantity):
        t *= u.day
    if np.isscalar(t.value):
        t = np.asarray((t.value,))
        t *= u.day
    # Make sure our query is not too long
    lefts = np.arange(0, len(t), 10)
    if len(t) == 1:
        rights = []
    else:
        rights = lefts[1:]
    rights = np.append(rights, None)
    for left, right in zip(lefts, rights):
        s = slice(left, right)
        # Using .value here for t[s] as a scalar gets around scalar
        # Quantity unhappiness in astroquery
        h = HorizonsClass(id=body,
                          epochs=t[s].value,
                          location='500@'+str(spacecraft))
        e = h.ephemerides(quantities=OBJ_COL_NUMS)
        t[s] += e['lighttime']
    # Put everything back into a Time object in FITS format
    t = Time(t, format='mjd')
    t.format = tformat
    return t

#scet2body(59945)
#pj_ts = scet2body(JUNO_PERIJOVES)
#
## Set up interpolation between PJ number and body time
#
#pj_list = np.arange(len(JUNO_PERIJOVES))
#a = np.interp(3, pj_list, pj_ts.mjd, left=-1, right=-1)
#a = Time(a, format='mjd')
#a.format = 'fits'
#print(a)
#def pj2mjd(pj):
#    np.interp(pj, pj_list, pj_ts.mjd, left=-1, right=-1)
#    pass

#def mjd2pj(mjd):
#    pass

class JunoTimes():
    def __init__(self,
                 body=399):
        self.body = body
        self._pj_list = None
        self._pj_ts = None

    @property
    def pj_list(self):
        if self._pj_list is None:
            self._pj_list = np.arange(len(JUNO_PERIJOVES))
        return self._pj_list
    
    @property
    def pj_ts(self):
        if self._pj_ts is None:
            self._pj_ts = scet2body(JUNO_PERIJOVES, body=self.body,
                                    tformat='mjd')
        return self._pj_ts

    #def pj2mjd(self, pj):
    #    if pj < 0:
    #        return self.pj_ts[0].mjd - pj*PRE_POST_JUNO_PJ
    #    if pj > len(JUNO_PERIJOVES):
    #        return self.pj_ts[-1].mjd + pj*PRE_POST_JUNO_PJ
    #    return np.interp(pj, self.pj_list, self.pj_ts.mjd)
    #
    #def mjd2pj(self, mjd):
    #    if mjd < self.pj_ts[0].mjd:
    #        return (self.pj_ts[0].mjd - mjd)/PRE_POST_JUNO_PJ
    #    if mjd > self.pj_ts[-1]:
    #        return (mjd - self.pj_ts[-1].mjd)/PRE_POST_JUNO_PJ
    #    return np.interp(mjd, self.pj_ts.mjd, self.pj_list)

    def pj2plt_date(self, pj):
        lowidx = np.flatnonzero(pj < 0)
        highidx = np.flatnonzero(pj > self.pj_list[-1])
        mjd = np.interp(pj, self.pj_list, self.pj_ts.mjd,
                        left=np.nan, right=np.nan)
        if not np.isscalar(mjd):
            # The axis likes to be monotonic, but single values for
            # the AXFormatter can be nan
            mjd[lowidx] = pj[lowidx]*PRE_POST_JUNO_PJ + self.pj_ts[0].mjd
            mjd[highidx] = ((pj[highidx] - self.pj_list[-1])*PRE_POST_JUNO_PJ
                            + self.pj_ts[-1].mjd)
        return mjd - MJD_MATPLOTLIB0

    def plt_date2pj(self, plt_date):
        t = Time(plt_date + MJD_MATPLOTLIB0, format='mjd')
        mjd = t.mjd
        lowidx = np.flatnonzero(mjd < self.pj_ts[0].mjd)
        highidx = np.flatnonzero(mjd > self.pj_ts[-1].mjd)
        pj = np.interp(mjd, self.pj_ts.mjd, self.pj_list,
                       left=np.nan, right=np.nan)
        if not np.isscalar(mjd):
            pj[lowidx] = -(self.pj_ts[0].mjd - mjd[lowidx])/PRE_POST_JUNO_PJ
            pj[highidx] = (self.pj_list[-1]
                           + (mjd[highidx]
                              - self.pj_ts[-1].mjd)
                           /PRE_POST_JUNO_PJ)
        return pj

class PJAXFormatter():
    """Provides second Y axis value interactive pyplot window with PJs"""
    def __init__(self, plt_date, y=None, body=399):
        if isinstance(y, u.Quantity):
            self.yunit = y.unit
        else:
            self.yunit = 1
        self.jts = JunoTimes(body=body)
    def __call__(self, plt_date, y):
        pj = self.jts.plt_date2pj(plt_date)
        dtime = Time(plt_date + MJD_MATPLOTLIB0, format='mjd')
        dtime.format = 'fits'
        #dtime = dtime.datetime64
        #return \
        #    f'date: {plt_date} ' \
        #    f'PJ: {pj} ' \
        #    f'y: {y}'
            #f'date: {dtime:%Y-%m-%d} ' \
        ptime = dtime.strftime('%Y-%m-%dT%H:%M:%S')
        return \
            f'date: {ptime}   ' \
            f'PJ: {pj:0.2f} ' \
            f'y: {y*self.yunit:0.2f}'

def juno_pj_axis(ax, position='top'):
    jts = JunoTimes()
    secax = ax.secondary_xaxis(position,
                               functions=(jts.plt_date2pj, jts.pj2plt_date))
    secax.xaxis.set_minor_locator(MultipleLocator(1))
    secax.set_xlabel('PJ')

