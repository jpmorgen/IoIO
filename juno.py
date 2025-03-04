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
    '2021-07-21T08:15:05', 
    '2021-09-02T22:42:52', 
    '2021-10-16T17:13:32', 
    '2021-11-29T14:13:30', 
    '2022-01-12T10:33:02', # Tweaked from here
    '2022-02-25T01:59:57', 
    '2022-04-09T15:49:18', 
    '2022-05-23T02:15:54', 
    '2022-07-05T09:17:22', 
    '2022-08-17T14:45:39', 
    '2022-09-29T17:11:56', 
    '2022-11-06T21:38:37', 
    '2022-12-15T03:23:23', 
    '2023-01-22T05:44:19',
    '2023-03-01T05:53:22', 
    '2023-04-08T08:13:34', 
    '2023-05-16T07:22:45', 
    '2023-06-23T06:55:08', 
    '2023-07-31T09:05:44', 
    '2023-09-07T11:58:02', 
    '2023-10-15T10:52:50', 
    '2023-11-22T12:17:18', 
    '2023-12-30T12:36:21', 
    '2024-02-03T21:47:30', 
    '2024-03-07T15:42:01', 
    '2024-04-09T08:48:18', 
    '2024-05-12T03:38:37',
    '2024-06-13T19:53:47', 
    '2024-07-16T14:33:56', 
    '2024-08-18T06:57:04', 
    '2024-09-20T02:28:50', 
    '2024-10-22T18:10:59', 
    '2024-11-24T13:05:23', 
    '2024-12-27T05:22:27',
    '2025-01-28T23:05:08',
    '2025-03-02T16:04:28',
    '2025-04-04T09:30:50',
    '2025-05-07T03:01:28',
    '2025-06-08T20:30:45',
    '2025-07-11T13:40:11',
    '2025-08-13T07:06:46',
    '2025-09-14T23:42:30']#,
#    '2025-10-20T15:37:23']

JUNO_APOJOVES = [              # These were not updated
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
    '2021-08-12T03:18:51',
    '2021-09-24T19:58:32',
    '2021-11-07T15:42:37',
    '2021-12-21T12:20:01',
    '2022-02-03T06:18:33',
    '2022-03-18T21:02:00',
    '2022-05-01T09:01:29',
    '2022-06-13T17:58:27',
    '2022-07-27T00:00:03',
    '2022-09-08T04:08:09',
    '2022-10-18T19:32:20',
    '2022-11-26T00:29:18',
    '2023-01-03T04:34:56',
    '2023-02-10T05:47:13',
    '2023-03-20T07:09:36',
    '2023-04-27T07:40:51',
    '2023-06-04T07:14:33',
    '2023-07-12T07:58:44',
    '2023-08-19T10:31:09',
    '2023-09-26T11:21:03',
    '2023-11-03T11:40:23',
    '2023-12-11T12:19:47',
    '2024-01-17T05:08:58',
    '2024-02-20T06:53:44',
    '2024-03-24T00:34:41',
    '2024-04-25T19:49:35',
    '2024-05-28T17:15:00',
    '2024-06-30T14:21:26',
    '2024-08-02T11:12:59',
    '2024-09-04T08:05:13',
    '2024-10-07T04:44:29',
    '2024-11-09T00:15:02',
    '2024-12-11T20:16:22',
    '2025-01-13T17:19:39',
    '2025-02-15T14:58:50',
    '2025-03-20T13:15:57',
    '2025-04-22T11:05:48',
    '2025-05-25T09:33:12',
    '2025-06-27T08:27:19',
    '2025-07-30T07:17:20',
    '2025-09-01T06:10:10',
    '2025-10-04T04:36:13']

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
        return \
            f'date: {dtime}   ' \
            f'PJ: {pj:0.2f} ' \
            f'y: {y*self.yunit:0.2f}'

def juno_pj_axis(ax, position='top'):
    jts = JunoTimes()
    secax = ax.secondary_xaxis(position,
                               functions=(jts.plt_date2pj, jts.pj2plt_date))
    secax.xaxis.set_minor_locator(MultipleLocator(1))
    secax.set_xlabel('PJ')

