import os
import glob
from datetime import datetime
from dateutil.relativedelta import relativedelta

from astropy import log
import astropy.units as u
from astropy.units import imperial

from boltwood_parser import EntryCollection

from IoIO.utils import dict_to_ccd_meta
from IoIO.cormultipipe import RAW_DATA_ROOT
from IoIO.cordata import CorData

AAG_DIRECTORY = os.path.join(RAW_DATA_ROOT, 'AAG')
CURRENT_AAG_FNAME = os.path.join(AAG_DIRECTORY,'AAG_SLD.log')
PAST_AAG_GLOB = 'AAG_SLD_*.log'
WEATHER_MAX_DELTA_TIME = 1800 # s


log.setLevel('DEBUG')

class CorBoltwood():
    def __init__(self,
                 raw_data_root=RAW_DATA_ROOT,
                 current_aag_fname=CURRENT_AAG_FNAME,
                 past_aag_glob=PAST_AAG_GLOB,
                 weather_max_delta_time=WEATHER_MAX_DELTA_TIME,
                 init_calc=False):
        self.raw_data_root = raw_data_root
        self.current_aag_fname = current_aag_fname
        self.past_aag_glob = past_aag_glob
        self.weather_max_delta_time = weather_max_delta_time
        self._aag_fdicts = None
        if init_calc:
            _ = self.aag_fdicts
    
    @property
    def aag_fdicts(self):
        """Returns a list of dictionaries in reverse cronological order
        that contain fname, start and stop times

        """
        if self._aag_fdicts is not None:
            return self._aag_fdicts
        flist = glob.glob(os.path.join(AAG_DIRECTORY, PAST_AAG_GLOB))
        flist.sort(reverse=True)
        past_fdicts = []
        for f in flist:
            bname = os.path.basename(f)
            root, _ = os.path.splitext(bname)
            _, year_quarter = root.split('AAG_SLD_')
            if 'q' not in year_quarter:
                continue
            year, quarter = year_quarter.split('q')
            quarter = int(quarter)
            month = (quarter-1)*3+1
            start_date = datetime(int(year), int(month), 1)
            # https://stackoverflow.com/questions/37135699/get-first-date-and-last-date-of-current-quarter-in-python
            stop_date = start_date + relativedelta(months=3, seconds=-0.001)
            past_fdicts.append({'fname': f,
                                'start': start_date,
                                'stop': stop_date})
            self._aag_fdicts = past_fdicts
        return self._aag_fdicts

    def aag_fname(self, ccd):
        """Returns AAG single-line format filename corresponding to data in ccd

        Parameters
        ----------
        ccd : CorObsData
            CorObsData for which weather data is desired

        Returns
        -------
        fname : str or None
            Returns fname of AAG_SLD file or None if ccd data were
            recorded before start of available AAG files 

        """
        last_fname = None
        for fdict in self.aag_fdicts:
            if fdict['stop'] < ccd.tavg.datetime:
                return last_fname or self.current_aag_fname
            last_fname = fdict['fname']
        return None

    def boltwood_entry(self, ccd):
        """Returns the boltwood_parser.WeatherEntry matching ccd

        Parameters
        ----------
        ccd : CorObsData
            CorObsData for which weather data is desired

        Returns
        -------
        weather_entry : WeatherEntry or None
            Returns WeatherEntry corresponding to data in ccd or None
            if ccd data were recorded before start of available AAG
            files

        """
        if self.aag_fname(ccd) is None:
            return None
        with open(self.aag_fname(ccd), mode='r') as aag_file:
            weather_data = aag_file.read()
        boltwood_collection = EntryCollection(weather_data)
        nearest_entry = boltwood_collection.find(ccd.tavg.datetime)
        if abs(nearest_entry.time.timestamp()
               - ccd.tavg.datetime.timestamp()) > self.weather_max_delta_time:
            return None
        return nearest_entry

    def boltwood_to_meta(self, ccd, bmp_meta=None, **kwargs):
        """Put Boltwood entries into CCD metadata (if missing) and
        transcribe into bmp_meta

        Parameters
        ----------
        ccd : CorData
            input ccd

        bmp_meta : dict or None
            bigmultipipe meta

        Returns
        -------
        ccd : CorData
            Copy of original CCD if CCD modified 

        """
        imperial.enable()
        if bmp_meta is None:
            bmp_meta = {}
        if ccd.meta.get('AMBTEMP'):
            # ACP was writing Boltwood values into FITS headers, but
            # not recording their units.  This is what I had things set for
            temp_unit = u.C
            wind_unit = u.imperial.mi/u.hr
            bmp_meta['AMBTEMP']  = ccd.meta['AMBTEMP']*temp_unit
            bmp_meta['DEWPOINT'] = ccd.meta['DEWPOINT']*temp_unit
            bmp_meta['WINDSPD']  = ccd.meta['WINDSPD']*wind_unit 
            bmp_meta['SKYTEMP']  = ccd.meta['SKYTEMP']*temp_unit
            return ccd
        be = self.boltwood_entry(ccd)
        if be is None:
            return ccd
        weather_meta = {}
        if be.temp_scale == 'F':
            temp_unit = u.imperial.deg_F
        else:
            temp_unit = u.Unit(be.temp_scale)
        if be.wind_scale == 'K':
            wind_unit = u.imperial.nauticalmile/u.hour
        elif be.wind_scale == 'M':
            wind_unit = u.imperial.mile/u.hour
        else:
            raise ValueError(f'Unknown wind unit {be.wind_scale}')
        weather_meta['AMBTEMP']  = be.ambient_temperature*temp_unit
        weather_meta['DEWPOINT'] = be.dew_point*temp_unit
        weather_meta['WINDSPD']  = be.wind_speed*wind_unit
        weather_meta['SKYTEMP']  = be.sky_temperature*temp_unit
        bmp_meta.update(weather_meta)
        return dict_to_ccd_meta(ccd, weather_meta)

bmp_meta = {}
cb = CorBoltwood(init_calc=True)
#ccd = CorData.read('/data/IoIO/raw/2018-05-16/SII_on-band_026.fits')
#ccd = CorData.read('/data/IoIO/raw/20210520/SII_on-band_002.fits')
#ccd = CorData.read('/data/IoIO/raw/20231208/0012P-S001-R001-C001-Na_off.fts')
ccd = CorData.read('/data/IoIO/raw/20240224/SII_on-band_004.fits')
ccd = cb.boltwood_to_meta(ccd, bmp_meta=bmp_meta)
print(bmp_meta)
print(ccd.meta)


#if ccd.meta.get('AMBTEMP'):
#    # Weather data has already been entered by ACP
#    print('continue')
#
#flist = glob.glob(os.path.join(AAG_DIRECTORY, PAST_AAG_GLOB))
#
#flist.sort(reverse=True)
#
#past_fdicts = []
#for f in flist:
#    # AAG_SLD.log is always the current log
#    bname = os.path.basename(f)
#    root, _ = os.path.splitext(bname)
#    _, year_quarter = root.split('AAG_SLD_')
#    if 'q' not in year_quarter:
#        continue
#    year, quarter = year_quarter.split('q')
#    quarter = int(quarter)
#    month = (quarter-1)*3+1
#    start_date = datetime(int(year), int(month), 1)
#    # https://stackoverflow.com/questions/37135699/get-first-date-and-last-date-of-current-quarter-in-python
#    stop_date = start_date + relativedelta(months=3, seconds=-0.001)
#    past_fdicts.append({'fname': f,
#                        'start': start_date,
#                        'stop': stop_date})
#
#last_fname = None
#for fdict in past_fdicts:
#    if fdict['stop'] < ccd.tavg.datetime:
#        aag_fname = last_fname or AAG_FNAME
#        break
#    last_fname = fdict['fname']
#    
#print(aag_fname)
#
#
#with open(aag_fname, mode='r') as file:
#    w = file.read()
#bc = EntryCollection(w)
#be = bc.find(ccd.tavg.datetime)
