import os
import re
import glob
from datetime import datetime
from dateutil.relativedelta import relativedelta

from astropy import log
import astropy.units as u
from astropy.units import imperial

from boltwood_parser import EntryCollection

from IoIO.utils import dict_to_ccd_meta
from IoIO.ioio_directories import RAW_DATA_ROOT
from IoIO.cordata import CorData

AAG_DIRECTORY = os.path.join(RAW_DATA_ROOT, 'AAG')
CURRENT_AAG_FNAME = os.path.join(AAG_DIRECTORY,'AAG_SLD.log')
PAST_AAG_GLOB = 'AAG_SLD_*.log'
WEATHER_MAX_DELTA_TIME = 1800 # s
# The first four of these are provided by the Boltwood II single-line
# data format.  PRESSURE and HUMIDITY are provided by the mount, which
# it is connected to MaxIm
WEATHER_KEYS = ['AMBTEMP', 'DEWPOINT', 'WINDSPD', 'SKYTEMP',
                'PRESSURE', 'HUMIDITY']

# Enable imperial units for this and all modules that import it
imperial.enable()

class CorBoltwood():
    def __init__(self,
                 raw_data_root=RAW_DATA_ROOT,
                 current_aag_fname=CURRENT_AAG_FNAME,
                 past_aag_glob=PAST_AAG_GLOB,
                 weather_max_delta_time=WEATHER_MAX_DELTA_TIME,
                 precalc=False):
        self.raw_data_root = raw_data_root
        self.current_aag_fname = current_aag_fname
        self.past_aag_glob = past_aag_glob
        self.weather_max_delta_time = weather_max_delta_time
        self._aag_fdicts = None
        if precalc:
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

    def boltwood_to_ccd_meta(self, ccd):
        """Put Boltwood entries into CCD metadata if missing.  If
        present, indicate likely units in comet area of FITS cards

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
        weather_meta = {}
        if ccd.meta.get('AMBTEMP'):
            # ACP was writing Boltwood values into FITS headers, but
            # not recording their units.  Write the units I configured
            # the weather station for in by hand into the FITS card
            # comment area
            temp_unit = u.C
            wind_unit = u.imperial.mi/u.hr
            weather_meta['AMBTEMP']  = ccd.meta['AMBTEMP']*temp_unit
            weather_meta['DEWPOINT'] = ccd.meta['DEWPOINT']*temp_unit
            weather_meta['WINDSPD']  = ccd.meta['WINDSPD']*wind_unit 
            weather_meta['SKYTEMP']  = ccd.meta['SKYTEMP']*temp_unit
            return dict_to_ccd_meta(ccd, weather_meta)

        try:
            be = self.boltwood_entry(ccd)
        except Exception as e:
            log.error(f'RAWFNAME of problem: {ccd.meta["RAWFNAME"]} {e}')
            return ccd
        if be is None:
            return ccd
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
        return dict_to_ccd_meta(ccd, weather_meta)

def weather_to_bmp_meta(ccd_in, bmp_meta=None, **kwargs):
    if isinstance(ccd_in, list):
        return [weather_to_bmp_meta(ccd, bmp_meta=bmp_meta,
                                 **kwargs)
                for ccd in ccd_in]
    if bmp_meta is None:
        bmp_meta = {}
    imperial.enable()
    for k in WEATHER_KEYS:
        # Units are of the format '[unit] <other possible text>'
        val = ccd_in.meta.get(k)
        if val is None:
            # Not all of the keys are present all the time
            continue
        unit = ccd_in.meta.comments[k]
        unit = re.search(r'\[(.*)\]', unit)
        unit = unit.group(1)
        unit = u.Unit(unit)
        bmp_meta[k] = val*unit
    return ccd_in
        
#cb = CorBoltwood(precalc=True)
##ccd = CorData.read('/data/IoIO/raw/2018-05-16/SII_on-band_026.fits')
##ccd = CorData.read('/data/IoIO/raw/20210520/SII_on-band_002.fits')
##ccd = CorData.read('/data/IoIO/raw/20231208/0012P-S001-R001-C001-Na_off.fts')
#ccd = CorData.read('/data/IoIO/raw/20240224/SII_on-band_004.fits')
#ccd = cb.boltwood_to_ccd_meta(ccd)
#print(ccd.meta)
#bmp_meta = {}
#ccd = weather_to_bmp_meta(ccd, bmp_meta)

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
