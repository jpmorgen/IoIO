import os
import glob
from datetime import datetime
from dateutil.relativedelta import relativedelta

from astropy import log

from boltwood_parser import EntryCollection

from IoIO.cormultipipe import RAW_DATA_ROOT
from IoIO.cordata import CorData

AAG_DIRECTORY = os.path.join(RAW_DATA_ROOT, 'AAG')
AAG_FNAME = 'AAG_SLD.log'
PAST_AAG_GLOB = 'AAG_SLD_*.log'

log.setLevel('DEBUG')


#ccd = CorData.read('/data/IoIO/raw/20231208/0012P-S001-R001-C001-Na_off.fts')
#ccd = CorData.read('/data/IoIO/raw/20240224/SII_on-band_004.fits')
ccd = CorData.read('/data/IoIO/raw/20210520/SII_on-band_002.fits')

if ccd.meta.get('AMBTEMP'):
    # Weather data has already been entered by ACP
    print('continue')

flist = glob.glob(os.path.join(AAG_DIRECTORY, PAST_AAG_GLOB))

flist.sort(reverse=True)

past_fdicts = []
for f in flist:
    # AAG_SLD.log is always the current log
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

last_fname = None
for fdict in past_fdicts:
    if fdict['stop'] < ccd.tavg.datetime:
        aag_fname = last_fname or AAG_FNAME
        break
    last_fname = fdict['fname']
    
print(aag_fname)


with open(aag_fname, mode='r') as file:
    w = file.read()
bc = EntryCollection(w)
be = bc.find(ccd.tavg.datetime)
