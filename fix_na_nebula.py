"""Renames columns to new column name standard""" 

import re
import os

from astropy.table import QTable

top = '/data/IoIO/Na_nebula'
fulldirs = [os.path.join(top, d) for d in os.listdir(top)]
dirs = [d for d in fulldirs
        if (not os.path.islink(d) and os.path.isdir(d))]

for directory in dirs:
    tname = os.path.join(directory, 'Na_nebula.ecsv')
    print(tname)
    try:
        ot= QTable.read(tname)
    except:
        print('Bad table')
        continue    
    if len(ot) == 0:
        print('Zero-length table')
        continue
    newt = ot.copy()
    ocolnames = ot.colnames
    old_na_ap_re = re.compile('Na_ap_.*')
    old_na_aps = list(filter(old_na_ap_re.match, ocolnames))

    for col in old_na_aps:
        split_col = col.split('_')
        newcol = f'{split_col[0]}_{split_col[4]}_{split_col[2]}_jupiterRad'
        newt.rename_column(col, newcol)

    #tname = '/tmp/test.ecsv'
    newt.write(tname, overwrite=True)
