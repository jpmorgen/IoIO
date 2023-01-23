"""Standardizes outnames in tables to .fits""" 

import re
import os

from astropy.table import QTable

def fix_outnames(top, ecsvname):
    fulldirs = [os.path.join(top, d) for d in os.listdir(top)]
    dirs = [d for d in fulldirs
            if (not os.path.islink(d) and os.path.isdir(d))]
    
    for directory in dirs:
        tname = os.path.join(directory, ecsvname)
        print(tname)
        try:
            ot= QTable.read(tname)
        except:
            print(f'Bad table: {tname}')
            continue    
        if len(ot) == 0:
            print(f'Zero-length table: {tname}')
            continue
        newt = ot.copy()
        n = [os.path.splitext(o)[0] + '.fits' for o in newt['outname']]    
        newt['outname'] = n
        #tname = '/tmp/test.ecsv'
        newt.write(tname, overwrite=True)

fix_outnames('/data/IoIO/Na_nebula', 'Na_nebula.ecsv')
fix_outnames('/data/IoIO/Torus', 'Characterize_Ansas.ecsv')
