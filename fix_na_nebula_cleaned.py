"""Renames columns to fix factor of 2 labeling error in aptures used
for Na nebula surface brightness measurements""" 

import re

from astropy.table import QTable

from IoIO.utils import ColnameEncoder
from IoIO.torus import add_mask_col

ot = QTable.read('/data/IoIO/Na_nebula/Na_nebula_cleaned_dists_too_big.ecsv')
newt = ot.copy()

ocolnames = ot.colnames

old_na_ap_re = re.compile('Na_ap_.*')
old_na_aps = list(filter(old_na_ap_re.match, ocolnames))

for col in old_na_aps:
    split_col = col.split('_')
    print(split_col)
    newcol = f'{split_col[0]}_{split_col[4]}_{split_col[2]}_jupiterRad'
    print(newcol)
    newt.rename_column(col, newcol)

old_sb_re = re.compile('.*_sb_.*')
old_sbs = list(filter(old_sb_re.match, ocolnames))

for col in old_sbs:
    split_col = col.split('_')
    print(split_col[-2])
    r = float(split_col[-2])
    r /= 2
    split_col[-2] = f'{r:.1f}'
    newcol = '_'.join(split_col)
    print(newcol)
    newt.rename_column(col, newcol)

add_mask_col(newt)

newt.write('/data/IoIO/Na_nebula/Na_nebula_cleaned.ecsv')
