import os
import glob

import numpy as np

from astropy import log


from IoIO.cormultipipe import RAW_DATA_ROOT

log.setLevel('DEBUG')

directory = os.path.join(RAW_DATA_ROOT, 'AAG')
glob_include = ['AAG*.log']

flist = []
for gi in glob_include:
    flist += glob.glob(os.path.join(directory, gi))
print(flist)

all_w = None
for fname in flist:
    w = np.loadtxt(fname, dtype=str)
    if all_w is None:
        all_w = w
    else:
        all_w = np.append(all_w, w, axis=0)
    #break

# https://interactiveastronomy.com/skyroof_help/Weatherdatafile.html

dim_idx = np.flatnonzero(all_w[:, 18] == '2')
dark_idx = np.flatnonzero(all_w[:, 18] == '1')
ambient = all_w[:, 5]
ambient = ambient.astype(float)
print(f'Maximum recorded temperature: {np.max(ambient)}')
print(f'Maximum while dim: {np.max(ambient[dim_idx])}')
print(f'Maximum while dark: {np.max(ambient[dark_idx])}')

print(f'Minimum recorded temperature: {np.min(ambient)}')
print(f'Minimum while dim: {np.min(ambient[dim_idx])}')
print(f'Minimum while dark: {np.min(ambient[dark_idx])}')


#wdata = []
#for fname in flist:
#    twdata = cached_csv(None, fname, read_csvs=True, write_csvs=False)
#    wdata.append(twdata)

