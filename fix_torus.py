"""Add mean_image column"""

import os
import numpy as np

import astropy.units as u
from astropy.table import QTable, MaskedColumn, vstack

from bigmultipipe import BigMultiPipe, prune_pout

from IoIO.cordata import CorData
from IoIO.cormultipipe import calc_obj_to_ND, mean_image
from IoIO.torus import add_mask_col


PROCESS_SIZE = 2E9 # one Na image seems to stay under 2G

class FixTorus(BigMultiPipe):
    def __init__(self, **kwargs):                 
        super().__init__(num_processes=0.8,
                         process_size=PROCESS_SIZE,
                         **kwargs)

    def file_read(self, in_name, **kwargs):
        if not os.path.exists(in_name):
            # Some files end erroneously in fit instead of fits
            in_name += 's'
            if not os.path.exists(in_name):
                log.warning(f'File not found: {in_name}')
                return None
        return CorData.read(in_name)

t = QTable.read('/data/IoIO/Torus/Torus.ecsv')

back_pipe = FixTorus()

flist = t['outname']#[0:10]
pout = back_pipe.pipeline(flist, post_process_list=[calc_obj_to_ND,
                                                    mean_image])
flist, pout = prune_pout(flist, pout)
_ , pipe_meta = zip(*pout)

t['mean_image'] = np.NAN * u.R
t['obj_to_ND'] = np.NAN * u.R

t.add_index('outname')
for outname, meta in zip(flist, pipe_meta):
    print(f"{outname} {meta['obj_to_ND']:.2f} {meta['mean_image']:.2f}")
    t.loc[outname]['obj_to_ND'] = meta['obj_to_ND']
    t.loc[outname]['mean_image'] = meta['mean_image']

#t = QTable.read('/data/IoIO/Torus/Torus_cleaned.ecsv')
add_mask_col(t)

t.write('/data/IoIO/Torus/Torus_cleaned.ecsv', overwrite=True)


#### This includes some of Jan 2023, but all are fixed
###t2022 = QTable.read('/data/IoIO/Torus/Torus_cleaned.ecsv')
#### Everything from the day beyond the last entry of t2022 (actually a
#### little before) to the present has mean_image and mask_col
###t2023 = QTable.read('/data/IoIO/Torus/Torus.ecsv')
###
###last_tavg = t2022[-1]['tavg']
###t2023 = t2023[t2023['tavg'] > last_tavg]
###
####loc = t2022['tavg'].location.copy()
####t2022['tavg'].location = None
####t2023['tavg'].location = None
###
#### Somehow this didn't get done properly
###add_mask_col(t2022)
###t2022['mask'] = MaskedColumn(t2022['mask'])
###
###t = vstack([t2022, t2023])
####t['tavg'].location = loc
###t.write('/data/IoIO/Torus/Torus_cleaned_2023.ecsv', overwrite=True)
