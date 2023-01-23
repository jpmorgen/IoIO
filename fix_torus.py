"""Add mean_image column"""

import numpy as np

import astropy.units as u
from astropy.table import QTable

from bigmultipipe import BigMultiPipe, prune_pout

from IoIO.cordata import CorData
from IoIO.cormultipipe import mean_image
from IoIO.torus import add_mask_col


PROCESS_SIZE = 2E9 # one Na image seems to stay under 2G

class AddBackground(BigMultiPipe):
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

back_pipe = AddBackground()

flist = t['outname']#[0:10]
pout = back_pipe.pipeline(flist, post_process_list=[mean_image])
flist, pout = prune_pout(flist, pout)
_ , pipe_meta = zip(*pout)

t['mean_image'] = np.NAN * u.R

for outname, meta in zip(flist, pipe_meta):
    print(outname, meta['mean_image'])
    idx = np.flatnonzero(t['outname'] == outname)
    t[idx[0]]['mean_image'] = meta['mean_image']

add_mask_col(t)

t.write('/data/IoIO/Torus/Torus_cleaned.ecsv', overwrite=True)
