#!/bin/python3
import os

import matplotlib.pyplot as plt

import moviepy.editor as mpy

import astropy.units as u
from astropy.time import Time
from astropy.table import QTable
from astropy.stats import biweight_location

from IoIO.utils import ColnameEncoder
from IoIO.torus import add_mask_col

SPEEDUP = 24000
FPS = 24

def make_movie(t, outname, reduction_suffix=None, start=None, stop=None):
    t = t[~t['mask']]
    if start is not None:
        start = Time(start, format='fits')
        mask = t['tavg'] >= start
        t = t[mask]
    if stop is not None:
        stop = Time(stop, format='fits')
        mask = t['tavg'] <= stop
        t = t[mask]
    print(len(t))

    t.sort('tavg')
    png_names = [os.path.splitext(fname)[0]
                 for fname in t['outname']]
    if reduction_suffix:
        # This is a little awkward, since I didn't keep with the pattern
        png_names = [fname.replace('-back-sub', reduction_suffix) 
                     for fname in png_names]
    print('first png_names')
    print(png_names[10])
    png_names = [fname + '.png'
                 for fname in png_names]
    # Sometimes creation of PNG files fails
    png_names = [fname for fname in png_names
                 if os.path.exists(fname)]
    print('second png_names')
    print(png_names[10])

    durations = (t['tavg'][1:] - t['tavg'][0:-1]).sec
    med_duration = biweight_location(durations)

    durations[durations > 40000] = med_duration

    # https://stackoverflow.com/questions/44732602/convert-image-sequence-to-video-using-moviepy
    clips = [mpy.ImageClip(m).set_duration(d/SPEEDUP)
             for m, d in zip(png_names, durations)]

    movie = mpy.concatenate_videoclips(clips, method="compose")
    movie.write_videofile(outname, fps=FPS)

start = None
stop = None

# # 2018-05-12
# start = '2018-05-11'
# stop = '2018-05-20'

# 2022-11-09
#start = '2022-10-21'
#stop = '2022-11-22'

## Looking for bright start of 2022 Na outburst
#start = '2022-05-29'
#stop =  '2022-09-17'

#start = '2023-06-01'

# Search for outburst during Juno
start = '2024-08-01'

t = QTable.read('/data/IoIO/Na_nebula/Na_nebula_cleaned.ecsv')
make_movie(t, '/tmp/test_Na.mp4', start=start, stop=stop)
#make_movie(t, '/tmp/test_Na_apertures.mp4', reduction_suffix='_apertures',
#           start=start, stop=stop)
t = QTable.read('/data/IoIO/Torus/Torus_cleaned.ecsv')
make_movie(t, '/tmp/test_SII.mp4', start=start, stop=stop)


#make_movie(t, '/tmp/test_SII_reduction.mp4', reduction_suffix='_reduction',
#           start=start, stop=stop)

