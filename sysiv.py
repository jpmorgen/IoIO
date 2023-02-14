"""Module to display periodograms and phase plots in search of sysIV"""

from astropy import log
import astropy.units as u
from astropy.time import Time

from IoIO.torus import periodogram, phase_plot

log.setLevel('DEBUG')

plotdir = '/home/jpmorgen/Papers/io/IoIO_2017--2022/'




#min_period = 40*u.hr
min_period = 9.5*u.hr
#min_period = 3*u.hr
#min_period = 1*u.hr
max_period = 11*u.hr
#max_period = 25*u.hr
#max_period = (40*u.day).to(u.hr)
#max_period = (180*u.day).to(u.hr)

fold_period = 9.925*u.hr

#start = Time('2017-01-01', format='fits')
#stop = Time('2017-12-31', format='fits')
#fold_period = 10.2*u.hr [ east ansa 10.29]

start = Time('2018-01-01', format='fits')
stop = Time('2018-12-31', format='fits')
#fold_period = 9.9*u.hr
#fold_period = 9.94*u.hr # null hypothesis -- still significant
#fold_period = 9.98*u.hr
#fold_period = 10.07*u.hr # 1/2 wave
#fold_period = 10.23*u.hr # 1/2 wave with bump
#fold_period = 10.30*u.hr # 1/2 wave with bump

#fold_period = 10.43*u.hr # most convincing full-ish wave
#fold_period = 10.15*u.hr # null hypothesis -- 3 lumps

#start = Time('2019-01-01', format='fits')
#stop = Time('2019-12-31', format='fits')
#fold_period = 10.10*u.hr # best yet!

#start = Time('2020-01-01', format='fits')
#stop = Time('2020-12-31', format='fits')
#stop = Time('2020-09-01', format='fits') # same with folding
#fold_period = 10.10*u.hr #\ one complex 10.12+-0.05
#fold_period = 10.15*u.hr #/
#fold_period = 10.12*u.hr # ugly 2-lump
#fold_period = 10.29*u.hr # ugly slope

## Yucky in Lomb-Scargle
#start = Time('2021-01-01', format='fits')
#stop = Time('2021-12-31', format='fits')
#fold_period = 10.11*u.hr # NICE!  Double-lumped sysIII

#start = Time('2022-01-01', format='fits')
#stop = Time('2023-04-01', format='fits')
#fold_period = 10.12*u.hr # Weak
#fold_period = 10.24*u.hr # Pretty clear
#fold_period = 10.36*u.hr # Not as nice a sin wave

#start = Time('2022-08-01', format='fits')
#start = Time('2022-09-15', format='fits')
#stop = Time('2023-11-01', format='fits')
# Peaks similar to whole dataset

#start = Time('2022-11-01', format='fits')
#stop = Time('2023-04-01', format='fits')
fold_period = 10.23*u.hr # no side-lobes in periodogram.  Phase is
                         # pretty tight, not perfect sin
#start = Time('2022-11-21', format='fits')
#stop = Time('2023-04-01', format='fits')
# East ansa 10.25, so basically the same

# # These months up to 4 don't make much sense
#start = Time('2018-01-01', format='fits')
#stop = Time('2018-03-01', format='fits')
#fold_period = 10.03*u.hr # Maybe, sysIII not enough data
#fold_period = 10.17*u.hr # Maybe
#fold_period = 10.38*u.hr # Maybe -- these are all slopes

#start = Time('2018-02-01', format='fits')
#start = Time('2018-03-01', format='fits')
#stop = Time('2018-04-01', format='fits')
#fold_period = 10.02*u.hr # Not bad
#fold_period = 10.11*u.hr # Not bad
#fold_period = 10.05*u.hr # null hypothesis sloping
#stop = Time('2018-05-15', format='fits') # A lot more data
#fold_period = 9.925*u.hr # multi-peaked
#fold_period = 9.98*u.hr # yes, but ragged
#fold_period = 10.07*u.hr # not too convincing
#fold_period = 10.22*u.hr # OK
#fold_period = 10.41*u.hr # not great


#start = Time('2018-05-15', format='fits')
#start = Time('2018-01-01', format='fits')
#stop = Time('2018-05-01', format='fits')
#start = Time('2018-05-01', format='fits')
#start = Time('2018-03-01', format='fits')
#start = Time('2018-04-01', format='fits')
#stop = Time('2018-07-01', format='fits')
#stop = Time('2018-12-31', format='fits')
#fold_period = 9.925*u.hr # multi-lump
#fold_period = 10.00*u.hr # No: very slight modulation + slope
#fold_period = 10.32*u.hr # yes modulation, but not even sin
#fold_period = 10.42*u.hr # double-lumped sin

#fold_period = 10.35*u.hr # complex of 10.32 & 10.42 except for narrow
#                         # lump, pretty good sin
#
#fold_period = 10.19*u.hr # null hypothesis, could still call that
#                         # convincing at a low level



##periodogram(start, stop, min_period=min_period, max_period=max_period,
##            false_alarm_level=0.001)
periodogram(start, stop, side='east', min_period=min_period,
            max_period=max_period, plot_type='frequency')#, plotdir=plotdir)#, autopower=True)




#phase_plot(start, stop, fold_period=fold_period, plotdir=plotdir)

