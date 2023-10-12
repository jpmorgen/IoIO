import numpy as np

import matplotlib.pyplot as plt

import astropy.units as u
from astropy.table import QTable
from astropy.convolution import Gaussian1DKernel
from astropy.coordinates import Angle
from astropy.modeling import models, fitting

from IoIO.utils import nan_biweight, nan_mad
from IoIO.torus import (IO_ORBIT_R, add_medfilt,
                        add_mask_col, add_interpolated)


MEDFILT_WIDTH = 21

t = QTable.read('/data/IoIO/Torus/Torus.ecsv')
add_mask_col(t)
mask = t['mask'] 
#mask = np.logical_or(mask, t['ansa_left_r_peak_err'] > 0.1*u.R_jup)
#mask = np.logical_or(mask, t['ansa_right_r_peak_err'] > 0.1*u.R_jup)
t = t[~t['mask']]    

t.sort('tavg')
add_medfilt(t, 'ansa_left_surf_bright', medfilt_width=MEDFILT_WIDTH)
add_medfilt(t, 'ansa_right_surf_bright', medfilt_width=MEDFILT_WIDTH)
add_medfilt(t, 'ansa_left_r_peak', medfilt_width=MEDFILT_WIDTH)
add_medfilt(t, 'ansa_right_r_peak', medfilt_width=MEDFILT_WIDTH)

kernel = Gaussian1DKernel(10)
add_interpolated(t, 'ansa_left_r_peak_medfilt', kernel)
add_interpolated(t, 'ansa_right_r_peak_medfilt', kernel)

t['ansa_left_surf_bright_medsub'] = (t['ansa_left_surf_bright']
         - t['ansa_left_surf_bright_medfilt'])
t['ansa_right_surf_bright_medsub'] = (t['ansa_right_surf_bright']
         - t['ansa_right_surf_bright_medfilt'])

t.write('/data/IoIO/analysis/Torus_for_Carl.ecsv', overwrite=True)

tavg = t['tavg']
left_sysIII = t['Jupiter_PDObsLon'] + 90*u.deg
right_sysIII = t['Jupiter_PDObsLon'] - 90*u.deg
left_pos = t['ansa_left_r_peak']
left_pos_err = t['ansa_left_r_peak_err']
right_pos = t['ansa_right_r_peak']
right_pos_err = t['ansa_right_r_peak_err']

left_pos_medfilt = t['ansa_left_r_peak_medfilt']
right_pos_medfilt = t['ansa_right_r_peak_medfilt']

## This doesn't change anything relative to the medfilts because we are
## trying to subtract it from data that doesn't exist
#left_pos_interp = t['ansa_left_r_peak_medfilt_interp']
#right_pos_interp = t['ansa_right_r_peak_medfilt_interp']

left_sb_medsub = t['ansa_left_surf_bright_medsub']
left_sb_err = t['ansa_left_surf_bright_err']
right_sb_medsub = t['ansa_right_surf_bright_medsub']
right_sb_err = t['ansa_right_surf_bright_err']

#plt.errorbar(tavg.datetime, left_medsub.value, left_err.value, fmt='b,')
#plt.errorbar(tavg.datetime, right_medsub.value, right_err.value, fmt='r,')
#plt.show()

#plt.errorbar(left_sysIII.value, left_medsub.value, left_err.value, fmt='b,')
#plt.errorbar(right_sysIII.value, right_medsub.value, right_err.value, fmt='r,')
#plt.xlim((0, 720))
#plt.show()

left_pos_biweight = nan_biweight(left_pos_medfilt)
right_pos_biweight = nan_biweight(right_pos_medfilt)

left_medsub_pos = -left_pos
right_medsub_pos = right_pos

#left_medsub_pos = -(left_pos - left_pos_medfilt + left_pos_biweight)
#right_medsub_pos = right_pos - right_pos_medfilt + right_pos_biweight

## This is exactly the same as medfilt, as per above
##left_medsub_pos = left_pos - (left_pos_interp
##                              + nan_biweight(left_pos_interp)
##                              + IO_ORBIT_R)
##right_medsub_pos = right_pos - (right_pos_interp
##                                + nan_biweight(right_pos_interp)
##                                - IO_ORBIT_R)


left_sysIII = Angle(left_sysIII)
right_sysIII = Angle(right_sysIII)
left_sysIII = left_sysIII.wrap_at(360*u.deg)
right_sysIII = right_sysIII.wrap_at(360*u.deg)
left_sysIII = np.append(left_sysIII, left_sysIII+360*u.deg)
right_sysIII = np.append(right_sysIII, right_sysIII+360*u.deg)


left_medsub_pos = np.append(left_medsub_pos, left_medsub_pos)
right_medsub_pos = np.append(right_medsub_pos, right_medsub_pos)
left_pos_err = np.append(left_pos_err, left_pos_err)
right_pos_err = np.append(right_pos_err, right_pos_err)


s95_left = (models.Cosine1D(amplitude=0.049*u.R_jup,
                          frequency=1/360,
                          phase=-167/360)
            + models.Polynomial1D(0, c0=5.85*u.R_jup))
s95_right = (models.Cosine1D(amplitude=0.073*u.R_jup,
                           frequency=1/360,
                           phase=-130/360)
             + models.Polynomial1D(0, c0=5.57*u.R_jup))

s18_left = (models.Cosine1D(amplitude=0.028*u.R_jup,
                          frequency=1/360,
                          phase=-142/360)
            + models.Polynomial1D(0, c0=5.862*u.R_jup))
s18_right = (models.Cosine1D(amplitude=0.044*u.R_jup,
                           frequency=1/360,
                           phase=-128/360)
             + models.Polynomial1D(0, c0=5.602*u.R_jup))

IoIO_left = (models.Cosine1D(amplitude=0.03*u.R_jup,
                           frequency=1/360,
                           phase=-142/360)
             + models.Polynomial1D(0, c0=-left_pos_biweight))
IoIO_right = (models.Cosine1D(amplitude=0.03*u.R_jup,
                            frequency=1/360,
                            phase=-128/360)
             + models.Polynomial1D(0, c0=right_pos_biweight))

fit = fitting.LevMarLSQFitter(calc_uncertainties=True)
IoIO_left_fit = fit(IoIO_left, left_sysIII.value, left_medsub_pos,
                    weights=left_pos_err, maxiter=500)
IoIO_right_fit = fit(IoIO_right, right_sysIII.value, right_medsub_pos,
                    weights=right_pos_err, maxiter=500)

#plt.errorbar(left_sysIII.value,
#             left_medsub_pos.value,
#             left_pos_err.value, fmt='b,')
#plt.errorbar(right_sysIII.value,
#             right_medsub_pos.value,
#             right_pos_err.value, fmt='r,')
#plt.xlim((0, 720)) 
#plt.show()

figmult = 2
f = plt.figure(figsize=[11*figmult, 8.5*figmult])
plt.errorbar(left_sysIII.value,
             left_medsub_pos.value,
             left_pos_err.value, fmt='b,')
plt.errorbar(right_sysIII.value,
             right_medsub_pos.value,
             right_pos_err.value, fmt='r,')

earth_sysIII = np.arange(-90, 720+90)
left_model_sysIII = earth_sysIII# + 90
right_model_sysIII = earth_sysIII# - 90
plt.plot(left_model_sysIII, s95_left(left_model_sysIII), 'b-.')
plt.plot(right_model_sysIII, s95_right(right_model_sysIII), 'r-.')

plt.plot(left_model_sysIII, s18_left(left_model_sysIII), 'b-.')
plt.plot(right_model_sysIII, s18_right(right_model_sysIII), 'r-.')


#plt.plot(left_model_sysIII, IoIO_left_fit(left_model_sysIII), 'b-')
#plt.plot(right_model_sysIII, IoIO_right_fit(right_model_sysIII), 'r-')

plt.plot(left_model_sysIII, IoIO_left(left_model_sysIII), 'b-')
plt.plot(right_model_sysIII, IoIO_right(right_model_sysIII), 'r-')

plt.xlim((0, 720))
#plt.ylim((5.52, 6.25))

plt.xlabel(r'$\lambda_\mathrm{III}$ (deg)')
plt.ylabel(r'Dawnward ribbon shift from Io orbit (R$_\mathrm{J}$)')

plt.show()

