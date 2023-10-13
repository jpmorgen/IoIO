import numpy as np

import matplotlib.pyplot as plt

import astropy.units as u
from astropy.table import QTable
from astropy.convolution import Gaussian1DKernel
from astropy.coordinates import Angle
from astropy.modeling import models, fitting

from IoIO.utils import nan_biweight, nan_mad
from IoIO.torus import (IO_ORBIT_R, nan_median_filter, add_medfilt,
                        add_mask_col, add_interpolated)


TIME_MEDFILT_WIDTH = 21
ANGLE_MEDFILT_WIDTH = 101

t = QTable.read('/data/IoIO/Torus/Torus.ecsv')
add_mask_col(t)
t['mask'] = np.logical_or(t['mask'], t['ansa_left_r_peak_err'] > 0.05*u.R_jup)
t['mask'] = np.logical_or(t['mask'], t['ansa_right_r_peak_err'] > 0.05*u.R_jup)
t = t[~t['mask']]    

t.sort('tavg')
add_medfilt(t, 'ansa_left_surf_bright', medfilt_width=TIME_MEDFILT_WIDTH)
add_medfilt(t, 'ansa_right_surf_bright', medfilt_width=TIME_MEDFILT_WIDTH)
add_medfilt(t, 'ansa_left_r_peak', medfilt_width=TIME_MEDFILT_WIDTH)
add_medfilt(t, 'ansa_right_r_peak', medfilt_width=TIME_MEDFILT_WIDTH)

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

#left_pos_medsub = -left_pos
#right_pos_medsub = right_pos

left_pos_medsub = -(left_pos - left_pos_medfilt + left_pos_biweight)
right_pos_medsub = right_pos - right_pos_medfilt + right_pos_biweight


## This is exactly the same as medfilt, as per above
##left_pos_medsub = left_pos - (left_pos_interp
##                              + nan_biweight(left_pos_interp)
##                              + IO_ORBIT_R)
##right_pos_medsub = right_pos - (right_pos_interp
##                                + nan_biweight(right_pos_interp)
##                                - IO_ORBIT_R)


left_sysIII = Angle(left_sysIII)
right_sysIII = Angle(right_sysIII)
left_sysIII = left_sysIII.wrap_at(360*u.deg)
right_sysIII = right_sysIII.wrap_at(360*u.deg)
left_sysIII = np.append(left_sysIII, left_sysIII+360*u.deg)
right_sysIII = np.append(right_sysIII, right_sysIII+360*u.deg)


left_pos_medsub = np.append(left_pos_medsub, left_pos_medsub)
right_pos_medsub = np.append(right_pos_medsub, right_pos_medsub)
left_pos_err = np.append(left_pos_err, left_pos_err)
right_pos_err = np.append(right_pos_err, right_pos_err)

left_sysIII_idx = np.argsort(left_sysIII)
right_sysIII_idx = np.argsort(right_sysIII)

left_sysIII_by_sysIII = left_sysIII[left_sysIII_idx]
right_sysIII_by_sysIII = right_sysIII[right_sysIII_idx]
left_pos_medsub_by_sysIII = left_pos_medsub[left_sysIII_idx]
right_pos_medsub_by_sysIII = right_pos_medsub[right_sysIII_idx]
left_pos_by_sysIII_err = left_pos_err[left_sysIII_idx]
right_pos_by_sysIII_err = right_pos_err[right_sysIII_idx]

left_pos_medsub_by_sysIII = nan_median_filter(left_pos_medsub_by_sysIII,
                                              size=ANGLE_MEDFILT_WIDTH)
right_pos_medsub_by_sysIII = nan_median_filter(right_pos_medsub_by_sysIII,
                                               size=ANGLE_MEDFILT_WIDTH)



s95_left = (models.Cosine1D(amplitude=0.049,
                          frequency=1/360,
                          phase=-167/360)
            + models.Polynomial1D(0, c0=5.85))
s95_right = (models.Cosine1D(amplitude=0.073,
                           frequency=1/360,
                           phase=-130/360)
             + models.Polynomial1D(0, c0=5.57))

s18_left = (models.Cosine1D(amplitude=0.028,
                          frequency=1/360,
                          phase=-142/360)
            + models.Polynomial1D(0, c0=5.862))
s18_right = (models.Cosine1D(amplitude=0.044,
                           frequency=1/360,
                           phase=-128/360)
             + models.Polynomial1D(0, c0=5.602))

IoIO_left = (models.Cosine1D(amplitude=0.03,
                           frequency=1/360,
                             phase=-142/360,
                             fixed={'frequency': True})
             + models.Polynomial1D(0, c0=-left_pos_biweight.value,
                                   fixed={'c0': True}))
IoIO_right = (models.Cosine1D(amplitude=0.03,
                              frequency=1/360,
                              phase=-128/360,
                              fixed={'frequency': True})
             + models.Polynomial1D(0, c0=right_pos_biweight.value,
                                   fixed={'c0': True}))

#fit = fitting.LevMarLSQFitter(calc_uncertainties=True)
#IoIO_left_fit = fit(IoIO_left, left_sysIII.value,
#                    left_pos_medsub_by_sysIII.value,
#                    weights=1/left_pos_err, maxiter=500)
#print(fit.fit_info['message'])
#IoIO_right_fit = fit(IoIO_right, right_sysIII.value,
#                     right_pos_medsub_by_sysIII.value,
#                    weights=1/right_pos_err, maxiter=500)
#print(fit.fit_info['message'])

fit = fitting.LevMarLSQFitter()
IoIO_left_fit = fit(IoIO_left, left_sysIII.value,
                    left_pos_medsub_by_sysIII.value,
                    weights=1/(0.05*u.R_jup))
print(fit.fit_info['message'])
IoIO_right_fit = fit(IoIO_right, right_sysIII.value,
                     right_pos_medsub_by_sysIII.value,
                     weights=1/(0.05*u.R_jup))

print(fit.fit_info['message'])



#plt.errorbar(left_sysIII.value,
#             left_pos_medsub.value,
#             left_pos_err.value, fmt='b,')
#plt.errorbar(right_sysIII.value,
#             right_pos_medsub.value,
#             right_pos_err.value, fmt='r,')
#plt.xlim((0, 720)) 
#plt.show()

figmult = 2
f = plt.figure(figsize=[11*figmult, 8.5*figmult])
plt.errorbar(left_sysIII.value,
             left_pos_medsub.value,
             left_pos_err.value, fmt='b,')
plt.errorbar(right_sysIII.value,
             right_pos_medsub.value,
             right_pos_err.value, fmt='r,')

plt.plot(left_sysIII_by_sysIII, left_pos_medsub_by_sysIII, 'k.')
plt.plot(right_sysIII_by_sysIII, right_pos_medsub_by_sysIII, 'k.')

earth_sysIII = np.arange(-90, 720+90)
left_model_sysIII = earth_sysIII# + 90
right_model_sysIII = earth_sysIII# - 90
plt.plot(left_model_sysIII, s95_left(left_model_sysIII), 'b-.')
plt.plot(right_model_sysIII, s95_right(right_model_sysIII), 'r-.')

plt.plot(left_model_sysIII, s18_left(left_model_sysIII), 'b-.')
plt.plot(right_model_sysIII, s18_right(right_model_sysIII), 'r-.')


plt.plot(left_model_sysIII, IoIO_left_fit(left_model_sysIII), 'b-')
plt.plot(right_model_sysIII, IoIO_right_fit(right_model_sysIII), 'r-')

#plt.plot(left_model_sysIII, IoIO_left(left_model_sysIII), 'b-')
#plt.plot(right_model_sysIII, IoIO_right(right_model_sysIII), 'r-')

plt.xlim((0, 720))
#plt.ylim((5.52, 6.25))

plt.xlabel(r'$\lambda_\mathrm{III}$ (deg)')
plt.ylabel(r'Dawnward ribbon shift from Io orbit (R$_\mathrm{J}$)')

plt.show()

