#!/usr/bin/python3

import numpy as np
from astropy.time import Time, TimeDelta
from astropy import units as u

def guide_calc(x1, y1, fits_t1=None, x2=None, y2=None, fits_t2=None, guide_dt=10, guide_dx=0, guide_dy=0, last_guide=None, aggressiveness=0.5, target_c = np.asarray((1297, 1100))):
    """ Calculate offset guider values given pixel and times"""

    # Pixel scales in arcsec/pix
    main_scale = 1.59/2
    guide_scale = 4.42
    typical_expo = 385 * u.s
    
    if last_guide == None:
        guide_dt = guide_dt * u.s
        previous_dc_dt = np.asarray((guide_dx, guide_dy)) / guide_dt
    else:
        guide_dt = last_guide[0]
        previous_dc_dt = np.asarray((last_guide[1], last_guide[2])) / guide_dt

    # Convert input time interval to proper units
    
    # time period to use in offset guide file
    new_guide_dt = 10 * u.s

    if fits_t1 == None:
        t1 = Time('2017-01-01T00:00:00', format='fits')
    else:
        t1 = Time(fits_t1, format='fits')
    if fits_t2 == None:
        # Take our typical exposure time to settle toward the center
        t2 = t1 + typical_expo
    else:
        if fits_t1 == None:
            raise ValueError('fits_t1 given, but fits_t1 not supplied')
        t2 = Time(fits_t2, format='fits')
    dt = (t2 - t1) * 24*3600 * u.s / u.day

    c1 = np.asarray((x1, y1))
    c2 = np.asarray((x2, y2))
    
    if x2 == None and y2 == None:
        latest_c = c1
        measured_dc_dt = 0
    else:
        latest_c = c2
        measured_dc_dt = (c2 - c1) / dt * main_scale / guide_scale

    # Despite the motion of the previous_dc_dt, we are still getting
    # some motion form the measured_dc_dt.  We want to reverse that
    # motion and add in a little more to get to the target center.  Do
    # this gently over the time scale of our previous dt, moderated by
    # the aggressiveness
    
    target_c_dc_dt = (latest_c - target_c) / dt * aggressiveness
    print(target_c_dc_dt * dt)

    r = new_guide_dt * (previous_dc_dt - measured_dc_dt - target_c_dc_dt)
    
    # Print out new_rates
    print('{0} {1:+.3f} {2:+.3f}'.format(new_guide_dt/u.s, r[0], r[1]))

    return(new_guide_dt, r[0], r[1])

## Totally cool way to be able to run fibo as a script
#if __name__ == "__main__":
#    import sys
#    guide_calc(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), sys.argv[6])

new_guide = guide_calc(1292, 1100)
#10.0 -0.001 -0.014

new_guide = guide_calc(1292, 1100, '2017-04-20T08:10:21', 1294, 1105)
#10.0 -0.006 -0.027

new_guide = guide_calc(1294, 1105, '2017-04-20T08:10:21', 1301, 1106, last_guide=new_guide)
#10.0 -0.024 -0.030

new_guide = guide_calc(1301, 1106, '2017-04-20T08:10:21', 1309, 1108, last_guide=new_guide)
#10.0 -0.045 -0.035

new_guide = guide_calc(1309, 1108, '2017-04-20T08:10:21', 1318, 1112, last_guide=new_guide)
