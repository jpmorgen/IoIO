"""Create smoothed version of APCC horizon for use in ACP, since
1-degree diagonal steps that APCC horizon editor produces are not
smooth enough for ACP #WAITINLIMITS logic

"""


import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

in_name = '/data/io/IoIO/IoIO1U1_backup/ProgramData/Astro-Physics/APCC/IoIO_5ft_pier_rings_X_corrected.hrz.hor'
with open(in_name, 'r') as f:
    data = f.read()
print(data)
dlist = data.split(' ')
print(dlist)
dflist = [float(d) for d in dlist]
print(dflist)
da = np.asarray(dflist)
print(da)
ds = savgol_filter(da, 9, 3)
print(ds)
ds[0] = 29.5
ds[-1] = 26.0

dssl = [f'{d:0.1f}' for d in ds]
print(dssl)
dss = ' '.join(dssl)
print(dss)

out_name = '/data/io/IoIO/IoIO1U1_backup/ProgramData/Astro-Physics/APCC/IoIO_5ft_pier_rings_X_corrected_smoothed.hor'
with open(out_name, 'w') as f:
    data = f.write(dss)


#plt.plot(da)
#plt.plot(ds)
#plt.show()

