The latest release of this software is available at

https://zenodo.org/badge/latestdoi/157604461

The Io Input/Output facility (IoIO) is a small-aperture (35 cm)
coronagraph constructed for the study of material ejected from
Jupiter's moon Io (the output) and entering Jupiter's magnetosphere
(the input).  A full description of the instrument and its first
results is contained in Morgenthaler, J., Rathbun, J., Schmidt, C.,
Baumgardner, J, Schneider, N. "Large Volcanic Event on Io Inferred
from Jovian Sodium Nebula Brightening."  This archive contains the
code for recording, reducing and presenting the data.


Portions of this software depend on define.py, a package maintained by
Daniel R. Morgenthaler available at
https://zenodo.org/badge/latestdoi/157608102

Brief description of files in this repository:

IoIO.py: version of code used to control telescope during 2019-2020
Jovian oppositions.  Imports precisionguide.py

precisionguide.py: start of a general package which will solve the
problem of differential flexure between boresite-mounted guide scopes
and a main scope. Provides tools for precise positioning of a target
in the FOV.

ReduceCorObs.py: code used to reduce IoIO observations.  Imports
precisionguide.py and define.py.  define.py is maintained by Daniel
R. Morgenthaler and useful for debugging

read_ap.py and read_off_Jup_ap.py: reads the CSV file created by
ReduceCorObs.py which has the individual image aperture surface
brightness values and reduction parameters.  NOTE: This code applies a
correction of a factor of ADU2R_adjust = 1.15 to the aperture sum data
output by ReduceCorObs to account for details of the calibration
proceedure not yet coded into ReduceCorObs

Na_im.py, SII_im.py, SII_Na_side-by-side.py: read and display images
for publication.  Imports ADU2R_adjust from read_ap

Na_support_table.py: generate a properly sorted CSV file containing
the individual image aperture surface brightness values and reduction
parameters.  Imports ADU2R_adjust from read_ap

ioio.notebk: software developement notebook

