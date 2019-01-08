The Io Input/Output facility (IoIO) is a small-aperture (35 cm)
coronagraph constructed for the study of material ejected from
Jupiter's moon Io (the output) and entering Jupiter's magnetosphere
(the input).  A full description of the instrument and its first
results is contained in Morgenthaler, J., Rathbun, J., Schmidt, C.,
Baumgardner, J, Schneider, N. "Large Volcanic Event on Io Inferred
from Jovian Sodium Nebula Brightening."  This archive contains the
code for recording, reducing and presenting the data.


Portions of this software depend on define.py, a package maintained by
Daniel R. Morgenthaler available at https://github.com/jpmorgen/Daniel

Brief description of files in this repository:

ioio.py: version of code used to control telescope during 2018 Jovian
opposition.  No new development is expected since precisionguide.py
and IoIO.py combine to take on its functions

ReduceCorObs.py: code used to reduce IoIO observations.  Imports
precisionguide.py and define.py, a package maintained by Daniel
R. Morgenthaler useful for debugging

read_ap.py: reads the CSV file created by ReduceCorObs.py which has
the individual image aperture surface brightness values and reduction
parameters.  NOTE: This code applies a correction of a factor of
ADU2R_adjust = 1.15 to the aperture sum data output by ReduceCorObs to
account for details of the calibration proceedure not yet coded into
ReduceCorObs

Na_im.py: read and display two images side-by-side for publication.
Imports ADU2R_adjust from read_ap

Na_support_table.py: generate a properly sorted CSV file containing
the individual image aperture surface brightness values and reduction
parameters.  Imports ADU2R_adjust from read_ap

precisionguide.py: start of a general package which will solve the
problem of differential flexure between boresite-mounted guide scopes
and amain scope.  Based on ioio.py, imported by ReduceCorObs.py,
imports ASCOM_namespace.py

ASCOM_namespace.py: I have not figured out how to get Python to link
into the same system that Visual Basic and related compilers use to
make the ASCOM namespace available to client programs, so I just
define them myself here.

IoIO.py: new version of ioio.py which will eventually control the
instrument during data acquisition.  Imports precisionguide.py.

ioio.notebk: software developement notebook

