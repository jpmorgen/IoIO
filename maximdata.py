#!/usr/bin/python3

# Class for controlling MaxIm DL via ActiveX/COM events

# Pattern object after ccdproc.CCDData and astropy.io.fits.HDUlist
# such that it is a container of higher-level classes (e.g. HDUlist
# contains numpy.ndarrays), not a subclass of those classes
# (e.g. HDUlist elements don't inherit np.ndarray properties/methods,
# you have to extract them from the HDUlist and then do the operations)

# DEBUGGING
import matplotlib.pyplot as plt

from astropy import log
from astropy.io import fits
import win32com.client
import numpy as np
import time

#Every day usage commands
def wait(seconds):
    time.sleep(seconds)
def AddListItem(List, Item):
    List.append(Item)
def Help():
    print("""
wait(seconds)

print('h')
wait(3)
print('i')
________

h


i
--------
AddListItem(List, Item)

h[]
AddListItem(h "hi")
print(h)
________

[ hi ]
--------
""")
def startup():
    print("These are every day usage commands.")
    Help()
def Error(text):
    raise ValueError(text)

#Change True to False for no text
if True:
    startup()

