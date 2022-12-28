#!/usr/bin/python3

"""Simple image previewer for `~astropy.nddata.CCDData"""

import sys

import matplotlib.pyplot as plt

from astropy.nddata import CCDData

# https://stackoverflow.com/questions/27704490/interactive-pixel-information-of-an-image-in-python
class CCDImageFormatter(object):
    """Provides the x,y,z formatting I like for CCD images in the
    interactive pyplot window"""
    def __init__(self, im):
        self.im = im
    def __call__(self, x, y):
        s = self.im.shape
        col = int(x + 0.5)
        row = int(y + 0.5)
        if col >= 0 and col < s[1] and row >= 0 and row < s[0]:
            z = self.im[row, col]
            return 'x=%1.1f, y=%1.1f, z=%1.1f' % (x, y, z)
        else:
            return 'x=%1.1f, y=%1.1f' % (x, y)        

def simple_show(im, **kwargs):
    fig, ax = plt.subplots()
    ax.imshow(im, origin='lower',
              cmap=plt.cm.gray,
              filternorm=0, interpolation='none',
              **kwargs)
    if isinstance(im, CCDData):
        ax.format_coord = CCDImageFormatter(im.data)
    else:
        ax.format_coord = CCDImageFormatter(im)
    plt.show()

if __name__ == '__main__':
    ccd = CCDData.read(sys.argv[1])
    simple_show(ccd)
