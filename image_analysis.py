#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Common image analysis using pyqtgraph
"""
#__version__ = 'v01 2017-12-20' #created
#__version__ = 'v02 2017-12-20' # fixed possible line padding in imageToArray
__version__ = 'v03 2017-12-26' #

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np
from timeit import default_timer as timer

def imageToArray(img, copy=False, transpose=True):
    """ Corrected pyqtgraph function, supporting Indexed formats.
    Convert a QImage into numpy array. The image must have format RGB32, ARGB32, or ARGB32_Premultiplied.
    By default, the image is not copied; changes made to the array will appear in the QImage as well (beware: if 
    the QImage is collected before the array, there may be trouble).
    The array will have shape (width, height, (b,g,r,a)).
    &RA: fix for Indexed8, take care of possible padding
    """
    fmt = img.format()
    ptr = img.bits()
    bpl = img.bytesPerLine() # the bpl is width + len(padding). The padding area is not used for storing anything,
    #dtype = np.ushort if fmt==img.Format_Indexed8 else np.ubyte
    dtype = np.ubyte
    #print 'bpl,dtype',bpl,dtype
    USE_PYSIDE = False
    if USE_PYSIDE:
        arr = np.frombuffer(ptr, dtype=dtype)
    else:
        ptr.setsize(img.byteCount())
        #arr = np.asarray(ptr)
        #arr = np.frombuffer(ptr, dtype=np.ubyte) # this is 30% faster than asarray
        arr = np.frombuffer(ptr, dtype=dtype)
        #print('imageToArray:'+str((fmt,img.byteCount(),arr.size,arr.itemsize)))
        #print(str(arr))
        #
        #if img.byteCount() != arr.size * arr.itemsize:
        #    # Required for Python 2.6, PyQt 4.10
        #    # If this works on all platforms, then there is no need to use np.asarray..
        #    arr = np.frombuffer(ptr, np.ubyte, img.byteCount())
    
    if fmt in (img.Format_Indexed8,24):
        #arr = arr.reshape(img.height(), img.width())
        arr = arr.reshape(img.height(), bpl)
    else:
        arr = arr.reshape(img.height(), img.width(), 4)
        #arr = arr.reshape(img.height(), bpl, 4)
    if fmt == img.Format_RGB32:
        arr[...,3] = 255
    
    if copy:
        arr = arr.copy()
        
    if transpose:
        return arr.transpose((1,0,2))
    else:
        return arr

def rgb2gray(data):
    # convert RGB to Grayscale using weighted sum
    if len(data.shape) < 3:
        return data
    else:
        r,g,b = data[:,:,0], data[:,:,1], data[:,:,2]
        return 0.2989 * r + 0.5870 * g + 0.1140 * b

# Interpret image data as row-major instead of col-major
pg.setConfigOptions(imageAxisOrder='row-major')

# parse arguments
import argparse
parser = argparse.ArgumentParser(description='''
  Common image analysis using pyqtgraph''')
parser.add_argument('-d','--dbg', action='store_true', help='turn on debugging')
parser.add_argument('-i','--iso', action='store_false', help='Disable Isocurve drawing')
parser.add_argument('-r','--roi', action='store_false', help='Disable Region Of Interest analysis')
parser.add_argument('-X','--sx', type=float, default = 1., help=
  'scale view horizontally by factor SX, use negative for mirroring')
parser.add_argument('-Y','--sy', type=float, default = 1., help=
  'scale view vertically by factor SY')    
parser.add_argument('-R','--rotate', type=float, default = 0., help=
  'rotate view by degree R')
parser.add_argument('-H','--hist', action='store_false', help='Disable histogram with contrast and isocurve contol')
parser.add_argument('file', nargs='*', default='avt23.png')

pargs = parser.parse_args()
if not pargs.hist: pargs.iso = False
if isinstance(pargs.file, list): pargs.file = pargs.file[0]
print(parser.prog+' version '+__version__)

pg.mkQApp()
win = pg.GraphicsLayoutWidget()

qGraphicsGridLayout = win.ci.layout
qGraphicsGridLayout.setColumnFixedWidth(1,150)

# setup image transformation
transform = QtGui.QTransform().scale(pargs.sx, -pargs.sy)
transform.rotate(pargs.rotate)

# A plot area (ViewBox + axes) for displaying the image
p1 = win.addPlot()

# Item for displaying image data
img = pg.ImageItem()
p1.addItem(img)

'''
# Generate image data
data = np.random.normal(size=(200, 100))
data[20:80, 20:80] += 2.
data = pg.gaussianFilter(data, (3, 3))
data += np.random.normal(size=(200, 100)) * 0.1
'''
# Get data from file
qimg = QtGui.QImage()
import collections
profilingState = collections.OrderedDict()
profilingStart = timer()
if not qimg.load(pargs.file): 
    print('ERROR loading image')
    sys.exit(1)
qimg = qimg.transformed(transform)
profilingState['load'] = timer()
if True:
    shape = qimg.width(),qimg.height()
    strides0 = qimg.bytesPerLine()
    format = qimg.format()
    fdict={4:'RGB32', 3:'Indexed8',5:'ARGB32',6:'ARGB32_Premultiplied'}
    try: fmt = fdict[format]
    except: fmt = 'fmt'+str(format)
    if pargs.dbg: print('image: '+str((fmt,shape,strides0)))
win.setWindowTitle('image:'+pargs.file+' '+fmt+str(shape))
data = imageToArray(qimg,transpose=False) #,copy=True)
profilingState['toArray'] = timer()
if pargs.dbg: print('array: '+str((data.shape,data)))

if pargs.hist:
    # Contrast/color control
    hist = pg.HistogramLUTItem()
    hist.setImageItem(img)
    hist.setLevels(data.min(), data.max())
    win.addItem(hist)
    profilingState['levels'] = timer()

if pargs.iso:
    # Isocurve drawing
    iso = pg.IsocurveItem(level=0.8, pen='g')
    iso.setParentItem(img)
    iso.setZValue(5)
    # Draggable line for setting isocurve level
    isoLine = pg.InfiniteLine(angle=0, movable=True, pen='g')
    hist.vb.addItem(isoLine)
    hist.vb.setMouseEnabled(y=False) # makes user interaction a little easier
    isoLine.setValue(0.8)
    isoLine.setZValue(1000) # bring iso line above contrast controls
    iso.setData(pg.gaussianFilter(rgb2gray(data), (2, 2)))
    
    # Callback for handling user interaction with isocurves
    def updateIsocurve():
        global isoLine, iso
        iso.setLevel(isoLine.value())
    
    # Connect callback to signal
    isoLine.sigDragged.connect(updateIsocurve)
    profilingState['iso'] = timer()

if pargs.roi:
    # Custom ROI for selecting an image region
    #roi = pg.ROI([-8, 14], [6, 5])
    h,w = data.shape[:2]
    roi = pg.ROI([w*0.25, h*0.25], [w*0.5, h*0.5])
    roi.addScaleHandle([0.5, 1], [0.5, 0.5])
    roi.addScaleHandle([0, 0.5], [0.5, 0.5])
    p1.addItem(roi)
    roi.setZValue(10)  # make sure ROI is drawn above image

    # Another plot area for displaying ROI data
    win.nextRow()
    p2 = win.addPlot(colspan=2)
    p2.setMaximumHeight(250)

    # Callback for handling user interaction in ROI
    def updatePlot():
        global img, roi, data, p2
        selected = roi.getArrayRegion(rgb2gray(data), img)
        p2.plot(selected.mean(axis=0), clear=True)#, stepMode=True)
    
    # Connect callback to signal
    roi.sigRegionChanged.connect(updatePlot)
    updatePlot()
    profilingState['roi'] = timer()

win.resize(800, 800)
win.show()

# set image
img.setImage(data)
profilingStop = timer()
profilingState['show'] = profilingStop

#if pargs.dbg:
if True:
    print('total time: '+'%0.3g' % (profilingStop-profilingStart))
    print('profile:')
    for item,value in profilingState.items():
        print(item+': '+'%0.3g' % (value - profilingStart))
        profilingStart = value

# set position and scale of image
#img.scale(0.2, 0.2)
#img.translate(-50, 0)

# zoom to fit image
#p1.autoRange()  

# enable Ctrl-C to kill application
import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)    

## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()

