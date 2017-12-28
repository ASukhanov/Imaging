#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Common image analysis using pyqtgraph and optionally, OpenCV.
The performance of pyqtgraph and OpenCV is similar.
OpenCV supports more image formats, including 16-bit/channel.
"""
#__version__ = 'v01 2017-12-20' #created
#__version__ = 'v02 2017-12-20' # fixed possible line padding in imageToArray
#__version__ = 'v03 2017-12-26' # profiling added
#__version__ = 'v04 2017-12-26' # opencv
#__version__ = 'r05 2017-12-27' # col-major orientation for QT to match openCV 
#__version__ = 'r06 2017-12-27' # row-major for ImageItem
''' Comparison of analysis of avt29.png (ARGB 1620x1220) 
on acnlinec and laptop Dell Latitude E6420, both with pyqtgraph 0.10.0:
+-----------------+----------+
|    acnlinec     |  laptop  |
+-----------------+----------+
 total time: 1.8    0.9
 profile:
 load: 0.227        0.075
 trans: 2.1e-05     2.1e-5
 toArray: 0.00132   0.00103
 image: 0.215       0.164
 levels: 0.0756     0.0437
 iso: 0.4           0.384
 roi: 0.862         0.226
 show: 0.0221       0.0065
------------------+----------+
'''
__version__ = 'r06 2017-12-27' # flipping corrected

import sys
import numpy as np
from timeit import default_timer as timer
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
# Interpret image data as row-major instead of col-majorwin = QtGui.QMainWindow()
#pg.setConfigOptions(imageAxisOrder='row-major')
import pyqtgraph.dockarea

#````````````````````````````Helper functions`````````````````````````````````
# not needed for OpenCV
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
    dtype = np.ubyte
    USE_PYSIDE = False
    if USE_PYSIDE:
        arr = np.frombuffer(ptr, dtype=dtype)
    else:
        ptr.setsize(img.byteCount())
        #arr = np.asarray(ptr)
        arr = np.frombuffer(ptr, dtype=dtype) # this is 30% faster than asarray
        #print('imageToArray:'+str((fmt,img.byteCount(),arr.size,arr.itemsize)))
        #print(str(arr))
        #
        #if img.byteCount() != arr.size * arr.itemsize:
        #    # Required for Python 2.6, PyQt 4.10
        #    # If this works on all platforms, then there is no need to use np.asarray..
        #    arr = np.frombuffer(ptr, np.ubyte, img.byteCount())

    if fmt in (img.Format_Indexed8,24):
        arr = arr.reshape(img.height(), bpl)
    else:
        arr = arr.reshape(img.height(), img.width(), 4)
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
#,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
#````````````````````````````parse arguments``````````````````````````````````
import argparse
parser = argparse.ArgumentParser(description='''
  Common image analysis using pyqtgraph''')
parser.add_argument('-d','--dbg', action='store_true', help='turn on debugging')
parser.add_argument('-i','--iso', action='store_false', help='Disable Isocurve drawing')
parser.add_argument('-r','--roi', action='store_false', help='Disable Region Of Interest analysis')
parser.add_argument('-x','--sx', type=float, default = 1, help=
  'scale view horizontally by factor SX, use negative for mirroring')
parser.add_argument('-y','--sy', type=float, default = 1, help=
  'scale view vertically by factor SY')    
parser.add_argument('-R','--rotate', type=float, default = 0., help=
  'rotate view by degree R')
parser.add_argument('-H','--hist', action='store_false', help='Disable histogram with contrast and isocurve contol')
parser.add_argument('-v','--cv', action='store_true', help='Use openCV')
parser.add_argument('file', nargs='*', default='avt23.png')
pargs = parser.parse_args()

if not pargs.hist: pargs.iso = False
needTrans = not (pargs.sx==1 and pargs.sy==1 and pargs.rotate==0)
if isinstance(pargs.file, list): pargs.file = pargs.file[0]
print(parser.prog+' using '+('openCV' if pargs.cv else 'pyqtgraph')+', version '+__version__)

if not pargs.cv:
    # setup image transformation
    transform = QtGui.QTransform().scale(pargs.sx, -pargs.sy)
    transform.rotate(pargs.rotate)

# set up profiling
import collections
profilingState = collections.OrderedDict()
profilingStart = timer()
#,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
#````````````````````````````Get data from file```````````````````````````````
windowTitle = 'image:'+pargs.file.split('/')[-1:][0]+' '

if pargs.cv: # get data array using OpenCV
    import cv2 as cv # import openCV
    data = cv.imread(pargs.file,-1)
    try:
        windowTitle += '(h,w,d):'+str(data.shape)+' of '+str(data.dtype)
        if pargs.dbg: print(windowTitle)
    except Exception as e:
        print('ERROR loading image '+pargs.file+str(e))
        sys.exit(1)
    profilingState['load'] = timer()
    
    data = cv.flip(data,0)
    if needTrans:
        flip = None
        # flip image
        if pargs.sx < 0: flip = 0; pargs.sx = -pargs.sx
        if pargs.sy < 0: flip = 1; pargs.sy = -pargs.sy
        if flip != None: data = cv.flip(data,flip)
        height,width = data.shape[:2]

        # scale image
        #data = cv.resize(data,(width*int(pargs.sx),height*int(pargs.sy))) #,interpolation=cv.INTER_CUBIC
        data = cv.resize(data,(0,0),fx=pargs.sx,fy=pargs.sy) #,interpolation=cv.INTER_CUBIC
        
        # rotate image
        transform = cv.getRotationMatrix2D((width/2,height/2),-pargs.rotate,1)
        height,width = data.shape[:2]
        data = cv.warpAffine(data,transform,(width,height))
        profilingState['trans'] = timer()
    
else: # get data array using QT
    qimg = QtGui.QImage()
    if not qimg.load(pargs.file): 
        print('ERROR loading image '+pargs.file)
        sys.exit(1)
    profilingState['load'] = timer()
    #if needTrans:
    qimg = qimg.transformed(transform)
    profilingState['trans'] = timer()

    # Convert image to numpy array
    shape = qimg.height(),qimg.width(),
    strides0 = qimg.bytesPerLine()
    format = qimg.format()
    fdict={4:'RGB32', 3:'Indexed8',5:'ARGB32',6:'ARGB32_Premultiplied'}
    try: fmt = fdict[format]
    except: fmt = 'fmt'+str(format)
    data = imageToArray(qimg,transpose=False) #,copy=True)
    windowTitle += fmt+' h,w,d:'+str(data.shape)
    if pargs.dbg: print(windowTitle)
    profilingState['toArray'] = timer()
#,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,    
if pargs.dbg: print('array: '+str((data.shape,data)))

# data was: height,width,depth, but setImage require width,height,depth
data = np.swapaxes(data,0,1)

# set image
imgItem = pg.ImageItem()
imgItem.setImage(data)
#''''''''''''''''''''''''''''Create craphic objects```````````````````````````
pg.mkQApp()
#win = pg.GraphicsLayoutWidget()
#qGraphicsGridLayout = win.ci.layout
#qGraphicsGridLayout.setColumnFixedWidth(1,150)
win = QtGui.QMainWindow()
area = pg.dockarea.DockArea()
win.setCentralWidget(area)
win.setWindowTitle(windowTitle)

dockImage = pg.dockarea.Dock("dockImage - Image", size=(600,800))
area.addDock(dockImage, 'left')
dockImage.hideTitleBar()

# Item for displaying image data
gl = pg.GraphicsLayoutWidget()
dockImage.addWidget(gl)
plottedImage = gl.addPlot()
# Item for displaying image data
plottedImage.addItem(imgItem)
profilingState['image'] = timer()

#````````````````````````````Analysis objects`````````````````````````````````
if pargs.hist:
    # Contrast/color control
    hist = pg.HistogramLUTItem()
    hist.setImageItem(imgItem)
    hist.setLevels(data.min(), data.max())
    #win.addItem(hist)
    gl.addItem(hist)
    profilingState['levels'] = timer()

if pargs.iso:
    # Isocurve drawing
    iso = pg.IsocurveItem(level=0.8, pen='g')
    iso.setParentItem(imgItem)
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
    dockPlot = pg.dockarea.Dock("dockPlot", size=(1,100))
    area.addDock(dockPlot, 'bottom')
    dockPlot.hideTitleBar() # dockPlot: Hide title bar on dock Plot
    roiPlot = pg.PlotWidget()
    dockPlot.addWidget(roiPlot)
    w,h = data.shape[:2]
    roi = pg.ROI([w*0.25, h*0.25], [w*0.5, h*0.5])
    roi.addScaleHandle([1, 1], [0, 0])
    plottedImage.addItem(roi)
    roi.setZValue(10)  # make sure ROI is drawn above image

    # Callback for handling user interaction in ROI
    def updatePlot():
        global imgItem, roi, data, p2
        selected = roi.getArrayRegion(rgb2gray(data), imgItem)
        roiPlot.plot(selected.mean(axis=0), clear=True)#, stepMode=True)
    
    # Connect callback to signal
    roi.sigRegionChanged.connect(updatePlot)
    profilingState['roi init'] = timer()
    updatePlot()
    profilingState['roi update'] = timer()
#,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
win.resize(1200, 800)
win.show()

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
#imgItem.scale(0.2, 0.2)
#imgItem.translate(-50, 0)

# zoom to fit image
#plottedImage.autoRange()  

# enable Ctrl-C to kill application
import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)    

## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()

