#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Common image analysis using pyqtgraph or optionally, PyPNG or OpenCV.
- Wide range of image formats.
- 16-bit images supported (requires PyPNG or OpenCV).
- Image orientation and rotation (program options: -x, -y, -R).
- Interactive zooming, panning.
- Contrast control: displays histogram of image data with movable region 
    defining the dark/light levels (program option: -H).
- ROI and embedded plot for measuring image color intensities (program option -r).
- The centroid of the ROI is marked on the image with the red cross.
- Isocurves (program option -i).
- Interactive console with access to image data, graphics objects and shell commands.
- The centroid position, RMS and sum of distribution inside the ROI is reported in the console dock.
- Export as PNG, TIFF, JPG,..., SVG?, Matplotlib, CSV, HDF5.

The pyqtgraph is fast in most cases but it only supports 8-bits/channel. 

The OpenCV (option -e cv) is as fast as pyqtgraph, supports 16-bits/channel, 
it is a large package, not widely available. 

The PyPNG (option -e png) supports 16-bit and more per channel for PNG images, 
it is pure python can be downloaded from: https://github.com/drj11/pypng.
Slow on color images.
"""
#__version__ = 'v01 2017-12-20' #created
#__version__ = 'v02 2017-12-20' # fixed possible line padding in imageToArray
#__version__ = 'v03 2017-12-26' # profiling added
#__version__ = 'v04 2017-12-26' # opencv
#__version__ = 'r05 2017-12-27' # col-major orientation for QT to match openCV 
#__version__ = 'r06 2017-12-27' # row-major for ImageItem
#__version__ = 'r06 2017-12-27' # flipping corrected
#__version__ = 'v07 2017-12-28' # corrected the axis selection for ROI histogram
#TODO-v07# the binning of the ROI is relative to the ROI itself, for example, if one pixel is saturated then in the roi hist its value will be splitted between the neighboring bins,
# the ROI binning should be synchronized with the image pixels.

#__version__ = 'v08 2017-12-29' # pyPNG file reader option
#__version__ = 'v09 2018-01-02' # fixed blue/red color swap for -eQT
#__version__ = 'v10 2018-01-03' # interactive console added, aspect ratio locked
#__version__ = 'v11 2018-01-03' # -ecv corrected
#__version__ = 'r12 2018-01-04' # grayData, color intensities in roi plot
__version__ = 'v13 2018-01-05' # calculation of centroid, width and sum of the ROI

import sys
import numpy as np
from timeit import default_timer as timer

#''''''''''''''''''''''''''''Create craphic objects```````````````````````````
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
# Interpret image data as row-major instead of col-major
pg.setConfigOptions(imageAxisOrder='row-major')
import pyqtgraph.dockarea

pg.mkQApp()
win = QtGui.QMainWindow()
area = pg.dockarea.DockArea()
win.setCentralWidget(area)
imgItem = pg.ImageItem()

dockImage = pg.dockarea.Dock("dockImage - Image", size=(600,800))
area.addDock(dockImage, 'left')
dockImage.hideTitleBar()

# Item for displaying image data
gl = pg.GraphicsLayoutWidget()
dockImage.addWidget(gl)
plotItem = gl.addPlot()

#````````````````````````````parse arguments``````````````````````````````````
import argparse
parser = argparse.ArgumentParser(description='''
  Common image analysis using pyqtgraph''')
parser.add_argument('-d','--dbg', action='store_true', help='Turn on debugging')
parser.add_argument('-i','--iso', action='store_false', help='Disable Isocurve drawing')
parser.add_argument('-r','--roi', action='store_false', help='Disable Region Of Interest analysis')
parser.add_argument('-x','--sx', type=float, default = 1, help=
  'scale view horizontally by factor SX, use negative for mirroring')
parser.add_argument('-y','--sy', type=float, default = 1, help=
  'scale view vertically by factor SY')    
parser.add_argument('-R','--rotate', type=float, default = 0., help=
  'rotate view by degree R')
parser.add_argument('-H','--hist', action='store_false', help='Disable histogram with contrast and isocurve contol')
parser.add_argument('-e','--extract',default='qt',help='image extractor: qt for QT, cv for OpenCV, png for pyPng')
parser.add_argument('-c','--console', action='store_true', help='Enable interactive python console')
parser.add_argument('-g','--gray', action='store_true', help='Show gray image')

parser.add_argument('file', default='avt23.png', help='image file') #nargs='*', 
pargs = parser.parse_args()

if not pargs.hist: pargs.iso = False
needTrans = not (pargs.sx==1 and pargs.sy==1 and pargs.rotate==0)
if isinstance(pargs.file, list): pargs.file = pargs.file[0]
extractor = {'qt':'QT','cv':'OpenCV','png':'PyPng'} 
print(parser.prog+' using '+extractor[pargs.extract]+', version '+__version__)

#````````````````````````````Helper functions`````````````````````````````````
if pargs.extract == 'qt':
    
    def imageToArray(img, copy=False, transpose=True):
        """ Corrected pyqtgraph function, supporting Indexed formats.
        Convert a QImage into numpy array. The image must have format RGB32, ARGB32, or ARGB32_Premultiplied.
        By default, the image is not copied; changes made to the array will appear in the QImage as well (beware: if 
        the QImage is collected before the array, there may be trouble).
        The array will have shape (width, height, (b,g,r,a)).
        &RA: fix for Indexed8, take care of possible padding
        """
        nplanes = img.byteCount()/img.height()/img.width()
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

        if fmt in (img.Format_Indexed8, 24):
            arr = arr.reshape(img.height(), bpl)
        else:
            arr = arr.reshape(img.height(), img.width(),nplanes)
        #TODO: not sure if it is needed (at least for PNG files):
        #if fmt == img.Format_RGB32:
        #    arr[...,3] = 255
        
        if copy:
            arr = arr.copy()
            
        if transpose:
            return arr.transpose((1,0,2))
        else:
            return arr

def rgb2gray(data):
    import math
    # convert RGB to Grayscale using weighted sum
    if len(data.shape) < 3: # no need to convert gray arrays
        return data
    else:
        #return np.dot(data[...,:3], [0.299, 0.587, 0.114]) # 1.6 times slower: timing: 0.169
        r,g,b = data[:,:,0], data[:,:,1], data[:,:,2] 
        #return (r+g+b)/3. # not nice on basi6a16.png, timing 0.036
        #return  np.sqrt(0.299*(r**2) + 0.587*(g**2) + 0.114*(b**2)) # ugly on basi6a16.png
        return 0.2989 * r + 0.5870 * g + 0.1140 * b # Luminance (perceived), timing: 0.0959
    
def cprint(msg): 
    if pargs.dbg: print(msg)  
#,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,


# set up profiling
import collections
profilingState = collections.OrderedDict()
profilingStart = timer()
#,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
#````````````````````````````Get numpy array from file````````````````````````
windowTitle = 'image:'+pargs.file.split('/')[-1:][0]+' '
qimg = None
if pargs.extract == 'cv': # get data array using OpenCV
    import cv2 as cv # import openCV
    profilingState['init'] = timer()
    data = cv.imread(pargs.file,-1)
    height,width,nplanes = data.shape
    try:
        windowTitle += '(w,h,p):'+str((width,height,nplanes))+' of '+str(data.dtype)
        print(windowTitle)
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
        data = cv.resize(data,(0,0),fx=pargs.sx,fy=pargs.sy) #,interpolation=cv.INTER_CUBIC
        
        # rotate image
        transform = cv.getRotationMatrix2D((width/2,height/2),-pargs.rotate,1)
        height,width = data.shape[:2]
        data = cv.warpAffine(data,transform,(width,height))
        profilingState['trans'] = timer()
    # data was: height,width,depth, but setImage require width,height,depth
    #data = np.swapaxes(data,0,1)
    if nplanes == 3: data = data[...,[2,1,0]] # convert BGRA to RGBA, it is very fast.
    
elif pargs.extract == 'qt': # get data array using QT
    qimg = QtGui.QImage()
    # setup image transformation
    transform = QtGui.QTransform().scale(pargs.sx, -pargs.sy)
    transform.rotate(pargs.rotate)
    profilingState['init'] = timer()
    if not qimg.load(pargs.file): 
        print('ERROR loading image '+pargs.file)
        sys.exit(1)
    profilingState['load'] = timer()
    format = qimg.format()
    fdict={4:'RGB32', 3:'Indexed8',5:'ARGB32',6:'ARGB32_Premultiplied'}
    #print qimg.numColors(),qimg.bitPlaneCount(),qimg.byteCount(),qimg.numBytes(),qimg.depth()
    height,width = qimg.height(), qimg.width()
    nplanes = qimg.byteCount()/height/width
    try: fmt = fdict[format]
    except: fmt = 'fmt'+str(format)
    windowTitle += fmt+' w,h,p:'+str((width,height,nplanes))+' %i-bit' % (qimg.depth()/nplanes)
    print(windowTitle)
    #if needTrans:
    qimg = qimg.transformed(transform)
    profilingState['trans'] = timer()

    # Convert image to numpy array
    #data = pg.imageToArray(qimg,transpose=False) # using standard pg function
    data = imageToArray(qimg,transpose=False) # using local, overloaded function
    if nplanes == 4: data = data[...,[2,1,0,3]] # convert BGRA to RGBA, it is very fast.

elif pargs.extract == 'png':
    #print('PyPNG extractor')
    import png
    reader = png.Reader(pargs.file)
    profilingState['init'] = timer()
    width,height,pix,meta = reader.read_flat() # slow for RGB images
    nplanes = meta['planes']
    profilingState['load'] = timer()
    windowTitle +=' w,h,p:'+str((width,height,nplanes))+' %i-bit' % (meta['bitdepth'])
    print(windowTitle)
    if pargs.dbg: print('meta: '+str(meta))    
    hwp = (height,width,nplanes) if nplanes > 1 else (height,width)
    d = np.reshape(pix,hwp)
    #data = np.flipud(d)
    data = d[::-1,...] # flip first axis, height
    if pargs.dbg: print(str(data))
    
else:
    print('ERROR: extractor '+str(pargs.extract)+' is not supported')
    sys.exit(1)
#,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,    
if pargs.dbg: print('array: '+str((data.shape,data)))
profilingState['gotData'] = timer()

if pargs.roi or pargs.iso:
    grayData = rgb2gray(data)
    profilingState['rgb2gray'] = timer()
    if pargs.gray: 
        data = grayData
        nplanes = 1
else:
    grayData = data

# set image
imgItem.setImage(data)
win.setWindowTitle(windowTitle)

# Item for displaying image data
plotItem.addItem(imgItem)
plotItem.autoRange(padding=0) # remove default padding
plotItem.setAspectLocked()
profilingState['image'] = timer()

#````````````````````````````Analysis objects`````````````````````````````````
if pargs.hist:
    # Contrast/color control
    hist = pg.HistogramLUTItem(fillHistogram=False)
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
    iso.setData(pg.gaussianFilter(grayData, (2, 2)))
    
    # Callback for handling user interaction with isocurves
    def updateIsocurve():
        global isoLine, iso
        iso.setLevel(isoLine.value())
    
    # Connect callback to signal
    isoLine.sigDragged.connect(updateIsocurve)
    profilingState['iso'] = timer()

roiPlot,meansV = None,None
if pargs.roi:
    # Custom ROI for selecting an image region
    dockPlot = pg.dockarea.Dock("dockPlot", size=(1,100))
    area.addDock(dockPlot, 'bottom')
    dockPlot.hideTitleBar() # dockPlot: Hide title bar on dock Plot
    roiPlot = pg.PlotWidget()
    dockPlot.addWidget(roiPlot)

    #roi = pg.ROI([width*0.25, height*0.25], [width*0.5, height*0.5])   
    #roi.addScaleHandle([1, 1], [0, 0])
    roi = pg.RectROI([width*0.25, height*0.25], [width*0.5, height*0.5])
    
    plotItem.addItem(roi)
    roi.setZValue(10)  # make sure ROI is drawn above image
    
    #centerLabel = pg.LabelItem('+',color='r',justify='center')
    centerLabel = pg.TextItem('+',color='r',anchor=(0.5,0.5))
    plotItem.addItem(centerLabel)
    

    def distMoments(data):
        ''' calculate first and second moments of a distribution'''
        x = np.arange(data.size)
        m1 = np.sum(x*data)/np.sum(data)
        m2 = np.sqrt(np.abs(np.sum((x-m1)**2*data)/np.sum(data)))
        return m1,m2
    
    # Callback for handling user interaction in ROI
    def updatePlot():
        global imgItem, roi, data, p2, meansV
        
        #````````Calculate centroid and width
        #print(str((roi.pos(),roi.size())))
        slices = roi.getArraySlice(grayData,imgItem)[0][:2]
        roiArray = grayData[slices]
        oy,ox = slices[0].start, slices[1].start
        
        ## find center of mass of the roi
        #import scipy.ndimage as ndi
        #cm = ndi.center_of_mass(roiArray) # ROI centerMass in ROI indexes
        #cmOfRoi = [i+k for i,k in zip(cm,(oy,ox))] # ROI centerMass in Image indexes
        #cprint('ROI Center of Mass: (%0.4g,%0.4g)'%(cmOfRoi[1],cmOfRoi[0]))
        
        meansV = roiArray.mean(axis=0)
        meansH = roiArray.mean(axis=1)      
        momentsH = distMoments(meansV)
        momentsV = distMoments(meansH)
        centroid = (momentsH[0] + ox +.5, momentsV[0] + oy +.5) # .5 is for pixel center
        width = (momentsH[1]*2,momentsV[1]*2)
        cprint('Centroid: (%0.4g,%0.4g)'%centroid+', width:(%0.4g,%0.4g)'%width+', sum:%0.4g'%roiArray.sum())
        
        # draw cross at the cetroid position
        centerLabel.setPos(*centroid)
        
        # plot the ROI histograms
        if nplanes == 1 :
            roiPlot.plot(meansV,clear=True)
        else:
            # plot color intensities
            meansV = data.mean(axis=0)
            roiPlot.plot(meansV[:,0],pen='r', clear=True) # plot red
            roiPlot.plot(meansV[:,1],pen='g',) # plot green
            roiPlot.plot(meansV[:,2],pen='b',) # plot blue
    
    # Connect callback to signal
    roi.sigRegionChanged.connect(updatePlot)
    profilingState['roi init'] = timer()
    updatePlot()
    profilingState['roi update'] = timer()
    
if pargs.console:
    import pyqtgraph.console
    #````````````````````````````Bug fix in pyqtgraph 0.10.0`````````````````````
    import pickle
    class CustomConsoleWidget(pyqtgraph.console.ConsoleWidget):
        """ Fixing bugs in pyqtgraph 0.10.0:
        Need to rewrite faulty saveHistory()
        and handle exception in loadHistory() if history file is empty."""
        def loadHistory(self):
            """Return the list of previously-invoked command strings (or None)."""
            if self.historyFile is not None:
                try:
                    pickle.load(open(self.historyFile, 'rb'))
                except Exception as e:
                    print('WARNING: history file '+' not open: '+str(e))

        def saveHistory(self, history):
            """Store the list of previously-invoked command strings."""
            #TODO: no sense to provide history argument, use self.input.history instead
            if pargs.dbg: print('>saveHistory')
            if self.historyFile is not None:
                #bug#pickle.dump(open(self.historyFile, 'wb'), history)
                pickle.dump(history,open(self.historyFile, 'wb'))
    #,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,            
    #````````````````````````Add the console widget```````````````````````````
    dockConsole = pg.dockarea.Dock("dockConsole - Console", size=(1,50), closable=True)
    area.addDock(dockConsole, 'bottom')
    
    def sh(s): # console-available metod to execute shell commands
        import subprocess,os
        print subprocess.Popen(s,shell=True, stdout = None if s[-1:]=="&" else subprocess.PIPE).stdout.read()
        
    #gWidgetConsole = pg.console.ConsoleWidget(
    gWidgetConsole = CustomConsoleWidget(
        namespace={'pg':pg, 'np': np, 'plot': roiPlot, 'roi':roi, 'roiData':meansV,
          'data':data, 'image': qimg, 'imageItem':imgItem, 'pargs':pargs, 'sh':sh},
        historyFile='/tmp/%s.pcl'%parser.prog,
        text="""This is an interactive python console. The numpy and pyqtgraph modules have already been imported  as 'np' and 'pg'
The shell command can be invoked as sh('command').
Accessible local objects: 'data': image array, 'roiData': roi array, 'plot': bottom plot, 'image': QImage, 'imageItem': image object.
For example, to plot vertical projection of the roi: plot.plot(roiData.mean(axis=1), clear=True).
to swap Red/Blue colors: imageItem.setImage(data[...,[2,1,0,3]])
""")
    #print 'gWidgetConsole:',gWidgetConsole.ui.historyList
    dockConsole.addWidget(gWidgetConsole)
    
    def cprint(msg): gWidgetConsole.write('#'+msg+'\n') # use it to inform the user
  
#,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
win.resize(1200, 800)
win.show()

profilingStop = timer()
profilingState['show'] = profilingStop

#if pargs.dbg:
if True:
    print('total time: '+'%0.3g' % (profilingStop-profilingStart))
    print('getData: '+'%0.3g' % (profilingState['gotData']-profilingState['init']))
    print('profile:')
    for item,value in profilingState.items():
        print(item+': '+'%0.3g' % (value - profilingStart))
        profilingStart = value

# enable Ctrl-C to kill application
import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)    

## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()

