#!/usr/bin/python
''' Interactive Image Analyzer. It is a feature-added version of the adoimage.py.
Features:
+ Input source: file, ADO parameter (requires pyado) or EPICS PV (requires pyepics).
+ Accepts data from vectored ADO parameters, user have to define the array shape (program options: -w --width,height,bits/channel).
+ Wide range of image formats.
+ 16-bit/channel images supported (requires PyPNG or OpenCV).
+ Image orientation and rotation (program options: -o and -R).
+ Interactive zooming, panning, rotation.
+ Contrast control: Displays histogram of image data with movable region defining the dark/light levels.
+ ROI and embedded plot for measuring image values.
+ Isocurves. The isocurve level defines the threshold for spot finding.
+ Fast multi-spot finder, reports and logs centroid position and integral of most intense spots in the ROI.
+ Export as PNG,TIFF, JPG..., SVG?, Matplotlib, CSV, HDF5.
+ Interactive python console with access to image data, graphics objects and shell commands (program option: -c).
+ Configuration and reporting in the parameter dock.
+ Image references: save/retrieve image to/from a reference slots.
+ Binary operation on current image and a reference: addition, subtraction.
+ The background subtraction can be achieved by subtraction of a blurred image.

The pyqtgraph is fast in most cases but it only supports 8-bits/channel. 

The OpenCV (option -e cv) is as fast as pyqtgraph, supports 16-bits/channel, 
it is a large package, not widely available. 

The PyPNG (option -e png) supports 16-bit and more per channel for PNG images, 
it is pure python can be downloaded from: https://github.com/drj11/pypng.
The PyPNG is slow on color images.
'''
#__version__ = 'v01 2018-01-17' # adopted from adoimage.py
#__version__ = 'v02 2018-01-18' # parameter tree dockPar, menu removed
#TODO-v02: async receiver takes 50% CPU for 1 Hz 1620*1220*12bit image. Improved in v08.
#__version__ = 'v03 2018-01-18' # dockPar Reference, store, retrieve, add
#__version__ = 'v04 2018-01-18' # spotLog
#__version__ = 'v05 2018-01-21' # findSpots is 100 faster than gaussian fitting, and only 3 times slower than projectionalmoments.
#__version__ = 'v05 2018-01-21' # only findSpots() left for spot finding
#__version__ = 'v06 2018-01-22' # react to cropped image, -o option for compatibility with bicview
#__version__ = 'v07 2018-01-22' # correct scaling for reference operations
#TODO-v07 multi-peak fitting on meanV could be useful when spots are overlapping
__version__ = 'v08 2018-01-23' # UseAdo=False for PVMonitorADO consumes 2-3 times less CPU. pvMonitor.clear() called in imager.stop()

import io
import sys
import time
import struct

#````````````````````````````Stuff for profiling``````````````````````````````
from timeit import default_timer as timer
import collections
profilingState = collections.OrderedDict() # keeps processing times for diagnostics

def profile(state):
    # store the state
    #global profilingState
    profilingState[state] = timer()

def profStates(first,last):
    # returns text lines with time differences between intermediate states
    txt = ''
    l = timer()
    t = 0
    for key,value in profilingState.items():
        if key == first: 
            t = value
        elif key == last:
            break
        if t:
            d = value - t    
            txt += 'time of '+key+' :%0.3g'%(d)+'\n'
            t = value
    return txt

def profDif(first,last):
    return '%0.3g'%(profilingState[last] - profilingState[first])
#,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
    
#from PyQt4 import QtGui, QtCore
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import pyqtgraph.dockarea
import pyqtgraph.console
# Interpret image data as row-major instead of col-major
pg.setConfigOptions(imageAxisOrder='row-major')

import numpy as np
import skimage.transform as st
import scipy

# if graphics is done in callback, then we need this:
QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_X11InitThreads)

app = QtGui.QApplication([])

#necessary explicit globals
pargs = None
#imager = None

#````````````````````````````PV Monitor Objects for different access systems``
''' The Monitor object is instantiated as:
pvm = PVMonitorXXX(pvname, callback, reader = readerName)
where 
- pvname is the name of the PV
- callback is a function callback(**kwargs) , which will be called when new 
data is available.
- reader defines the image reader, the supported readers are 'file', 'qt', 'png', 'cv'
'''
#````````````````````````````Base PVMonitor class`````````````````````````````
#class PVMonitor(object): # need to be derived from object for super to work
class PVMonitor(QtCore.QThread): # inheritance from QtCore.QThread is needed for qt signals
    def __init__(self):
        super(PVMonitor,self).__init__()
        self.firstEvent = True
        self.data = None
        self.paused = False
        self.timestamp = None
    
    def convert_png_to_ndArray(self,pngReader):
        width,height,pix,meta = pngReader.read_flat() # slow for RGB images
        printd('meta: '+str(meta))    
        if self.firstEvent:
            self.firstEvent = False
            #self.bitsPerChannel = meta['bitdepth']
            p = meta['planes']
            self.hwp = [height,width,p]
        printd('hwp:'+str(self.hwp))
        return np.reshape(pix, self.hwp if self.hwp[2]>1 else self.hwp[:2])
        
    def nextImage(self):
        return None

    def monitor(self):
        '''starts a monitor on the named PV by pvmonitor().'''
        printi('pvmonitor.monitor() is not instrumented') 
        
    def clear(self):
        '''clears a monitor set on the named PV by pvmonitor().'''
        printi('pvmonitor.clear() is not instrumented') 
            
#````````````````````````````Monitor of data from a file``````````````````````
class PVMonitorFile(PVMonitor):
    def __init__(self,pvname,callback,**kwargs):
        super(PVMonitorFile,self).__init__()
        self.pvsystem = 'File' # for Accelerator Device Objects, ADO
        self.qimg = QtGui.QImage() # important to have it persistent
        reader = kwargs['reader']
        if reader == 'qt':
            if not self.qimg.load(pvname): 
                printe('Loading image '+pvname)
                sys.exit(1)
            self.data = convert_qimg_to_ndArray(self.qimg)
        elif reader == 'png':
            pngReader = png.Reader(pvname)
            self.data = self.convert_png_to_ndArray(pngReader)
        elif reader == 'raw':
            print('Raw format is not yet implemented for --access file')
        
        # inform the caller that new data is available
        callback(data=self.data)
        
#````````````````````````````Monitor of a Process Variable from ADO system````
UsePyAdo = False # False for low level cns module which is twice less CPU hungry
if not UsePyAdo:
    # use low level cns module
    import threading
    from cad import cns
    
class PVMonitorAdo(PVMonitor):
    # define signal on data arrival
    signalDataArrived = QtCore.pyqtSignal(object)
    
    def __init__(self,pvname,callback,**kwargs):
        super(PVMonitorAdo, self).__init__() # for signal/slot paradigm we need to call the parent init
        try:
            from cad import pyado
        except:
            printe('pyado module not available')
            sys.exit(1)
            
        self.pvsystem = 'ADO' # for Accelerator Device Objects, ADO
        self.qimg = QtGui.QImage() # important to have it persistent
        self.iface = pyado.useDirect()
        #self.reader = reader
        self.pvname = pvname
        self.callback = callback
        self.kwargs = kwargs
        
        # get the first event
        ado,par = pvname.split(':')
        printi('Getting first event')
        s = timer()
        try:
            r = self.iface.get(ado,par)
            blob = r[pvname]['value']
        except Exception as e:
           print('could not get '+str((pvname))+', item0:'+str(r.items()[0][0]))
           sys.exit(1)
        printi('First event received in %0.4g s'%(timer() - s))

        self.monitor()

        #self.data = np.array(blob,'u1')
        self.data = self.blobToNdArray(blob)
             
        # invoke the callback function
        self.callback(data=self.data)

    def mySlot(self,a):
        self.callback(data=self.data)
        
    def blobToNdArray(self,blob):
        data = np.array(blob,'u1')
        reader = self.kwargs['reader']
        if reader == 'qt':
            self.qimg = QtGui.QImage.fromData(data.data)
            data = convert_qimg_to_ndArray(self.qimg)
        elif reader == 'png':
            import io, array
            # TODO: it should be faster way to define the reader
            #blob = struct.pack(str(len(self.blob))+'b',*self.blob) # using struct
            pngReader = png.Reader(io.BytesIO(bytearray(data)))
            data = self.convert_png_to_ndArray(pngReader)
        elif reader == 'raw': pass
        else: printe('ungnown reader '+reader)
            
        printd('data: '+str(data.shape)+' of '+str(data.dtype)+':\n'+str(data))
        return data
        
    if UsePyAdo:
        def _asyncCallback(self,cargs):
            ''' called when new data have arrived'''
            if self.paused:
                printd('newData')
                return # Note: more efficient would be to cancelAsync()
            profile('start')
            profile('newData')
            #print 'got: ',cargs
            printd('in callback')
            if not isinstance(cargs,dict):
                printe('callback argument is not dict')
                return
            returned_dictionary = cargs
            props = returned_dictionary[self.pvname]
            printd('got props: '+str([i for i in props]))
            # check if data were delivered correctly,
            if 'timestampSeconds' not in props:
                printe('timestampSeconds not in props')
                return
            blob = props['value']
            # convert blob data to bytes
            profile('blob')
            self.data = self.blobToNdArray(blob)
            profile('unpack') # 80ms
            self.timestamp = props['timestampSeconds']+props['timestampNanoSeconds']*1e-9
            
            printd('sending signal to update_image')
            #self.signalDataArrived.emit('newData',data=self.data)
            self.signalDataArrived.emit('newData')
            printd('done')

        def monitor(self):
            '''starts a monitor on the named PV using pyado wrapper'''
            try:
                r = self.iface.getAsync(self._asyncCallback,*self.pvname.split(':'))
            except: r = None
            if not r:
                printe('in getAsync('+self.pvname+')')
                exit(1)
            # connect signal to slot
            self.signalDataArrived.connect(self.mySlot) # use mySlot because cannot connect external slot
            printi('ADO Monitor for '+self.pvname+' using pyado started')

        def clear(self):
            self.iface.cancelAsync()
            self.signalDataArrived.disconnect(self.mySlot) # use mySlot because cannot connect external slot
            printi('ADO Monitor cleared')
    else:
    # using low level cns module
        def _asyncCallback(self,*args):
            profile('start')
            profile('newData')
            #for datalist, tid, names in args: #
            datalist, tid, names = args[0] # expect only one item
            printd('cb cns:'+str((datalist[0][:20],tid,names)))
            profile('blob')
            self.data = self.blobToNdArray(datalist[0])
            profile('unpack')
            #self.timestamp = props['timestampSeconds']+props['timestampNanoSeconds']*1e-9
            
            printd('sending signal to update_image')
            #self.signalDataArrived.emit('newData',data=self.data)
            self.signalDataArrived.emit('newData')
            printd('done')
            
        def _getAsync_thread(self,aServer):
            # separate thread to wait for data
            aServer.waitfordata()
            printi('receiver_thread finished, number of threads: '+str(threading.active_count()))
            # try the more controllable way:
            #for d in aServer.newdataT():
            #    process_request_async( d )

        def monitor(self):
            '''starts a monitor on the named PV using low level cns library'''
            self.receiver = None
            #self.tid = None # by some reason we have to keep track of the tid
            adoname,p = self.pvname.split(':')
            ado = cns.CreateAdo( adoname )
            if ado is None:
                printe('cannot create '+adoname)
                sys.exit(1)
            request = [(ado, p, 'value')]
            areceiver = cns.asyncReceiver(self._asyncCallback)
            areceiver.start()
            if pargs.dbg:
                for ado, parameter, property in request:
                    print "requesting", ado.systemName, parameter + ':' + property
            status, self.tid = cns.adoGetAsync( list = ( request, areceiver ) )
            if status == 0:
                if pargs.dbg:
                    print 'status = OK', "tid =", tid
            else:
                print 'error = {0} while requesting {1} parameter from {2}'.format \
                    ( cns.getErrorString(status), request[0][1], request[0][0].systemName )
                return None
            thread = threading.Thread(target=self._getAsync_thread, args=(areceiver,))
            thread.start()            
            self.receiver = areceiver
            
            # connect signal to slot
            self.signalDataArrived.connect(self.mySlot) # use mySlot because cannot connect external slot
            printi('ADO Monitor for '+self.pvname+' using cns started')

        def clear(self):
            rc = cns.adoStopAsync(self.receiver,self.tid)
            if rc == None:
                msg = 'async server stopped: '+str(rc)
                printi(''+msg)
            else:
                try: msg = ' Tid '+ str(self.tid)+' for ado '+self.ado.genericName
                except: msg = ''
                printw('cannot adoStopAsync'+msg)
            printi('stopped async server tid:'+str(self.tid)+', number of threads: '+str(threading.active_count()))
                
#````````````````````````````Monitor of a Process Variable from EPICS system``
class PVMonitorEpics(PVMonitor):
    # define signal on data arrival
    signalDataArrived = QtCore.pyqtSignal(object)       
    def __init__(self,pvname,callback,**kwargs):
        print('PVMonitorEpics is not implemented yet')
#,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
#````````````````````````````Helper Functions`````````````````````````````````        
def printi(msg): print('info: '+msg)
    
def printw(msg): print('WARNING: '+msg)
    
def printe(msg): print('ERROR: '+msg)

def printd(msg): 
    if pargs.dbg: print('dbg: '+msg)
    
def rgb2gray(data):
    # convert RGB to Grayscale
    if len(data.shape) < 3:
        return data
    else:
        r,g,b = data[:,:,0], data[:,:,1], data[:,:,2]
        if pargs.graysum:  # using perception-based weighted sum 
            return 0.2989 * r + 0.5870 * g + 0.1140 * b
        else: # uniform sum
            return r/3 + g/3 + b/3

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
    
    if copy:
        arr = arr.copy()
        
    if transpose:
        return arr.transpose((1,0,2))
    else:
        return arr

def convert_qimg_to_ndArray(qimg):
    w,h = qimg.width(),qimg.height()
    if w == 0:
        printe('Width unknown, use -w to specify it')
        exit(-1)
    planes = qimg.byteCount()/w/h
    t = False
    if planes == 4: 
        return imageToArray(qimg,transpose=t)[...,[2,1,0,3]] # convert BGRA to RGBA
    else:
        return imageToArray(qimg,transpose=t)

def blur(a):
    return scipy.ndimage.gaussian_filter(a,(2,2)) # 10 times faster than pg.gaussianFilter

def distMoments(data):
    ''' calculate first and second moments of a distribution'''
    x = np.arange(data.size)
    m1 = np.sum(x*data)/np.sum(data)
    m2 = np.sqrt(np.abs(np.sum((x-m1)**2*data)/np.sum(data)))
    return m1,m2

gWidgetConsole = None
def cprint(msg): # print to Console
    if gWidgetConsole:
        gWidgetConsole.write('#'+msg+'\n') # use it to inform the user
    #else: print(msg)

def sh(s): # console-available metod to execute shell commands
    import subprocess,os
    print subprocess.Popen(s,shell=True, stdout = None if s[-1:]=="&" else subprocess.PIPE).stdout.read()

#````````````````````````````Spot processing stuff````````````````````````````

def centroid(data): # it is possible to speed it up
    h,w = np.shape(data)   
    x = np.arange(0,w)
    y = np.arange(0,h)

    X,Y = np.meshgrid(x,y)

    s = np.sum(data)
    cx = np.sum(X*data)/s
    cy = np.sum(Y*data)/s

    return (cx,cy)


def findSpots(region,threshold,maxSpots):
    # find up to maxSpots in the ndarray region and return its centroids and sum.

    # Set everything below the threshold to zero:
    z_thresh = np.copy(blur(region))
    profile('blurring')
    z_thresh[z_thresh<threshold] = 0
    profile('thresholding')
    
    # now find the objects
    labeled_image, number_of_objects = scipy.ndimage.label(z_thresh)
    profile('labeling')
    
    # sort the objects according to its sum
    sums = scipy.ndimage.sum(z_thresh,labeled_image,index=range(1,number_of_objects+1))
    #printd('sums:'+str(sums))
    sumsSorted = sorted(enumerate(sums),key=lambda idx: idx[1],reverse=True)
    #printd('sums:'+str(sums))
    labelsSortedBySum = [i[0] for i in sumsSorted]
    #printd(str(labelsSortedBySum))
    profile('sums')
    peak_slices = scipy.ndimage.find_objects(labeled_image)
    largestSlices = [(peak_slices[i],sums[i]) for i in labelsSortedBySum]
    profile('find spots')
    
    # calculate centroids
    centroids = []
    for peak_slice,s in largestSlices[:maxSpots]:
        dy,dx  = peak_slice
        x,y = dx.start, dy.start
        p = centroid(z_thresh[peak_slice])
        centroids.append((x+p[0],y+p[1],s))
    profile('centroids')
    return centroids

#,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,    
Console = True # needed only when the interactive console is used
if Console: 
    #````````````````````````````Bug fix in pyqtgraph 0.10.0`````````````````````
    import pickle
    class CustomConsoleWidget(pyqtgraph.console.ConsoleWidget):
        ''' Fixing bugs in pyqtgraph 0.10.0:
        Need to rewrite faulty saveHistory()
        and handle exception in loadHistory() if history file is empty.'''
        def loadHistory(self):
            '''Return the list of previously-invoked command strings (or None).'''
            if self.historyFile is not None:
                try:
                    pickle.load(open(self.historyFile, 'rb'))
                except Exception as e:
                    printw('History file '+' not open: '+str(e))

        def saveHistory(self, history):
            '''Store the list of previously-invoked command strings.'''
            #TODO: no sense to provide history argument, use self.input.history instead
            printd('>saveHistory')
            if self.historyFile is not None:
                #bug#pickle.dump(open(self.historyFile, 'wb'), history)
                pickle.dump(history,open(self.historyFile, 'wb'))
#,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
#```````````````````````````Imager````````````````````````````````````````````
class Imager(QtCore.QThread): # for signal/slot paradigm the inheritance from QtCore.QThread is necessary
    def __init__(self,pvname):
        super(Imager, self).__init__() # for signal/slot paradigm we need to call the parent init
        self.pvname = pvname
        self.qimg = QtGui.QImage()
        self.blob = None
        self.hwpb = [0]*4 # height width, number of planes, bits/channel
        self.plot = None
        self.roi = None
        self.data = None
        self.imageItem = None
        self.dockParRotate = 0
        self.degree = 0
        self.spotLog = None
        self.events = 0
        self.threshold = 0
        self.maxSpots = 4
        self.spots = []                
        profile('start')

    def show(self):
        ''' Display the widget, called once, when the first image is available'''
        self.win = QtGui.QMainWindow()
        area = pg.dockarea.DockArea()
        self.win.setCentralWidget(area)
        
        # image info
        self.winTitle = 'image:'+self.pvname+' hwpb:'+str(self.hwpb)
        self.win.setWindowTitle(self.winTitle)

        #````````````````````````````parameters data``````````````````````````
        import pyqtgraph.parametertree.parameterTypes as pTypes
        from pyqtgraph.parametertree import Parameter, ParameterTree, ParameterItem, registerParameterType
        self.numRefs = 5
        params = [
            {'name': 'Configuration', 'type': 'group', 'children': [
                {'name':'Event','type':'int','value':0,
                  'tip':'Accepted events','readonly':True},
                {'name':'Monitor', 'type': 'bool', 'value': True,
                  'tip': 'Enable/disable parameter monitoring'},
                {'name':'Rotate', 'type': 'float', 'value': 0,
                  'tip':'Rotate image by degree clockwise'},
                {'name':'Blur', 'type': 'bool', 'value': False,
                  'tip':'Blur the image using gaussian filter with sigma 2'},
                {'name':'Debug', 'type': 'bool', 'value': False},
                #{'name':'Test', 'type': 'str', 'value': 'abcd'},
                #{'name':'Debug Action', 'type': 'action'},
            ]},
            {'name':'SpotFinder', 'type':'group', 'children': [
                {'name':'MaxSpots', 'type': 'int', 'value':self.maxSpots,
                  'limits':(0,self.maxSpots),
                  'tip': 'Max number of spots to find in the ROI'},
                {'name':'Threshold', 'type': 'float', 'value':self.threshold,
                  'tip': 'Threshold level for spot finding, changed with isoCurve level'},
                {'name':'Spots', 'type':'str','value':'(0,0)',
                  'readonly': True,'tip':'X,Y and integral of found spots'},
                {'name':'SpotLog', 'type': 'bool', 'value': False,
                  'tip':'Log the spots parameters to a file'},
            ]},
            {'name':'Reference', 'type': 'group','children': [
                {'name':'Slot','type':'list','values': range(self.numRefs),
                  'tip':'Slot to store/retrieve/ reference image'},
                #{'name':'Info','type':'str','value':'','readonly': True},
                {'name':'Operation', 'type':'list','values':['','store','retrieve',
                  'add','subtract'],'tip':'Binary operation on current image and the reference'}
            ]},
        ]
        #```````````````````````````Create parameter tree`````````````````````````````
        ## Create tree of Parameter objects
        self.pgPar = Parameter.create(name='params', type='group', children=params)
        
        # Handle any changes in the parameter tree
        def handle_change(param, changes):
            global args
            printd('tree changes:')
            for param, change, itemData in changes:
                path = self.pgPar.childPath(param)
                if path is not None:
                    childName = '.'.join(path)
                else:
                    childName = param.name()
                printd('  parameter: %s'% childName)
                printd('  change:    %s'% change)
                if change == 'options': continue # do not print lengthy text
                printd('  itemData:      %s'% str(itemData))
                printd('  ----------')
            
                parGroupName,parItem = childName,''
                try: 
                    parGroupName,parItem = childName.split('.')
                except: None
                if parGroupName == 'Configuration':               
                    if parItem == 'Monitor':
                        printd('Monitor')
                        pvMonitor.paused = not itemData # pvMonitor.paused is obsolete
                        if itemData: pvMonitor.monitor()
                        else: pvMonitor.clear()
                        self.updateTitle()                       
                    elif parItem == 'Rotate':
                        self.dockParRotate = float(itemData)
                        self.rotate(self.dockParRotate)
                    elif parItem == 'Blur':
                        if itemData:
                            self.data = blur(self.data)
                        else:
                            self.data = self.receivedData
                        self.updateImageItemAndRoi()
                    elif parItem == 'Debug':
                        pargs.dbg = itemData
                        printi('Debugging is '+('en' if pargs.dbg else 'dis')+'abled')
                    if parItem == 'Debug Action':
                        printi('Debug Action pressed')
                        # add here the action to test
                        #pvMonitor.clear()
                        # crop image to ROI
                        print 'rd:',self.receivedData.shape
                        self.receivedData = self.receivedData[:600,:1000]
                        print 'rd:',self.receivedData.shape
                        self.data = self.receivedData
                        self.updateImageItemAndRoi()
                        
                if parGroupName == 'SpotFinder':               
                    if parItem == 'MaxSpots':
                        self.maxSpots = itemData
                    elif parItem == 'Threshold':
                        #print 'thr:',itemData
                        self.threshold = itemData
                    elif parItem == 'SpotLog':
                        if itemData:
                            try:
                                fn = pargs.logdir+'sl_'+self.pvname.replace(':','_')\
                                  +time.strftime('_%y%m%d%H%M.log')
                                self.spotLog = open(fn,'w',1)
                                printi('file: '+self.spotLog.name+' opened')
                            except: 
                                printe('opening '+fn)
                                self.spotLog = None
                        else:
                            try: 
                                self.spotLog.close()
                                printi('file:'+str(self.spotLog.name)+' closed')
                            except: pass
                            self.spotLog = None
                    
                elif parGroupName == 'Reference':
                    if parItem == 'Operation':
                        slot = self.pgPar.child('Reference').child('Slot').value()
                        fn = 'slot'+str(slot)+'.png'
                        printd('Operation:'+str(itemData)+' Reference '+str(slot))
                        if itemData == 'store':
                            img = self.imageItem.qimage
                            if not img.save(fn): 
                                printe('saving '+fn)
                        elif itemData == 'retrieve':
                            data = self.scaleData(self.load(fn))
                            self.data = data
                            self.updateImageItemAndRoi()
                        elif itemData == 'add':
                            data = self.scaleData(self.load(fn))
                            self.data = (data + self.data)/2
                            self.updateImageItemAndRoi()
                        elif itemData == 'subtract':
                            data = self.scaleData(self.load(fn))
                            self.data = self.data - data
                            self.updateImageItemAndRoi()
                           
        self.pgPar.sigTreeStateChanged.connect(handle_change)
           
        # Too lazy for recursion:
        '''
        def valueChanging(param, value):
            printi('Value changing (not finalized):'+str((param, value)))

        for child in self.pgPar.children():
            child.sigValueChanging.connect(valueChanging)
            for ch2 in child.children():
                ch2.sigValueChanging.connect(valueChanging)
        '''        
        def valueChanged(param, value):
            printi('Value changed:'+str((param, value)))
        
        for child in self.pgPar.children():
            child.sigValueChanged.connect(valueChanged)
            for ch2 in child.children():
                if not ch2.readonly:
                    ch2.sigValueChanged.connect(valueChanged)       

        ## Create ParameterTree widgets, both accessing the same data
        printd('Create ParameterTree widgets')
        pgParTree = ParameterTree()
        pgParTree.setParameters(self.pgPar, showTop=False)
        pgParTree.setWindowTitle('Parameter Tree')
        dockPar = pg.dockarea.Dock('dockPar', size=(50,10))
        dockPar.addWidget(pgParTree)
        area.addDock(dockPar, 'left')
        #,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,

        ## Create docks, place them into the window one at a time.
        ## Note that size arguments are only a suggestion; docks will still have to
        ## fill the entire dock area and obey the limits of their internal widgets.
        dockImage = pg.dockarea.Dock('dockImage - Image', size=(600,800))
        area.addDock(dockImage, 'left')
        dockImage.hideTitleBar()
                
        #````````````````````Add widgets into each dock```````````````````````
        # dockImage: a plot area (ViewBox + axes) for displaying the image
        gl = pg.GraphicsLayoutWidget()
        plotItem = gl.addPlot()
        dockImage.addWidget(gl)
        # Item for displaying image data
        #print 'adding imageItem:',self.imageItem.width(),self.imageItem.height()
        plotItem.addItem(self.imageItem)
        plotItem.autoRange(padding=0) # remove default padding
        plotItem.setAspectLocked()
        #,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,

        if pargs.hist:
            # Contrast/color control
            self.contrast = pg.HistogramLUTItem()
            self.contrast.setImageItem(self.imageItem)
            grayData = rgb2gray(self.data)
            self.contrast.setLevels(grayData.min(), grayData.max())
            
            #TODO dockHist#
            #hw = pg.GraphicsView()
            #hw.addItem(self.contrast)
            #dockHist.addWidget(hw)
            
            gl.addItem(self.contrast)

        if pargs.iso:
        # Isocurve drawing
            self.iso = pg.IsocurveItem(level=0.8, pen='g')
            self.iso.setParentItem(self.imageItem)
            self.iso.setZValue(5)
            # Draggable line for setting isocurve level
            self.isoLine = pg.InfiniteLine(angle=0, movable=True, pen='g')
            self.contrast.vb.addItem(self.isoLine)
            self.contrast.vb.setMouseEnabled(y=False) # makes user interaction a little easier
            self.isoLine.setValue(0.8)
            self.isoLine.setZValue(1000) # bring iso line above contrast controls
            # Connect callback to signal
            #self.isoLine.sigDragged.connect(self.updateIsocurve)
            self.isoLine.sigPositionChangeFinished.connect(self.updateIsocurve)
            self.updateIso()

        if pargs.roi:
        # Custom ROI for selecting an image region
            dockPlot = pg.dockarea.Dock('dockPlot', size=(1,100))
            area.addDock(dockPlot, 'bottom')
            dockPlot.hideTitleBar()
            self.plot = pg.PlotWidget()
            dockPlot.addWidget(self.plot)

            h,w,p,b = self.hwpb
            #self.roi = pg.ROI([w*0.25, h*0.25], [w*0.5, h*0.5])
            #self.roi.addScaleHandle([1, 1], [0, 0])
            self.roi = pg.RectROI([w*0.25, h*0.25], [w*0.5, h*0.5], sideScalers=True)
            plotItem.addItem(self.roi)
            self.roi.setZValue(10)  # make sure pargs.roi is drawn above image
            
            # create max number of spot labels
            self.spotLabels = [pg.TextItem('*',color='r',anchor=(0.5,0.5)) for i in range(self.maxSpots)]
            for sl in self.spotLabels:
                plotItem.addItem(sl)

            # Connect callback to signal
            self.roi.sigRegionChangeFinished.connect(self.updateRoi)
            self.updateRoi()

        if pargs.console:
        # interactive python console
            dockConsole = pg.dockarea.Dock('dockConsole - Console', size=(1,50), closable=True)
            area.addDock(dockConsole, 'bottom')
            ## Add the console widget
            global gWidgetConsole
            gWidgetConsole = CustomConsoleWidget(
                namespace={'pg': pg, 'np': np, 'plot': self.plot, 'roi':self.roi, #'roiData':meansV,
          'data':self.data, 'image': self.qimg, 'imageItem':self.imageItem, 'pargs':pargs, 'sh':sh, 'io':self.iface},
                historyFile='/tmp/pygpm_console.pcl',
            text="""This is an interactive python console. The numpy and pyqtgraph modules have already been imported  as 'np' and 'pg'
The shell command can be invoked as sh('command').
Accessible local objects: 'data': image array, 'roiData': roi array, 'plot': bottom plot, 'image': QImage, 'imageItem': image object.
For example, to plot vertical projection of the roi: plot.plot(roiData.mean(axis=1), clear=True).
to swap Red/Blue colors: imageItem.setImage(data[...,[2,1,0,3]])
""")
            dockConsole.addWidget(gWidgetConsole)

        self.win.resize(1200, 800) 
        self.win.show()
        
    def load(self,fn):
    # load image from file
        data = []
        img = QtGui.QImage()
        if img.load(fn):
            printd('image: '+str((img.width(),img.height())))
            data = convert_qimg_to_ndArray(img)
            printd('loaded:'+str(data))
        else: printe('loading '+fn)        
        return data
    
    def scaleData(self,data):
    # scale data to the range of self.data
        # return float array
        # if current image is gray, then take only one channel from saved PNG image 
        if len(self.data.shape) == 2:
            data = data[:,:,0] # saved files are always color PNG
        return data.astype(float) * np.max(self.data) / np.max(data)

    def rotate(self,degree):
        self.data = self.receivedData
        self.data = st.rotate(self.data, degree, preserve_range = True)
        self.updateImageItemAndRoi()
        self.updateIso()
        
    def set_dockPar(self,child,grandchild,value):
        self.pgPar.child(child).child(grandchild).setValue(value)

    def stop(self):
        pvMonitor.clear()
        printi('imager stopped')
        try: self.spotLog.close()
        except: pass

    def updateTitle(self):
        self.win.setWindowTitle(('Waiting','Paused')[pvMonitor.paused]+' '+self.winTitle)
                
    def updateRoi(self):
    # callback for handling ROI
        profile('init roi')
        # the following is much faster than getArrayRegion
        slices = self.roi.getArraySlice(self.grayData,self.imageItem)[0][:2]
        roiArray = self.grayData[slices]
        oy,ox = slices[0].start, slices[1].start
        profile('roiArray')
        
        # find spots using isoLevel as a threshold
        if self.threshold>1 and self.maxSpots>0:
            self.spots = findSpots(roiArray,self.threshold,self.maxSpots)
            profile('findSpots')
            if pargs.profile:
                print(profStates('roiArray','findSpots'))
                print('FindSpot time: '+profDif('roiArray','findSpots'))
            #self.spots = [(x+ox+0.5,y+oy+0.5,s) for x,y,s in self.spots]
            self.spots = [(x+ox,y+oy,s) for x,y,s in self.spots]
            msg = ''
            for i,spot in enumerate(self.spots):
                #print 'l%i:(%0.4g,%0.4g)'%(i,spot[0],spot[1])
                self.spotLabels[i].setPos(spot[0],spot[1])
                msg += '%0.4g,%0.4g,%0.4g,,'%spot
            for j in range(i+1,len(self.spotLabels)):
                #print 'reset spot ',j
                self.spotLabels[j].setPos(0,0)
            printd('findSpots: '+msg)
            self.set_dockPar('SpotFinder','Spots',msg)
            if self.spotLog: spotLogTxt = time.strftime('%y-%m-%d %H:%M:%S, ')\
              +msg
              
        if self.spotLog: 
            self.spotLog.write(spotLogTxt+'\n')
        
        # plot the ROI histograms
        meansV = self.data[slices].mean(axis=0) # vertical means
        #x = range(len(meansV)+1); s = True
        x = range(len(meansV)); s = False
        if self.hwpb[2] == 1: # gray image
            self.plot.plot(x,meansV,clear=True,stepMode=s)
        else: # color image
            # plot color intensities
            self.plot.plot(x,meansV[:,0],pen='r', clear=True,stepMode=s) # plot red
            self.plot.plot(x,meansV[:,1],pen='g',stepMode=s) # plot green
            self.plot.plot(x,meansV[:,2],pen='b',stepMode=s) # plot blue
            meansVG = self.grayData[slices].mean(axis=0)
            self.plot.plot(x,meansVG,pen='w',stepMode=s) # plot white
        profile('roiPlot')
        
    def updateIsocurve(self):
    # callback for handling ISO
        printd('>uIso')
        #profile('init iso')
        self.threshold = self.isoLine.value()
        #printi('isolevel:'+str(self.threshold))
        self.iso.setLevel(self.threshold)
        #profile('iso')
        self.set_dockPar('SpotFinder','Threshold',self.threshold)
         
    def updateIso(self):
    # build isocurves from smoothed data
        self.iso.setData(blur(self.grayData))
        self.updateIsocurve()

    def updateImageItemAndRoi(self):
        self.imageItem.setImage(self.data)
        if pargs.roi or pargs.iso:
            self.grayData = rgb2gray(self.data)
        else: self.grayData = None
        self.contrast.regionChanged() # update contrast
        if pargs.roi: 
            self.updateRoi()

    def update_image(self, **kargs):
        global profilingState
        profile('image')
        printd('uimage:'+str(kargs))
        data = kargs['data']

        if pargs.width:
            #``````````The source is vector parameter with user-supplied shape
            l = len(data)
            if self.imageItem == None: # do it once
                tokens = pargs.width.split(',')
                w = int(tokens[0])
                h = int(tokens[1])
                try:    bitsPerChannel = int(tokens[2])
                except: bitsPerChannel = 8
                bytesPerChannel = (bitsPerChannel-1)/8 + 1
                self.hwpb = [h, w, l/w/h/bytesPerChannel,bytesPerChannel]
                printd('de-vectoring:'+str((l, self.hwpb)))
            
            if self.hwpb[3] > 1: # we need to merge pairs of bytes to integers
                #data = np.array(struct.unpack('<'+str(l/self.hwpb[3] )+'H', data.data),'u2')
                data = struct.unpack('<'+str(l/self.hwpb[3] )+'H', data.data) 
                #data = struct.unpack('<'+str(l/self.hwpb[3] )+'H', bytearray(data)) 
                profile('merge')
            shape = (self.hwpb[:3] if self.hwpb[2]>1 else self.hwpb[:2])
            printd('shape:'+str(shape))
            data = np.reshape(data,shape)
        #,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
        #````````````````````Got numpy array from data````````````````````````
        data = data[::-1,...] # flip first axis, the height            
        printd('array: '+str((data.shape,data.dtype,data)))
        profile('array')

        degree = pargs.rotate + self.dockParRotate  
        if degree != self.degree:
            self.degree = degree
            data = st.rotate(data, degree, preserve_range = True)            
            printd('first rotation:'+str(pargs.rotate)+' data:'+str(data))
            
        if pargs.flip:
            if   pargs.flip == 'V': data = data = data[::-1,...]
            elif pargs.flip == 'H': data = data = data[:,::-1,...]
                    
        if pargs.gray: 
            self.data = rgb2gray(data)
        else:
            self.data = data
        #````````````````````Data array is ready for analysys`````````````````
        self.receivedData = self.data # store received data
        h,w = self.data.shape[:2]
        if self.imageItem == None: # first event, do the show() only once
            if self.hwpb[0] == 0:
                try: p = self.data.shape[2]
                except: p = 1
                b = self.data.dtype.itemsize
                self.hwpb = [h,w,p,b]
            printd('hwpb:'+str(self.hwpb))
            printd('self.array: '+str((self.data.shape,self.data)))
            self.imageItem = pg.ImageItem(self.data)
            if pargs.roi or pargs.iso:
                self.grayData = rgb2gray(self.data)
            else: self.grayData = None
            self.show()
        else:  # udate data
            if [h,w] != self.hwpb[:2]: # shape changed, change self.hwpb[:2]
               self.hwpb[:2] = h,w
               printi('data shape changed '+str(self.hwpb))
            self.updateImageItemAndRoi()
        self.events += 1
        self.set_dockPar('Configuration','Event',self.events) # concern: time=0.5ms
        #,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,            
        if pargs.profile:
            print('### Total time: '+'%0.3g' % (timer()-profilingState['start']))
            print(profStates('start','finish'))
        
#````````````````````````````Main Program`````````````````````````````````````
pvMonitor = None
#png = None
def main():
    global pargs, imager, pvMonitor, png
    # parse program arguments
    import argparse
    parser = argparse.ArgumentParser(description='''
      Interactive analysis of images from ADO, file (or EPICS).''')
    parser.add_argument('-d','--dbg', action='store_true', help='turn on debugging')
    parser.add_argument('-R','--rotate', type=float, default=0, help='Rotate image by ROT degree')
    parser.add_argument('-F','--flip', help="flip image, 'V':vertically, 'H':horizontally")
    parser.add_argument('-o','--o',type=int, default=0, help='''
To confirm with RHIC camera orientation convention, it specifies the order in which pixels are drawn, to adjust for camera mounting orientation, mirrors, etc.
        0 = As captured
        1 = Flip Vertical
        2 = Flip Horizontal
        3 = Flip Vertical & Horizontal
        4 = Rotate 90 deg clockwise
        5 = Rotate & Flip Vertical
        6 = Rotate & Flip Horizontal
        7 = Rotate, Flip Horizontal & Vertical''')
    parser.add_argument('-i','--iso', action='store_false', help='Disable Isocurve drawing')
    parser.add_argument('-r','--roi', action='store_false', help='Disable Region Of Interest analysis')
    #parser.add_argument('-f','--file', help='Process input file instead of ADO')
    parser.add_argument('-c','--console', action='store_true', help='Enable interactive python console')
    parser.add_argument('-H','--hist', action='store_false', help='Disable histogram with contrast and isocurve contol')
    parser.add_argument('-w','--width', help='For blob data: width,height,bits/channel i.e 1620,1220,12. The bits/channel is needed only if it is > 8')
    parser.add_argument('-p','--profile', action='store_true', help='Enable code profiling')
    parser.add_argument('-e','--extract',default='qt',help='image extractor: qt for QT (default), png for pyPng (for 16-bit+ images)') #cv for OpenCV, 
    parser.add_argument('-g','--gray', action='store_true', help='Show gray image')
    #parser.add_argument('-s','--spot', action='store_true', help='Enable Spot Finder to estimate spot and background parameters inside the ROI. It could be slow on some images')
    parser.add_argument('-G','--graysum', action='store_true', help='Use summed color-to-gray conversion, rather than perceptional')
    parser.add_argument('-a','--access', default = 'ado', help='pv access system: file/ado/epics') 
    parser.add_argument('-l','--logdir', default = '/operations/app_store/imageAnalyzer/logs/', help='pv access system: file/ado/epics') 
    #parser.add_argument('pname',default='ebic.avt29:gmImageM',
    #  help=''' ADOName ParameterName i.e: ebic.avt29:gmImageM''')
    parser.add_argument('pname', nargs='*', 
      default=['ebic.avt29','gmImageM'],
      help=''' ADOName ParameterName i.e: ebic.avt29:gmImageM or ebic.avt29 gmImageM''')

    pargs = parser.parse_args()
    extractor = {'qt':'QT','cv':'OpenCV','png':'PyPng','raw':'raw'}
    if pargs.width: # if width is provided, the pargs.extract should be set to 'raw'
        pargs.extract = 'raw'
    if len(pargs.pname) == 2:
        pname = pargs.pname[0]+':'+pargs.pname[1]
    else:
        pname = pargs.pname[0]
        
    print(parser.prog+' '+pname+' using '+extractor[pargs.extract]+', version '+__version__)
    if not pargs.hist: pargs.iso = False
                
    if pargs.extract == 'png':
        import png
    elif pargs.extract == 'cv':
        import cv

    # convert -o option to -R and -F combination
    odict = {0:(pargs.rotate,pargs.flip), 1:(0,'V'), 2:(0,'H'), 3:(180,'N'),
      4:(90,'N'), 5:(90,'V'), 6:(90,'H'), 7:(270,'N')}
    pargs.rotate, pargs.flip = odict[pargs.o]

    # instantiate the imager
    imager = Imager(pname)

    # instantiate the data monitor
    if pargs.access == 'file':
        pvMonitor = PVMonitorFile(pname,imager.update_image,
                                reader=pargs.extract)
    elif pargs.access == 'ado':
        pvMonitor = PVMonitorAdo(pname,imager.update_image,
                                reader=pargs.extract)
    elif pargs.access == 'epics':
        pvMonitor = PVMonitorEpics(pname,imager.update_image,
                                reader=pargs.extract)
    #print 'pvMonitor:',pvMonitor        
#,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
if __name__ == '__main__':

    # enable Ctrl-C to kill application
    import signal
    signal.signal(signal.SIGINT, signal.SIG_DFL)    
    
    main()

    try:
        # Start Qt event loop unless running in interactive mode or using pyside
        #print('starting QtGUI'+('.' if pargs.file else ' Waiting for data...'))
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QtGui.QApplication.instance().exec_()
    except KeyboardInterrupt:
        print('keyboard interrupt: exiting')
    print('Done')
    imager.stop()




