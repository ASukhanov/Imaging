#!/usr/bin/python
''' Interactive Image Analyzer of streamed images.
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
+ Continuous background subtraction.

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
#__version__ = 'v08 2018-01-23' # UseAdo=False for PVMonitorADO consumes 2-3 times less CPU. pvMonitor.clear() called in imager.stop()
#__version__ = 'v09 2018-01-24' # pyado.py interface removed in favor of low level cns.py, it is 2-3 times faster, fixing minor bag with reading raw image
__version__ = 'v10 2018-02-11' # Release.

import io
import sys
import time
import struct
import threading
import subprocess
import os

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
from scipy import ndimage

# if graphics is done in callback, then we need this:
QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_X11InitThreads)

app = QtGui.QApplication([])

#necessary explicit globals
pargs = None
#imager = None
MaxSpotLabels = 16
#````````````````````````````Helper Functions`````````````````````````````````        
def printi(msg): print('info: '+msg)
    
def printw(msg): print('WARNING: '+msg)
    
def printe(msg): print('ERROR: '+msg)

def printd(msg): 
    if pargs.dbg: print('dbg: '+msg)

gWidgetConsole = None
def cprint(msg): # print to Console
    if gWidgetConsole:
        gWidgetConsole.write('#'+msg+'\n') # use it to inform the user
    #print(msg)

def cprinte(msg): # print to Console
    if gWidgetConsole:
        gWidgetConsole.write('#ERROR: '+msg+'\n') # use it to inform the user
    printe(msg)

def cprintw(msg): # print to Console
    if gWidgetConsole:
        gWidgetConsole.write('#WARNING: '+msg+'\n') # use it to inform the user
    printw(msg)

def rgb2gray(data):
    # convert RGB to Grayscale
    if len(data.shape) < 3:
        return data
    else:
        r,g,b = data[:,:,0], data[:,:,1], data[:,:,2]
        if pargs.graysum: # uniform sum
            return r/3 + g/3 + b/3
        else: # using perception-based weighted sum 
            return 0.2989 * r + 0.5870 * g + 0.1140 * b

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
        exit(5)
    planes = qimg.byteCount()/w/h
    t = False
    if planes == 4: 
        return imageToArray(qimg,transpose=t)[...,[2,1,0,3]] # convert BGRA to RGBA
    else:
        return imageToArray(qimg,transpose=t)

def rotate(data,degree):
    import math    
    if pargs.flip: degree = -degree
    degree += pargs.rotate
    fracp,n = math.modf(degree/90)
    #s = timer()
    if fracp == 0:
        data = np.rot90(data,n) # fast rotate by integral of 90 degree
    else: 
        #if degree: data = st.rotate(data, degree, resize = True, preserve_range = True)
        if degree: data = ndimage.rotate(data, degree, reshape = True, order=1)
    #printi('rotate time:'+str(timer()-s))
    if pargs.flip:
        if   pargs.flip == 'V': return data[::-1,...]
        elif pargs.flip == 'H': return data[:,::-1,...]
    return data

def blur(a):
    if len(a.shape) == 2:
        return ndimage.gaussian_filter(a,(2,2)) # 10 times faster than pg.gaussianFilter
    else:
        cprintw('blurring of color images is not implemented yet')
        return a       
    
def sh(s): # console-available metod to execute shell commands
    print subprocess.Popen(s,shell=True, stdout = None if s[-1:]=="&" else subprocess.PIPE).stdout.read()

#````````````````````````````Spot processing stuff````````````````````````````

def centroid(data): # the fastest.
    centroid =  np.array([0.,0.])
    for axis in (0,1):
        i = np.arange(data.shape[axis],dtype=float)#.astype(float)
        oppositeAxis = int(not axis)
        projection = data.sum(axis = oppositeAxis).astype(float)
        centroid[oppositeAxis] = np.dot(projection/projection.sum(),i)
    return centroid

def findSpots(region,threshold,maxSpots):
    # find up to maxSpots in the ndarray region and return its centroids and sum.
    profile('startFind')
    # Set everything below the threshold to zero:
    z_thresh = np.copy(blur(region))
    profile('blurring')
    z_thresh[z_thresh<threshold] = 0
    profile('thresholding')
    
    # now find the objects
    labeled_image, number_of_objects = ndimage.label(z_thresh)
    profile('labeling')
    
    # sort the objects according to its sum
    sums = ndimage.sum(z_thresh,labeled_image,index=range(1,number_of_objects+1))
    #printd('sums:'+str(sums))
    sumsSorted = sorted(enumerate(sums),key=lambda idx: idx[1],reverse=True)
    #printd('sums:'+str(sums))
    labelsSortedBySum = [i[0] for i in sumsSorted]
    #printd(str(labelsSortedBySum))
    profile('sums')
    peak_slices = ndimage.find_objects(labeled_image)
    largestSlices = [(peak_slices[i],sums[i]) for i in labelsSortedBySum]
    profile('find')
    
    # calculate centroids
    centroids = []
    for peak_slice,s in largestSlices[:maxSpots]:
        dy,dx  = peak_slice
        x,y = dx.start, dy.start
        #reg = z_thresh[peak_slice]
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
#,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
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
        
    def monitor(self):
        '''starts a monitor on the named PV by pvmonitor().'''
        printi('pvmonitor.monitor() is not instrumented') 
        
    def clear(self):
        '''clears a monitor set on the named PV by pvmonitor().'''
        printi('pvmonitor.clear() is not instrumented') 
            
    def getTimeStamp(self):
        '''returns timestamp, used for polling data delivery'''
        printi('pvmonitor.getTimeStamp() is not instrumented')
        
    def getData(self):
        '''returns the image ndarray, used for polling data delivery'''
        printi('pvmonitor.getData() is not instrumented')
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
        self.timestamp = time.time()
        
        # inform the caller that new data is available
        #print '>file cb:',self.data.shape
        #callback(data=self.data)

    def getTimeStamp(self):
        return self.timestamp
        
    def getData(self):
        return self.data
        
#````````````````````````````Monitor of a Process Variable from ADO system````
    
class PVMonitorAdo(PVMonitor):
    # define signal on data arrival
    signalDataArrived = QtCore.pyqtSignal(object)
    
    def __init__(self,pvname,callback,**kwargs):
        super(PVMonitorAdo, self).__init__() # for signal/slot paradigm we need to call the parent init
            
        self.pvsystem = 'ADO' # for Accelerator Device Objects, ADO
        self.qimg = QtGui.QImage() # important to have it persistent
        #self.reader = reader
        self.pvname = pvname
        self.callback = callback
        self.kwargs = kwargs
        
        # create handle to ADO
        try: adoName,par = pvname.split(':')
        except:
            printe("Not valid 'ADO:Parameter': "+str(pvname))
            sys.exit(1)
        if self.handle is None:
            printe('cannot create '+adoName)
            sys.exit(1)

        # check if parameter exists
        metaData = cns.adoMetaData(self.handle)
        if not isinstance(metaData, dict):
            printe("while trying to get metadata"+str(metaData))
            sys.exit(2)
        if (par,'value') not in metaData:
            printe('ado '+adoName+' does not have '+self.par)
            sys.exit(3)
        self.pvname = adoName+':'+self.par

        # store the requests for future use
        self.dataRequest = [(self.handle, par, 'value'),]
          #(self.handle,self.par,'timestampSeconds'),
          #(self.handle,self.par,'timestampNanoSeconds')]

        # check if parameter has timestamp property
        ts = int(str(cns.adoGet(self.handle,self.par,'timestampSeconds')[0][0][0]))
        p = 'dataIdM' if ts == 0 else self.par 
        self.tsRequest = [(self.handle,p,'timestampSeconds'),
          (self.handle,p,'timestampNanoSeconds')]

        profile('start')

        '''
        # get the first event
        printi('Getting first event')
        s = timer()

        where = cns.cnslookup(ado)
        if where == None:
            printe('no such name:'+ado)
            sys.exit(2)
        handle = cns.adoName.create( where )
        blob, status = cns.adoGet( handle, par, 'value' )
        if status != 0: 
            printe(' ADO '+ado+' does not have '+par+'.value')
            sys.exit(3)
        blob = blob[0]
        printd('blob['+str(len(blob))+str(blob[:20]))
        printi('First event received in %0.4g s'%(timer() - s))

        #self.data = np.array(blob,'u1')
        self.data = self.blobToNdArray(blob)
             
        # process the event using monitor's callback
        self.callback(data=self.data)
        
        # start parameter monitoring
        self.monitor()
        '''

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
            pngReader = png.Reader(io.BytesIO(bytearray(data)))
            data = self.convert_png_to_ndArray(pngReader)
        elif reader == 'raw': pass
        else: printe('ungnown reader '+reader)            
        printd('data: '+str(data.shape)+' of '+str(data.dtype)+':\n'+str(data))
        return data
        
    def getTimeStamp(self):
        try:
            r = cns.adoGet( list = self.tsRequest )
            v = r[0][0][0] + r[0][1][0]/1e9
        except Exception as e:
            printe('getting timestamp')
            v = 0
        return v
        
    def getData(self):
        data = None
        blob, status = cns.adoGet(list = self.dataRequest)
        blob = blob[0]
        printd('got from '+self.pvname+': blob['+str(len(blob))+str(blob[:20]))
        if len(blob) == 0:
            printe('no data from '+self.pvname+' is manager all right?')
        else:
            data = self.blobToNdArray(blob)
        return data

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
            printd('status = OK tid: '+str(self.tid))
        else:
            printe('error = {0} while requesting {1} parameter from {2}'.format \
                ( cns.getErrorString(status), request[0][1], request[0][0].systemName ))
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
        printw('PVMonitorEpics is not implemented yet')
#,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
#```````````````````````````Imager````````````````````````````````````````````
class Imager(QtCore.QThread): # for signal/slot paradigm the inheritance from QtCore.QThread is necessary
    # define signal on data arrival
    signalDataArrived = QtCore.pyqtSignal(object)
    def __init__(self,pvname):
        super(Imager, self).__init__() # for signal/slot paradigm we need to call the parent init
        self.pvname = pvname
        self.qimg = QtGui.QImage()
        self.blob = None
        self.hwpb = [0]*4 # height width, number of planes, bits/channel
        self.plot = None
        self.roi = None
        self.contrast = None # Contrast histogram
        try: self.roiRect = [float(i) for i in pargs.ROI.split(',')]
        except: self.roiRect = None
        self.iso = None # Isocurve object
        self.isoInRoi = False
        self.data = None
        self.grayData = None
        self.mainWidget = None
        self.imageItem = None
        self.dockParRotate = 0
        self.degree = 0
        self.spotLog = None
        self.events = 0
        self.threshold = pargs.threshold # threshold for image thresholding
        self.maxSpots = pargs.maxSpots # number of spots to find
        self.spots = [] # calculated spot parameters
        self.roiArray = [] # array of ROI-selected data
        self.save = False # enable the continuous saving of imahes 
        self.refresh = 1 # refresh period in seconds
        self.paused = False # pause processing
        self.sleep = 0 # debugging sleep
        self.timestamp = -1 # timestamp from the source
        self.blocked = False # to synchronize event loops in GUI and procThread 
        self.rawData = None # rawData from reader
        self.stopProcThread = False # to stop procThread
        self.background = None
        profile('start')
        # connect signal to slot
        self.signalDataArrived.connect(self.process_image)

    def start(self):
        if not os.path.exists(pargs.logdir):
                os.makedirs(pargs.logdir)        
        self.savePath = pargs.logdir
        print('Processing thread for '+self.pvname+' started')
        thread = threading.Thread(target=self.procThread)
        thread.start()

    def qMessage(self,text):
        ans = QtGui.QMessageBox.question(self.gl, 'Confirm', text)
        #,                QtGui.QMessageBox.Yes, QtGui.QMessageBox.No)
        return 1 if ans == QtGui.QMessageBox.Ok else 0

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
        refs = ['ref'+str(i) for i in range(self.numRefs)]
        params = [
            {'name': 'Control', 'type': 'group', 'children': [
                {'name':'Event','type':'int','value':0,
                  'tip':'Accepted events','readonly':True},
                {'name':'Pause', 'type': 'bool', 'value': False,
                  'tip': 'Enable/disable receiving of images, '},
                {'name':'Saving', 'type': 'bool', 'value': False,
                  'tip': 'Enable/disable saving of images'},
                {'name':'View saved images', 'type': 'action',
                  'tip':'View saved images using GraphicMagic, use space/backspace for next/previous image'},
            ]},
            {'name': 'Configuration', 'type': 'group', 'children': [
                {'name':'Color', 'type':'list','values':['Native','Gray','Red','Green','Blue'],
                  'tip':'Convert image to grayscale or use only one color channel'},
                {'name':'Refresh', 'type':'list','values':['1Hz','0.1Hz','10Hz'],#,'Instant'],
                  'tip':'Refresh rate'},
                {'name':'Rotate', 'type': 'float', 'value': 0,
                  'tip':'Rotate image view by degree clockwise'},
                {'name':'Blur', 'type': 'action',
                  'tip':'Convert the current image to gray and blur it using gaussian filter with sigma 2'},
                {'name':'Background', 'type':'list','values':['None',] + refs,
                  'tip':'Streaming subtraction of a reference image inside the ROI'},
                #{'name':'Debug', 'type': 'bool', 'value': False},
                #{'name':'Sleep', 'type': 'float', 'value': 0},
                #{'name':'Test', 'type': 'str', 'value': 'abcd'},
                #{'name':'Debug Action', 'type': 'action'},
            ]},
            {'name':'SpotFinder', 'type':'group', 'children': [
                {'name':'MaxSpots', 'type': 'int', 'value':self.maxSpots,
                  'limits':(0,MaxSpotLabels),
                  'tip': 'Max number of spots to find in the ROI'},
                {'name':'Found:', 'type': 'int', 'value':0,'readonroiArrayly':True,
                  'tip': 'Number of spots found in the ROI'},
                {'name':'Threshold', 'type': 'float', 'value':self.threshold,
                  'tip': 'Threshold level for spot finding, changed with isoCurve level'},
                {'name':'Spots', 'type':'str','value':'(0,0)',
                  'readonly': True,'tip':'X,Y and integral of found spots'},
                {'name':'SpotLog', 'type': 'bool', 'value': False,
                  'tip':'Log the spots parameters to a file'},
            ]},
            {'name':'Reference images', 'type': 'group','children': [
                {'name':'Slot','type':'list','values': refs,
                  'tip':'Slot to store/retrieve/ reference image to/from local file "slot#.png"'},
                {'name':'View', 'type': 'action',
                  'tip':'View reference image, use space/backspace for next/previous image'},
                {'name':'Store', 'type': 'action'},
                {'name':'Retrieve', 'type': 'action'},
                {'name':'Add', 'type': 'action'},
                {'name':'Subtract', 'type': 'action'},
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
                if parGroupName == 'Control':
                    if parItem == 'Pause':
                        printd('Pause')
                        self.paused = itemData
                        self.updateTitle()
                        if not self.paused:
                            self.timestamp = 0 # invalidate timestamp to get one event 
                    elif parItem == 'Saving':
                        self.save = itemData
                        if self.save:
                            self.saveImage()
                            cprint('Saving images to '+self.savePath)
                        else:
                            cprint('Stopped saving to '+self.savePath)
                    elif parItem == 'View saved images':
                        # view saved images by spawning external viewer GraphicMagic
                        if not os.path.exists(self.savePath):
                            cprinte('opening path: '+self.savePath)
                            return
                        cmd = ['gm','display',self.savePath+'IV_'+self.pvname+'_*']
                        #printi('spawning: '+str(cmd))
                        cprint('viewing from '+self.savePath+', use Backspace/Space for browsing')
                        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                        
                elif parGroupName == 'Configuration':               
                    if parItem == 'Color':
                        if itemData == 'Gray':
                            pargs.gray = True
                            self.savedData = self.data
                            self.data = rgb2gray(self.data)
                            self.updateImageItemAndRoi()
                        elif itemData == 'Native':
                            pargs.gray = False
                            self.data = self.savedData
                            self.grayData = rgb2gray(self.data)
                            self.updateImageItemAndRoi()
                        else:
                            cprintw('Color = '+itemData+' reserved for future updates')
                    elif parItem == 'Rotate':
                        self.dockParRotate = float(itemData)
                        self.data = rotate(self.receivedData,self.dockParRotate)
                        self.grayData = rgb2gray(self.data)
                        self.updateImageItemAndRoi()
                        self.updateIsocurve()
                    elif parItem == 'Blur':
                        #TODO: blur color images
                        self.data = blur(self.data)
                        self.updateImageItemAndRoi()
                    elif parItem == 'Refresh':
                        self.refresh = {'1Hz':1,'0.1Hz':10,'10Hz':0.1,'Instant':0}[itemData]
                    elif parItem == 'Background':
                        if itemData == 'None':
                            self.background = None
                        else:
                            fn = pargs.logdir+self.pvname+'_'+itemData+'.png'
                            self.background = self.shapeData(self.load(fn))
                    elif parItem == 'Debug':
                        pargs.dbg = itemData
                        printi('Debugging is '+('en' if pargs.dbg else 'dis')+'abled')
                    elif parItem == 'Sleep':
                        self.sleep = itemData
                    elif parItem == 'Debug Action':
                        printi('Debug Action pressed')
                        print self.qMessage('confirm debug')
                        # add here the action to test
                        #printi('rd:'+str(self.receivedData.shape))
                        #self.receivedData = self.receivedData[:600,:1000]
                        #printi('rd:'+str(self.receivedData.shape))
                        #self.data = self.receivedData
                        #self.updateImageItemAndRoi()
                        
                if parGroupName == 'SpotFinder':               
                    if parItem == 'MaxSpots':
                        self.maxSpots = itemData
                    elif parItem == 'Threshold':
                        self.threshold = itemData
                    elif parItem == 'SpotLog':
                        if itemData:
                            try:
                                fn = pargs.logdir+'sl_'+self.pvname.replace(':','_')\
                                  +time.strftime('_%y%m%d%H%M.log')
                                self.spotLog = open(fn,'w',1)
                                cprint('file: '+self.spotLog.name+' opened')
                            except Exception as e: 
                                cprinte('opening '+fn+': '+str(e))
                                self.spotLog = None
                            if self.spotLog:
                                cmd = ['xterm','-e','tail -f '+fn]
                                #print 'tail:',cmd
                                #time.sleep(.5)
                                try:
                                    p = subprocess.Popen(cmd)
                                except Exception as e:
                                    cprintw('spawning '+str(cmd)+' : '+str(e))
                        else:
                            try: 
                                self.spotLog.close()
                                cprint('file:'+str(self.spotLog.name)+' closed')
                                p = subprocess.Popen(['pkill','-9','-f',self.spotLog.name])
                            except Exception as e: printe('in spotLog '+str(e))
                            self.spotLog = None
                    
                if parGroupName == 'Reference images':
                    prefix = pargs.logdir+self.pvname+'_'
                    if parItem == 'Slot': pass
                    elif parItem == 'View':
                        # view saved images by spawning external viewer GraphicMagic
                        cmd = ["gm",'display',prefix+'*']
                        cprint('viewing from '+prefix+'*'+', use Backspace/Space for browsing')
                        try:
                            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                        except Exception as e:
                            cprintw('spawning viewer '+str(cmd)+' : '+str(e))
                    else:
                        child = self.pgPar.child(parGroupName).child(parItem)
                        slot = self.pgPar.child(parGroupName).child('Slot').value()
                        fn = prefix+slot+'.png'
                        if parItem == 'Store':
                            if slot == 'ref0':
                                self.qMessage('cannot store to '+slot+', it is reserved for fixed background')
                            else:
                                if os.path.exists(fn):
                                    if not self.qMessage('Are you sure you want to overwrite '+slot+'?'):
                                        return
                                img = self.imageItem.qimage
                                if img.mirrored().save(fn,"PNG"): 
                                    cprint('Current image stored to '+fn)
                                else:    cprinte('saving '+fn)
                        else:
                            self.referenceOperation(fn,parItem)
                           
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
        #,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,        ## Create docks, place them into the window one at a time.
        ## Note that size arguments are only a suggestion; docks will still have to
        ## fill the entire dock area and obey the limits of their internal widgets.
        h,w = self.data.shape[:2]
        imageHSize = float(w)/float(h)*pargs.vertSize
        winHSize = float(w+100)/float(h)*pargs.vertSize # correct for the with of the contrast hist
        dockImage = pg.dockarea.Dock('dockImage - Image', size=(imageHSize,pargs.vertSize))
        area.addDock(dockImage, 'left')
        dockImage.hideTitleBar()
                
        #````````````````````Add widgets into each dock```````````````````````
        # dockImage: a plot area (ViewBox + axes) for displaying the image
        self.gl = pg.GraphicsLayoutWidget()
        self.mainWidget = self.gl.addPlot()
        dockImage.addWidget(self.gl)
        # Item for displaying image data
        #print 'adding imageItem:',self.imageItem.width(),self.imageItem.height()
        self.mainWidget.addItem(self.imageItem)
        self.mainWidget.autoRange(padding=0) # remove default padding
        self.mainWidget.setAspectLocked()
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
            
            self.gl.addItem(self.contrast)

        if pargs.iso != 'Off':
        # Isocurve drawing
            if pargs.iso == 'ROI':
                self.isoInRoi = True
                printw('iso == ROI is not fully functional yet')
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
            #self.updateIso()

        if pargs.roi:
        # Custom ROI for selecting an image region
            #dockPlot = pg.dockarea.Dock('dockPlot', size=(1,100))
            dockPlot = pg.dockarea.Dock('dockPlot', size=(0,0))
            area.addDock(dockPlot, 'bottom')
            dockPlot.hideTitleBar()
            self.plot = pg.PlotWidget()
            dockPlot.addWidget(self.plot)

            h,w,p,b = self.hwpb
            #self.roi = pg.ROI([w*0.25, h*0.25], [w*0.5, h*0.5])
            #self.roi.addScaleHandle([1, 1], [0, 0])
            rect = (w*0.25, h*0.25, w*0.5, h*0.5) if self.roiRect == None else self.roiRect
            self.roi = pg.RectROI(rect[:2], rect[2:], sideScalers=True)
            self.mainWidget.addItem(self.roi)
            self.roi.setZValue(10)  # make sure pargs.roi is drawn above image
            
            # create max number of spot labels
            self.spotLabels = [pg.TextItem('*',color='r',anchor=(0.5,0.5)) 
              for i in range(MaxSpotLabels)]
            for sl in self.spotLabels:
                self.mainWidget.addItem(sl)

            # Connect callback to signal
            self.roi.sigRegionChangeFinished.connect(self.updateRoi)
            self.updateRoi()

        if pargs.console:
        # interactive python console
            dockConsole = pg.dockarea.Dock('dockConsole - Console', size=(1,50), closable=True)
            area.addDock(dockConsole, 'bottom')
            ## Add the console widget
            global gWidgetConsole
            histFile = '/tmp/imageAnalyzer_console.pcl'
            gWidgetConsole = CustomConsoleWidget(
                namespace={'pg': pg, 'np': np, 'plot': self.plot, 'roi':self.roi, #'roiData':meansV,
          'data':self.data, 'image': self.qimg, 'imageItem':self.imageItem, 'pargs':pargs, 'sh':sh},
                historyFile=histFile,
            text="""This is an interactive python console. The numpy and pyqtgraph modules have already been imported  as 'np' and 'pg'
The shell command can be invoked as sh('command').
Accessible local objects: 'data': image array, 'roiData': roi array, 'plot': bottom plot, 'image': QImage, 'imageItem': image object.
For example, to plot vertical projection of the roi: plot.plot(roiData.mean(axis=1), clear=True).
to swap Red/Blue colors: imageItem.setImage(data[...,[2,1,0,3]])
""")
            dockConsole.addWidget(gWidgetConsole)

        self.win.resize(winHSize, pargs.vertSize) 
        self.win.show()
        #,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
        
    def load(self,fn):
    # load image from file
        data = None
        img = QtGui.QImage()
        if img.load(fn):
            #printd('image: '+str((img.width(),img.height())))
            data = convert_qimg_to_ndArray(img.mirrored())
            #printd('loaded:'+str(data))
            #cprint('Loaded image from '+fn)
        else: cprinte('loading '+fn)        
        #print 'load:',data.dtype,self.data.dtype
        return data
    
    def saveImage(self):
        import datetime
        fmt = '_%Y%m%d_%H%M%S.png'
        if self.timestamp:
            strtime = datetime.datetime.fromtimestamp(self.timestamp).strftime(fmt)
        else:
            strtime = time.strftime(fmt)
        #self.checkPath()
        fn = self.savePath+'IV_'+self.pvname+strtime
        img = self.imageItem.qimage
        if not img.mirrored().save(fn,"PNG"): 
            cprinte('saving '+fn)
            # does not work: self.set_dockPar('Control','Saving',False)
    
    def shapeData(self,data):
        if data is not None:
            if len(self.data.shape) == 2:
                data = data[:,:,0] # saved files are always color PNG
        return data

    def referenceOperation(self,fn,operation):
        ''' binary operation with current and restored image '''
        try: 
            data = self.shapeData(self.load(fn))
            if data is None: return
            if operation == 'Retrieve':
                self.data = data
                cprint('retrieved image '+fn)
            elif operation == 'Add':
                self.data = (self.data.astype(int) + data)/2
                cprint('added image '+fn+' to current image')
            elif operation == 'Subtract':
                self.data = self.data.astype(int) - data
                cprint('subtracted image '+fn+' from current image')
            else: pass
            self.grayData = rgb2gray(self.data)
            self.updateImageItemAndRoi()
        except Exception as e: printe(str(e))
        
    def set_dockPar(self,child,grandchild,value):
        self.pgPar.child(child).child(grandchild).setValue(value)

    def stop(self):
        self.stopProcThread = True
        printi('imager stopped')
        try: self.spotLog.close()
        except: pass

    def updateTitle(self):
        self.win.setWindowTitle(('Waiting','Paused')[self.paused]+' '+self.winTitle)
                
    def updateRoi(self):
    # callback for handling ROI
        profile('initRoi')
        # the following is much faster than getArrayRegion
        slices = self.roi.getArraySlice(self.grayData,self.imageItem)[0][:2]
        self.roiArray = self.grayData[slices]
        oy,ox = slices[0].start, slices[1].start
        profile('roiArray')
        
        # find spots using isoLevel as a threshold
        if self.threshold>1 and self.maxSpots>0:
            self.spots = findSpots(self.roiArray,self.threshold,self.maxSpots)
            profile('spotsFound')
            if pargs.profile:
                print(profStates('initRoi','spotsFound'))
                print('FindSpot time: '+profDif('initRoi','spotsFound'))
            #self.spots = [(x+ox+0.5,y+oy+0.5,s) for x,y,s in self.spots]
            self.spots = [[x+ox,y+oy,s] for x,y,s in self.spots]
            msg = ''
            h,w = self.data.shape[:2]
            conv = [(0,1),(0,1)]
            
            for i,spot in enumerate(self.spots):
                #print 'l%i:(%0.4g,%0.4g)'%(i,spot[0],spot[1])
                self.spotLabels[i].setPos(spot[0],spot[1])
                spot[:2] = [(pos-c[0])/c[1] for pos,c in zip(spot[:2],conv)]
                msg += '%5.1f,%5.1f,%0.4g,\t,'%tuple(spot)
            # reset outstanding spotLabels
            for j in range(len(self.spots),len(self.spotLabels)):
                self.spotLabels[j].setPos(0,0)
            printd('findSpots: '+msg)
            self.set_dockPar('SpotFinder','Found:',len(self.spots))
            self.set_dockPar('SpotFinder','Spots',msg)
            if self.spotLog: 
                spotLogTxt = time.strftime('%y-%m-%d %H:%M:%S,\t')+msg
                self.spotLog.write(spotLogTxt+'\n')
        
        # plot the ROI histograms
        meansV = self.data[slices].mean(axis=0) # vertical means
        #x = range(len(meansV)+1); s = True
        x = range(len(meansV)); s = False
        #if self.hwpb[2] == 1: # gray image
        if len(self.data.shape) == 2: # gray image
            self.plot.plot(x,meansV,clear=True,stepMode=s)
        else: # color image
            # plot color intensities
            self.plot.plot(x,meansV[:,0],pen='r', clear=True,stepMode=s) # plot red
            self.plot.plot(x,meansV[:,1],pen='g',stepMode=s) # plot green
            self.plot.plot(x,meansV[:,2],pen='b',stepMode=s) # plot blue
            meansVG = self.grayData[slices].mean(axis=0)
            self.plot.plot(x,meansVG,pen='w',stepMode=s) # plot white
        #profile('roiPlot')
        
    def updateIsocurve(self):
    # callback for handling ISO
        printd('>uIsoCurve')
        #profile('init iso')
        #if len(self.roiArray):
        if self.isoInRoi:
            #TODO: need to relocate the isocurves to ROI origin
            self.iso.setData(blur(self.roiArray))
        else:
            self.iso.setData(blur(self.grayData))
        self.threshold = self.isoLine.value()
        #printi('isolevel:'+str(self.threshold))
        self.iso.setLevel(self.threshold)
        #profile('iso')
        self.set_dockPar('SpotFinder','Threshold',self.threshold)
        if self.roi: 
            self.updateRoi()
         
    def updateImageItemAndRoi(self):
        self.imageItem.setImage(self.data)
        if self.contrast is not None: self.contrast.regionChanged() # update contrast histogram
        if self.roi: 
            self.updateRoi()

    def process_image(self, **kargs):
        global profilingState
        profile('image')
        printd('uimage:'+str(kargs))
        data = self.rawData

        if pargs.width:
            #``````````The source is vector parameter with user-supplied shape
            l = len(data)
            if self.imageItem == None: # do it once
                tokens = pargs.width.split(',')
                w = int(tokens[0])
                h = int(tokens[1])
                bytesPerPixel = l/w/h
                try:    bitsPerPixel = int(tokens[2])
                except: bitsPerPixel = 8*bytesPerPixel                
                # we cannot decide exactly about nPlanes and bytesPerChannel based on bitsPerPixel
                # here is assumption:
                nPlanes = 3 if bitsPerPixel > 16 else 1 #
                bytesPerChannel = bytesPerPixel / nPlanes
                self.hwpb = [h, w, nPlanes,bytesPerChannel]
            
            if self.hwpb[3] > 1: # we need to merge pairs of bytes to integers
                #data = np.array(struct.unpack('<'+str(l/self.hwpb[3] )+'H', data.data),'u2')
                data = struct.unpack('<'+str(l/self.hwpb[3] )+'H', data.data) 
                #data = struct.unpack('<'+str(l/self.hwpb[3] )+'H', bytearray(data)) 
                profile('merge')
            shape = (self.hwpb[:3] if self.hwpb[2]>1 else self.hwpb[:2])
            data = np.reshape(data,shape)
        #,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
        #````````````````````Got numpy array from data````````````````````````
        data = data[::-1,...] # flip vertical axis            
        self.receivedData = data # store received data
        printd('array: '+str((data.shape,data.dtype,data)))
        profile('array')
                                        
        self.data = rotate(self.receivedData,self.dockParRotate)
        #profile('rotate'); print('rotation time:'+profDif('array','rotate'))

        if pargs.gray: 
            self.data = rgb2gray(self.data)
            self.grayData = data
        else:
            self.grayData = rgb2gray(self.data)

        #````````````````````Data array is ready for analisys`````````````````
        h,w = self.data.shape[:2]
        if self.imageItem == None: # first event, do the show() only once
            if self.hwpb[0] == 0: # get p,b: number of planes and bytes/channel
                try: p = self.data.shape[2]
                except: p = 1
                b = self.data.dtype.itemsize
                self.hwpb = [h,w,p,b]
            printd('hwpb:'+str(self.hwpb))
            printd('self.array: '+str((self.data.shape,self.data)))
            self.imageItem = pg.ImageItem(self.data)
            self.show()
        else:  # udate data
            #TODO: react on shape change correctly, cannot rely on self.hwpb because of possible rotationg2-cec.laser-relay-cam_ref1.png
            if self.background is not None:
                self.data = self.data.astype(int) - self.background
                self.grayData = rgb2gray(self.data)
            if self.save: self.saveImage() #TODO shouldn't it be after update
            self.updateImageItemAndRoi()
        self.events += 1
        self.set_dockPar('Control','Event',self.events) # concern: time=0.5ms
        #,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,            
        if pargs.profile:
            print('### Total time: '+'%0.3g' % (timer()-profilingState['start']))
            print(profStates('start','finish'))
        self.blocked = False
            
    def procThread(self):
    # data processing thread
        if not pvMonitor:
            printe('pvMonitor did not start, processing stopped')
            sys.exit(10)
        pvMon = pvMonitor
        printi('Processing started')
        while not self.stopProcThread:
            profile('start')
            profile('tmp')
            t = pvMon.getTimeStamp() if pvMon else 0.
            profile('getTS')
            dt = t - self.timestamp
            if dt >= self.refresh and not self.paused:
                self.timestamp = t
                #print('Data available')
                self.rawData = pvMon.getData()
                profile('getData')
                self.blocked = True
                #self.process_image(data = data) # not thread-safe at high rate       
                self.signalDataArrived.emit('newData')
                while self.blocked:
                    time.sleep(0.1)
                time.sleep(self.sleep) # adjustable sleep for debugging
                #print('Data processed')
            time.sleep(self.refresh/2.)
        printi('Processing finished')
        
#````````````````````````````Main Program`````````````````````````````````````
pvMonitor = None
#png = None
def main():
    global pargs, imager, pvMonitor, png
    # parse program arguments
    import argparse
    parser = argparse.ArgumentParser(description='''
      Interactive analysis of images from ADO or file''')
    parser.add_argument('-d','--dbg', action='store_true', help=
      'turn on debugging')
    parser.add_argument('-R','--rotate', type=float, default=0, help=
      'Rotate image by ROT degree')
    parser.add_argument('-F','--flip', help=
      "flip image, 'V':vertically, 'H':horizontally")
    #parser.add_argument('-i','--iso', action='store_false', help=
    #  'Disable Isocurve drawing')
    parser.add_argument('-i','--iso',default='Image',help=
      'Isocurve drawing options: ROI - only in ROI (default), Image - in full image, Off - no isocurevs')
    parser.add_argument('-r','--roi', action='store_false', help=
      'Disable Region Of Interest analysis')
    parser.add_argument('-f','--fullsize', action='store_true', help=
      'use full-size full-speed imageM parameter')
    parser.add_argument('-c','--console', action='store_false', help=
      'Disable interactive python console')
    parser.add_argument('-H','--hist', action='store_false', help=
      'Disable histogram with contrast and isocurve contol')
    parser.add_argument('-w','--width', help=
      'For blob data: width,height,bits/pixel i.e 1620,1220,12. The bits/pixel may be omitted for standard images')
    parser.add_argument('-p','--profile', action='store_true', help=
      'Enable code profiling')
    parser.add_argument('-e','--extract',default='qt',help=
      'image extractor: qt for QT (default), png for pyPng (for 16-bit+ images)') #cv for OpenCV, 
    parser.add_argument('-g','--gray', action='store_true', help=
      'Show gray image')
    #parser.add_argument('-s','--spot', action='store_true', help='Enable Spot Finder to estimate spot and background parameters inside the ROI. It could be slow on some images')
    parser.add_argument('-G','--graysum', action='store_false', help='Use perceptional color-to-gray conversion, rather than uniform')
    parser.add_argument('-a','--access', default = 'file', help='pv access system: file/ado') 
    parser.add_argument('-l','--logdir', default = '/tmp/imageViewer/',help=
      'Directory for logging and rererences')
    parser.add_argument('-m','--maxSpots',type=int,default=4,
      help='Maximum number of spots to find')
    parser.add_argument('-t','--threshold',type=float,default=0,
      help='Threshold for spot finding')
    parser.add_argument('-O','--ROI',default='',
      help='ROI rectangle: posX,pozY,sizeX,sizeY')
    parser.add_argument('-v','--vertSize',type=float,default=800,
      help='Vertical size of the display window')
    parser.add_argument('pname', nargs='*', 
      default=['hubble_deep_field.jpg'],
      help=''' image_source''')

    pargs = parser.parse_args()
    #print pargs.access, pargs.pname
    
    pname = pargs.pname[0]

    extractor = {'qt':'QT','cv':'OpenCV','png':'PyPng','raw':'raw'}
    if pargs.width: # if width is provided, the pargs.extract should be set to 'raw'
        pargs.extract = 'raw'
                
    print(parser.prog+' '+pname+' using '+extractor[pargs.extract]+
      ', version '+__version__)
    if not pargs.hist: pargs.iso = 'Off'
                
    if pargs.extract == 'png':
        import png

    elif pargs.extract == 'cv':
        import cv
        
    # instantiate the imager
    imager = Imager(pname)

    # instantiate the data monitor
    if pargs.access == 'file':
        pvMonitor = PVMonitorFile(pname,imager.process_image,
                                reader=pargs.extract)
    elif pargs.access == 'ado':
        # use low level cns module
        try: 
            from cad import cns
            print('cns version '+cns.__version__)
        except Exception as e:
            printe('importing cns: '+str(e))
            cns = None
        if not cns: 
            exit(6)
        pvMonitor = PVMonitorAdo(pname,imager.process_image,
                                reader=pargs.extract)
    imager.start()
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
