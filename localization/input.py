import numpy as np
import json
from StringIO import StringIO
import re
import os
import sys
import cv2
import cv
from random import randint

class Box:
    def __init__(self, box):
        for i in xrange(len(box)):
            box[i] = int(box[i])
            
        
        
        xres = 640
        yres = 352
        
        self.box = box
        self.width = int(box[2])
        self.height = int(box[3])
        
        if box[0] > xres:
            box[0] = xres
        if box[1] > yres:
            box[1] = yres
            
        maxX = min([box[0]+self.width, xres])
        maxY = min([box[1]+self.height, xres])
            

        cenX = (box[0]+box[2]/2)
        cenY = (box[1]+box[3]/2)
        self.centroid = (cenX, cenY)

        
        self.upperLeft = (box[0], box[1])
        self.lowerRight = (maxX, maxY)
        self.lowerLeft = (box[0], maxY)
        
        
        self.humanID = None
        self.hits = 0
        self.color = (randint(0, 255), randint(0, 255), randint(0, 255))
        self.area = box[2]*box[3]
        self.picked = False
        self.bM = (cenX, cenY) #(cenX, maxY-30)##best guess for contact with ground
        self.tM = (cenX, box[1])
        self.world_bM = None
        self.w = None
        
    def shape(self, box):
        newBox=[]
        

             




        
class Frame:
    def __init__(self, fid, loc=None, quat=None):
        self.ID = fid #should match index, here for debugging
        self.image = None
        self.boxes = []
        self.slices = []
        self.location = (loc)
        self.quaterion = (quat)
        self.img = None

class Video:
    def __init__(self):
        self.boxesFile = "data/boxes.txt"
        self.locationFile = "data/cameraLocation.json"
        self.imageDir = "data/Test1/"
        self.imageFiles = [s for s in os.listdir(self.imageDir) if s.endswith('.png')]
        self.imageFiles.sort()  
        
        
        self.boxesOffset = 3732    #Need to add this to the boxes ID to make them match
        self.y_crop = 4 #amount of pix that are cropped in LSDSlam (4)
        self.x_crop = 0
        self.resample = .5 #image in LSDSlam is 1/2   #assuming resample, then crop

        
        
       
        self.kalibrationM = np.zeros((3,3))
        self.frames = []  #[ID]=Frame()
        self.features = []  # [personID]=keypoints
        
        
    def readFiles(self):
        with open(self.boxesFile, 'r') as f:
            self.boxesText = f.readlines()
        with open(self.locationFile, 'r') as f:
            self.locationJSON = f.readlines()
        
        
    def addNextFrame(self):
        frameCtr = 0
        
        for line in self.locationJSON:
            nextLoc = json.loads(line)
            
            loc = (float(nextLoc[u'Frame'][u'Loc'][u'X']), 
                   float(nextLoc[u'Frame'][u'Loc'][u'Y']), 
                   float(nextLoc[u'Frame'][u'Loc'][u'Z']))
            
            quat = (float(nextLoc[u'Frame'][u'Quat'][u'x']),
                    float(nextLoc[u'Frame'][u'Quat'][u'y']),
                    float(nextLoc[u'Frame'][u'Quat'][u'z']),
                    float(nextLoc[u'Frame'][u'Quat'][u'w']))
            
            fid = int(nextLoc[u'Frame'][u'ID']) - 1
            #print fid
            
            
            while frameCtr < fid:
                #print fid
                #print frameCtr            
                self.frames.append(Frame(frameCtr))
                frameCtr += 1

            
            frameCtr = fid + 1
            
            #print len(self.frames)
            #print fid
            
            self.frames.append(Frame(fid, loc, quat))
            

        for line in self.boxesText:
            boxes = re.findall("\[(.*?)\]", line)
            fid = re.match("[^\s\[]+", line)
            
            fid = int(line[:fid.end()]) - self.boxesOffset
            
            
            processedBoxes = self.processBox(boxes)
            
            #try:

            if fid < len(self.frames):
                self.frames[fid].boxes = processedBoxes
            #except:
            #    print "error" + str(fid)
          
        for file in self.imageFiles:

            fid = re.match("[^(\.png$)]+", file)
            if fid != None:
                
                fid = int(file[:fid.end()]) - self.boxesOffset
                #print fid
                
                if fid < len(self.frames):
                    self.frames[fid].img = self.imageDir + file

            
    def processBox(self, rawBoxes):
        processedBoxes = []
        for box in rawBoxes:
            newBox = box.split()
            croppedBox = []
            for idx, coord in enumerate(newBox):
                coord = int(coord)*self.resample
                if coord < 0:
                    coord == 0
                
                if idx % 2 == 1: #y-values every other starting at index 1
                    coord = coord - self.y_crop
                if idx % 2 == 0: #x-values every other starting at index 0
                    coord = coord - self.x_crop
                croppedBox.append(coord)
            bx = Box(croppedBox)
            if bx.height < 200:
                processedBoxes.append(bx)
        return processedBoxes
    
            
            
        
        
    

        
        
        



        
        
    
        