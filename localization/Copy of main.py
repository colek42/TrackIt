from input import *
import os
import sys
import math



# for idx, i in enumerate(vid.frames):
# 
#     print i.ID
#     for box in i.boxes:
#         print box.__dict__

class Human:
    def __init__(self):
        self.keypoints = []  #list of descriptors for this person
        self.dectectedFrames = []
        self.imageSlices = []
        self.boxes = []



class DetectOverlap(object):
    def __init__(self, vid):
        self.imageFiles = os.listdir(vid.imageDir)
        self.imageFiles.sort()   
        self.video = vid
        self.currFrame = 0
        self.nextHumanID = 0
        self.boxCache = []
        self.Humans = dict()
        self.lookbackSize = 15 #number of frames to search for matches
    
    def overlapBoxes(self):
        if self.currFrame == 0:
            self.currFrame += self.lookbackSize
            return
        
        currBoxes = self.video.frames[self.currFrame].boxes
        prevBoxes = self.video.frames[self.currFrame-1].boxes
        
        
        for currBox in currBoxes:
            minDist = max([currBox.height/2, currBox.width/2]) #threshhold for detecting overlap
            possibleMatches = []
            for prevBox in prevBoxes:
                dist = self.distance(currBox, prevBox)
                if dist < minDist**2:  #faster than sqrt
                    possibleMatches.append(prevBox)
            
            if len(possibleMatches) > 0:  #try to match from prev frame
                match = min(possibleMatches)
                if match.humanID != None:
                    currBox.humanID = match.humanID
            
            
            else:
                currBox.humanID = self.nextHumanID
                self.nextHumanID += 1
                
            self.updateHuman(currBox.humanID, currBox, self.currFrame)
        self.currFrame +=1
            
                

    def distance(self, box1, box2):
        f1 = (box1.centroid[0]-box2.centroid[0])**2
        f2 = (box1.centroid[1]-box2.centroid[1])**2
        return f1+f2
    
    def updateHuman(self, hid, box, frame):
        if self.Humans.has_key(hid):
            self.Humans[hid].boxes.append((box, frame))
            print "Frame: " + str(frame) + "\tID: " + str(hid) + "\tCentroid: " + str(box.centroid)
        else:
            self.Humans[hid] = Human()
            self.updateHuman(hid, box, frame)
            
        
vid = Video()
vid.readFiles()
vid.addNextFrame()
start = DetectOverlap(vid)

for i in vid.frames:
    start.overlapBoxes()  
    
    



