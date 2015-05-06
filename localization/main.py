from input import *
import os
import sys
import math
import cv2
import cv
import numpy as np
import operator
import camera as otherCam
import matplotlib.pyplot as plt

MODE = '3D'



def ThreeDeeDist(p1, p2):
    m1 = np.array(p1[0])
    m2 = np.array(p2[0])
    
    dist = np.linalg.norm(m1-m2)
    
    return dist






class Camera:
    def __init__(self):
        self.fx_w = 0.527334
        self.fy_h = 0.827306
        self.cx_w = 0.473568
        self.cy_h = 0.499436
        self.pix_x = 640
        self.pix_y = 352
        
        self.calibMatrix()
        
        
    def quatToRotation(self, x=0, y=0, z=0, w=0):
        matrix = np.zeros((3,3))
    
        # Repetitive calculations.
        q4_2 = w**2
        
        q12 = x * y
        q13 = x * z
        q14 = x * w
        q23 = y * z
        q24 = y * w
        q34 = z * w
    
        # The diagonal.
        matrix[0, 0] = 2.0 * (x**2 + q4_2) - 1.0
        matrix[1, 1] = 2.0 * (y**2 + q4_2) - 1.0
        matrix[2, 2] = 2.0 * (z**2 + q4_2) - 1.0
    
        # Off-diagonal.
        matrix[0, 1] = 2.0 * (q12 - q34)
        matrix[0, 2] = 2.0 * (q13 + q24)
        matrix[1, 2] = 2.0 * (q23 - q14)
    
        matrix[1, 0] = 2.0 * (q12 + q34)
        matrix[2, 0] = 2.0 * (q13 - q24)
        matrix[2, 1] = 2.0 * (q23 + q14)
        
        r, trash = cv2.Rodrigues(matrix)
        
        r[2] = r[2] - math.radians(0)
        r[1] = r[1] - math.radians(180)
        self.R, trash = cv2.Rodrigues(r)
    
    #world coords of camera
    def transVec(self, x=0, y=0, z=0):
        matrix = np.zeros((3,1))
        
        matrix[0] = x
        matrix[1] = y
        matrix[2] = z
        
        self.t = matrix
    
    
    def calibMatrix(self):
        matrix = np.zeros((3,3))
        
        matrix[0,0] = self.fx_w
        matrix[1,1] = self.fy_h
        matrix[2,2] = 1
        
        matrix[0,2] = self.cx_w
        matrix[1,2] = self.cy_h
        
        self.K = matrix
        
    def projectionMatrix(self):
        
        #self.P = self.K.dot(np.hstack((self.R, self.t)))
        
        
        tmpP1 = np.transpose(self.R).dot(self.t)
        tmpP2 = np.hstack((self.R, np.negative(tmpP1)))
        tmpP3 = self.K.dot(tmpP2)
        self.P = tmpP3
        self.P2 = self.K.dot(np.hstack((self.R, self.t)))

    def getWorldPoint(self, u, v, z=0):
        
        camCoords = np.ones((1, 3))
        camCoords[0,0] = u
        camCoords[0,1] = v
        camCoords = np.array([u, v, z])
        
        
        
        col3tmp= self.P2[:,2] * z
        col3 = self.P2[:,3] - col3tmp
        P = self.P2
        P[:,2] = col3
        tmpP = P[:,:3]
        
        
        
        #tmpP = np.hstack((self.P[:, [0, 1]], self.P[:, 2, np.newaxis] * z + self.P[:, 3, np.newaxis]))
        tmpP2 = np.linalg.inv(tmpP).dot(camCoords.transpose())
        
        x = (tmpP2[0]/tmpP2[2])
        y = (tmpP2[1]/tmpP2[2])
        
#         print "X=", x 
#         print "Y=", y
#         print "A=", z
        ret = (x, y, z)
        #print self.getCameraPoint(x, y, z)
        #print "U,V" + str(u) + "   " + str(v)
        
        
        
        
        
        return ret         
        
    def getCameraPoint(self, x, y, z):
        
        world = [[x, y, 1]]
        world = np.array(world)

#         world = np.array(1, 3)
#         world[0] =float(x)
#         world[1] =float(y)
#         world[2] =float(z)
        
        h, jac = cv2.projectPoints(world, self.R, self.t, self.K, None)
        #print "H: " + str(h)
        
        
        
        
# 
#         camera_coords = self.R.dot(world.transpose()) + self.t
#         xy = camera_coords[0:2, :]
#         z = camera_coords[2, :]
# # #         
#         imgCoords = xy / z
#         M = np.ones((1,3))
#         M[0,0] = imgCoords[0,0]
#         M[0,1] = imgCoords[1,0]
#         M[0,2] = 1
#         
#         print "m: " + str(imgCoords)
#          
#         ret = M.dot(self.K)
        
        
        
        
        return h
        
    
    

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
        self.video = vid
        self.currFrame = 0
        self.nextHumanID = 0
        self.boxCache = []
        self.Humans = dict()
        self.lookbackSize = 20 #number of frames to search for matches
        self.cam = otherCam.Camera()
        self.heightAvg=0
        self.heightCnt=0
        self.skip = 0
        self.try3D = 0
        matrix = np.zeros((3,3))
        
        
        fx_w = 0.527334
        fy_h = 0.827306
        cx_w = 0.473568
        cy_h = 0.499436
        pix_x = 640
        pix_y = 352
        
               
        
        matrix[0,0] = fx_w
        matrix[1,1] = fy_h
        matrix[2,2] = 1
        
        matrix[0,2] = cx_w
        matrix[1,2] = cy_h
        
        self.cam.set_K = matrix
        
        
        
    
    def threeD(self):
        global MODE
        
        
        try:
            loc = self.video.frames[self.currFrame].location
            quat = self.video.frames[self.currFrame].quaterion
            loc = np.array(loc[:3]).transpose()
            
            locArr = np.zeros((3,1))
            locArr[0] = loc[0]
            locArr[1] = loc[1]
            locArr[2] = loc[2]
            
            self.cam.set_t(locArr)
            self.cam.quatToRotation(*quat)
        except:
            MODE = "NOT TRACKING"
        #self.cam.projectionMatrix()
        

        
        
    
    
    
    def overlapBoxes(self):
        global MODE
        for idx2, currBoxTmp in enumerate(self.video.frames[self.currFrame].boxes):
            tmp = self.cam.image_to_world(*currBoxTmp.bM)
            currBoxTmp.world_bM = (float(tmp[0][0]), float(tmp[0][1]), float(tmp[0][2]))
            currBoxTmp.w = tmp[1]
              
            for i in range(0, len(self.video.frames[self.currFrame].boxes)-1):
                try:
                    dist = self.distance(currBoxTmp, self.video.frames[self.currFrame].boxes[i])
                    #print float(dist[2])
                      
#                       
#                     if float(dist[0]) < .0000001:
#                         pass
                      
                    if float(dist[0]) <  .00001:
                        self.video.frames[self.currFrame].boxes.pop(i) #boxes too close
                        i+=1
                           
#                           
#                     elif float(dist[0]) <  .000001 or float(dist[0]) > 300:
#                         self.video.frames[self.currFrame].boxes.pop(i) #boxes too close                
#                         i+=1
                except:
                    pass
                #    print "here"
        
        
        
        if self.currFrame == 0:
            self.currFrame += self.lookbackSize
            return
        
        try:
            currBoxes = self.video.frames[self.currFrame].boxes
        except:
            return
        


            
        
        for idx1, currBox in enumerate(currBoxes):
            #max([currBox.height/2, currBox.width/2]) #threshhold for detecting overlap
            match = None
            if MODE == '3D':
                minDist = .1
            else:
                minDist = 100
                print "################# LOST TRACKING ################"
            for i in range(1, self.lookbackSize):
                prevBoxes = self.video.frames[self.currFrame-i].boxes
                prevBoxes.sort(key=operator.attrgetter('hits'))
                #prevBoxes.reverse()
                
                for idx, prevBox in enumerate((prevBoxes)):
                    
                    if prevBox.picked != True:
                        
                    
                        if MODE == '2D' or MODE == 'NOT TRACKING':
                            self.try3D +=1
                            
                            dist = self.distance(currBox, prevBox)
                            areaDiff = abs(currBox.area - prevBox.area)
                            if (dist < minDist or dist < 5) and areaDiff < .20 * currBox.area:
                                minDist = dist
                                match = (idx, self.currFrame-i) #box index, frame
                            if MODE == 'NOT TRACKING' and self.try3D>90:
                                self.try3D = 0
                                MODE = '3D'
                                
                        if MODE == '3D':
                            dist = self.distance(currBox, prevBox)
                            #currBox.world_bM=float(dist[0])
                            
                            #self.video.frames[self.currFrame].boxes[idx1].world_bM=dist[1]
                            
                            #print "Dist:" +str(dist[0]) + "  Min:" + str(minDist)
                            if float(dist[0]) < minDist:
                                minDist = float(dist[0])
                                
                                match = (idx, self.currFrame-i) #box index, frame
                        
                
            if match != None:  #try to match from prev frame
                
                if self.video.frames[match[1]].boxes[match[0]].humanID != None:
                    currBox.humanID = self.video.frames[match[1]].boxes[match[0]].humanID
                    currBox.color = self.video.frames[match[1]].boxes[match[0]].color
                    currBox.hits = self.video.frames[match[1]].boxes[match[0]].hits + 1
                    self.video.frames[match[1]].boxes[match[0]].picked = True
                    
                else:
                    currBox.humanID = self.nextHumanID
                    self.nextHumanID += 1

            
            
            else:
                currBox.humanID = self.nextHumanID
                self.nextHumanID += 1
                
            self.video.frames[self.currFrame].boxes[idx1] = currBox    
            self.updateHuman(currBox.humanID, currBox, self.currFrame)
        self.currFrame +=1
            
                

    def distance(self, box1, box2):
        if MODE == '2D':        
            f1 = (box1.centroid[0]-box2.centroid[0])**2
            f2 = (box1.centroid[1]-box2.centroid[1])**2
            return math.sqrt(f1+f2)
        if MODE == '3D':
            
            
            b1 = box1.world_bM
            b2 = box2.world_bM
            #print b2
            
            if b2 == None:
                b2 = self.cam.image_to_world(*box2.bM)
                
            if b1 == None:
                b1 = self.cam.image_to_world(*box1.bM)
            
            
            #ll = self.cam.image_to_world(*box1.lowerLeft)
            #lr = self.cam.image_to_world(*box1.lowerRight)
                
            #print "WIDTH: "  + str(ThreeDeeDist(ll, lr))
            
            #h1 = self.cam.image_to_world(box1.bM[0], box1.bM[1], box1.box[1])
            
            
            dist = ThreeDeeDist(b1, b2)
            #height = ThreeDeeDist(b1, h1)
            #print height
            
            return (dist, b1, None)
    
    def updateHuman(self, hid, box, frame):
        if self.Humans.has_key(hid):
            self.Humans[hid].boxes.append((box, frame))
            #print "Frame: " + str(frame) + "\tID: " + str(hid) + "\tCentroid: " + str(box.centroid)
        else:
            self.Humans[hid] = Human()
            self.updateHuman(hid, box, frame)
            




class Visulize(object):
    def __init__(self, record=False, filename=None):
        global start
        
        self.record = record
        self.filename = filename
        self.hotBoxes = dict()
        self.tracks = []
        self.start = start
        
        plt.figure(figsize =(18,16))
        plt.axis([-.3, .7, -.3, .5])
        plt.ion()
        plt.show()
    
    
    def plot(self, x, y, clr='red', mkr="o"):
        plt.scatter(x, y, s=8, color=clr, marker=mkr)
        
        plt.draw()
    
    
        
    def showNext(self, frame):
        #try:
        img = cv2.imread(frame.img, cv2.CV_LOAD_IMAGE_COLOR)
        img = img[4:356:, 0:640] #crop to match LSDSlam
        rendered = []
        for box in frame.boxes:
            if box.hits > 25 and MODE == "3D":
                self.hotBoxes[box.humanID] = box
                rendered.append(box.humanID)
                
#                 if box.hits > 25:
#                     self.tracks.append((box.world_bM, box.color, box.w))
#                     if len(self.tracks) > 50:
#                         self.tracks.pop(0)
                
                cv2.rectangle(img, box.upperLeft, box.lowerRight, box.color, 3)
                try:
                    self.plot(-box.world_bM[0], box.world_bM[1], clr=tuple(i / 255.0 for i in box.color))
                    
                    loc = (self.start.cam.t[0][0], self.start.cam.t[1][0])
                    
                    self.plot(-loc[0], loc[1], clr='blue', mkr="x")
                except:
                    #print "NO"
                    pass
                
        delList = []
        for key in self.hotBoxes:
            try:
                if key not in rendered:
                    if self.hotBoxes[key].hits > 2:
                        cv2.rectangle(img, self.hotBoxes[key].upperLeft, self.hotBoxes[key].lowerRight, self.hotBoxes[key].color, 3)
                        self.hotBoxes[key].hits = self.hotBoxes[key].hits * .80
                    else:
                        delList.append(key)
            except: 
                pass
        for key in delList:
            del self.hotBoxes[key]        
        
#         for track in self.tracks:
#             try:
#                 pnt = np.zeros((3, 1))
#                 x = track[0][0]#*track[2]
#                 y = track[0][1]#*track[2]
#                 z = track[2]
#                 
#                 
#                 
#                 
#                 
#                 #if self.start.cam.is_visible_world(np.array(pnt)):
#                 camCoords = self.start.cam.world_to_image(x, y, z)
#                 u = int(camCoords[0][0])
#                 v = int(camCoords[1][0])
#                 #print camCoords
#                 
#                 
#                 if u > 100 and u < 640 and v > 100 and v < 352:
#                     cv2.circle(img, (u,v), 4, track[1])
#             except:
#                 pass
            
           
        cv2.imshow("Test", img)
        cv2.waitKey(30)

        
        
        
        
        
     
        
    










 
vid = Video()
vid.readFiles()
vid.addNextFrame()
start = DetectOverlap(vid)
vis = Visulize()


try:
    for i in vid.frames:
        
        start.threeD()
        start.overlapBoxes()
        currFrame = start.currFrame
             
        vis.showNext(vid.frames[currFrame-1])
except:
    plt.savefig("geoLocs.png")
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    #                             f1 = (self.heightAvg * self.heightCnt) / (self.heightCnt + 1)
#                             f2 = (dist[2] /(self.heightCnt + 1)) #* (1/(self.heightAvg[1] + 1))
#                             self.heightAvg = f1 + f2
#                             self.heightCnt += 1
#                             print self.heightAvg
#                             print dist[2]
#                             
#                             
#                             if self.heightCnt > 20:
#                                 if abs(dist[2]-self.heightAvg) < .5 * self.heightAvg:
 



