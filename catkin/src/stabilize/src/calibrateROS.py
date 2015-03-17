#!/usr/bin/env python
import cv2
from cv2 import cvtColor
cv2.namedWindow("Image", flags=cv2.WINDOW_NORMAL)
import roslib
import traceback
from sys import argv
from gtk._gtk import Frame
roslib.load_manifest('stabilize')
import sys
import rospy
import sensor_msgs.msg
from cv_bridge import CvBridge, CvBridgeError
import threading
from kalmanFilt import KalFilter
import numpy as np
import cv
import sensor_msgs.msg
from time import sleep
from humanDetect import findHumans
from denseFlow import DenseFlow
from flow2 import Flow
import calibrate

def threaded(fn):
    def wrapper(*args, **kwargs):
        threading.Thread(target=fn, args=args, kwargs=kwargs).start()
    return wrapper



class calib:
    def __init__(self):
        self.bridge = CvBridge()

        self.detect_interval = 4
        self.tracks = []
        self.frame_idx = 0
        self.timerx = 0
        self.prev = None
        self.i=0
        self.p0 = None
        self.use_ransac = True
        self.me = 0
        self.frame2 = None
        self.image = None
        self.final = None
        self.hello = None
        self.features1 = None
        self.features2 = None
        self.featuresLast = None
        self.guess = 0
        fourcc=cv2.cv.CV_FOURCC('I', 'Y', 'U', 'V')
        self.vw = None
        rospy.Subscriber("ardrone/front/image_raw" , sensor_msgs.msg.Image, self.CallBack)
        self.imgCorr = calibrate.ImageCorrection()
        
        
        
        self.gotImage()
        
        

    def rec(self):
        if self.i > 100:
            fourcc=cv2.cv.CV_FOURCC('I', 'Y', 'U', 'V')
            self.vw = cv2.VideoWriter('/home/cole/calibrate.avi', fourcc , 30, (self.frame.shape[:2][1], self.frame.shape[:2][0]))
        else:
           self.rec()
        
        
    def CallBack(self, image):
        self.image = self.bridge.imgmsg_to_cv2(image)
        
        
    def gotImage(self):
        #self.rec() 
        sleep(3)
        numImg = 0
        stop = False
        
        while not stop:
            ret = self.imgCorr.calibratedImage(self.image)
            
            
            ret = self.imgCorr.findCorners(self.image)
            try:
                if ret != False:
                    k = cv2.waitKey(10) & 0xFF
                    if k == 32:
                        cv2.destroyWindow('Image') 
                        stop = True

            except:
                print "Image Here"
                numImg +=1
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(ret, 'Space To Stop', (100,100), font, 2, (0,0,0), 5)
                cv2.imshow('Image',ret)
                k = cv2.waitKey(1) & 0xFF
                if k == 32:
                    cv2.destroyWindow('Image') 
                    stop = True
        self.imgCorr.getCorrectionMatrix()
            

#         elif tries < 1000:
#             tries +=1
#             self.gotImage(tries)
        
def main():
    rospy.init_node('sabilize', anonymous=True)
    start = calib()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print "Shutting down on node."

if __name__ == '__main__':

    main()         
            
            
        
        