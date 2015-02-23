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


def threaded(fn):
    def wrapper(*args, **kwargs):
        threading.Thread(target=fn, args=args, kwargs=kwargs).start()
    return wrapper


class stabilize:
    def __init__(self):
        self.bridge = CvBridge()
        self.track_len = 20
        self.detect_interval = 4
        self.tracks = []
        self.frame_idx = 0
        self.timerx = 0
        self.prev = None

        self.p0 = None
        self.use_ransac = True
        self.me = 0
        self.frame2 = None
        self.image = None
        self.kf = kalman()
        self.final = None
        self.hello = None
        #self.img = cv2.VideoCapture('Feb162015.mp4')

        self.featuresLast = None
        self.guess = 0
        fourcc=cv2.cv.CV_FOURCC('I', 'Y', 'U', 'V')
        self.vw = cv2.VideoWriter('/home/cole/outkalman2.avi', fourcc , 30, (20, 20))
        rospy.Subscriber("ardrone/front/image_raw" , sensor_msgs.msg.Image, self.CallBack)
        self.gotImage()
        
        self.rec()
        
        ##Set Up Recorder
        
    
    @threaded
    def rec(self):
        try:
            fourcc=cv2.cv.CV_FOURCC('I', 'Y', 'U', 'V')
            self.vw = cv2.VideoWriter('/home/cole/outkalman2.avi', fourcc , 30, (self.frame.shape[:2]))
        except:
            self.rec()
        
        
     

        
        
        
        
        
        #imu = Subscriber("/tf", sensor_msgs.msg.Imu)
#         self.ts = ApproximateTimeSynchronizer([self.img, self.tf], 10, (1.0/30))
#         self.ts.registerCallback(self.gotImage)
        

    def CallBack(self, image):
        img = self.bridge.imgmsg_to_cv2(image)
        #img = cv2.VideoCapture('Feb162015.mp4')


        self.image= cv2.copyMakeBorder(img,250,250,1000,1000,cv2.BORDER_CONSTANT,value=(0,0,0))   
    
    #@threaded
    def gotImage(self):

        i=0
        while True:
           
            
            #self.image= cv2.copyMakeBorder((self.img.read()[1]),250,250,700,700,cv2.BORDER_CONSTANT,value=(0,0,0))
            #self.image = cv2.resize(self.image, None, np.size(self.image), 0.5, 0.5, cv.CV_INTER_AREA);   
            
            
            
            
            
            
            if self.image != None:
                try:
#                      
                    show = self.find_motion_vector(self.image)
                    if self.hello == None:
                        self.hello = show
                    if i%100 == 0:
                    
                        print i
                    #grey =  cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
                    
                    if self.final == None:
                        self.hello = show
                        self.final = show

                    
                    ret,self.final = cv2.threshold(self.final,0,20,cv2.THRESH_TOZERO)
                    
                    self.final = cv2.addWeighted(self.hello, .9, show, .1, 0)
                    tmp = cvtColor(self.final, cv2.COLOR_BGR2GRAY)
                    tmp2 = np.float32(tmp)
                  
                    
                    #self.final=(self.final * .5)
                    self.hello = np.where(show==0, self.final, show)
                         
#                     self.final = cv2.addWeighted(self.final, .1, show, .9, 0)
#                     self.final = cv2.overlayImage(self.final, .9, old, .1, 0)
                    #self.final = cv2.add(self.final, show, .5)                
                    #    self.imLst = self.imLst[0:-40]
                    
                        #for idx, pic in enumerate(self.imLst):
                            
                            
                            #self.final = cv2.addWeighted(self.final, 0, self.imLst[idx+5], 0, 0)
                            
                    #else: self.final = show
                    
                    b = self.vw.write(self.hello)
                    cv2.imshow("Image", self.hello)
                    
                    #cv2.imshow("Image", mask)
                    
                    cv2.waitKey(1)
                    i+=1
                except:
                    print traceback.format_exc()
                    self.final = None  

                    self.frame2 = None
                    self.featuresLast = None

                    
#         try:
#             t = self.tf.getLatestCommonTime('/cam_front', '/map')
#             (trans,rot) = self.tf.lookupTransform('/cam_front', '/map', t)
#             print trans
#         except:
#             print 'pass'
# 
#         
# 
# 
#         
#         
#         #self.control()
# #        except:
# #            pass
#         #cv2.imshow('flow', draw_flow(self.gray, self.flow))
# 
# 
# 
#     def stabil(self):
#         pass
# 
# 
    def find_motion_vector(self, imCurr):
        LK_WINDOW_SIZE = (7, 7)
        LK_MAX_LEVEL = 200
        LK_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        GF_MAX_CORNERS = 2000
        GF_QUALITY_LEVEL = 0.1
        GF_MIN_DISTANCE = 5
        GF_BLOCK_SIZE = 7
         
         
        """given 2 contiguous images in an image stream, returns the motion vector \
         between them"""
          
        self.frame = cv2.cvtColor(imCurr, cv2.COLOR_BGR2GRAY)
          
        if self.frame2 == None:
            self.frame2 = self.frame.copy()

        rows, cols = self.frame.shape[:2]
#         if self.featuresLast != None:  
#             if self.featuresLast.size >= 10:
#                 features1 = self.featuresLast
#             else:
#                 features1 = cv2.goodFeaturesToTrack(self.frame, GF_MAX_CORNERS, GF_QUALITY_LEVEL, GF_MIN_DISTANCE, GF_BLOCK_SIZE)
#         else:
#             features1 = cv2.goodFeaturesToTrack(self.frame, GF_MAX_CORNERS, GF_QUALITY_LEVEL, GF_MIN_DISTANCE, GF_BLOCK_SIZE)
# #                 
# #                     
# #             #return numpy.array([0.0, 0.0])
#         #features1 = cv2.goodFeaturesToTrack(self.frame, GF_MAX_CORNERS, GF_QUALITY_LEVEL, GF_MIN_DISTANCE, GF_BLOCK_SIZE)
#         features2, st, err = cv2.calcOpticalFlowPyrLK(self.frame2, self.frame, features1, \
#             nextPts=None, winSize=LK_WINDOW_SIZE, maxLevel=LK_MAX_LEVEL, \
#             criteria=LK_CRITERIA)
# 
#         if features2.size >= 8:  #if we dont have enough point to track dont even try.
#         #delete bad features
#             for idx, row in enumerate(features2):
#                 if (st[idx][0]) != 1:
#                     #cv2.circle(imCurr, (int(row[0][0]), int(row[0][1])), 2, red, -1)
#                     #print "circle"
#                     try:                
#                         features1 = np.delete(features1, (idx), axis=0)
#                     except:
#                         pass
#                     
#                     try:               
#                         features2 = np.delete(features2, (idx), axis=0)
#                     except:
#                         pass
                    
                
            
            
            #if features2.size < 10:
            #    trans = self.transLast
            #else:
            #trans = cv2.estimateRigidTransform((features2), (features1), fullAffine=False)
        
        
#         H = cv2.findHomography(features1, features2, cv.CV_RANSAC, 2) 
#         if np.sum(H[1]) < 20:
#             print "No Transform Found!"
#             self.final = None  
#             self.frame2 = None
#             self.featuresLast = None
#             return
#         
#         trans = H[0]#cv2.estimateRigidTransform(H, features2, fullAffine=False)
        
        #features2 = self.featuresLast
            
        n=2
        for i in range(0, n):
            trans = cv2.estimateRigidTransform(self.frame, self.frame2, fullAffine=False)
            if trans is None:
                break
             
            if i < n-1:
                self.frame2 = cv2.warpAffine(self.frame, trans, (cols, rows))
        
        if trans != None:
            self.kf.newValue(trans)  ##if we know we have bad data we to not want to add it to our kalman set
        else:
            print "No Transform Found - Guessing!"
            self.guess +=1
            if self.guess > 5:
                print "Too Many Guesses, Re Centering Image and Starting Over"
                self.final = None  
                self.frame2 = None
                self.featuresLast = None
                self.guess = 0
#     
#         else:
#             print "Not very many Points"
#             self.final = None  
#             self.frame2 = None
#             self.featuresLast = None
#       
#         
#         try:
    #self.transLast = trans
    #self.featuresLast = features2
           
    

        
    
        filtTrans = np.float32(self.kf.getValue())

        ret = cv2.warpAffine(imCurr, filtTrans, (cols, rows))
    #         mask = self.bs.apply(ret)
    #         mask = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
        return ret
            
            
            #print 'it Worked!'
#         except:
#             return imCurr
#             except:
#                 ret = cv2.warpAffine(imCurr, trans, (cols, rows))
                 
#         except:
#             print "e"
#             ret = cv2.warpAffine(imCurr, trans, (cols, rows))
     
        
        #return ret #ret.copy()            
            
            

        

        
        
        
        
        
        
        
        
        
        
        
        #trans = cv2.getAffineTransform(features2, features1)
         
         
        #good_features2 = features2[st==1]
        #good_features1 = features1[st==1]
         
        #diff = good_features1 - good_features2
         
        # if no correspondences are found:
        #if len(diff) == 0: 
            #
            #print 'none2'
            #return numpy.array([0.0, 0.0])
         





class kalman():
    def __init__(self):
        self.A = list()
        for i in range(0, 2):
            self.A.append([])
            for j in range(0, 3):
                self.A[i].append(KalFilter(0,0))
       
        

            
    def newValue(self, M):
                
        for i in range(0, 2):
            for j in range(0, 3):
                self.A[i][j].input_latest_noisy_measurement(M[i][j])
                

                

    
    def getValue(self):
        M = list()
        
        
        for i in range(0, 2):
            M.append(list())
            for j in range(0, 3):
                M[i].append(self.A[i][j].posteri_estimate)
                

        return np.float32(np.array(M))
        
        











def main():
    rospy.init_node('sabilize', anonymous=True)
    start = stabilize()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print "Shutting down on node."

if __name__ == '__main__':
   
    main()  

        #print np.mean(diff, axis=0, dtype=np.float32)
        #return
    
    
#     def features(self):
#         
#         frame = self.cv_image
#         row, col = frame.shape[:2]
#         frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         self.vis = frame.copy()
#         if self.p0 is not None:
#             p2, trace_status = checkedTrace(self.gray1, frame_gray, self.p1)
# 
#             self.p1 = p2[trace_status].copy()
#             self.p0 = self.p0[trace_status].copy()
#             self.gray1 = frame_gray
# 
#             if len(self.p0) < 9:
#                 self.p0 = None
#                 return
#             H, status = cv2.findHomography(self.p0, self.p1, (0, cv2.RANSAC)[self.use_ransac], 15.0)
#             h, w = frame.shape[:2]
#             overlay = cv2.warpPerspective(self.frame0, H, (w, h))
#             self.vis = cv2.addWeighted(self.vis, 0.5, overlay, 0.5, 0.0)
#             
#             for (x0, y0), (x1, y1), good in zip(self.p0[:,0], self.p1[:,0], status[:,0]):
#                 if good:
#                     cv2.line(self.vis, (x0, y0), (x1, y1), (0, 128, 0))
#                 cv2.circle(self.vis, (x1, y1), 2, (red, green)[good], -1)
#             draw_str(self.vis, (20, 20), 'track count: %d' % len(self.p1))
#             if self.use_ransac:
#                 draw_str(self.vis, (20, 40), 'RANSAC')
#                 self.me = 0
#         else:
#             p = cv2.goodFeaturesToTrack(frame_gray, **feature_params)
#             if p is not None:
#                 for x, y in p[:,0]:
#                     cv2.circle(self.vis, (x, y), 2, green, -1)
#                 draw_str(self.vis, (20, 20), 'feature count: %d' % len(p))
# 
#         cv2.namedWindow("Window", flags=cv2.WINDOW_OPENGL)
#         cv2.imshow("Window", self.vis)
# 
#         ch = cv2.waitKey(1)
#         if ch == 27:
#             return
#         self.me += 1
#         print self.me
#         if self.me > 50  and self.p0 is None: ##Change this to acceleration trigger
#         #if ch == ord(' '):
#             print "Space"
#             self.frame0 = frame.copy()
#             self.p0 = cv2.goodFeaturesToTrack(frame_gray, **feature_params)
#             if self.p0 is not None:
#                 self.p1 = self.p0
#                 self.gray0 = frame_gray
#                 self.gray1 = frame_gray
#             self.me = 0
#                 
#         if ch == ord('r'):
#             self.use_ransac = not self.use_ransac
#             print "R"


            #this = centroid(p[0,:])
            #for i in this:
            #    try:
            #        cv2.circle(self.vis, (int(i[0]), int(i[1])), 2, red, -1)
            #        #cv2.circle(self.vis, (int(i[2]), int(i[3])), 2, red, -1)
            #    except:
            #        pass
            
                





#                 tri = Delaunay(p[:,0])
#                 
#                 x1=0
#                 x2=0
#                 y1=0
#                 y2=0
#                 itr=0
#                 for x, y in p[:,0]:
#                     if itr % 2 == 0:
#                         x1 = x
#                         y1 = y
#                     if itr % 2 != 0:
#                         x2 = x
#                         y2 = y
#                     cv2.line(self.vis, (x1, y1), (x2, y2), red)
#                     itr+=1

                
                
            
            
        
        #cv2.waitKey(2)

    

    #def control(self):


    
    
    
    
    
    
    
    
    
#     def run2(self):
#         #if time() - self.timerx < .05: return
#         
#         
#         if self.prev is None:
#             self.prev = self.cv_image
#         prevgray = cv2.cvtColor(self.prev, cv2.COLOR_BGR2GRAY)
#         self.hsv = np.zeros_like(self.prev)
#         self.hsv[...,1] = 255
#     
# 
#         img = self.cv_image
#         self.gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         self.flow = cv2.calcOpticalFlowFarneback(prevgray, self.gray, 0.5, 3, 15, 3, 5, 1.2, 0)
#         mag, ang = cv2.cartToPolar(self.flow[...,0], self.flow[...,1])
#         self.hsv[...,0] = ang*180/np.pi/2
#         self.hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
#         self.rgb = cv2.cvtColor(self.hsv,cv2.COLOR_HSV2BGR)
#         
#         angle = ang.mean()
#         magnitude = mag.mean()        
#         
#         
#         K=0
#         
#         rows, cols = self.gray.shape
#         
#         M = cv2.getRotationMatrix2D((cols/2,rows/2),angle, 1)
#         print magnitude
#         
#         dst = cv2.warpAffine(self.gray,M,(cols,rows))        
#         
#         
#     
#         
#         
#         self.prevgray = self.gray
#         cv2.imshow('flow', dst)
#         #cv2.imshow('flow2', draw_flow(self.gray, self.flow))
#         cv2.waitKey(1)
#         #self.timerx = time()

    


    
#             ch = 0xFF & cv2.waitKey(5)
#             if ch == 27:
#                 break
#             if ch == ord('1'):
#                 show_hsv = not show_hsv
#                 print 'HSV flow self.visualization is', ['off', 'on'][show_hsv]
#             if ch == ord('2'):
#                 show_glitch = not show_glitch
#                 if show_glitch:
#                     cur_glitch = img.copy()
#                 print 'glitch is', ['off', 'on'][show_glitch]
#         cv2.destroyAllWindows()
        
        
        
        
        


# 
# 
# 
# def draw_flow(img, flow, step=16):
#     h, w = img.shape[:2]
#     y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1)
#     fx, fy = flow[y,x].T
#     lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
#     lines = np.int32(lines + 0.5)
#     self.vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
#     cv2.polylines(self.vis, lines, 0, (0, 255, 0))
#     for (x1, y1), (x2, y2) in lines:
#         cv2.circle(self.vis, (x1, y1), 1, (0, 255, 0), -1)
#     return self.vis
# 
# def draw_hsv(flow):
#     h, w = flow.shape[:2]
#     fx, fy = flow[:,:,0], flow[:,:,1]
#     ang = np.arctan2(fy, fx) + np.pi
#     v = np.sqrt(fx*fx+fy*fy)
#     hsv = np.zeros((h, w, 3), np.uint8)
#     hsv[...,0] = ang*(180/np.pi/2)
#     hsv[...,1] = 255
#     hsv[...,2] = np.minimum(v*4, 255)
#     bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
#     return bgr
# 
# def warp_flow(img, flow):
#     h, w = flow.shape[:2]
#     flow = -flow
#     flow[:,:,0] += np.arange(w)
#     flow[:,:,1] += np.arange(h)[:,np.newaxis]
#     res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
#     return res

     


