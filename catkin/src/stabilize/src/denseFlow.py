import cv2
import numpy as np





class DenseFlow:
    def __init__(self):
        pass

    def getFlow(self, prvs, next):
        #hsv = np.zeros_like(prvs[0])
        #hsv[...,1] = 255
        #ret, frame2 = cap.read()
        #next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
        hsv = np.zeros((610, 1340, 3), np.uint8)  ##change
        
#         prvs = cv2.buildOpticalFlowPyramid(prvs, (20,20), 5)
#         prvs = cv2.buildOpticalFlowPyramid(next, (20,20), 5)
#         
        prvs = cv2.cvtColor(prvs, cv2.COLOR_BGR2GRAY)
        next = cv2.cvtColor(next, cv2.COLOR_BGR2GRAY)

        
        flow = cv2.calcOpticalFlowFarneback(prvs, next, 0.5, 1, 4, 8, 7, 1.5, 0)


     
    
     
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        hsv[...,0] = ang*180/np.pi/2
        hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
     
        cv2.imshow("Flow", rgb)
        return rgb
        
       