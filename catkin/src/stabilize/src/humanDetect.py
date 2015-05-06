import numpy as np
import cv2








class findHumans:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
        
        #self.cap = cv2.VideoCapture("""Video file name/directory goes here""")
        self.fgbg = cv2.BackgroundSubtractorMOG()
        self.kernel = np.ones((1,1),np.uint8)

    def getBox(self, frame):
        fgmask = self.fgbg.apply(frame, learningRate=1.0/5)
        opening = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, self.kernel)
        opening= cv2.dilate(opening,self.kernel)
        opening= cv2.dilate(opening,self.kernel)
        opening= cv2.dilate(opening,self.kernel)
        opening= cv2.dilate(opening,self.kernel)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret,thresh1 = cv2.threshold(opening,170,255,cv2.THRESH_BINARY)
        ret1,thresh2=cv2.threshold(opening,170,255,cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        cnts = sorted(contours, key = cv2.contourArea, reverse = True)[0:10]
        cv2.drawContours(frame, cnts, -1, (0,0,255), 2)
        return frame
    
    