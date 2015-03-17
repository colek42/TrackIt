#CV 2.4.8
import cv2
import numpy as np
import glob







class ImageCorrection:
    def __init__(self, dim = (8,6), mtx = None, dist = None):
        # termination criteria
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        pattern_size = dim
        square_size = 1.0
        
        self.pattern_points = np.zeros( (np.prod(pattern_size), 3), np.float32 )
        self.pattern_points[:,:2] = np.indices(pattern_size).T.reshape(-1, 2)
        self.pattern_points *= square_size
        

        # Arrays to store object points and image points from all the images.
        self.obj_points = [] # 3d point in real world space
        self.img_points = [] # 2d points in image plane.

        
        self.matches = []
        
        self.mtx = mtx
        self.dist = dist
        self.dim = dim
        
        
        
    def findCorners(self, img):
        
        try:
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        except:
            print "No Image Yet"
            return False
        
        ret, corners = cv2.findChessboardCorners(gray, (self.dim),None)
        
        if ret == True:
            cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),self.criteria)


            self.matches.append(img)
    
            # Draw corners
            cv2.drawChessboardCorners(img, (self.dim), corners, ret)
            self.img_points.append(corners.reshape(-1, 2))
            self.obj_points.append(self.pattern_points)
            
            
            
            return img
        else:
            return False
        
    def getCorrectionMatrix(self):
        font = cv2.FONT_HERSHEY_SIMPLEX
        goodImgPnts = []
        goodObjPnts = []
        lastImg = None
        
        for idx, img in enumerate(self.matches):
            cv2.putText(img, 'Use This Image For Calibration Y/Any Key For No', (100,100), font, 1, (0,255,0), 2)
            cv2.drawChessboardCorners(img, (self.dim), self.img_points[idx], True)
            cv2.imshow('Select Images', img)
            k = cv2.waitKey(0) & 0xFF
            if k == 121 or k == 89:
                goodImgPnts.append(self.img_points[idx])
                goodObjPnts.append(self.obj_points[idx])
                lastImg = img
            else:
                pass

        h,  w = lastImg.shape[:2]
        
        ret, self.mtx, self.dist, rvecs, tvecs = cv2.calibrateCamera(goodObjPnts, goodImgPnts, (w, h))
         
        self.writeCalibrationFile()
               

    def calibratedImage(self, img, calibFile='/home/cole/test.npz'):
        try:
            if img == None:
                return
        except:
            pass
        
        data = np.load(calibFile)
        
        
        
        h,  w = img.shape[:2]
        newcameramtx, roi=cv2.getOptimalNewCameraMatrix(data['a'],data['b'],(w,h),1,(w,h))
        dst = cv2.undistort(img, data['a'], data['b'], None, newcameramtx)
                        
        return dst 
        
    def writeCalibrationFile(self, fn='/home/cole/test.npz'):
        if fn == None:
            fn = input('File Name: ')
            
        np.savez_compressed(fn, a=self.mtx, b=self.dist)

        
        
        