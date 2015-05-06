import numpy as np
import cv2
import cv2.cv as cv
from common import anorm2, draw_str
from time import clock
import math
import traceback


help_message = '''
USAGE: lk_track.py [<video_source>]

Keys:
  1 - toggle old/new CalcOpticalFlowPyrLK implementation
  SPACE - reset features
'''
class Flow():
    def __init__(self):
        self.tracks = []
        self.itr = 0
        
        self.lk_params = dict( winSize  = (15, 15), 
                          maxLevel = 5, 
                          criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
                          #derivLambda = 0.0 )    
        
        self.feature_params = dict( maxCorners = 500, 
                               qualityLevel = .1,
                               minDistance = 5,
                               blockSize = 10 )
        
    def calc_flow_old(self, img0, img1, p0):
        p0 = [(x, y) for x, y in p0.reshape(-1, 2)]
        h, w = img0.shape[:2]
        img0_cv = cv.CreateMat(h, w, cv.CV_8U)
        img1_cv = cv.CreateMat(h, w, cv.CV_8U)
        np.asarray(img0_cv)[:] = img0
        np.asarray(img1_cv)[:] = img1
        t = clock()
        features, status, error  = cv.CalcOpticalFlowPyrLK(img0_cv, img1_cv, None, None, p0, 
            self.lk_params['winSize'], self.lk_params['maxLevel'], self.lk_params['criteria'], 0, p0)
        return np.float32(features), status, error, clock()-t

    def flow(self, old, new):
        self.itr +=1
        track_len = 500
        

        old_mode = False
        mask = cv2.resize(new, None, np.size(new), .85, .85, cv.CV_INTER_AREA)
        frame = new
        prev_frame = old
        vis = frame.copy()
        if len(self.tracks) > 0:
            p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
            img0 = cv2.cvtColor(prev_frame, cv.CV_BGR2GRAY)
            img1 = cv2.cvtColor(frame, cv.CV_BGR2GRAY)
            if old_mode:
                p1,  st, err, dt = self.calc_flow_old(img0, img1, p0)
            else:
                t = clock()
                p1,  st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **self.lk_params)
                dt = clock()-t
            
            for tr, (x, y) in zip(self.tracks, p1.reshape(-1, 2)):
                tr.append((x, y))
                delTr = False
                for tup in tr:
                    cnt=0
                    try:
                        if np.count_nonzero(mask[int(tup[1])][int(tup[0])]) == 0:
                           cnt+=1
                           if cnt > 1:
                               #delTr = True
                               self.delTracks(tr)
                               break
                           
                    except:
                        delTr = True
                if delTr == True:
                    self.delTracks(tr)
                    del tr
                    
                    continue

                if len(tr) > track_len:
                    #self.delTracks(tr)
                    del tr[0]
                    
                if self.pruneTracks(tr) == True:
                    self.delTracks(tr)
                    del tr
            
                
            


#                             
                         
                
                
                else:
                    if len(tr) > 15:
                        cv2.circle(vis, (x, y), 2, (0, 0, 255), -1)
                        cv2.polylines(vis, [np.int32(tr)], False, (255, 0, 0))
                
                    
                #cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)

            draw_str(vis, (20, 20), ['new', 'old'][old_mode]+' mode')
            draw_str(vis, (20, 40), 'time: %.02f ms' % (dt*1000))


        #cv2.imshow('lk_track', vis)
        if len(self.tracks) == 0 or self.itr == 30:
            self.itr = 0
            gray = cv2.cvtColor(frame, cv.CV_BGR2GRAY)
            p = cv2.goodFeaturesToTrack(gray, **self.feature_params)
            p = [] if p is None else p.reshape(-1, 2)
            #self.tracks = []
            for x, y in np.float32(p):
                self.tracks.append([(x, y)])
        return vis
    
    
    
    
    
    def delTracks(self, tr):
        for idxi, i in enumerate(self.tracks):
            for idxj, j in enumerate(i):
                if j in tr:
                    self.tracks.pop(idxi)
                    break
        
    
    
    
    
    
    
    
    
    def pruneTracks(self, tr):
        try:
            length = 5
            n1 = abs(tr[0][0] - tr[-1][0])
            n2 = abs(tr[0][1] - tr[-1][1])
            n3 = math.sqrt(n1**2+n2**2)
            length = length + n3
            if length < len(tr):  #*(100/tr[-1][1]):

                return True
            else:
                return False
        except:
            pass

        

