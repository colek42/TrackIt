"""
This program takes an input directory that contains video files of the sample data that needs to be processed for HOG Training.
All positive and negative images will be created in the same directory as the videos.
Positive images are noted with _p/_n for positive and negative respectively, and _f, where f is the frame number the image was derived from.
 """

import cv2
import numpy as np
import os

# Global Variables 
gx=0
gy=0
visible=False
init=False

# Function to get positive and negative images from directory of videos
def VideoCropper(filename):
    cv2.namedWindow('frame')
    cap = cv2.VideoCapture(filename)
    global gx,gy,visible,init
    gx=0
    gy=0
    visible=False
    init=False
    frameno=0
    jmpframe=int(raw_input("Enter frame number to start(0 to start at beginning):"))     
    while (frameno!=jmpframe):
        ret,frame=cap.read()
        frameno=frameno+1
    
    while(cap.isOpened()):
        ret, frame = cap.read()
        tmpframe=np.array(frame)
        cv2.setMouseCallback('frame',draw_roi)
        print("Frame number: "+ str(frameno))
        while(True):
            cv2.imshow('frame',tmpframe)

            if (visible):
                if(init):
                    tmpframe=np.array(frame)
                    cv2.rectangle(tmpframe,(gx,gy),(gx+64,gy+128),(255,0,0),1)
                    init=False
                    
            if cv2.waitKey(10) & 0xFF == ord(' ') and visible:
                print("Positive")
                roi=frame[gy:gy+128,gx:gx+64]
                saveName=filename[0:-4]+"_"+str(frameno)+"_p.jpeg"
                cv2.imwrite(saveName,roi)
                break
            if cv2.waitKey(20) & 0xFF == ord('n') and visible:
                print("Negative")
                roi=frame[gy:gy+128,gx:gx+64]
                saveName=filename[0:-4]+"_"+str(frameno)+"_n.jpeg"
                cv2.imwrite(saveName,roi)
                break
        visible=False
        frameno=frameno+1
        
    cap.release()
    cv2.destroyAllWindows()
    
# Mouse callback event for getting area to be cropped
def draw_roi(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        global gx,gy,visible,init
        gx=x
        gy=y
        visible=True
        init=True        
        
# Main start of program        
inputDir=raw_input("Enter the name of the directory where the video files are located:\n")
os.chdir(inputDir)
print(os.getcwd())
for files in os.listdir(os.getcwd()):
    print files
    if files.endswith(".mp4") or files.endswith(".wmv") or files.endswith(".avi") or files.endswith(".MOV"): 
        print(os.getcwd()+"\\"+files)
        VideoCropper(os.getcwd()+"\\"+files)
    else:
        continue
