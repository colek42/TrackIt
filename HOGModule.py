from sklearn import metrics
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import Imputer
from sklearn import cross_validation
import pickle
import cv2
import numpy as np
import os
from numpy import linalg
import xml.etree.ElementTree as ET

# Global variables.
gx=0
gy=0
visible=False
init=False

# Class that handles all aspects of training an HOG descriptor, from creating sample images to actual SVM training.
class H_O_G():

# Initializes the class.    
    def __init__(self, directory):
        self.currentDirectory=directory
        self.positiveSampleImages=0
        self.negativeSampleImages=0
        os.chdir(directory)
        self.homography = []
        print(os.getcwd())
        
# Tests a video with the trained svm model.
    def ViewSVMVideo(self):
        videoName=raw_input("Enter name of video: ")
        hog = cv2.HOGDescriptor()
        model=pickle.load( open( "svm.p", "rb" ) )
        svm = cv2.SVM()
        svm.load("hog_classifier.xml")
        tmpOCV=[]
        a=0
        while (a<=3779):
            tmpOCV.append([model.coef_[0][a]])
            a=a+1
        tmpOCV.append([model.intercept_])    
        tmpOCV=np.array(tmpOCV)
        
        tree = ET.parse('hog_classifier.xml')
        XMLarray = []
        sVectors=[]
        modXMLarray=[]
        rho=0
        for sp in tree.iter('_'):
            sVectors.append(sp.text)
        for element in tree.iter('rho'):
            rho=element.text
            rho=rho[0:10]+rho[-5:]
        XMLarray=sVectors[0].split()
        XMLarray.append(rho)
        for elements in XMLarray:
            modXMLarray.append(float(elements))
        tmpSKL=np.asarray(modXMLarray)
        tmpSKL=np.reshape(tmpSKL,(3781,1))
        
        print tmpSKL
        print cv2.HOGDescriptor_getDefaultPeopleDetector()
        print tmpOCV
        
        
#        hog.setSVMDetector(tmpSKL)
        hog.setSVMDetector(tmpOCV)
#        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        
        hogParams = {'winStride': (8, 8), 'padding': (32, 32), 'scale': 1.1}
        cap = cv2.VideoCapture(videoName)
        framenum=0
        while(True):
            text=open("video.txt","a")
            writeout=""
            ret, frame = cap.read()
            if not ret:
                break
            frameClone = frame.copy()
            result = hog.detectMultiScale(frame, **hogParams)
#            print result
            writeout=writeout+str(framenum)+" "
            for r in result[0]:
                fX, fY, fW, fH = r
                writeout=writeout+str(r)
                cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH), (0, 255, 0), 2)
            cv2.imshow('HOGframe',frameClone)
            writeout=writeout+"\n"
            print(writeout)
            text.write(writeout)
            framenum=framenum+1
            #cv2.imshow("frame",frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            text.close()
        print("Program finished...")
        cap.release()
        cv2.destroyAllWindows()
        
# Helper function for computing a image's feature vector using opencv's built in HOG feature vector compute function.
    def ImageHOGComputer(self,img, hog):
        tmpImg = cv2.imread(img)
        featureVector = hog.compute(tmpImg)
        return featureVector

# Displays the amount of positive and negative samples in the directory.    
    def HOGSampleCount(self):
        if (self.positiveSampleImages==0 and self.negativeSampleImages==0):
            positive=0
            negative=0
            for files in os.listdir(os.getcwd()):
    
                if files.endswith("p.jpeg"): 
                    positive=positive+1
                elif files.endswith("n.jpeg"):
                    negative=negative+1
                else:
                    continue
                self.positiveSampleImages=positive
                self.negativeSampleImages=negative
            print("The amount of positive samples in directory: "+str(positive))
            print("The amount of negative samples in directory: "+str(negative))
        else:
            print("The amount of positive samples in directory: "+str(self.positiveSampleImages))
            print("The amount of negative samples in directory: "+str(self.negativeSampleImages))

# Creates a CSV file in the working directory, which contains the feature vectors from all the sample images.            
    def FeatureVectorCreator(self):
        positive=0
        negative=0
        hog = cv2.HOGDescriptor()
        DS=open("Dataset.csv",'a')
        for files in os.listdir(os.getcwd()):
            if files.endswith("p.jpeg"):
                tmpp=self.ImageHOGComputer(files,hog)
                print(tmpp) 
                positive=positive+1
                tmpstr=""
                for i in tmpp:
                    tmpstr=tmpstr+str(i)[2:-1]+","
                tmpstr=tmpstr+"1\n"
                DS.write(tmpstr)
                
            elif files.endswith("n.jpeg"):
                tmpn=self.ImageHOGComputer(files,hog)
                print(tmpn) 
                negative=negative+1
                tmpstr=""
                for i in tmpp:
                    tmpstr=tmpstr+str(i)[2:-1]+","
                tmpstr=tmpstr+"0\n"
                DS.write(tmpstr)
                
            else:
                continue
        DS.close()
        print("The amount of positive images processed: "+str(positive)+"\nThe amount of negative images processed "+str(negative))

# Trains and test a SVM using the CSV file generated from the FeatureVectorCreator method.        
    def Train_And_Test(self):
        HOG_data=np.loadtxt('dataset.csv',dtype = str, delimiter=",")
        tmpdata = np.array(HOG_data[:,0:-1],dtype = float) #[float(i) for i in HOG_data[:,0:-5]] , -5 because of extra values added to dataset.csv (i.e (0 or 1, frame num, x and y values, and filename.))
        target= [float(i) for i in HOG_data[:,-1]] #loads the pos or neg value of 0 or 1. 
        target=np.asarray(target)
        tmpdata[tmpdata==0]=np.nan
        imp=Imputer(missing_values='NaN',strategy='mean')
        data=imp.fit_transform(tmpdata)
        
        final_model=SVC(C=0.01,kernel='linear', class_weight='auto')
        data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.20)
        cv = cross_validation.KFold(len(data_train), n_folds=5)
        results = []
        for traincv, testcv in cv:
            model=SVC(C=0.01,kernel='linear', class_weight='auto')
            model.probability=True
            model.fit(data_train[traincv], target_train[traincv])
            r=model.score(data_train[testcv], target_train[testcv], sample_weight=None)
            results.append(r)

        print(results)
        print(np.mean(results))
        final_model.fit(data_train,target_train)
        opencv_data_train=np.float32(data_train)
        opencv_target_train=np.float32(target_train)     
        svm_params = dict( kernel_type = cv2.SVM_LINEAR,
                    svm_type = cv2.SVM_C_SVC,
                    C=0.01)
        svm = cv2.SVM()
        svm.train(opencv_data_train,opencv_target_train, params=svm_params)
        svm.save("hog_classifier.xml")  
        
        print(final_model)
        expected=target_test
        predicted=final_model.predict(data_test)
        target_names = ['Not Human', 'Human']
        
        print(metrics.classification_report(expected,predicted,target_names=target_names))
        print(metrics.confusion_matrix(expected,predicted))
        pickle.dump(model, open( "svm.p", "wb" ) )

# Helper function that listens for mouse actions for the VideoCropperHelper method.    
    def draw_roi(self,event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            global gx,gy,visible,init
            gx=x
            gy=y
            visible=True
            init=True    

# Cycles through video files in working directory to extract positive and negative sample images from.        
    def VideoCropper(self):
        for files in os.listdir(os.getcwd()):
            print files
            if files.endswith(".mp4") or files.endswith(".wmv") or files.endswith(".avi") or files.endswith(".MOV"): 
                print(os.getcwd()+"\\"+files)
                self.VideoCropperHelper(os.getcwd()+"\\"+files)
            else:
                continue
            
# Helper function that processes video with openCV in order to extract positive and negative sample images.
# Keyboard Commands:
# =========================================================================================================
# V = Next frame
# N = Save as negative image
# SpaceBar = Save as positive image
# S = Save video image for Homography
# < = Decrease roi
# > = Increase roi 
# Q = Quit        
    def VideoCropperHelper(self,filename):
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
            width=64
            height=128
            ret, frame = cap.read()
            tmpframe=np.array(frame)
            tmpframe2=np.array(frame)
            cv2.setMouseCallback('frame',self.draw_roi)
            print("Frame number: "+ str(frameno))
            while(True):
                cv2.imshow('frame',tmpframe)

                if (visible):
                    if(init):
                        tmpframe=np.array(frame)
                        cv2.rectangle(tmpframe,(gx,gy),(gx+width,gy+height),(255,0,0),1)
                        init=False             
                k = cv2.waitKey(10) & 0xFF
# Save roi defined by mouse as a 64x128 JPEG image; Image name is annotated as a positive image.
                if k== ord(' ') and visible:
                    print("Positive")
                    roi=frame[gy:gy+height,gx:gx+width]
                    resized_roi = cv2.resize(roi, (64,128))
                    saveName=filename[0:-4]+"_"+str(frameno)+ "_" + str(gx)+ "_" + str(gy) + "_P.jpeg"
                    cv2.imwrite(saveName,resized_roi)
                    break
# Save Image for Homography 
                if k== ord('s') and visible:
                    print("Homography Image")
                    print(frame)
                    #roi=frame[gy:gy+height,gx:gx+width]
                    #resized_roi = cv2.resize(roi, (64,128))
                    saveName=filename[0:-4]+"_"+str(frameno)+".jpeg"  
                    cv2.imwrite(saveName, tmpframe2)                    
                    break                
# Save roi defined by mouse as a 64x128 JPEG image; Image name is annotated as a negative image.
                if k== ord('n') and visible:
                    print("Negative")
                    roi=frame[gy:gy+height,gx:gx+width]
                    resized_roi = cv2.resize(roi, (64,128))
                    saveName=filename[0:-4]+"_"+str(frameno)+ "_" + str(gx)+ "_" + str(gy) + "_N.jpeg"  
                    cv2.imwrite(saveName,resized_roi)                    
                    break
# Decreases size of roi.
                if k== ord(','):
                    if width>=16:
                        width=width-4
                        height=height-8
                        tmpframe=np.array(frame)
                        cv2.rectangle(tmpframe,(gx,gy),(gx+width,gy+height),(255,0,0),1)
                        print("Width:"+str(width)+ " Height:"+str(height))
# Increases size of roi.
                if k== ord('.'):
                    if width<=128:
                        width=width+4
                        height=height+8
                        tmpframe=np.array(frame)
                        cv2.rectangle(tmpframe,(gx,gy),(gx+width,gy+height),(255,0,0),1)
                        print("Width:"+str(width)+ " Height:"+str(height))
# Skips to next frame.
                if k== ord('v'):
                    break
# Exits out of video.            
                if k== ord('q'):
                    cap.release()
                    break
                         
            visible=False
            frameno=frameno+1
        cap.release()
        cv2.destroyAllWindows()
# code for selecting points on homography plane to get world coordinates. 
    def drawHom(self,event,x,y,flags,param):
        if event == cv2.EVENT_RBUTTONDOWN:
            global gx,gy,visible,init
            gx=x
            gy=y
            visible=True
            init=True
            
# code for creating homography from selecting 4 points on ground plane
    def HomographyGet(self):
        imageName= raw_input("Enter name of image: ") #VIRAT_S_000005_4758.jpeg
        image = cv2.imread(imageName,1)
        print("Select points in same order as they appear in corresponding world coordinate array.")
        print("Click left mouse to select point,[ h = add point to list," )
        print(" d = remove a point from the list, w = select first point")
        print("for distance measurement, e = select second point for distance")
        print("measurement, ' ' = compute homography, p = find world coord.,")
        print("q = quit]")
        cv2.namedWindow('frame')
        global xA,yA,x,y,gx,gy,visible,init
        gx="None"
        gy="None"
        x = "None"
        y = "None"
        xA = "None"
        yA = "None"
        visible=False
        init=False
        imagePoints = []
        while(image is not None):
            tmpframe=np.array(image)
            cv2.setMouseCallback('frame',self.draw_roi)
            while(True):
                
                cv2.imshow('frame',tmpframe)
                #if (visible):
                    #if(init):
                                     
                k = cv2.waitKey(10) & 0xFF
                if k== ord('h') and visible and init and gx != "None":
                    cv2.circle(tmpframe,(gx,gy),2,(255,0,0), 2) 
                    cv2.putText(tmpframe, str(gx)+','+str(gy),(gx+10,gy), 0, 1, (255,0,0), 1,8, False)
                    imagePoints.append((gx,gy))
                    init=False
                    print(imagePoints)

                if k== ord('p') and visible and init and gx!= "None":
                    cv2.circle(tmpframe,(gx,gy),2,(255,0,0), 2) 
                    coordText = HOGClass.PointToWorld((gx,gy))
                    cv2.putText(tmpframe, coordText,(gx+10,gy), 0, 1, (255,0,0), 1,8, False)
                    init=False
                    print(coordText)

                if k== ord('w') and visible and init and gx!= "None":
                    x,y = HOGClass.PointToWorld((gx,gy))
                    xA,yA = gx,gy
                    print("First coordinate selected.")
                    init=False
                    

                if k== ord('e') and visible and init and x != "None":
                    x2,y2 = HOGClass.PointToWorld((gx,gy))
                    #length = np.sqrt(np.sum(((x,y)-(x2,y2))**2))
                    #from scipy.spatial import distance
                    #a = (1,2,3)
                    #b = (4,5,6)
                    #dst = distance.euclidean(a,b)
                    a = np.array((x ,y, 1))
                    b = np.array((x2, y2, 1))
                    length = np.linalg.norm(a-b)
                    print(length)
                    #length2 = dist(a,b)
                    cv2.line(tmpframe,(xA,yA),(gx,gy),(255,0,0), 2) 
                    coordText = HOGClass.PointToWorld((gx,gy))
                    cv2.putText(tmpframe, str(length),(gx+10,gy), 0, 1, (255,0,0), 1,8, False)
                    #print(length)
                    x,y,gx,gy = "None"
                    xA = "None"
                    yA = "None"
                    init=False
                    #print(length)

                if k== ord(' ') and visible and len(imagePoints) >= 4:
                    print("Computing homography matrix of ground plane.")
                    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
                    #cv2.polylines(vis, [np.int32(tracked.quad)], True, (255, 255, 255), 2)
                    # Arrays to store object points and image points from all the images.
                    worldPoints = [(0,0),(0,6.62),(-1.1641,12.1343),(-20.7816,11.8242),(-37.7347,6.4976)] # 3d point in real world space taken from VIRAT provided data
                    #worldPoints = [ (12.2198,-2.5840),(9.4356,-2.456),(0,0),(0.1404,6.4885),(-7.245,6.7323)]#VIRAT_S_000100_homography.jpg
                    HOGClass.CalculateHom(imagePoints, worldPoints)

                if k== ord('d') and visible and len(imagePoints) >= 1:
                    print(imagePoints.pop(-1))
                    print(imagePoints)
                    print("Removed last chosen pixel homography point.")
                    
                if k== ord('q'):
                    image = None
                    break
        
        cv2.destroyAllWindows()

# code for calculating world coord from pixel coord

    def PointToWorld(self,imagePoint):
        if not self.homography == []:
            point = ([[imagePoint[0]],[imagePoint[1]],[1]])
            pointMatrix = np.array(point,dtype=np.float32)
            worldXyz = np.dot(self.homography, pointMatrix)
            world = np.divide(worldXyz,worldXyz[2])
            (worldX, worldY) = world[0],world[1]
            worldVal = str(world[0])+","+str(world[1])
            print(worldX,worldY)
            return (worldX, worldY)
        else:
            print("You must first create homography matrix.")
        
#code for calculating homography matrix
        
    def CalculateHom(self,imagePoints,worldPoints): # need to verify that the matrix is correct
        """Calculates the homography if there are 4+ point pairs"""
        n = len(imagePoints)
        # This calculation is from the paper, A Plane Measuring Device
        # by A. Criminisi, I. Reid, A. Zisserman.  For more details, see:
        # http://www.robots.ox.ac.uk/~vgg/presentations/bmvc97/criminispaper/
        A = np.zeros((n*2,8))
        B = np.zeros((n*2,1))
        for i in range(0,n):
            A[2*i][0:2] = imagePoints[i]
            A[2*i][2] = 1
            A[2*i][6] = -imagePoints[i][0]*worldPoints[i][0]
            A[2*i][7] = -imagePoints[i][1]*worldPoints[i][0]
            A[2*i+1][3:5] = imagePoints[i]
            A[2*i+1][5] = 1
            A[2*i+1][6] = -imagePoints[i][0]*worldPoints[i][1]
            A[2*i+1][7] = -imagePoints[i][1]*worldPoints[i][1]
            B[2*i] = worldPoints[i][0]
            B[2*i+1] = worldPoints[i][1]
        
        X = linalg.lstsq(A,B)
        H = np.reshape(np.vstack((X[0],[1])),(3,3))
        
##        while len(worldPoints) > 4:
##            worldPoints.pop(-1)
##        while len(imagePoints) > 4:
##            imagePoints.pop(-1)
        src = np.array(imagePoints, dtype=np.float32)      # using cv2.getPerspectiveTransform returns the same matrix as the code above but only allows for 4 points.
        dest = np.array(worldPoints, dtype=np.float32)
        #H2 = cv2.getPerspectiveTransform(src, dest)
        H3, mask = cv2.findHomography(src, dest, cv2.RANSAC,5.0)
        self.homography = H3
        origin = ([[imagePoints[0][0]],[imagePoints[0][1]],[1]])
        originMatrix = np.array(origin,dtype=np.float32)
        #print(originMatrix)
        originTest = np.dot(H3, originMatrix)
        originTestXY = np.divide(originTest,originTest[2])
        #print(originTestXY)
        print("Selected Image pixel coordinates.")
        print(src)
        print("Static world coordinates.")
        print(dest)
##        print("Matrix H derived from method provided by http://www.robots.ox.ac.uk/~vgg/presentations/bmvc97/criminispaper/.")
##        print(H)
##        print("Matrix H2 derived from cv2.getPerspectiveTransform(src, dest).")
##        print(H2)
        print("Matrix H3 derived from cv2.findHomography(src, dest, cv2.RANSAC,5.0).")
        print(H3)
        print("Image coordinates of world origin.")
        print(originMatrix)
        print("Origin test result world coordinates.")
        print(originTestXY)
        #return H3
        
# Displays all possible functions for users.        
    def DisplayOptions(self):
        print("=============================================================================================")
        print("HOGModule for use in ML training, creating samples, and extracting feature vectors.")
        print("----Options----")
        print("1. Display amount of samples in directory")
        print("2. Create feature vectors from samples in directory")
        print("3. Test and train the SVM")
        print("4. Create positive and negative samples from videos in given directory")
        print("5. View the svm model on a video.")
        print("6. Create Homography")
        print("7. Exit")
        
# Start of script.    
if __name__ == '__main__':
    inputDir=raw_input("Enter the name of the directory where the files are located for HOG Training:\n")
    HOGClass=H_O_G(inputDir)
    end=False
    HOGClass.DisplayOptions()
    while(not end):
        inputCommand=raw_input("Ready for command: ")
        if (inputCommand=="1"):
            HOGClass.HOGSampleCount()           
        elif(inputCommand=="2"):
            HOGClass.FeatureVectorCreator()
        elif(inputCommand=="3"):
            HOGClass.Train_And_Test()
        elif(inputCommand=="4"):
            HOGClass.VideoCropper()
        elif(inputCommand=="5"):
            HOGClass.ViewSVMVideo()
        elif(inputCommand=="6"):
            HOGClass.HomographyGet()
        elif(inputCommand=="7"):
            end=True
        else:
            print("Not a valid command. Try Again...")
        HOGClass.DisplayOptions()
    print("Ending program...")
else:
    print 'HOGModule is being imported from another module.'
