from sklearn import metrics
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import Imputer
import pickle
import cv2
import numpy as np
import os

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
        print(os.getcwd())
        
# Tests a video with the trained svm model.
    def ViewSVMVideo(self):
        videoName=raw_input("Enter name of video: ")
        hog = cv2.HOGDescriptor()
        model=pickle.load( open( "svm.p", "rb" ) )
        tmp=[]
        a=0
        while (a<=3779):
            print model.coef_[0][a]
            tmp.append([model.coef_[0][a]])
            print(str(a))
            a=a+1
        tmp.append([model.intercept_])    
        tmpNum=np.array(tmp)
        hog.setSVMDetector(tmpNum)
        #hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        hogParams = {'winStride': (8, 8), 'padding': (32, 32), 'scale': 1.05}
        cap = cv2.VideoCapture(videoName)
        while(True):
            ret, frame = cap.read()
            frameClone = frame.copy()
            if not ret:
                break
            result = hog.detectMultiScale(frame, **hogParams)
            print result
            for r in result[0]:
                fX, fY, fW, fH = r
                cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH), (0, 255, 0), 2)
            cv2.imshow('HOGframe',frameClone)
            #cv2.imshow("frame",frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
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
                tmpstr=tmpstr+"1,"+files[-11:-7]+"\n"
                DS.write(tmpstr)
                
                #DS.write(str(tmpp[0])[2:-1]+","+str(tmpp[1])[2:-1]+","+str(tmpp[2])[2:-1]+","+str(tmpp[3])[2:-1]+","+str(tmpp[4])[2:-1]+","+str(tmpp[5])[2:-1]+","+str(tmpp[6])[2:-1]+",1\n")
            elif files.endswith("n.jpeg"):
                tmpn=self.ImageHOGComputer(files,hog)
                print(tmpn) 
                negative=negative+1
                
                tmpstr=""
                for i in tmpp:
                    tmpstr=tmpstr+str(i)[2:-1]+","
                tmpstr=tmpstr+"0,"+files[-11:-7]+"\n"
                DS.write(tmpstr)
                
                #DS.write(str(tmpn[0])[2:-1]+","+str(tmpn[1])[2:-1]+","+str(tmpn[2])[2:-1]+","+str(tmpn[3])[2:-1]+","+str(tmpn[4])[2:-1]+","+str(tmpn[5])[2:-1]+","+str(tmpn[6])[2:-1]+",0\n")
            else:
                continue
        DS.close()
        print("The amount of positive images processed: "+str(positive)+"\nThe amount of negative images processed "+str(negative))

# Trains and test a SVM using the CSV file generated from the FeatureVectorCreator method.        
    def Train_And_Test(self):
        HOG_data=np.loadtxt('dataset.csv',delimiter=",")
        tmpdata=HOG_data[:,0:-2]
        target=HOG_data[:,-2]
        print(target)
        tmpdata[tmpdata==0]=np.nan
        imp=Imputer(missing_values='NaN',strategy='mean')
        data=imp.fit_transform(tmpdata)
        data_train,data_test,target_train,target_test=train_test_split(data,target,test_size=0.3)
        model=SVC(C=0.01,kernel='linear', class_weight='auto')
        model.fit(data_train,target_train)
        print(data_train)
        print(target_train)    
        opencv_data_train=np.float32(data_train)
        opencv_target_train=np.float32(target_train)     
        svm_params = dict( kernel_type = cv2.SVM_LINEAR,
                    svm_type = cv2.SVM_C_SVC,
                    C=2.67, gamma=5.383)
        svm = cv2.SVM()
        svm.train(opencv_data_train,opencv_target_train, params=svm_params)
        svm.save("hog_classifier.xml")  
        print(model)
        expected=target_test
        predicted=model.predict(data_test)
        target_names = ['Not Human', 'Human']
        
        #-------Currently a work in progress----Graphing the ROC Curve
        #target_train = label_binarize(target_train, classes=[0, 1, 2])
        #n_classes = target_train.shape[1]
        #classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True))
        #target_score = classifier.fit(data_train, target_train).decision_function(data_test)
        #fpr = dict()
        #tpr = dict()
        #roc_auc = dict()
        #for i in range(n_classes):
        #    fpr[i], tpr[i], _ = roc_curve(target_test[:, i], target_score[:, i])
        #fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
        
        #plt.figure()
        #plt.plot(fpr["micro"], tpr["micro"],
        #label='micro-average ROC curve (area = {0:0.2f})'
        #      ''.format(roc_auc["micro"]))
        #for i in range(n_classes):
        #    plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'
        #                           ''.format(i, roc_auc[i]))
        
        print(metrics.classification_report(expected,predicted,target_names=target_names))
        print(metrics.confusion_matrix(expected,predicted))
        print(metrics.roc_curve(expected,predicted))
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
# < = Decrease roi
# > = Increase roi         
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
                    saveName=filename[0:-4]+"_"+str(frameno)+"_p.jpeg"
                    cv2.imwrite(saveName,resized_roi)
                    break
# Save roi defined by mouse as a 64x128 JPEG image; Image name is annotated as a negative image.
                if k== ord('n') and visible:
                    print("Negative")
                    roi=frame[gy:gy+height,gx:gx+width]
                    resized_roi = cv2.resize(roi, (64,128))
                    saveName=filename[0:-4]+"_"+str(frameno)+"_n.jpeg"
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
            visible=False
            frameno=frameno+1
        cap.release()
        cv2.destroyAllWindows()

# Start of script.    
if __name__ == '__main__':
    inputDir=raw_input("Enter the name of the directory where the files are located for HOG Training:\n")
    HOGClass=H_O_G(inputDir)
    end=False
    print("=============================================================================================")
    print("HOGModule for use in ML training, creating samples, and extracting feature vectors.")
    print("----Options----")
    print("1. Display amount of samples in directory")
    print("2. Create feature vectors from samples in directory")
    print("3. Test and train the SVM")
    print("4. Create positive and negative samples from videos in given directory")
    print("5. View the svm model on a video.")
    print("6. Exit")
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
            end=True
        else:
            print("Not a valid command. Try Again...")
    print("Ending program...")
else:
    print 'HOGModule is being imported from another module.'
