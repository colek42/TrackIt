# TrackIt
This is a project I did in college.  It uses HOG to track humans and plot them on a map.  What makes this interesting is that the ARDrone has no GPS, so locations need to be inferred.  I used LSD Slam for this.  Please see the paper in this repo for details.  https://github.com/colek42/TrackIt/blob/master/human-detection-localization.pdf

####Depends on
ROS Indigo/Ubuntu 14.04


Might need to install dependencies

> sudo apt-get install freeglut3-dev build-essential git  
> sudo apt-get install ros-indigo-joy ros-indigo-joystick-drivers  
  
-

DO NOT CHANGE DIRECTORIES INTO src.  The repository has the correct directory structure fpr the project.  By default git will create a new directory named TrackIt in whatever directory you run the command, if you want the project in a different directory simply add an argument with the desired directory name.

>git clone https://github.com/colek42/TrackIt.git  
>cd catkin/src  
>catkin_init_workspace  
>cd ..  
>rosdep update  
>rosdep install tum_ardrone  
>rosdep install tum_simulator  
>catkin make  


Shell Scripts to start nodes are included in the catkin directory.


