#Environment Setup Instructions

##Set Up Ports for NAT is using VM

--add image here

    UDP 5556
    UDP 5554
    UDP 5555 ArDrone v1
    TCP 5555 ArDrone v2
    TCP 5559

##Install Dependencies and other neccisities

    sudo apt-get install freeglut3-dev build-essential git

    
##Install ROS

    sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu trusty main" > /etc/apt/sources.list.d/ros-latest.list'
    sudo apt-get update
    sudo apt-get install ros-indigo-desktop-full
    sudo rosdep init
    rosdep update

    echo "source /opt/ros/indigo/setup.bash" >> ~/.bashrc
    source ~/.bashrc

    sudo apt-get install python-rosinstall

#Note about sourcing the workspace

Everytime you want to run a command that is in the workpace you need to source
the environment with the following command.

    source devel/setup.bash



#Refer to readme.md for AR_Drone setup instructions 


