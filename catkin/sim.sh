
#!/bin/bash

##Set Working Directory To Location of Script
script="$0"
basename="$(dirname $script)"
cd $basename


#source environment
source devel/setup.bash

#Start ROS Master
roscore &

sleep 5
#start joy node
rosrun joy joy_node &

#start simulation
roslaunch cvg_sim_gazebo ardrone_testworld.launch & 
echo "Sleeping for 20 seconds to load simulation"
sleep 10

#start gui and state estimation node
roslaunch tum_ardrone tum_ardrone.launch
