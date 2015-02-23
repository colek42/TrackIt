
#!/bin/bash

##Set Working Directory To Location of Script
script="$0"
basename="$(dirname $script)"
cd $basename


#source environment
source devel/setup.bash

#start joy node
rosrun joy joy_node &

#start ar_drone driver
roslaunch tum_ardrone ardrone_driver.launch & 
echo "Sleeping for 20 seconds to wait for connection"
sleep 20

rosservice call /ardrone/setrecord 1 &

#start gui and state estimation node
roslaunch tum_ardrone tum_ardrone.launch
