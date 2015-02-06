
#!/bin/bash

##Set Working Directory To Location of Script
script="$0"
basename="$(dirname $script)"
cd $basename



source devel/setup.bash
roslaunch tum_ardrone ardrone_driver.launch & 
echo "Sleeping for 20 seconds to wait for connection"
sleep 20

roslaunch tum_ardrone tum_ardrone.launch
