
#!/bin/bash

cd ~/workspace
source devel/setup.bash
roslaunch tum_ardrone ardrone_driver.launch & 
echo "Sleeping for 20 seconds to wait for connection"
sleep 20

roslaunch tum_ardrone tum_ardrone.launch
