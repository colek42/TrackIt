#!/bin/bash

##Set Working Directory To Location of Script
script="$0"
basename="$(dirname $script)"
cd $basename


source devel/setup.bash
mkdir bagfiles
cd bagfiles
echo "Press CTRL-C to Stop"
rosbag record -a 
