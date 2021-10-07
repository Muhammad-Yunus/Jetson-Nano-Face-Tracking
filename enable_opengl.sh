#!/bin/bash

sudo service gdm3 stop
sudo X
export DISPLAY=:0
xrandr
glxinfo | egrep -i '(version|nvidia)'
