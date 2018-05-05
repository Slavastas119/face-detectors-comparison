#!/usr/bin/env bash

sudo apt-get update

bash install_opencv.sh

sudo pip install numpy
sudo pip3 install numpy

sudo pip install dlib
sudo pip3 install dlib

echo "Finished Installation, now rebooting!"
sudo reboot