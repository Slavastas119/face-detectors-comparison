#!/usr/bin/env bash

echo "===== Installing OpenCV 3.3 ... ====="
sudo apt-get install -y cmake build-essential
sudo apt-get install -y libjpeg-dev libtiff5-dev libjasper-dev libpng12-dev
sudo apt-get install -y libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
sudo apt-get install -y libxvidcore-dev libx264-dev
sudo apt-get install -y libgtk2.0-dev libgtk-3-dev
sudo apt-get install -y libatlas-base-dev gfortran
sudo apt-get install -y python2.7-dev

sudo pip install numpy
sudo apt-get -y autoremove

echo "Downloading the OpenCV source..."
cd ~
wget -O opencv.zip https://github.com/Itseez/opencv/archive/3.3.0.zip
unzip opencv.zip
rm opencv.zip

#wget -O opencv_contrib.zip https://github.com/Itseez/opencv_contrib/archive/3.3.0.zip
#unzip opencv_contrib.zip
#rm opencv_contrib.zip

cd ~/opencv-3.3.0/
mkdir build
cd build

cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D INSTALL_C_EXAMPLES=OFF \
    -D INSTALL_PYTHON_EXAMPLES=OFF -D BUILD_EXAMPLES=OFF ..

#cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D INSTALL_C_EXAMPLES=OFF \
#    -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib-3.3.0/modules \
#    -D INSTALL_PYTHON_EXAMPLES=OFF -D BUILD_EXAMPLES=OFF ..

#echo "== Temporarily Increasing SWAP file size =="
#sudo sed -i -- 's/CONF_SWAPSIZE=100/CONF_SWAPSIZE=512/g' /etc/dphys-swapfile
#sudo /etc/init.d/dphys-swapfile stop
#sudo /etc/init.d/dphys-swapfile start

make
sudo make install
sudo ldconfig

cd ~
sudo rm -r opencv-3.3.0/
#sudo rm -r opencv_contrib-3.3.0/

#sudo sed -i -- 's/CONF_SWAPSIZE=512/CONF_SWAPSIZE=100/g' /etc/dphys-swapfile
#sudo /etc/init.d/dphys-swapfile stop
#sudo /etc/init.d/dphys-swapfile start