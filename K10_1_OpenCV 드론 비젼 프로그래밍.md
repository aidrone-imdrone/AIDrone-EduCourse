## 1. 기본 라이브러리 설치

<br/>

     sudo apt update  &&  sudo apt upgrade -y     
     sudo apt install -y build-essential cmake git pkg-config libjpeg-dev libtiff-dev libpng-dev libavcodec-dev libavformat-dev libswscale-dev libv4l-dev  libxvidcore-dev libx264-dev  libgtk2.0-dev libgtk-3-dev imagemagick libv4l-dev libatlas-base-dev gfortran python3-dev libopenblas-dev imagemagick subversion python3-pil gstreamer1.0-tools  libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev 

<br/>

## 2. 스왑 사이즈를 2048로 변경

<br/>

       sudo dphys-swapfile swapoff
       
       sudo nano /etc/dphys-swapfile

<bt/>

![image](https://github.com/user-attachments/assets/1beb6134-5fe6-456f-94fe-a48627fcde77)

       sudo  dphys-wapfile  setup

       sudo  dphys-wapfile  swapon
       
<br/>        

![image](https://github.com/user-attachments/assets/9e22ae3b-588c-40d1-8732-50a163778078)

<br/>
       

## 3. 간단하게 OpenCV 설치  for C++ and Python 

     sudo apt install libopencv-dev

     sudo apt install python3-opencv

#### OpenCV 실행 파일을 파이썬 가상환경으로 이동 

     sudo  find  /  -type  f  -name  "cv2*.so"

![image](https://github.com/user-attachments/assets/6113320f-6d1f-4a84-9098-6b6e8be3ce14)

     cp /usr/lib/python3/dist-packages/cv2.cpython-39-arm-linux-gnueabihf.so    ~/myvenv/lib/python3.9/site-packages/

#### OpenCV의 버젼 확인

![image](https://github.com/user-attachments/assets/473ac68a-f880-4d9e-a3f6-503a26e5823e)

<br/>

## 4. 소스를 다운받아서 OpenCV 설치 (If using  C/C++ and Python )  : 추천하는 방법

       git clone https://github.com/opencv/opencv.git
   
       git clone https://github.com/opencv/opencv_contrib.git

       cd  ~/opencv

       mkdir build

       cd build

       cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D WITH_TBB=OFF -DWITH_LIBV4L=ON -D WITH_IPP=OFF -D WITH_1394=OFF -D BUILD_WITH_DEBUG_INFO=OFF -D BUILD_DOCS=OFF -D  INSTALL_C_EXAMPLES=ON -D INSTALL_PYTHON_EXAMPLES=ON -D BUILD_EXAMPLES=OFF -D BUILD_TESTS=OFF -D BUILD_PERF_TESTS=OFF -D ENABLE_NEON=ON -D WITH_QT=OFF -D WITH_GTK=ON -D WITH_OPENGL=ON -D  OPENCV_ENABLE_NONFREE=ON -D  OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules -D WITH_V4L=ON -D WITH_FFMPEG=ON -D WITH_XINE=ON -D ENABLE_PRECOMPILED_HEADERS=OFF -D BUILD_NEW_PYTHON_SUPPORT=ON -D  OPENCV_GENERATE_PKGCONFIG=ON -D  PYTHON3_PACKAGES_PATH=/usr/lib/python3.9/dist-packages -D WITH_GSTREAMER=0N -D WITH_GSTREAMER_0_10=OFF  ../
 
#### you have to change PYTHON3_PACKAGES_PATH's python version to your python version.

<br/>

       make –j4

       sudo make install

       sudo  ldconfig

<br/>

## 5. Rasbperry Pi Zero 2w 에 MJPG-Streamer 설치 ( Bullseye OS Version)

<br/>

#### 1) mjpg-streamer 소스 빌드드 

     sudo apt update && sudo apt upgrade -y

     git clone  https://github.com/jacksonliam/mjpg-streamer.git

     cd mjpg-streamer/mjpg-streamer-experimental

     make CMAKE_BUILD_TYPE=Debug

     sudo make install 
     
     cd ~     
    
<br/>  

#### 3) MJPG-Streamer 테스트
      cd mjpg-streamer/mjpg-streamer-experimental

      ./mjpg_streamer -i "input_uvc.so -r 640x480 -f 30 -rot 180" -o "output_http.so -w ./www"


##### Wep page in PC 

      https://<raspberry pi Address>:8080

![image](https://github.com/user-attachments/assets/fd233dcf-a3a0-400d-9f30-4f9a15ec14f6)

<br/>

## 6. Make Raspberry Pi automatically become MJPG STREAMER when it starts 

#### Change  www's index.html to  stream_simple.html 

      cd mjpg-streamer/mjpg-streamer-experimental/www
      
      mv index.html  index.html_backup
      
      nano  index.html 

<br/>

      <!DOCTYPE html>
      <html>
      <head>
             <meta http-equiv="refresh" content="0; URL=/stream_simple.html">
      </head>
      <body>
      </body>
      </html>

![image](https://github.com/user-attachments/assets/c3c19b79-f056-4f1f-87b0-a5aa60577309)

##### Save index.html 

<br/>      
 
####  mjpg-streamer.service 편집집 

     sudo nano /etc/systemd/system/mjpg-streamer.service
 
     < change from user id to your user >
    
<br/> 

     [Unit]
     Description=MJPG Streamer Service
     fter=network.target

     [Service]
     ExecStartPre=/usr/bin/v4l2-ctl --set-ctrl=horizontal_flip=1
     ExecStartPre=/usr/bin/v4l2-ctl --set-ctrl=vertical_flip=1
     ExecStart=/home/your_id/mjpg-streamer/mjpg-streamer-experimental/mjpg_streamer -i "input_uvc.so -r 640x480 -f 30" -o "output_http.so -w ./www -p 80"
     WorkingDirectory=/home/your id/mjpg-streamer/mjpg-streamer-experimental
     User=root
     Restart=always

     [Install]
     WantedBy=multi-user.target

     

<br/>
               
####  mjpg-streamer service 시작작 
<br/>

   sudo systemctl daemon-reload
   
   sudo systemctl enable mjpg-streamer
   
   sudo systemctl start mjpg-streamer
   
   sudo reboot

<br/>

####  You can see the camera video in Website on PC

     http://<raspberry pi wifi address>
     
![image](https://github.com/user-attachments/assets/e9f3a5fb-403b-48a2-8bf2-5143b5beb6e8)

<br/>

####  Stop  mjpg-streamer service 

#####  for stop

     sudo systemctl stop mjpg-streamer

#####  for disalbe if you need

     sudo systemctl disable mjpg-streamer

