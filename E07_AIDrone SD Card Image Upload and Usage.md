# 1. SD Card Format
  
### 1) Download Formatter For SD Card Format

   https://www.sdcard.org/downloads/formatter/

<br/><br/>

<img src="https://github.com/user-attachments/assets/ae597e88-2d88-4e03-9a48-f9a8e278eb83"  width="500">

<br/>  

<img src="https://github.com/user-attachments/assets/2e7de94a-ed19-4117-8b05-72f53f65dcf1" width="800">

<br/><br/>

<img src="https://github.com/user-attachments/assets/fc95b894-46af-4fef-96e7-bffe7f034f39"  width="400">

<br/><br/>

### 2) Download  Win32DiskImager for Upload Image to SD Card.

     https://win32diskimager.org/

<br/>

<img src="https://github.com/user-attachments/assets/2c61b0d3-c712-4de4-84a3-9451225d83f6" width="700">

<br/><br/>

<img src="https://github.com/user-attachments/assets/cda9a45c-2f98-45d3-b3b6-9c2ba314539b" width="700">

<br/><br/>

### 3) Download  AIDrone Image

<br/>

    https://drive.google.com/file/d/1ibLG-2X8MOwmSodx6xZGig9wf8txTiWn/view?usp=drive_link

<br/>

### 4) AIDrone Image's  ID and Password

     ID :  aidrone
     
     passwd :  1234

<br/>

### 5) Connect AIDrone with PC Putty

<br/>

 

<br/>

### 6) Start  mjpg-streamer service 

<br/>

   sudo systemctl enable mjpg-streamer
   
   sudo systemctl start mjpg-streamer
   
   sudo reboot

<br/>

- **You can see the camera video in Website on PC**

     http://<raspberry_pi_wifi_address>
     
<br/>

<img src="https://github.com/user-attachments/assets/e9f3a5fb-403b-48a2-8bf2-5143b5beb6e8" width="500">

<br/><br/>

- **Stop  mjpg-streamer service** 

#####  for stop

     sudo systemctl stop mjpg-streamer

#####  for disalbe if you need

     sudo systemctl disable mjpg-streamer

<br/>

### 7) AP Mode to STA or  STA to AP Mode

<br/>

- **AT Mode**

    sudo   /usr/local/bin/switch_wifi_mode.sh   AP   my_custom_SSID

<br/>

- **STA**

    sudo   /usr/local/bin/switch_wifi_mode.sh   STA



