# 1. SD Card Format

### 1) Download  AIDrone Image

<br/>

https://drive.google.com/file/d/1ibLG-2X8MOwmSodx6xZGig9wf8txTiWn/view?usp=drive_link

- **Download the file and extract the compressed archive**
  
<br/>
 
### 2) Download Formatter For SD Card Format

   https://www.sdcard.org/downloads/formatter/

<br/><br/>

<img src="https://github.com/user-attachments/assets/ae597e88-2d88-4e03-9a48-f9a8e278eb83"  width="500">

<br/>  

<img src="https://github.com/user-attachments/assets/2e7de94a-ed19-4117-8b05-72f53f65dcf1" width="800">

<br/><br/>

<img src="https://github.com/user-attachments/assets/fc95b894-46af-4fef-96e7-bffe7f034f39"  width="400">

<br/><br/>

### 3) Download  Win32DiskImager for Upload Image to SD Card.

     https://win32diskimager.org/

<br/>

<img src="https://github.com/user-attachments/assets/2c61b0d3-c712-4de4-84a3-9451225d83f6" width="700">

<br/><br/>

<img src="https://github.com/user-attachments/assets/cda9a45c-2f98-45d3-b3b6-9c2ba314539b" width="700">

<br/><br/>

### 4) Connect your PC or laptop's Wi-Fi to AIDrone (Using CMD)

<br/>

<img src="https://github.com/user-attachments/assets/c5da5e4a-faec-4688-a49b-52d0ab133b1d"  width="400">

<br/><br/>

<img src="https://github.com/user-attachments/assets/272a299c-92cf-45f5-bf4e-d9323c63360f"  width="800">

<br/>

- **1 :  ssh    aidrone@192.168.4.1**
- **2 :  yes**
- **3 :  aidrone password  1234**

<br/><br/>

### 4) AIDrone Image's  ID and Password (for Futty)

     ID :  aidrone
     
     passwd :  1234

<br/>

<img src="https://github.com/user-attachments/assets/ee786acc-ad38-4f8e-b5eb-0e8d84972c47" width="500">

<br/>

<img src="https://github.com/user-attachments/assets/3e2de486-9c8a-4747-95ed-3f31ec1bcbec"  width="500">

<br/><br/>

### 5) Open the Command Prompt(CMD) and connect via SSH

- **
<br/>


<br/>


### 6) Set up your personal Wi-Fi connection

<br/>


<br/>

### 7) Start  mjpg-streamer service 

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

### 8) Stop  mjpg-streamer service

#####  for stop

     sudo systemctl stop mjpg-streamer

#####  for disalbe if you need

     sudo systemctl disable mjpg-streamer

<br/>

### 9) AP Mode to STA or  STA to AP Mode

<br/>

- **AT Mode**

    sudo   /usr/local/bin/switch_wifi_mode.sh   AP   my_custom_SSID

<br/>

- **STA**

    sudo   /usr/local/bin/switch_wifi_mode.sh   STA


