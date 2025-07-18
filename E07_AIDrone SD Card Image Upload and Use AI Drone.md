# 1. SD Card Format

### 1) Download  AIDrone Image

<br/>

https://drive.google.com/file/d/15wb1UVy0Z0z3OZVcA2XJ-jUYWJrShO_d/view?usp=drive_link

<br/>
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

### 3)  Download Win32DiskImager and upload it to the SD card.

        https://win32diskimager.org/

<br/>

<img src ="https://github.com/user-attachments/assets/ab7c2235-c839-4656-a462-33cb18b86fae" width="700">

<br/>

<img src="https://github.com/user-attachments/assets/3ebca0d9-c6a0-4b6b-81fe-485ef0137fe6" width="700">

<br/><br/>

-**After finished Image Upload, Put the SD card in Drone and Connect with Cable**

<br/>

### 4) Connect your PC or laptop's Wi-Fi to AIDrone (Using CMD)

<br/>

<img src="https://github.com/user-attachments/assets/c5da5e4a-faec-4688-a49b-52d0ab133b1d"  width="400">

<br/><br/>

-**Wi-Fi Password : 1234567890**
  
<br/><br/>

<img src="https://github.com/user-attachments/assets/272a299c-92cf-45f5-bf4e-d9323c63360f"  width="800">

<br/>

- **1 :  ssh    aidrone@192.168.4.1**
- **2 :  yes**
- **3 :  aidrone password  1234**

<br/><br/>

### 5) AIDrone Image's  ID and Password (for Futty)

     ID :  aidrone
     
     passwd :  1234

<br/>

<img src="https://github.com/user-attachments/assets/ee786acc-ad38-4f8e-b5eb-0e8d84972c47" width="500">

<br/>

<img src="https://github.com/user-attachments/assets/3e2de486-9c8a-4747-95ed-3f31ec1bcbec"  width="500">

<br/><br/>

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

### 7) Stop  mjpg-streamer service

#####  for stop

     sudo systemctl stop mjpg-streamer

#####  for disalbe if you need

     sudo systemctl disable mjpg-streamer

<br/>

### 8) AP Mode to STA or  STA to AP Mode (Basic setup is AP Mode)

<br/>

- **AT Mode: The AIDrone itself becomes a router. If there are multiple drones, modify each SSID separately. However, the address remains the same.**

<pre>
  sudo   /usr/local/bin/switch_wifi_mode.sh      AP       my_custom_SSID
</pre>   

<br/>

- **STA Mode: After switching to STA mode, you must configure the Raspberry Pi's Wi-Fi settings. Enter the SSID and password of the router available in your location.**

<pre>
  sudo   /usr/local/bin/switch_wifi_mode.sh     STA
</pre>    

<br/><br/>

<img src="https://github.com/user-attachments/assets/53dbb7fc-d563-4959-8741-557a0ca000f7" width="500">

<br/><br/>

<img src="https://github.com/user-attachments/assets/0ce9aba6-6e03-4913-9a39-4ff488fa4cff"  width="500">

<br/><br/>

<img src="https://github.com/user-attachments/assets/80d621d7-c3fd-4118-8a1b-9355ca942b25" width="500">

<br/><br/>

<img src="https://github.com/user-attachments/assets/c153b9d4-f225-41f9-ac74-a2dbf2510e26" width="500">

<br/><br/>

<img src="https://github.com/user-attachments/assets/8cc4f4b6-2598-44e2-928e-f88ca2503ed1" width="500">

<br/><br/>

<img src="https://github.com/user-attachments/assets/87247d24-7484-444e-9a10-ee80756c5d1e"  width="500">

<br/><br/>

- **Finish Enter**

<img src="https://github.com/user-attachments/assets/058aed39-cf0b-4332-93da-2f17d000c49e" width="500">
       
<br/><br/>
 
- **reboot**

<img src="https://github.com/user-attachments/assets/ecf387c8-c96b-4130-b0a2-04c515ce704d" width="500">

<br/><br/>

<img src="https://github.com/user-attachments/assets/96821cb0-264e-40c5-8ca8-3c4c77125328" width="500">




