# 1. AIDrone의 라즈베리파이 Zero 2W 이미지 다운로드 및  SD Card에 올리기

### 1) AIDrone의 라즈베리파이 Zero 2W 이미지 다운로드 

<br/>

https://drive.google.com/file/d/1ibLG-2X8MOwmSodx6xZGig9wf8txTiWn/view?usp=drive_link

- **압축 파일을 다운로드 받은 후 압축을 푼다**
  
<br/>
 
### 2) SD Card 포멧을 위해서 포멧터 프로그램 다운로드 

   https://www.sdcard.org/downloads/formatter/

<br/><br/>

<img src="https://github.com/user-attachments/assets/ae597e88-2d88-4e03-9a48-f9a8e278eb83"  width="500">

<br/>  

<img src="https://github.com/user-attachments/assets/2e7de94a-ed19-4117-8b05-72f53f65dcf1" width="800">

<br/><br/>

<img src="https://github.com/user-attachments/assets/fc95b894-46af-4fef-96e7-bffe7f034f39"  width="400">

<br/><br/>

### 3) 라즈베리파이 Imager를 다운받아서 SD Card에 업로드하기

         https://www.raspberrypi.com/software/

<br/>

<img src="https://github.com/user-attachments/assets/8ef0861a-491e-4faf-959f-32154c948d87" width="700">

<br/><br/>

<img src="https://github.com/user-attachments/assets/b4337fc4-bd0f-4404-aef3-0cfe27e5fad2" width="700">

<br/><br/>

<img src="https://github.com/user-attachments/assets/eadefe73-6202-42ca-9a17-943aa502b4b1" width="500">

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

### 8) AP Mode to STA or  STA to AP Mode  (Basic setup is AP Mode)

<br/>

- **AT Mode**

    sudo   /usr/local/bin/switch_wifi_mode.sh   AP   my_custom_SSID

<br/>

- **STA**

    sudo   /usr/local/bin/switch_wifi_mode.sh   STA


