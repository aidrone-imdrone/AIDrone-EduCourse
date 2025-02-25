# 1. AIDrone의 라즈베리파이 Zero 2W 이미지 다운로드 및  SD Card에 올리기

### 1) AIDrone의 라즈베리파이 Zero 2W 이미지 다운로드 

<br/>

https://drive.google.com/file/d/1ibLG-2X8MOwmSodx6xZGig9wf8txTiWn/view?usp=drive_link

- **압축 파일을 다운로드 받은 후 압축을 푼다**
  
<br/>
 
### 2) SD Card 포멧을 위해서 포멧 프로그램 다운로드 

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

<img src="https://github.com/user-attachments/assets/8ef0861a-491e-4faf-959f-32154c948d87" width="800">

<br/><br/>

<img src="https://github.com/user-attachments/assets/b4337fc4-bd0f-4404-aef3-0cfe27e5fad2" width="800">

<br/><br/>

<img src="https://github.com/user-attachments/assets/eadefe73-6202-42ca-9a17-943aa502b4b1" width="800">

<br/><br/>

### 4)  PC 또는 노트북에서 AIDrone Wi-Fi 연결 (CMD 창에서)

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

### 5)  AIDrone 라즈베리파이의 ID와  비번

     ID :  aidrone
     
     passwd :  1234

<br/>

<img src="https://github.com/user-attachments/assets/ee786acc-ad38-4f8e-b5eb-0e8d84972c47" width="500">

<br/>

<img src="https://github.com/user-attachments/assets/3e2de486-9c8a-4747-95ed-3f31ec1bcbec"  width="500">

<br/><br/>

### 6) mjpg-streamer 서비스 시작하기 (영상 스트리밍)

<br/>

   sudo systemctl enable mjpg-streamer
   
   sudo systemctl start mjpg-streamer
   
   sudo reboot

<br/>

- **PC에서 영상 확인하기**

     http://<raspberry_pi_wifi_address>
     
<br/>

<img src="https://github.com/user-attachments/assets/e9f3a5fb-403b-48a2-8bf2-5143b5beb6e8" width="500">

<br/><br/>

### 7) mjpg-streamer 서비스 정지

#####  멈춤 (다시 라즈베리파이 시작하면 서비스 계속)

     sudo systemctl stop mjpg-streamer

#####  라즈베리파이를 다시 시작해도 서비스 멈추기 

     sudo systemctl disable mjpg-streamer

<br/>

### 8) AIDrone의 Wi-Fi 모드는 AP 모드임 (주소는 192.168.4.1)

<br/>

- **AT Mode : AIDrone 자체가 라우터가 됨. 여러 드론이 있을 경우 각각 SSID 수정하세요, 단 주소는 같음.    WI-FI Password : 1234567890)**

<pre>
   sudo /usr/local/bin/switch_wifi_mode.sh   AP    my_custom_SSID  
</pre>

<br/>

- **STA Mode: STA 모드로 변경 후에는 라즈베리파이 WI-FI 설정을 해야 함.  공간에 있는 라우터의 SSID와 비번을 입력하세요**

<pre>
  sudo   /usr/local/bin/switch_wifi_mode.sh   STA  
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

<img src="https://github.com/user-attachments/assets/87247d24-7484-444e-9a10-ee80756c5d1e"  width="500">

<br/><br/>

- **Finish Enter**

<img src="https://github.com/user-attachments/assets/058aed39-cf0b-4332-93da-2f17d000c49e" width="500">
       
<br/><br/>
 
- **reboot**

<img src="https://github.com/user-attachments/assets/ecf387c8-c96b-4130-b0a2-04c515ce704d" width="500">

<br/><br/>

<img src="https://github.com/user-attachments/assets/96821cb0-264e-40c5-8ca8-3c4c77125328" width="500">






  


