
## 1. Upload the Raspberry Pi image to SD Card

### 1) Install Raspberry Pi Imager 
     
        https://www.raspberrypi.com/software/

<br/>

### 2) Open the Raspberry Pi imager
<br/>

![image](https://github.com/user-attachments/assets/d6adaf82-cb23-4591-a327-02b50e31c1f0)

<br/>

### 3) chose Raspberry Pi Device 
<br/>

![image](https://github.com/user-attachments/assets/b246d343-347b-4cc3-a6ef-47d7405ee8ee)

<br/>

### 4) Chose Raspberry Pi OS  (32bit Lite Bullseye)
<br/>

![image](https://github.com/user-attachments/assets/f1281366-08a0-4369-b96a-878e51597191)

<br/>

### 5) Chose SD Card
<br/>

![image](https://github.com/user-attachments/assets/c6fc4541-b942-458b-849f-fecb9c68d80a)

<br/>

### 6) Click Next Button and then Edit Settings
<br/>

![image](https://github.com/user-attachments/assets/ce82bddf-c6f1-46c3-b6b0-94cb20308ad9)

<br/>

![image](https://github.com/user-attachments/assets/c5f6abee-a97a-48ff-8911-501568b37132)

<br/>

![image](https://github.com/user-attachments/assets/6e7acfbd-59b5-4a4b-93e9-087103e202da)

<br/>

![image](https://github.com/user-attachments/assets/caabdb49-396c-409e-94b6-7b6ca63442b2)

<br/>

### 7) Write
<br/>

![image](https://github.com/user-attachments/assets/a032f216-b661-4ef1-b360-e21cd62a258d)

<br/>

![image](https://github.com/user-attachments/assets/9961da59-25d6-4a5b-85f3-0f2ed74acfc4)

<br/>

![image](https://github.com/user-attachments/assets/eaa06532-2ae4-4194-bfb0-6ba9fe1f47fb)

<br/>

### 8) For more information, please check this link
<br/>

https://www.raspberrypi.com/documentation/computers/getting-started.html

<br/><br/>

---

# 1. Raspberry Pi Basic Setup
### 1) Raspberry pi update and upgrade

       sudo  apt  update        
       sudo  apt  upgrade  -y 

<br/>

### 2) raspi-config setup
       
       sudo  nano   raspi-config

<br/>       

#### 1) Boot Option

![image](https://github.com/user-attachments/assets/f53304aa-5dc7-4588-808d-fb1c5708475c)

<br/>

#### 2) Interface Options 

![image](https://github.com/user-attachments/assets/2cd6db25-bcef-43d1-b211-556e92528a1c)

<br/>

#### 3) Localisation Options 

![image](https://github.com/user-attachments/assets/045760c3-3b15-4e27-9f3b-1f535b9c3344)

<br/>

#### 4) Advanced Options 

![image](https://github.com/user-attachments/assets/4245f56e-95a6-4c91-b6d2-f99a04c23a34)

<br/>

# 2. Install Putty in PC (If there is no putty in your pc)

### https://www.putty.org   Download Putty 

![image](https://user-images.githubusercontent.com/122161666/224391267-617a2dac-400b-4983-8a47-6379163ee5f6.png)

<br/>

### show up below page,  choose the putty for your PC's OS

![image](https://user-images.githubusercontent.com/122161666/224391765-02c437fb-357f-4e3b-9b01-a0e164b7015f.png)

<br/>

### After installed Putty, execute the putty and write your AIDrone Wifi's Number, Click open

![image](https://user-images.githubusercontent.com/122161666/224396899-08673c1b-b173-496a-ad1f-3d1d8a5c5929.png)

<br/>

### When connected AIDrone, write down your id and password on putty

![image](https://user-images.githubusercontent.com/122161666/224398030-60dc599c-4a61-47d1-87ce-2fb846f5133f.png)

<br/>

# 3. Set up Samba service for excahnge files between a PC and raspberry pi

     sudo apt install samba  samba-common-bin
     
     sudo smbpasswd -a  pi  (When User ID is pi.)
 
#####  =>  write donw password  two times

<br/>

###  samba set up 

      sudo  nano  /etc/samba/smb.conf

##### =>  alt + /  :  cursor move to the last line

<br/>
      
####  write like that

      [UserID]
              comment = userID home
              path = /home/userID
              valid users = userID
              guest ok = no
              browseable = yes
              writeable = yes
              create mask = 0777
              
#### ctrl+o  -> save
#### ctrl+x  -> end

#### samba service start

      sudo service smbd restart
      
<br/>

# 4. connecting samba on PC

![image](https://user-images.githubusercontent.com/122161666/224478786-c3a66388-0c7c-4635-ad17-22c3629327f4.png)

<br/>

# 5. Install pip for python programing

     sudo apt install python3-pip  
     
     python3 -m pip install --upgrade pip
     
<br/>

# 6. Create Python Virtual Environment

     sudo apt install python3-venv
     
     python3  -m  venv  myvenv

<br/>

# 7. Make sure the Python virtual environment starts automatically when the Raspberry Pi starts.

#### 1)  sudo  nano  ~/.bashrc

#### 2)  Alt + /   ( cursor move to last line)

#### 3)  Write down like this : 

            if  [ -d  "$HOME/myvenv" ]; then
                 source  $HOME/myvenv/bin/activate
            fi

#### 4)  Ctrl + x  ->  Y   (Saving File)

#### 5)  source  ~/.bashrc  (bashrc's change comment applied )

#### 6)  sudo reboot  

![image](https://github.com/user-attachments/assets/f4480cd5-2670-4328-bb85-20a7a7090933)

<br/>

# 8. Install python Libraries

        sudo  rm  /usr/lib/python3/dist-packages/gpg-1.14*
       
         myvenv/bin/pip install setuptools wheel numpy==1.23.5  flask

![image](https://github.com/user-attachments/assets/25cefbf6-dcaf-4465-9830-789605f3b111)


 
    
    

             
     
     
     

     
     
     






       
























