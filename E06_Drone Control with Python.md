# Flight AIDrone with python code in PC

## 1. Connect Transmitter with PC  by USB Cable

<br/>    

<img src="https://github.com/irbrain/AIDrone/assets/122161666/e6effc21-2827-4180-9aa8-3b1d017a3179" width="500">

- Please, Check Serial Port
- 
<br/><br/>

## 2. Install AIDrone's Python library

#### cmd window 
     
       pip install pyaidrone

<img src="https://github.com/user-attachments/assets/86719370-333d-479c-8e81-58139d644b6d" width="500">

<br/><br/>

#### Anaconda Python Virtual Environment

       conda activate aidrone

       pip install pyaidrone

- Anaconda must be run with administrator privileges.
  
<img src="https://github.com/user-attachments/assets/050b059c-a5df-4e56-b481-d2c224620fe7" width="500">

<br/><br/>

- Create Python Virtual Environment
  
<img src="https://github.com/user-attachments/assets/7dc8388a-2e52-4f9a-b91c-b9f70dfee319" width="500">

<br/>

- Install pyaidrone in Python Virtual Environment
  
<img src="https://github.com/user-attachments/assets/d0a785b5-6a11-4fa5-8061-e22c19afaf7d" width="500">

<br/><br/>

## 3. Python Examples

#### 1) move AIDrone a certain distance

       from time import sleep
       from pyaidrone.aiDrone import *
       from pyaidrone.deflib import *
       
       ready = -1
       
       def receiveData(packet):
            global ready
            ready = packet[7] & 0x03
            
       if __name__ = '__main__':
       
            aidrone = AIDrone(recevieData)
            aidrone.Open("COM3")   # change to the your ports number
            aidrone.setOption(0)
            sleep(0.5)
            
            while ready != 0:
                  sleep(0.1)
                  
            aidrone.takeoff()
            sleep(5)
            aidrone.move(FRONT, 200)    # 200 means 2m
            sleep(5)
            aidrone.move(BACK, 200) 
            sleep(5)
            aidrone.landing()
            sleep(3)
            aidrone.Close()

<br/>

#### 2) rotate AIDrone 

       from time import sleep
       from pyaidrone.aiDrone import *
       from pyaidrone.deflib import *
       
       ready = -1
       
       def receiveData(packet):
            global ready
            ready = packet[7] & 0x03
            
       if __name__ = '__main__':
       
            aidrone = AIDrone(recevieData)
            aidrone.Open("COM3")   # change to the your ports number
            aidrone.setOption(0)
            sleep(0.5)
            
            while ready != 0:
                  sleep(0.1)
                  
            aidrone.takeoff()
            sleep(5)
            aidrone.rotation(90)    # Turn to the right about 90 degree
            sleep(5)
            aidrone.rotation(-90) 
            sleep(5)
            aidrone.landing()
            sleep(3)
            aidrone.Close()

<br/>

 #### 3) up and down AIDrone 

       from time import sleep
       from pyaidrone.aiDrone import *
       from pyaidrone.deflib import *
       
       ready = -1
       
       def receiveData(packet):
            global ready
            ready = packet[7] & 0x03
            
       if __name__ = '__main__':
       
            aidrone = AIDrone(recevieData)
            aidrone.Open("COM3")   # change to the your ports number
            aidrone.setOption(0)
            sleep(0.5)
            
            while ready != 0:
                  sleep(0.1)
                  
            aidrone.takeoff()
            sleep(5)
            aidrone.altitude(150)    # up to the 1.5m. basic altitude is 70cm
            sleep(5)
            aidrone.altitude(50) 
            sleep(5)
            aidrone.altitude(100)
            sleep(8)
            aidrone.landing()
            sleep(3)
            aidrone.Close()

<br/>

 #### 4) flight's velocity change 

       from time import sleep
       from pyaidrone.aiDrone import *
       from pyaidrone.deflib import *
       
       ready = -1
       
       def receiveData(packet):
            global ready
            ready = packet[7] & 0x03
            
       if __name__ = '__main__':
       
            aidrone = AIDrone(recevieData)
            aidrone.Open("COM3")   # change to the your ports number
            aidrone.setOption(0)
            sleep(0.5)
            
            while ready != 0:
                  sleep(0.1)
                  
            aidrone.takeoff()
            sleep(5)
            aidrone.velocity(FRONT, 100)   
            sleep(2)
            aidrone.velocity(FRONT, 0) 
            sleep(5)
            aidrone.velocity(BACK, 100)
            sleep(2)
            aidrone.velocity(BACK, 0)
            sleep(5)
            aidrone.landing()
            sleep(5)
            aidrone.Close()

<br/>

 #### 5) RC AIDrone by direction keyboard 
 
      from time import sleep
      from pyaidrone.aiDrone import *
      from pyaidrone.deflib import *
      from pyaidrone.ikeyevent import *

      Height = 70
      Degree = 0

      if __name__ == '__main__':
            aidrone = AIDrone()
            ikey = IKeyEvent()
            aidrone.Open("COM3")
            aidrone.setOption(0)
            sleep(0.5)

             while not ikey.isKeyEscPressed():        
        
            if ikey.isKeyEnterPressed():             
                  aidrone.takeoff()

            if ikey.isKeySpacePressed():
                  aidrone.landing()
      
            if ikey.isKeyUpPressed():
                  aidrone.velocity(FRONT, 100)
            elif ikey.isKeyDownPressed():
                  aidrone.velocity(BACK, 100)
            else:
                  aidrone.velocity(FRONT, 0)

            if ikey.isKeyRightPressed():
                  aidrone.velocity(RIGHT, 100)
            elif ikey.isKeyLeftPressed():
                  aidrone.velocity(LEFT, 100)
            else:
                  aidrone.velocity(RIGHT, 0) 
        
            if ikey.isKeyWPressed():
                  Height = Height + 10
                  aidrone.altitude(Height)
            elif ikey.isKeyXPressed():
                  Height = Height - 10
                  aidrone.altitude(Height)

             if ikey.isKeyDPressed():
                  Degree = Degree + 10
                  aidrone.rotation(Degree)            
            elif ikey.isKeyAPressed():
                  Degree = Degree +10
                  aidrone.rotation(-Degree)            

            sleep(0.1)
      aidrone.Close()


<br/>

## 4. Python Examples Download  

https://drive.google.com/file/d/1mfU4Bj8GvPiZbgU38dHEQquj-FGMWcO8/view?usp=drive_link

<br/><br/>

## 5. Controlling Drone Flight on Raspberry Pi Zero 2W  
*(Applied from aidrone Image version 250309)*  

<br/>

### ⚠️ **Note: If it is an aidrone Image version before 250309, edit it directly**  

- **In `raspi-config`, set serial to `No` in Interface Options.**  

- **You must stop the Bluetooth service to use serial:**  
  ```bash
  sudo systemctl stop hciuart
  sudo systemctl disable hciuart

- **Modify the /boot/config.txt file (add to the last line):**
  ```bash
  dtoverlay=disable-bt

<br/>

### 1) Copy the example code to the Raspberry Pi using Samba.  (Samba ID: aidrone  ,  Password :  samba )

### 2) Modify the example code:

        aidrone.Open("COM3")   to   aidrone.Open("/dev/serial0")
        
### 3) Run the modified example on the Raspberry Pi using PuTTY to autonomously control the drone.

      (ex:  move, velocity, rotation, altitude, altitude_mode, etc.)
