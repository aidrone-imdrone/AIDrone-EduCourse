
# Control AIDrone Using Teachable Machine

### 1) Start Teachable Machine

<br/>

<img src="https://github.com/user-attachments/assets/766422b4-08dc-457e-94e9-928b495263ae" width="800">

<br/>

###  Make Class 5  -> TakeOff, Landing, Up, Down, Left, Right, BackGround, Hovering

![image](https://user-images.githubusercontent.com/122161666/227949247-011c49d0-adc2-4404-80d6-54166aa11d8a.png)

<br/>

![image](https://github.com/user-attachments/assets/efb0f62b-25c1-4111-a6c9-600c9eeb2247)

-**Please capture various environments in the background..**

-**Take 900 to 1000 images per class**

<br/>

![image](https://github.com/user-attachments/assets/b482bf37-10c3-4259-8f37-9b55db6d4cba)

<br/>

### After take direction Picture all and then Click Learning Model, If finished Learning Model, Click exporting the model 

<br/>

![image](https://user-images.githubusercontent.com/122161666/227953634-227318ce-7fee-40fe-a35e-26b4e10d64fd.png)

<br/>

### When download Model from Teachable Machine, it has two file inside(gesture_keras_model.h5 and labels.txt)

<br/>

<img src="https://github.com/user-attachments/assets/810d92b6-bdc9-4c53-bbb3-6ef1f56ad75b" width="800">

<br/><br/>

- **Download Link for the Resulting TensorFlow Keras Model from the Above Example**

    https://drive.google.com/file/d/176WYku7cJY5hpOT9xwC2uys1dby9OCCH/view?usp=drive_link

<br/>

### 2)  Control Code for Using the Model File Obtained from Teachable Machine (gesture_drone_contorl_pi.py)

<br/>

- **The control code, model file, and label text must be in the same folder. (Modify the model path accordingly in the code below.)**
- **This control code allows you to connect to the drone by plugging a Bluetooth dongle into the PC or connecting the controller to the PC via a USB cable. It enables drone control using hand gestures detected by the laptop or PC camera.**
  
<br/>

       from time import sleep, time
       from pyaidrone.aiDrone import *
       from pyaidrone.deflib import *
       from pyaidrone.ikeyevent import *
       import cv2
       import numpy as np
       from keras.models import load_model
       
       # Disable scientific notation for clarity
       np.set_printoptions(suppress=True)
       
       # Global variables
       Height = 100
       ready = -1
       tof = 0
       
       # Function to handle receiving data from the drone
       def receiveData(packet):
           global ready
           global tof
       
           ready = packet[7] & 0x03
           if packet[7] & 0x20 == 0x20 and len(packet) > 13:
               tof = packet[13]
               print("tof : ", tof)
       
       def main():
           # Load the model
           model = load_model("F:\AIDrone\converted_keras\gesture_keras_model.h5", compile=False)
       
           # Load the labels
           class_names = open("F:\AIDrone\converted_keras\labels.txt", "r").readlines()
       
           # Initialize webcam
           camera = cv2.VideoCapture(0)
       
           # Initialize drone
           aidrone = AIDrone()
           ikey = IKeyEvent()
           aidrone.Open("COM3")
           aidrone.setOption(0)
           sleep(0.5)
       
           # Variables for gesture detection
           current_gesture = "Background"
           last_gesture = "Background"
           gesture_start_time = 0
           confidence_threshold = 90  # 90% confidence threshold
           gesture_duration_threshold = 1.0  # 1 second duration threshold
       
           print("=== Hand Gesture Drone Control ===")
           print("ESC - Exit program")
           print("Available gestures:")
           for name in class_names:
               print(f"- {name[2:]}", end="")
       
           while True:
               # Check for ESC key to exit
               if ikey.isKeyEscPressed():
                   break
       
               # Grab the webcamera's image
               ret, image = camera.read()
               if not ret:
                   print("Failed to capture image from camera")
                   continue
       
               # Display the original image
               display_img = image.copy()
               
               # Resize the raw image for the model
               processed_img = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
       
               # Make the image a numpy array and reshape it to the models input shape
               processed_img = np.asarray(processed_img, dtype=np.float32).reshape(1, 224, 224, 3)
       
               # Normalize the image array
               processed_img = (processed_img / 127.5) - 1
       
               # Predict the gesture
               prediction = model.predict(processed_img, verbose=0)
               index = np.argmax(prediction)
               class_name = class_names[index][2:].strip()  # Remove index and strip whitespace
               confidence_score = prediction[0][index] * 100
       
               # Display the prediction and confidence on the image
               status_text = f"Gesture: {class_name}, Confidence: {confidence_score:.2f}%"
               cv2.putText(display_img, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
       
               # Show the image in a window
               cv2.imshow("Hand Gesture Drone Control", display_img)
               cv2.waitKey(1)
       
               # Check if the prediction is confident enough
               if confidence_score >= confidence_threshold:
                   current_gesture = class_name
                   
                   # Check if this is a new gesture
                   if current_gesture != last_gesture:
                       gesture_start_time = time()
                       last_gesture = current_gesture
                   
                   # Check if the gesture has been maintained for the required duration
                   elif time() - gesture_start_time >= gesture_duration_threshold:
                       # Execute the drone command based on the gesture
                       print(f"Executing: {current_gesture} (Confidence: {confidence_score:.2f}%)")
                       
                       if current_gesture == "TakeOff":
                           aidrone.takeoff()
                           print("Taking off...")
                       
                       elif current_gesture == "Landing":
                           aidrone.landing()
                           print("Landing...")
                       
                       elif current_gesture == "Left":
                           aidrone.velocity(LEFT, 100)
                           sleep(0.5)
                           aidrone.velocity(LEFT, 0)
                           print("Moving left...")
                       
                       elif current_gesture == "Right":
                           aidrone.velocity(RIGHT, 100)
                           sleep(0.5)
                           aidrone.velocity(RIGHT, 0)
                           print("Moving right...")
                       
                       elif current_gesture == "Hovering":
                           aidrone.velocity(FRONT, 0)
                           aidrone.velocity(RIGHT, 0)
                           print("Hovering...")
                       
                       # Reset the timer to prevent continuous execution
                       gesture_start_time = time() + 1  # Add delay to prevent repeated triggers
               
               # If confidence is low, consider it background/no gesture
               else:
                   current_gesture = "Background"
                   if current_gesture != last_gesture:
                       last_gesture = current_gesture
                       gesture_start_time = time()
       
           # Clean up
           camera.release()
           cv2.destroyAllWindows()
           aidrone.Close()
       
       if __name__ == '__main__':
           main()

<br/>

#### Code Features
- **In the above code, the confidence threshold is set to 90% to ensure the accuracy of gesture recognition.**
- **Additionally, the action is triggered only when a gesture is continuously recognized for more than 1 second.**


