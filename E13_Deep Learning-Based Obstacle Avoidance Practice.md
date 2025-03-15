# **Deep Learning-Based Mini Drone Obstacle Avoidance Practice**

## **1. Overview**
### **Project Objective**
This project implements **an obstacle avoidance function using a mini drone equipped with a Raspberry Pi Zero 2W and a camera**. The goal is to detect obstacles using real-time video analysis and machine learning models and enable autonomous flight that avoids detected obstacles.

### **Required Technologies and Libraries**
This project utilizes the following key technologies and libraries:
- **Python**: Programming language
- **OpenCV**: Real-time image processing
- **TensorFlow/Keras**: Deep learning model training and inference
- **pyaidrone**: Drone control library
- **NumPy**: Data preprocessing and transformation
- **Raspberry Pi Zero 2W**: Computing unit for the drone
- **Camera Module**: Image data acquisition

### **Features**
- **Real-time video data collection using a camera**
- **Training a CNN model with collected data**
- **Integrating the trained model into drone control code for obstacle detection and avoidance**
- **Automatic obstacle detection with lateral movement to avoid obstacles and resume forward motion**

---

## **2. Data Collection**
To collect obstacle data using the drone camera, the `rc.py` code is modified to allow controlling the drone with a keyboard while saving images.

```python
import cv2
import os
from pyaidrone.aiDrone import *
from pyaidrone.deflib import *
from pyaidrone.ikeyevent import *

# Set up the dataset directory
dataset_path = "./dataset"
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

cap = cv.VideoCapture("http://<your AIDrone IP>/?action=stream")  # Use drone camera  EX:  cv.VideoCapture("http://192.168.1.12/?action=stream")  

# Initialize the drone
aidrone = AIDrone()
ikey = IKeyEvent()
aidrone.Open("COM3")
aidrone.setOption(0)

img_count = 0

while not ikey.isKeyEscPressed():
    ret, frame = cap.read()
    if not ret:
        break
    
    cv2.imshow("Camera", frame)

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
        aidrone.altitude(10)
    elif ikey.isKeyXPressed():
        aidrone.altitude(-10)

    if ikey.isKeyDPressed():
        aidrone.rotation(10)
    elif ikey.isKeyAPressed():
        aidrone.rotation(-10)
    
    if ikey.isKeySPressed():  # Press 'S' key to save an image
        img_path = os.path.join(dataset_path, f"image_{img_count}.jpg")
        cv2.imwrite(img_path, frame)
        img_count += 1
        print(f"Saved {img_path}")
    
    elif ikey.isKeyQPressed():  # Press 'Q' key to exit
        break

cap.release()
cv2.destroyAllWindows()
aidrone.Close()
```

Now, pressing the `S` key while controlling the drone with the keyboard saves obstacle images.

---

## **3. Deep Learning Model Training**
### **Building and Training a CNN Model**
The collected data is used to train a CNN model.

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set up data path
data_gen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_data = data_gen.flow_from_directory(
    'dataset',
    target_size=(64, 64),
    batch_size=16,
    class_mode='binary',
    subset='training'
)

val_data = data_gen.flow_from_directory(
    'dataset',
    target_size=(64, 64),
    batch_size=16,
    class_mode='binary',
    subset='validation'
)

# Define CNN model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
model.fit(train_data, validation_data=val_data, epochs=10)

# Save model
model.save("drone_obstacle_model.h5")
```

Now, after collecting obstacle images while controlling the drone, the CNN model can be trained.

---

## **4. Integrating the Model with Drone Control Code**
### **Using the Trained Model to Control the Drone**
```python
import cv2
import tensorflow as tf
import numpy as np
from pyaidrone.aiDrone import *
from pyaidrone.deflib import *

# Load trained deep learning model
model = tf.keras.models.load_model("drone_obstacle_model.h5")

# Initialize drone
aidrone = AIDrone()
aidrone.Open("COM3")

# Set up camera
cap =  cv.VideoCapture("http://<your AIDrone IP>/?action=stream") 

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess image
    img = cv2.resize(frame, (64, 64))
    img = np.expand_dims(img, axis=0) / 255.0

    # Perform prediction
    prediction = model.predict(img)

    # Determine if an obstacle is detected
    if prediction > 0.5:
        print("Obstacle detected! Avoiding...")
        aidrone.velocity(LEFT, 100)  # Obstacle avoidance action
    else:
        print("Moving forward...")
        aidrone.velocity(FRONT, 100)

    cv2.imshow("Drone Camera", frame)

    # Exit condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Shutdown
cap.release()
cv2.destroyAllWindows()
aidrone.Close()
```

---

## **5. Project Summary**
### **Key Features**
- **Obstacle detection using a camera**
- **CNN model training for obstacle recognition**
- **Automatic obstacle avoidance when detected**

### **Potential Improvements**
- **Applying YOLO or other object detection models**
- **Using reinforcement learning for autonomous flight optimization**
- **Combining GPS to optimize obstacle avoidance paths**

This practice provides experience in **deep learning-based real-time image processing and autonomous drone flight**. 🚀

