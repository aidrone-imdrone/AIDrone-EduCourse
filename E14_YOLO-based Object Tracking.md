# **YOLO-Based Specific Object Tracking Project**

## **1. Overview**
### **Project Objective**
This project implements a **real-time object tracking drone system using YOLO (You Only Look Once)**. The YOLO model detects specific objects (e.g., people, cars) in a camera stream, and the user can select an object to be tracked by the drone.

### **Required Technologies and Libraries**
- **Python**: Programming language
- **OpenCV**: Real-time image processing
- **YOLOv4/v5**: Object detection model
- **pyaidrone**: Drone control library
- **NumPy**: Data processing and array operations
- **Raspberry Pi Zero 2W**: Drone's computing unit
- **Camera Module**: Image data acquisition

### **Features**
- **Real-time object detection using YOLO**
- **User-selectable object tracking**
- **Drone movement tracking based on object position**
- **Ability to train YOLO with new objects**

---

## **2. Environment Setup**
### **Installing Required Packages**
```bash
pip install opencv-python numpy torch torchvision pyyaml pyaidrone
```

### **Downloading YOLO Model**
Download the model from the [YOLOv5 official repository](https://github.com/ultralytics/yolov5):
```bash
git clone https://github.com/ultralytics/yolov5
cd yolov5
pip install -r requirements.txt
```

---

## **3. Adding a New Object to YOLO**
If the YOLO model does not recognize the object you want to track, follow these steps to train a custom model.

### **Step 1: Collect Data**
- Capture images containing the new object from different angles. (See Example in E13)
- Alternatively, use publicly available datasets.
- Save images in `dataset/images/`.

### **Step 2: Resize Images**
To ensure consistent training performance, resize images to 640x640 pixels.
```python
import cv2
import os

dataset_path = "dataset/images/"
output_path = "dataset/resized/"
os.makedirs(output_path, exist_ok=True)

for file in os.listdir(dataset_path):
    img = cv2.imread(os.path.join(dataset_path, file))
    img_resized = cv2.resize(img, (640, 640))
    cv2.imwrite(os.path.join(output_path, file), img_resized)
```

### **Step 3: Label Data for YOLO**
- Use [LabelImg](https://github.com/heartexlabs/labelImg) to annotate images.
```bash
pip install labelImg
labelImg
```
- Open `LabelImg`, select `dataset/resized/`, and label objects.
- Save annotations in YOLO format (`.txt` files with bounding box information).

### **Step 4: Prepare YOLO Dataset**
Organize the dataset as follows:
```
yolo_custom_dataset/
  ├── images/
  │   ├── train/   # Training images
  │   ├── val/     # Validation images
  ├── labels/
  │   ├── train/   # Training labels
  │   ├── val/     # Validation labels
```
Move `80%` of images to `train/` and `20%` to `val/`.

Create `custom_data.yaml`:
```yaml
train: /path_to_dataset/yolo_custom_dataset/images/train
val: /path_to_dataset/yolo_custom_dataset/images/val
nc: 1  # Number of classes
names: ['custom_object']  # Name of your object
```

### **Step 5: Train the YOLO Model**
```bash
python train.py --img 640 --batch 16 --epochs 50 --data custom_data.yaml --weights yolov5s.pt
```
- The trained model is saved as `runs/train/exp/weights/best.pt`.

### **Step 6: Test the Trained Model**
```bash
python detect.py --weights runs/train/exp/weights/best.pt --img 640 --conf 0.4 --source test.jpg
```
```bash
python detect.py --weights runs/train/exp/weights/best.pt --img 640 --conf 0.4 --source video.mp4
```

### **Step 7: Integrating the Model with YOLO**
```python
import torch
import cv2
import numpy as np

# Load trained YOLO model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='runs/train/exp/weights/best.pt', source='local')

# Load image
detect_img = cv2.imread('test.jpg')
results = model(detect_img)

# Display results
results.show()
```

## **4. Implementing User-Selectable Object Tracking**
This code allows the user to click on a detected object to track it.

```python
import torch
import cv2
import numpy as np
from pyaidrone.aiDrone import *
from pyaidrone.deflib import *

# Load YOLO model (use custom-trained model if available)
model = torch.hub.load('ultralytics/yolov5', 'custom', path='custom_model.pt', source='local')

# Initialize drone
aidrone = AIDrone()
aidrone.Open("COM3")

# Set up camera
cap =  cv.VideoCapture("http://<your AIDrone IP>/?action=stream") 
selected_object = None
selected_bbox = None

def select_object(event, x, y, flags, param):
    global selected_object, selected_bbox
    if event == cv2.EVENT_LBUTTONDOWN:
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            if x1 < x < x2 and y1 < y < y2:
                selected_object = model.names[int(cls)]
                selected_bbox = (x1, y1, x2, y2)
                print(f"Selected Object: {selected_object}")
                break

cv2.namedWindow("YOLO Object Tracking")
cv2.setMouseCallback("YOLO Object Tracking", select_object)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    detections = results.xyxy[0].cpu().numpy()
    center_x, center_y = frame.shape[1] // 2, frame.shape[0] // 2
    target_x, target_y = None, None

    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        label = model.names[int(cls)]
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2, y2)), (0, 255, 0), 2)
        cv2.putText(frame, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        if selected_object and label == selected_object:
            target_x = (x1 + x2) / 2
            target_y = (y1 + y2) / 2

    if target_x and target_y:
        if target_x < center_x - 50:
            aidrone.velocity(LEFT, 50)  # Move left
        elif target_x > center_x + 50:
            aidrone.velocity(RIGHT, 50)  # Move right
        else:
            aidrone.velocity(FRONT, 50)  # Move forward
    else:
        aidrone.velocity(FRONT, 0)  # Stop

    cv2.imshow('YOLO Object Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
aidrone.Close()
```

Now, clicking on a detected object will enable the drone to track it. Additionally, custom-trained models can be used to track new objects.

---

## **5. Project Summary**
### **Key Features**
- **YOLO-based object detection**
- **User-selectable object tracking**
- **Capability to train YOLO for new objects**
- **Drone follows the selected object**

### **Potential Improvements**
- **Improve tracking accuracy using YOLOv7 or newer models**
- **Expand object detection and tracking capabilities**
- **Use reinforcement learning for optimal tracking paths**
- **Integrate GPS and LiDAR for outdoor tracking**

This project demonstrates how to use YOLO for object detection and autonomous drone tracking. 🚀


