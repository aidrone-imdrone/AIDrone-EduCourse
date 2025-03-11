# Raspberry Pi Camera-based Drone Control System

This project is a system that recognizes hand gestures using a Raspberry Pi camera and controls a drone through the recognized gestures. It uses a TensorFlow Lite model to run efficiently in lightweight environments.

## Features

- Real-time hand gesture recognition through Raspberry Pi camera
- Efficient inference using TensorFlow Lite model
- Recognition of hand gestures with over 90% confidence only
- Drone commands execute only when the same hand gesture is maintained for more than 1 second
- Control drone through various hand gestures (takeoff, landing, left/right movement, hovering)

## Requirements

- Raspberry Pi (3 or above recommended)
- Raspberry Pi camera module
- Compatible drone (supporting aiDrone library)
- Following Python packages:
  - tflite_runtime
  - picamera2
  - opencv-python
  - numpy
  - pyaidrone (library for drone control)

## Installation

```bash
# Install required packages
pip install tflite_runtime
pip install picamera2
pip install opencv-python
pip install numpy

# Install pyaidrone library (install the library appropriate for your drone model)
# Example: pip install pyaidrone
```

## Usage

1. Prepare TensorFlow Lite model and label file
   - Create a hand gesture model using Teachable Machine and export as TFLite model
   - Place the model file (`model.tflite`) and label file (`labels.txt`) in the project directory

<br/>

<img src="https://github.com/user-attachments/assets/e6a6ef49-487f-4f16-866d-165b9f7d097f"  width="800">

<br/><br/>

-**Tensorflow Lite Model Download Link for the example above** 

  https://drive.google.com/file/d/1as4y0XBxWTalh1ZJsO-hn06-9TRT9Kvw/view?usp=drive_link

<br/>

2. Run the code
   ```bash
   python gesture_drone_control_pi.py
   ```
- **Available in the tflite folder**

<br/>

3. Control drone with hand gestures
   - "TakeOff": Drone takeoff
   - "Landing": Drone landing
   - "Left": Move drone to the left
   - "Right": Move drone to the right
   - "Hovering": Drone hovers in place

<br/>

## Code Explanation

### Main Components

- **Camera Processing**: Uses Picamera2 library to get images from the Raspberry Pi camera.
- **Model Loading**: Uses TFLite interpreter to load the model and perform inference.
- **Hand Gesture Recognition**: Preprocesses camera input and recognizes hand gestures through the model.
- **Duration Check**: Executes commands only when the same hand gesture is maintained for more than 1 second.
- **Drone Control**: Sends commands to the drone according to the recognized hand gesture.

### Core Algorithm

```python
# Check if this is a new gesture
if current_gesture != last_gesture:
    gesture_start_time = time()  # Set start time to current time
    last_gesture = current_gesture
    
# Check if the gesture has been maintained for the required time (1 second)
elif time() - gesture_start_time >= gesture_duration_threshold:
    # Execute drone command
    # ...
```

This algorithm resets the timer whenever the user changes hand gestures, and executes drone commands only when the same hand gesture is maintained for more than 1 second. This prevents incorrect command execution due to accidental recognition errors.

## Customization

- **Port Setting**: Modify the port connected to the drone to match your environment (`aidrone.Open("COM3")` part)  =>  aidrone.Open("/dev/serial0")
- **Confidence Threshold**: Adjust confidence threshold as needed (`confidence_threshold = 90`)
- **Duration Time**: Adjust hand gesture maintenance time (`gesture_duration_threshold = 1.0`)
- **Additional Gestures**: Add more hand gestures and drone commands to expand functionality

## Troubleshooting

- **Camera Error**: Check camera connection and settings
- **Model Loading Failure**: Check model and label file paths
- **Drone Connection Error**: Check drone battery and connection port
- **Low Recognition Rate**: Retrain the model with more training data or improve lighting conditions
