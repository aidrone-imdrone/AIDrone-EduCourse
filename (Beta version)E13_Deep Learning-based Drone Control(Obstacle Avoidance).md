# Visual Obstacle Detection and Avoidance System for Mini Drones

This project implements a lightweight obstacle detection and avoidance system for Raspberry Pi Zero 2W-based mini drones with 8520 DC motors. The system uses computer vision and deep learning to detect obstacles in real-time and navigate around them.

![Obstacle Detection Demo](https://github.com/user-attachments/assets/placeholder-image.jpg)

## Features

- Real-time obstacle detection using MobileNet SSD Lite
- Lightweight implementation optimized for Raspberry Pi Zero 2W
- Simple avoidance algorithm with directional decision-making
- Low latency processing for responsive flight control
- Support for different obstacle classes (people, furniture, walls, etc.)

## Hardware Requirements

- Raspberry Pi Zero 2W
- Raspberry Pi Camera Module (v2 recommended)
- Mini drone frame with 8520 DC motors
- Flight controller compatible with Raspberry Pi
- 3.7V LiPo battery
- Lightweight chassis for mounting the Raspberry Pi

## Software Requirements

- Raspberry Pi OS (Lite version recommended)
- Python 3.7+
- TensorFlow Lite
- OpenCV
- RPi.GPIO or appropriate motor control library

## Installation

```bash
# Update system packages
sudo apt update
sudo apt upgrade -y

# Install dependencies
sudo apt install -y python3-pip python3-opencv libjpeg-dev libopenjp2-7-dev

# Install Python packages
pip3 install RPi.GPIO numpy tflite-runtime opencv-python-headless picamera2

# Clone this repository
git clone https://github.com/yourusername/drone-obstacle-detection.git
cd drone-obstacle-detection

# Download the pre-trained model
./download_model.sh
```

## How It Works

### 1. Object Detection

The system uses a pre-trained MobileNet SSD model that has been optimized for TensorFlow Lite to detect common obstacles. The model has been quantized to improve inference speed on the Raspberry Pi Zero 2W.

### 2. Obstacle Analysis

Once obstacles are detected, the system analyzes their position, size, and class to determine the level of threat. Objects directly in the drone's path are given highest priority.

### 3. Avoidance Strategy

The system implements a simple yet effective avoidance strategy:
- If obstacles are detected on the left, move right
- If obstacles are detected on the right, move left
- If obstacles are directly ahead, hover and then move upward if possible
- When multiple obstacles are detected, choose the path with maximum clearance

### 4. Control Integration

Commands are sent to the flight controller to adjust the drone's trajectory based on the avoidance strategy.

## Usage

```bash
# Basic usage
python3 obstacle_avoidance.py

# With visualization (requires HDMI connection or VNC)
python3 obstacle_avoidance.py --visualize

# Adjust detection confidence threshold
python3 obstacle_avoidance.py --threshold 0.5

# For testing without motor control
python3 obstacle_avoidance.py --simulation
```

## Code Structure

- `obstacle_avoidance.py`: Main script for the obstacle detection and avoidance system
- `drone_control.py`: Functions for controlling the drone motors
- `detection_utils.py`: Utility functions for object detection and processing
- `models/`: Directory containing the TensorFlow Lite model and labels

## Implementation Example

Below is a simplified implementation of the core obstacle detection and avoidance system:

```python
import time
import numpy as np
import cv2
from picamera2 import Picamera2
import tflite_runtime.interpreter as tflite
from drone_control import DroneController

class ObstacleAvoidanceSystem:
    def __init__(self, model_path="models/detect.tflite", 
                 label_path="models/labelmap.txt", 
                 threshold=0.5,
                 visualize=False):
        # Initialize camera
        self.camera = Picamera2()
        config = self.camera.create_preview_configuration(main={"size": (640, 480)})
        self.camera.configure(config)
        self.camera.start()
        time.sleep(1)  # Give camera time to initialize
        
        # Load TFLite model
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        # Get model details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.height = self.input_details[0]['shape'][1]
        self.width = self.input_details[0]['shape'][2]
        
        # Load labels
        with open(label_path, 'r') as f:
            self.labels = [line.strip() for line in f.readlines()]
            
        # Initialize drone controller
        self.drone = DroneController()
        
        # Set detection threshold
        self.threshold = threshold
        self.visualize = visualize
        
        print("Obstacle Avoidance System Initialized")
        
    def preprocess_image(self, image):
        # Resize and convert to RGB
        input_image = cv2.resize(image, (self.width, self.height))
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        
        # Normalize and add batch dimension
        input_image = input_image.astype(np.float32) / 127.5 - 1
        input_image = np.expand_dims(input_image, axis=0)
        
        return input_image
        
    def detect_obstacles(self, image):
        # Preprocess image
        input_data = self.preprocess_image(image)
        
        # Set input tensor
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        
        # Run inference
        self.interpreter.invoke()
        
        # Get results
        boxes = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        classes = self.interpreter.get_tensor(self.output_details[1]['index'])[0]
        scores = self.interpreter.get_tensor(self.output_details[2]['index'])[0]
        
        # Filter detections based on threshold
        valid_detections = scores > self.threshold
        
        return boxes[valid_detections], classes[valid_detections], scores[valid_detections]
    
    def calculate_avoidance_direction(self, image, boxes):
        h, w, _ = image.shape
        center_x = w // 2
        
        # Analyze where obstacles are located
        left_zone_obstacles = 0
        right_zone_obstacles = 0
        center_zone_obstacles = 0
        
        for box in boxes:
            y_min, x_min, y_max, x_max = box
            
            # Convert normalized coordinates to pixel values
            x_min = int(x_min * w)
            x_max = int(x_max * w)
            
            # Calculate center of the obstacle
            obj_center_x = (x_min + x_max) // 2
            
            # Determine which zone the obstacle is in
            if obj_center_x < center_x - w//6:  # Left zone
                left_zone_obstacles += 1
            elif obj_center_x > center_x + w//6:  # Right zone
                right_zone_obstacles += 1
            else:  # Center zone
                center_zone_obstacles += 1
        
        # Decide avoidance direction
        if center_zone_obstacles > 0:
            # If obstacle directly ahead, prioritize going up or to the side with fewer obstacles
            if left_zone_obstacles <= right_zone_obstacles:
                return "LEFT"
            else:
                return "RIGHT"
        elif left_zone_obstacles > right_zone_obstacles:
            return "RIGHT"
        elif right_zone_obstacles > left_zone_obstacles:
            return "LEFT"
        else:
            return "FORWARD"
    
    def apply_avoidance_control(self, direction):
        if direction == "LEFT":
            print("Obstacle detected on right - Moving left")
            self.drone.move_left()
        elif direction == "RIGHT":
            print("Obstacle detected on left - Moving right")
            self.drone.move_right()
        elif direction == "UP":
            print("Obstacle detected ahead - Moving up")
            self.drone.move_up()
        else:
            print("Path clear - Moving forward")
            self.drone.move_forward()
    
    def visualize_detection(self, image, boxes, classes, scores):
        h, w, _ = image.shape
        for i in range(len(boxes)):
            # Get box coordinates
            y_min, x_min, y_max, x_max = boxes[i]
            
            # Convert normalized coordinates to pixel values
            x_min = int(x_min * w)
            x_max = int(x_max * w)
            y_min = int(y_min * h)
            y_max = int(y_max * h)
            
            # Draw rectangle around detected object
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            
            # Add label
            class_id = int(classes[i])
            label = f"{self.labels[class_id]}: {scores[i]:.2f}"
            cv2.putText(image, label, (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return image
    
    def run(self):
        try:
            while True:
                # Capture frame
                image = self.camera.capture_array()
                
                # Detect obstacles
                boxes, classes, scores = self.detect_obstacles(image)
                
                if len(boxes) > 0:
                    # Calculate avoidance direction
                    direction = self.calculate_avoidance_direction(image, boxes)
                    
                    # Apply avoidance control
                    self.apply_avoidance_control(direction)
                    
                    # Visualize if enabled
                    if self.visualize:
                        image = self.visualize_detection(image, boxes, classes, scores)
                        cv2.imshow("Obstacle Detection", image)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                else:
                    # No obstacles, continue forward
                    self.drone.move_forward()
                
                # Sleep to control frame rate
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("Stopping obstacle avoidance system")
        
        finally:
            # Clean up
            self.camera.stop()
            if self.visualize:
                cv2.destroyAllWindows()
            self.drone.stop()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Drone Obstacle Avoidance System')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Detection confidence threshold')
    parser.add_argument('--visualize', action='store_true',
                        help='Enable visualization')
    parser.add_argument('--simulation', action='store_true',
                        help='Run in simulation mode (no motor control)')
    
    args = parser.parse_args()
    
    # Create and run the system
    obstacle_system = ObstacleAvoidanceSystem(
        threshold=args.threshold,
        visualize=args.visualize
    )
    
    obstacle_system.run()
```

## Drone Control Module Example

Below is a simplified example of the drone control module:

```python
import time
import RPi.GPIO as GPIO

class DroneController:
    def __init__(self):
        # GPIO setup for motor control
        GPIO.setmode(GPIO.BCM)
        
        # Define GPIO pins for motors
        self.MOTOR_PINS = {
            'front_left': {'pin': 17, 'direction': 18},
            'front_right': {'pin': 22, 'direction': 23},
            'back_left': {'pin': 24, 'direction': 25},
            'back_right': {'pin': 27, 'direction': 26}
        }
        
        # Setup GPIO pins
        for motor in self.MOTOR_PINS.values():
            GPIO.setup(motor['pin'], GPIO.OUT)
            GPIO.setup(motor['direction'], GPIO.OUT)
            
            # Initialize PWM for speed control
            motor['pwm'] = GPIO.PWM(motor['pin'], 100)
            motor['pwm'].start(0)
        
        print("Drone Controller Initialized")
    
    def _set_motor_speed(self, motor_name, speed, direction):
        """Set speed and direction of a specific motor"""
        motor = self.MOTOR_PINS[motor_name]
        GPIO.output(motor['direction'], direction)
        motor['pwm'].ChangeDutyCycle(speed)
    
    def move_forward(self):
        """Move drone forward"""
        print("Moving forward")
        self._set_motor_speed('front_left', 70, 1)
        self._set_motor_speed('front_right', 70, 1)
        self._set_motor_speed('back_left', 70, 1)
        self._set_motor_speed('back_right', 70, 1)
    
    def move_left(self):
        """Move drone left"""
        print("Moving left")
        self._set_motor_speed('front_left', 40, 1)
        self._set_motor_speed('front_right', 70, 1)
        self._set_motor_speed('back_left', 40, 1)
        self._set_motor_speed('back_right', 70, 1)
    
    def move_right(self):
        """Move drone right"""
        print("Moving right")
        self._set_motor_speed('front_left', 70, 1)
        self._set_motor_speed('front_right', 40, 1)
        self._set_motor_speed('back_left', 70, 1)
        self._set_motor_speed('back_right', 40, 1)
    
    def move_up(self):
        """Move drone up"""
        print("Moving up")
        self._set_motor_speed('front_left', 90, 1)
        self._set_motor_speed('front_right', 90, 1)
        self._set_motor_speed('back_left', 90, 1)
        self._set_motor_speed('back_right', 90, 1)
    
    def hover(self):
        """Hover in place"""
        print("Hovering")
        self._set_motor_speed('front_left', 60, 1)
        self._set_motor_speed('front_right', 60, 1)
        self._set_motor_speed('back_left', 60, 1)
        self._set_motor_speed('back_right', 60, 1)
    
    def stop(self):
        """Stop all motors"""
        print("Stopping motors")
        self._set_motor_speed('front_left', 0, 0)
        self._set_motor_speed('front_right', 0, 0)
        self._set_motor_speed('back_left', 0, 0)
        self._set_motor_speed('back_right', 0, 0)
        
        # Clean up GPIO
        GPIO.cleanup()
```

## Model Download Script

Here's a utility script to download the pre-trained model:

```bash
#!/bin/bash

MODEL_DIR="models"
mkdir -p $MODEL_DIR

echo "Downloading TFLite model..."
wget -O $MODEL_DIR/detect.tflite https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip
unzip -j $MODEL_DIR/detect.tflite -d $MODEL_DIR
rm $MODEL_DIR/detect.tflite

echo "Downloading label map..."
wget -O $MODEL_DIR/labelmap.txt https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/lite/examples/object_detection/android/app/src/main/assets/labelmap.txt

echo "Download complete!"
```

## Performance Optimization

For Raspberry Pi Zero 2W, which has limited computational resources, several optimizations have been implemented:

1. **Model Quantization**: Using 8-bit quantized models to reduce memory and computational requirements
2. **Reduced Resolution**: Processing 320x240 or smaller frames to reduce computational load
3. **Frame Skipping**: Processing only every other frame to maintain real-time performance
4. **ROI Processing**: Only analyzing regions of interest rather than the entire frame
5. **Simplified Decision Logic**: Using efficient decision algorithms rather than complex path planning

## Calibration and Testing

Before deploying the system on the drone:

1. Test the object detection model with the camera while the drone is stationary
2. Calibrate the motor responses to different commands
3. Test the system in a controlled environment with soft obstacles
4. Gradually increase flight complexity as reliability is confirmed

## Troubleshooting

- **High Latency**: Reduce image resolution or increase frame skipping
- **False Detections**: Increase the confidence threshold or filter specific classes
- **Erratic Movements**: Adjust the motor control parameters and smoothing algorithms
- **Battery Drain**: Reduce processing frequency or optimize model further
- **Camera Issues**: Ensure proper camera connection and adequate lighting

## Future Improvements

- Implement depth estimation using stereo vision or IR sensors
- Add path memory to avoid oscillating between avoidance directions
- Implement more sophisticated path planning algorithms
- Add learning capabilities to improve avoidance strategies over time
- Integrate with GPS for outdoor navigation scenarios

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- TensorFlow Lite team for the pre-trained models
- Raspberry Pi Foundation
- The open-source drone community for inspiration and code references
