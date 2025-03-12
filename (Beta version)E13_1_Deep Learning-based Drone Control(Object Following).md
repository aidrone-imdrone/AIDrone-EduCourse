
# Object Tracking Drone System User Guide

This document explains how to install and use a system that automatically tracks objects selected by users with a Raspberry Pi Zero 2W-based mini drone.

![Drone Object Tracking Example](https://github.com/user-attachments/assets/placeholder-tracking-image.jpg)

## System Overview

This system consists of two main components:

1. **Raspberry Pi Drone Controller** (`object_tracking_drone.py`): Runs on the Raspberry Pi to control the drone and track objects.
2. **PC Object Selection Client** (`pc_object_selector.py`): Runs on a PC to allow users to select objects to track and send tracking commands to the drone.

The system provides video streaming through mjpg-streamer and tracks objects based on HSV color ranges.

## Requirements

### Hardware Requirements
- Raspberry Pi Zero 2W
- Raspberry Pi Camera Module
- Mini drone with 8520 DC motors
- Drone-to-Raspberry Pi connection cable

### Software Requirements
- Raspberry Pi
  - Raspberry Pi OS
  - Python 3.7 or higher
  - OpenCV
  - NumPy
  - pyaidrone library
  - mjpg-streamer (installed and registered as a service)
- PC
  - Python 3.7 or higher
  - OpenCV
  - NumPy
  - tkinter (for GUI)
  - Pillow

## Installation

### Raspberry Pi Setup

1. Install required packages:
```bash
sudo apt update
sudo apt install -y python3-pip python3-opencv python3-numpy
pip3 install opencv-python-headless numpy
```

2. Verify that mjpg-streamer is running:
```bash
sudo systemctl status mjpg-streamer.service
```

3. Download the drone object tracking code:
```bash
git clone https://github.com/yourusername/drone-object-tracking.git
cd drone-object-tracking
```

### PC Setup

1. Install required packages:
```bash
pip install opencv-python numpy pillow
```

2. Download the PC client code:
```bash
git clone https://github.com/yourusername/drone-object-tracking.git
cd drone-object-tracking
```

## Usage

### 1. Run Object Tracking Drone Server on Raspberry Pi

Connect to your Raspberry Pi via SSH and run:

```bash
cd ~/drone-object-tracking
python3 object_tracking_drone.py
```

Additional options:
```bash
# Simulation mode (test without a drone)
python3 object_tracking_drone.py --simulation

# Configure different ports or URL
python3 object_tracking_drone.py --mjpg-url http://localhost:8080/?action=stream --cmd-port 8081 --port /dev/serial0
```

When the program starts, note the Raspberry Pi's IP address and port numbers.

### 2. Run Object Selection Client on PC

On your PC, run:

```bash
cd drone-object-tracking
python pc_object_selector.py
```

### 3. Connect to the Drone

In the PC client GUI:

1. Enter the Raspberry Pi's IP address and ports.
2. Click the "Connect" button.
3. When the connection is successful, a "Connected" message will be displayed.

### 4. Select and Track an Object

1. Select the object you want to track by dragging with your mouse on the video stream.
2. The color of the selected object will be automatically analyzed, and the HSV color range will be displayed.
3. Adjust the HSV sliders if necessary to fine-tune the object detection range.
4. Click the "Start Tracking" button.
5. The drone will automatically begin tracking the selected object.

### 5. Stop Tracking

To stop tracking, click the "Stop Tracking" button.

## Code Structure

### Raspberry Pi Code (`object_tracking_drone.py`)

```
object_tracking_drone.py
├── Main function
├── ObjectTrackingDrone class
│   ├── open_mjpg_stream(): Connect to mjpg-streamer
│   ├── read_mjpg_frame(): Read video frames
│   ├── detect_object(): Detect objects based on HSV color
│   ├── control_drone(): Drone control logic
│   └── run(): Main loop
└── CommandServer class: Receive commands from PC
```

### PC Client Code (`pc_object_selector.py`)

```
pc_object_selector.py
├── Main function
└── DroneObjectSelector class
    ├── create_widgets(): Set up GUI
    ├── connect_to_drone(): Connect to drone
    ├── select_object(): Process object selection
    ├── analyze_selection(): Analyze selected area colors
    ├── start_tracking(): Send tracking commands
    └── stream_video(): Process video stream
```

## How It Works

1. **Object Selection**: The user selects an object to track using the PC client.
2. **Color Analysis**: The HSV color range of the selected area is analyzed.
3. **Tracking Command**: The PC client sends the HSV color range and tracking command to the drone.
4. **Object Detection**: The drone uses the received HSV color range to detect the object in the camera feed.
5. **Drone Control**: The drone automatically moves to track the detected object based on its position.

## Troubleshooting

### Connection Issues
- Verify that the Raspberry Pi IP address and ports are correct.
- Check that mjpg-streamer is running.
- Check firewall settings.

### Object Detection Issues
- Adjust HSV sliders to fine-tune the color range.
- Ensure lighting conditions are adequate.
- Check that there is sufficient color contrast between the object and background.

### Drone Control Issues
- Verify that the serial port is correctly configured.
- Check drone battery status.
- Test in simulation mode first.

## Advanced Configuration

### Fine-tuning HSV Values
You can adjust the HSV (Hue, Saturation, Value) values to optimize detection ranges for various lighting environments and object colors:

- **H (Hue)**: Represents the color (0-179).
- **S (Saturation)**: Represents the purity of the color (0-255).
- **V (Value)**: Represents brightness (0-255).

### Adjusting Dead Zone
You can adjust the dead zone size (the area where the object is considered centered) for stable drone tracking:

```python
# In the Raspberry Pi code
self.dead_zone = 50  # Adjust this value to change dead zone size
```

## License

This project is provided under the MIT License.

## Contributing

You can contribute to this project by submitting issues or pull requests.
