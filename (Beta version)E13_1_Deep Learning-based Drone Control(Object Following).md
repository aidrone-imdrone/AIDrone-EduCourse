# Drone Object Tracking Program User Manual

## Overview

This program is a UI application designed to control a mini coding drone built with a Raspberry Pi Zero 2W, camera, and FC (Flight Controller). It receives video from the Raspberry Pi through an MJPG-streamer service, allowing users to select and track specific objects.

## System Requirements

- Python 3.6 or higher
- Required libraries:
  - PyQt5
  - OpenCV (cv2)
  - NumPy
  - pyaidrone

## Installation

1. Install the required libraries:
```bash
pip install PyQt5 opencv-python numpy
```

2. Make sure the pyaidrone library or your drone control library is installed on your system.

3. Save the program code as `drone_tracking.py`.

## Running the Program

In a terminal or command prompt, run:

```bash
python drone_tracking.py
```

## Program Interface

The program UI is divided into two parts:
- Left: Control panel (settings, status, buttons, etc.)
- Right: Video display area

### Left Control Panel

#### 1. Connection Settings
- **Drone IP**: Enter the IP address of your Raspberry Pi Zero 2W.
- **MJPG Port**: Port number for MJPG-streamer (default: 80)
- **Command Port**: Port for drone control commands (default: 8081)
- **Connect/Disconnect** buttons: Connect to/disconnect from the drone and video stream

#### 2. Status
- Current connection status
- Tracking status
- FPS (Frames Per Second) display

#### 3. Object Selection Method
- Guide for users on how to select and track objects

#### 4. Object Tracking Control
- **Start Tracking** button: Begin tracking the selected object
- **Stop Tracking** button: Stop object tracking (drone maintains hovering)

#### 5. Detailed Object Information
- Displays information about the tracked object including position, size, HSV color values, etc.
- **Save Object** button: Save information about the currently tracked object
- **Load Object** button: Load saved object information

#### 6. Flight Control Settings
- **Take Off** button: Launch the drone
- **Land** button: Land the drone
- **Forward/Backward** buttons: Manual drone control

### Right Video Screen
- Displays real-time video from the drone camera
- Shows a green rectangle around the tracked object
- Displays a blue crosshair at the center of the screen (drone center point)

## How to Use

### 1. Connection Setup
1. Enter the drone IP address and ports.
2. Click the "Connect" button.
3. When the status changes to "Connection Successful", you are properly connected.

### 2. Drone Take Off
1. Click the "Take Off" button or press the Enter key.
2. The drone will take off and hover at a height of approximately 1m.

### 3. Object Selection
1. Select the object you want to track by dragging a rectangle around it with your mouse on the video screen.
2. The selected object will be automatically analyzed and its HSV color information extracted.
3. After object selection, the status will change to "Object Selection Complete - Please Press 'Start Tracking' Button".

### 4. Start Tracking
1. Click the "Start Tracking" button or press the P key.
2. The drone will begin tracking the selected object.
3. The drone will automatically move to keep the object centered in the frame.

### 5. Stop Tracking
1. Click the "Stop Tracking" button or press the O key.
2. The drone will stop tracking and maintain hovering at its current position.

### 6. Landing
1. Click the "Land" button or press the Space key.
2. The drone will stop tracking and land.

### 7. Disconnect
1. Click the "Disconnect" button or press the Esc key.
2. The drone connection and video stream will be disconnected.

## Keyboard Shortcuts

- **Enter**: Take off
- **Space**: Land
- **P**: Start tracking
- **O**: Stop tracking
- **Esc**: Disconnect

## Troubleshooting

### Connection Failure
- Verify that the Raspberry Pi's IP address is correct.
- Check that MJPG-streamer is running.
- Make sure the port numbers are correct.
- Check the network connection status.

### Object Lost During Tracking
- If the object moves too quickly, tracking may be difficult.
- If the object's color is similar to the background, tracking may be challenging.
- Changes in lighting conditions may cause color-based tracking to fail.

## Notes

- The drone can track objects after takeoff, and if no object is visible on screen, it will continue hovering at a height of 1m.
- During object tracking, pressing the "Stop Tracking" button will cause the drone to maintain its hovering state.
- While tracking an object, pressing the land button will stop tracking and land the drone.
- Always land the drone and disconnect before closing the program.
