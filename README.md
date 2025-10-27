# Vision-Based Lane Detection and Obstacle-Aware Decision Making

An Advanced Driver Assistance System (ADAS) prototype that integrates lane detection, object detection, and speed calibration modules using OpenCV, YOLO, and image processing.  
---

## Author
- Shanmukha Vinayak M
---

## Features

- Lane Detection using Canny edge detection and Hough Transform.
- Object Detection using YOLOv5 for vehicle and pedestrian awareness.
- Speed Calibration using pixel-distance and frame-time estimation.
- Voice Alerts through gTTS for real-time warnings.
- Visualization and video output generation using MoviePy.

---

## Tech Stack

| Category | Tools/Libraries |
|-----------|-----------------|
| Programming | Python 3.10 |
| Computer Vision | OpenCV, NumPy, Matplotlib |
| Object Detection | YOLOv5 (Ultralytics) |
| Voice Alerts | gTTS, playsound |
| Visualization | moviepy |
| Deep Learning | PyTorch, TorchVision |


---

## Example Results

| Module | Output |
|---------|---------|
| Lane Detection | Identified lane boundaries with region masking |
| Object Detection | YOLOv5 detection bounding boxes |
| Combined Output | Lane + Object + Speed info overlaid on video frames |



---

## Results Summary

- Multi-module ADAS prototype integration successful.
- Reliable lane detection under normal lighting.
- YOLO-based obstacle detection operational in real-time.
- Voice alerts triggered appropriately for detected hazards.

---

## Future Scope

- Deep Learning-based lane models (SCNN/LaneNet).
- Sensor Fusion with GPS and IMU.
- Weather and night vision adaptability.
- Embedded optimization on Jetson or Raspberry Pi.
- Decision-making logic for autonomous control.

---

## Installation and Usage

### 1. Clone the Repository
```bash
git clone https://github.com/ShanmukhaGit0101/Lane_Detection-Obstacle-AwareDecision-Making.git
cd Lane_Detection-Obstacle-AwareDecision-Making
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Main Script
```bash
python main.py
```





## Repository Link
https://github.com/ShanmukhaGit0101/Lane_Detection-Obstacle-AwareDecision-Making
