# Lane_Detection-Obstacle-AwareDecision-Making
# 🚗 Advanced Lane Detection using Computer Vision

A robust **lane detection system** built using Python and OpenCV, inspired by the Advanced Lane Lines project.  
This implementation performs **camera calibration, perspective transformation, thresholding, and lane overlay** on road videos — ideal for autonomous driving research and ADAS experimentation.

---

## 🎯 Project Overview

This project demonstrates a complete **lane detection pipeline** that identifies lane lines in road images and videos.  
It combines multiple computer vision techniques to detect lane curvature, lane deviation, and overlay visual cues for driver assistance.

---

## ⚙️ Features

- 📷 **Camera Calibration** — Corrects lens distortion using chessboard images.  
- 🎨 **Thresholding** — Applies gradient and color thresholds to isolate lane pixels.  
- 🔁 **Perspective Transformation** — Warps images to a top-down “bird’s-eye” view.  
- 🧠 **Lane Detection** — Identifies lane lines using sliding window search and polynomial fitting.  
- 🎥 **Video Processing** — Detects lanes frame-by-frame and saves annotated output.

---

## 🧩 System Flow

```text
Input Frame → Undistortion → Thresholding → Perspective Transform
             → Lane Detection → Inverse Transform → Overlay → Output
