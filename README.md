# AI-Object-Detection-Alert-System-using-YOLOv3-Tiny-and-OpenCV
Real-time object detection using YOLOv3-Tiny and OpenCV with automated video recording and email alert notifications.

## Project Overview

This project implements a real-time object detection and alert system using **YOLOv3-Tiny**, **OpenCV**, and **Python**. The system captures live video from a webcam, detects objects using a pre-trained YOLO model, and automatically records a short video clip when an object is detected. The recorded clip is then sent via email to the user as an alert notification.

The system is designed for **basic surveillance, intrusion monitoring, and automated alert generation**.

## Key Features

* Real-time object detection using YOLOv3-Tiny
* Webcam video monitoring using OpenCV
* Automatic video recording when an object is detected
* Email alert system with recorded video attachment
* Detection timestamp and object labeling
* Lightweight and fast detection using YOLOv3-Tiny model

## Technologies Used

* Python
* OpenCV (cv2)
* YOLOv3-Tiny Object Detection Model
* SMTP Email Protocol
* Deep Learning (DNN module in OpenCV)

## Working Principle

1. The system captures video from the webcam using OpenCV.
2. Each frame is processed and passed to the YOLOv3-Tiny neural network model.
3. The model detects objects using the **COCO dataset class labels**.
4. If an object is detected with confidence greater than the threshold, the system:

   * Identifies the object label
   * Records a 5-second video clip
5. The recorded video is saved with a timestamp.
6. An email alert is automatically sent with the video clip attached.

## System Components

* Python program
* YOLOv3-Tiny weights and configuration files
* COCO dataset class names
* Webcam for live video capture
* Email notification system using SMTP

## Output

* Detected object label displayed on console
* Recorded video clip of detection event
* Email notification with video evidence

## Applications

* Home security systems
* Smart surveillance monitoring
* Intrusion detection systems
* Automated alert systems
* AI-based monitoring solutions

## Future Improvements

* Integration with IoT cloud platforms
* Mobile notification system
* Multi-object tracking
* Real-time dashboard for monitoring
