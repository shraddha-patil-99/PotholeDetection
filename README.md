# RealRoadMap – Pothole Detection using Edge AI (Bharat AI SOC)

This project is a **real-time pothole detection system** built for the **Bharat AI SOC Challenge**.  
It runs **entirely on-device** using a **Raspberry Pi (CPU only)** and detects **road anomalies (potholes)** from video or camera input using a **lightweight deep learning model**.


---

## What this project does

- Takes video input (dashcam / file / webcam)
- Looks only at the **road region** in the frame
- Classifies it as:
  - Normal road  
  - Pothole
- Uses **TensorFlow / TensorFlow Lite**
- Works in **real time** on Raspberry Pi (about 5–7 FPS)
- Uses **temporal filtering** to avoid false detections

---

##  Important Files

```
detect.py        → Run pothole detection on PC/laptop  
detect_pi.py     → Run pothole detection on Raspberry Pi  
train_model.py   → Train the CNN model  
convertTflite.py → Convert trained model to TFLite   
road.mp4, test.mp4     → Sample videos
```

---
## Models and Dataset

The trained models and dataset are not included in this repository due to GitHub file size limits.

Download them from:
https://drive.google.com/drive/folders/1TRrAZIa5LWgLQ6TrAMivyLb6GAZYsCFN?usp=sharing
https://drive.google.com/drive/folders/1up8Bql_ce2CPcAn_g5rIB4InMtYaHPv3?usp=sharing
https://drive.google.com/file/d/10YhEoVxkbLUGk0olHhs5aYRMQZCQKSyn/view?usp=sharing
https://drive.google.com/file/d/1xtItK3YE3YSWDnMwYfvvstHt5QXU76EI/view?usp=sharing

After downloading, place the model files in the project root folder.

##  Requirements

```bash
pip install opencv-python numpy tensorflow
```

For Raspberry Pi (TFLite):

```bash
pip install tflite-runtime
```

---

##  How to Run

### On PC / Laptop
```bash
python detect.py
```
Edit inside `detect.py`:
```python
VIDEO_SOURCE = "road.mp4"  # or 0 for webcam
```

---

###  On Raspberry Pi
```bash
python detect_pi.py
```
This uses:
- `pothole_model.tflite`
- CPU-only inference
- Optimized ROI processing

---

## Model Idea (Simple Explanation)

- Problem is treated as **binary classification**:  
  `Pothole` vs `Normal Road`
- Uses a **small CNN** (fast and lightweight)
- Input size: **224×224**
- Only bottom part of the frame (road) is processed
- Multiple frames are checked before confirming a pothole

---

## Performance (Approx)

- FPS: **5–7**
- Accuracy: **~82%**
- Runs fully on **Raspberry Pi CPU**

---

## Authors

- Shraddha Patil  
- Anshika Yadav  

---

## Why this matters

This project shows that **useful AI can run on low-cost edge devices** and can help in:
- Road monitoring
- Smart cities
- Low-cost infrastructure inspection
- Real-time anomaly detection systems
