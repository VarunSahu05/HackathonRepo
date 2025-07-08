
# 🚦 UrbanFlux: YOLOv3-Based Intelligent Traffic Regulation Platform

UrbanFlux is a real-time, AI-powered traffic signal control system that adapts dynamically based on live vehicle density using **YOLOv3** and **OpenCV**—eliminating outdated fixed-timer traffic systems and transforming urban traffic management.

## 📌 Problem Statement

Urban traffic congestion stems from **rigid, fixed-duration traffic lights** that fail to adapt to live vehicle flow. This results in:

* Increased wait times
* Higher fuel consumption
* Elevated emissions
* Wasted signal time in underutilized lanes

## 💡 Our Solution

UrbanFlux reimagines traffic signaling using **computer vision and deep learning**, offering:

* Real-time vehicle detection via YOLOv3
* Adaptive traffic light durations per lane
* Traffic data logging for future optimization
* Priority to high-density/critical lanes
* No external sensors—only vision-based input

---

## 🔧 Functional Workflow

```plaintext

Start → System Init → Capture Live Feed → YOLOv3 Detection → Vehicle Count → Signal Timing Logic → Update Signals → Store Data → Repeat

```

### ⚙️ Modules:

1. **Camera Module**: Captures live video feeds per lane
2. **YOLOv3 Engine**: Detects and counts vehicles in each frame
3. **Traffic Signal Logic**:

   * Calculates green time dynamically (10–60s)
   * Gives less time to empty lanes, more to busy ones
4. **Signal Update Engine**: Updates hardware/controller signals
5. **Data Logger**: Stores vehicle count and signal timing for analytics

---

## 🚀 Features

* Real-time adaptive signal control
* YOLOv3 for high-speed object detection
* Lane-wise dynamic logic (no static timer)
* Camera-only solution (no external sensors)
* Traffic data archive for trend analysis
* Continuous operation powered by OpenCV

---

## 🧪 Novelty & Innovation

| Feature                | Traditional | UrbanFlux |
| ---------------------- | ----------- | --------- |
| Static Signal Timing   | ✅ Yes       | ❌ No      |
| Real-Time Adjustment   | ❌ No        | ✅ Yes     |
| Vision-Only System     | ❌ No        | ✅ Yes     |
| Priority-Based Lanes   | ❌ No        | ✅ Yes     |
| Scalable Across Cities | ⚠️ Limited  | ✅ Yes     |

---

## ⚠️ Drawbacks

* Requires high-performance edge devices (GPU/TPU)
* Camera reliability affects accuracy (weather, vandalism)
* Real-time processing delays can hinder performance
* Privacy concerns due to continuous video capture

---

## ❌ Showstoppers (Critical Failures)

| Failure Type          | Impact                                      |
| --------------------- | ------------------------------------------- |
| Camera Malfunction    | Vehicle detection halts                     |
| Low Frame Rate or Lag | Late signal updates                         |
| Power Outage          | System fails unless supported by UPS        |
| Blocked Vision        | Miscount or default to inefficient patterns |

---

## 📊 Performance Benchmarks

| Method                 | Accuracy   |
| ---------------------- | ---------- |
| YOLOv3 (Deep Learning) | **90–95%** |
| Background Subtractor  | 75–90%     |
| Traditional Methods    | 60–75%     |

---

## 📌 Setup (Basic Overview)

> Full source code and detailed setup instructions will be available in the repository.

* Python (>= 3.7)
* OpenCV
* YOLOv3 weights + config
* GPU-enabled system (for live YOLO inference)
* Video streams or IP cameras
* Traffic light controller (simulation or hardware interface)

---

## 🛠️ Future Enhancements

* Integrate license plate recognition for smart tolling
* Weather adaptation model for vision enhancement
* Add edge computing for on-site decisions

---

## 👨‍💻 Team Byte Bandits

| Name            | Email                                                       | Contact    |
| --------------- | ----------------------------------------------------------- | ---------- |
| Alekhya Kanderi | [alekhyak383@gmail.com](mailto:alekhyak383@gmail.com)       | 7396803962 |
| Varun Sahu      | [varunsahu91825@gmail.com](mailto:varunsahu91825@gmail.com) | 8897818655 |
| Akash Dupathi   | [akashdupathi@gmail.com](mailto:akashdupathi@gmail.com)     | 6305448977 |

---
