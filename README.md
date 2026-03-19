
# Smart Traffic Violation Detection System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/YOLOv8-Computer%20Vision-00FFFF?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/OpenCV-27338e?style=for-the-badge&logo=opencv&logoColor=white"/>
  <img src="https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white"/>
  <img src="https://img.shields.io/badge/EasyOCR-Number%20Plate-orange?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge"/>
</p>

> A real-time AI-powered traffic surveillance system that automatically detects traffic violations — red light jumping, helmet violations, and number plate recognition — using YOLOv8 and computer vision.

---

## 📌 Overview

Smart Traffic Violation Detection System is a complete computer vision pipeline for intelligent traffic surveillance. It processes live camera feeds or video files to detect three types of violations simultaneously — saving violation snapshots, reading number plates, and logging all incidents to a live web dashboard in real time.

---

## 🚦 Features

- 🚦 **Red Light Violation Detection** — detects vehicles crossing the stop line during a red signal
- 🏍️ **Helmet Detection** — identifies riders without helmets using YOLOv8 classification
- 🚗 **Number Plate Recognition** — reads vehicle number plates using EasyOCR
- 📸 **Violation Snapshots** — auto-saves timestamped images of every violation
- 📊 **Live Dashboard** — real-time violation log accessible from any browser via Flask
- 🗃️ **Violation Database** — all incidents stored with plate number, type, and timestamp

---

## 🏗️ System Architecture

```
[Traffic Camera / Video File]
         ↓
[Smart Traffic Violation Detection System]
    ├── Red Light Module    → Signal state detection + stop line crossing
    ├── Helmet Module       → YOLOv8 — helmet / no-helmet classification
    ├── Number Plate Module → EasyOCR — reads plate text
    ├── Snapshot Engine     → Saves violation images with timestamp
    └── Flask REST API      → Serves live violation log + dashboard
         ↓
[WiFi / LAN Network]
         ↓
[Traffic Violation Dashboard — any browser]
```

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Object Detection | YOLOv8 (Ultralytics) |
| Helmet Classification | YOLOv8 Custom Model |
| Number Plate OCR | EasyOCR |
| Video Processing | OpenCV |
| Signal Detection | HSV Color Segmentation |
| Backend API | Flask (Python) |
| Frontend Dashboard | HTML + Chart.js |
| Database | SQLite |

---

## 🚀 Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

### Run on Live Camera
```bash
git clone https://github.com/SAM-WESLEY/Smart-Traffic-Violation-Detection
cd Smart-Traffic-Violation-Detection
python main.py --source 0
```

### Run on Video File
```bash
python main.py --source traffic_video.mp4
```

### Access Dashboard
```
http://localhost:5000
```

---

## 📊 Detection Capabilities

| Violation Type | Method | Accuracy |
|---|---|---|
| Red Light Jumping | HSV Signal Detection + Stop Line | ~92% |
| No Helmet | YOLOv8 Custom Classification | ~94% |
| Number Plate Reading | EasyOCR | ~88% |

---

## 🎛️ Keyboard Controls

| Key | Action |
|---|---|
| `S` | Toggle signal state (RED / GREEN) manually |
| `R` | Reset violation counter |
| `ESC` | Quit |

---

## 🗂️ Project Structure

```
Smart-Traffic-Violation-Detection/
├── main.py                        # Main detection pipeline + Flask stream
├── modules/
│   ├── red_light_detector.py      # Signal state + stop line violation
│   ├── helmet_detector.py         # YOLOv8 helmet detection
│   ├── plate_reader.py            # EasyOCR number plate recognition
│   └── violation_logger.py        # SQLite violation database + snapshot saver
├── dashboard/
│   └── index.html                 # Live violation dashboard
├── models/
│   └── helmet_model.pt            # Custom trained YOLOv8 helmet model
├── violations/                    # Auto-saved violation snapshots
├── violations.db                  # SQLite violation database
├── requirements.txt
└── README.md
```

---

## 📋 Sample Violation Log

```
╔══════════════════════════════════════════════════════════════╗
║           SMART TRAFFIC VIOLATION LOG                        ║
╠══════╦════════════════╦═══════════════╦════════════════════╣
║  ID  ║   Plate No.    ║  Violation    ║  Timestamp         ║
╠══════╬════════════════╬═══════════════╬════════════════════╣
║  001 ║  TN 09 AB 1234 ║  Red Light    ║  2025-03-15 09:14  ║
║  002 ║  MH 12 CD 5678 ║  No Helmet    ║  2025-03-15 09:15  ║
║  003 ║  KA 05 EF 9012 ║  Red Light    ║  2025-03-15 09:16  ║
╚══════╩════════════════╩═══════════════╩════════════════════╝
```

---

## 📬 Contact

**Sam Wesley S**
📧 samwesley@karunya.edu.in
🔗 [LinkedIn](https://linkedin.com/in/samwesleys)
🐙 [GitHub](https://github.com/SAM-WESLEY)

---

<p align="center">
  <i>Built with ❤️ at Karunya Institute of Technology and Sciences</i>
</p>

<p align="center">If this project helped you, please give it a ⭐</p>
