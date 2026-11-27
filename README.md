# AI-Based Smart Traffic Management System

A student-friendly prototype using YOLO object detection, ByteTrack tracking, OpenCV and Streamlit.

## Features
- Detects and classifies cars, bikes/motorcycles, buses and trucks.
- Tracks vehicles across video frames (where the tracker assigns IDs).
- Estimates congestion from vehicles detected per analyzed frame.
- Displays traffic-count trends and congestion frequency.
- Exports an annotated MP4 and frame-level CSV report.

## Requirements
- Python 3.10 or 3.11 recommended
- Internet connection on first run so Ultralytics can download `yolov8n.pt`
- A traffic video (`.mp4`, `.avi`, `.mov` or `.mkv`)

## Run on Windows
Open PowerShell or the VS Code terminal in this folder:

```bash
python -m venv .venv
.venv\\Scripts\\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Your browser will open the dashboard. Upload a traffic video and click **Analyze traffic video**.

## Run on macOS/Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## How it works
1. Streamlit accepts a traffic video.
2. OpenCV reads the video frame by frame.
3. YOLO detects COCO vehicle classes: car (2), motorcycle (3), bus (5), truck (7).
4. ByteTrack associates detections between frames.
5. The app counts vehicles in each analyzed frame and assigns Low/Moderate/High congestion using the thresholds in the sidebar.
6. Results are plotted and exported as CSV and annotated MP4.

## Important project limitations
- Congestion thresholds are heuristic and depend on camera angle, road width, traffic density and frame sampling. Calibrate them using sample videos.
- A vehicle may be counted in more than one frame in the class detection totals. The unique tracked vehicle metric uses tracker IDs and is a better estimate, but IDs can reset or be missed.
- This prototype does not calculate real-world speed or automatically control traffic lights. Speed estimation requires camera calibration and a known road-distance reference.
- Do not use this prototype for enforcement or safety-critical traffic control without validation.

## Suggested academic extensions
- Add a virtual counting line and count each tracked vehicle only once when it crosses.
- Add separate lane-wise counts and peak-hour comparisons.
- Calibrate congestion thresholds with manually labeled sample clips.
- Add a dashboard database (SQLite) and compare traffic across dates.
- For a signal-control simulation, use SUMO rather than connecting this prototype to real traffic lights.

## Suggested report structure
1. Abstract
2. Problem statement and objectives
3. Literature review (computer vision, YOLO, object tracking)
4. System architecture and tools
5. Dataset/video collection and preprocessing
6. Implementation
7. Results and evaluation
8. Limitations and future scope
9. Conclusion and references

## Evaluation plan
Manually label a small test clip (e.g., 100 sampled frames), compare detected counts with ground truth, and report:
- Detection precision = TP / (TP + FP)
- Detection recall = TP / (TP + FN)
- F1-score = 2 * precision * recall / (precision + recall)
- Count error = absolute(predicted count - ground-truth count)

Do not report invented accuracy numbers; measure them on your own test videos.
