# 🛳️ Ship Detection in SAR Imagery

A comprehensive **Maritime Intelligence System** that detects, tracks, classifies, and analyzes ships in Synthetic Aperture Radar (SAR) imagery using YOLOv8n, ByteTrack, and advanced analytics — all presented via an interactive Streamlit dashboard.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green.svg)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red.svg)

---

## 🚀 Features

### Core (Must-Build)
| Feature | Description | Stack |
|---------|-------------|-------|
| YOLOv8n + ByteTrack | Real-time ship detection & multi-object tracking | ultralytics, OpenCV |
| Threat Scoring | 0-100 risk score (confidence, zone, speed, dwell) | NumPy, Shapely |
| Multi-Zone Alerts | Named zones with cooldown — no alert spam | Shapely, deque |
| SAR Speckle Filter | Lee/Frost adaptive despeckling | OpenCV, NumPy |

### Unique
| Feature | Description | Stack |
|---------|-------------|-------|
| Dark Vessel Detection | Flag ships with no AIS match | Pandas, Shapely |
| Ship Type Classifier | Cargo/Tanker/Fishing/Military (EfficientNet) | timm, torchvision |
| Fleet Detection | DBSCAN convoy/formation clustering | scikit-learn |
| Trajectory Prediction | Kalman filter forecast arrows | filterpy, NumPy |

### Wow Factor
| Feature | Description | Stack |
|---------|-------------|-------|
| GeoJSON Map Overlay | Ships on interactive Folium/Leaflet map | Folium, pyproj |
| NL Query Interface | "Show ships in zone B from last 10 min" | regex parser |
| Temporal Heatmap | Animated ship density GIF export | matplotlib, imageio |
| PDF/HTML Reports | One-click downloadable session summary | ReportLab |

---

## 📦 Installation

```bash
# Clone / navigate to project
cd ship-detection-sar

# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

---

## 🏃 Quick Start

### 1. Launch Dashboard
```bash
streamlit run dashboard/app.py
```

### 2. Upload a SAR Image
- Go to **🔍 Detection** page
- Upload a SAR image (.png, .jpg, .tif)
- Click **Run Detection**

### 3. Explore
- **🗺️ Map View** — Ships plotted on interactive map
- **📊 Analytics** — Fleet detection, trajectory, heatmap
- **⚠️ Alerts** — Zone violations, dark vessels, NL queries
- **📄 Reports** — Download PDF/HTML reports

---

## 🏋️ Training

```bash
# Train YOLOv8n on SSDD dataset
python scripts/train.py --data data/yolo_format/dataset.yaml --epochs 50

# Evaluate
python scripts/evaluate.py --weights models/yolov8n_sar.pt --data data/yolo_format/dataset.yaml
```

---

## 📁 Project Structure

```
ship-detection-sar/
├── config/           # Settings, zone definitions
├── data/             # Datasets, mock AIS data
├── models/           # Trained model weights
├── src/
│   ├── preprocessing/  # Speckle filter, augmentation, Sentinel-1
│   ├── detection/      # YOLOv8 detector, tracker, classifier
│   ├── analytics/      # Threat scoring, zones, dark vessel, fleet, trajectory
│   ├── visualization/  # Renderer, Folium map, heatmap
│   ├── reporting/      # NL query, PDF/HTML reports
│   └── pipeline.py     # End-to-end orchestration
├── dashboard/          # Streamlit UI (5 pages)
├── scripts/            # Training & evaluation
└── requirements.txt
```

---

## 🔬 Technical Approach

1. **Preprocessing**: SAR images are despeckled using Lee/Frost adaptive filters, then normalized and tiled for inference.
2. **Detection**: YOLOv8n detects ship candidates with configurable confidence thresholds; ByteTrack assigns persistent track IDs.
3. **Classification**: Each detection is cropped and classified (cargo/tanker/fishing/military) via EfficientNet.
4. **Analytics**: Multi-factor threat scoring, zone violation alerts, AIS-based dark vessel detection, DBSCAN fleet clustering, and Kalman trajectory prediction.
5. **Visualization**: All results are rendered on images, plotted on Folium maps, and exportable as GIFs and PDF/HTML reports.

### Limitations
- Training data is limited (SSDD ~1,160 images); mAP may be lower than production systems
- Ship classification uses heuristic fallback without trained EfficientNet weights
- AIS data is simulated for demo; real AIS integration requires maritime data feeds
- Sentinel-1 preprocessing uses simplified radiometric calibration

---

## 📄 License

This project is for educational and research purposes (IIIT Sri City).
