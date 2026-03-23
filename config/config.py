"""
Global configuration for Ship Detection in SAR Imagery project.
"""
import os
from pathlib import Path

# ── Project Paths ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

SSDD_DIR = DATA_DIR / "ssdd"
SENTINEL_DIR = DATA_DIR / "sentinel1"
YOLO_DATA_DIR = DATA_DIR / "yolo_format"
MOCK_AIS_PATH = DATA_DIR / "mock_ais.csv"
ZONES_PATH = PROJECT_ROOT / "config" / "zones.json"

# ── Model Settings ─────────────────────────────────────────────────────────
YOLO_WEIGHTS = MODELS_DIR / "yolov8n_sar.pt"
CLASSIFIER_WEIGHTS = MODELS_DIR / "ship_classifier.pt"
YOLO_PRETRAINED = "yolov8n.pt"  # Base pretrained weights for fine-tuning

# ── Detection Settings ─────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.25
NMS_IOU_THRESHOLD = 0.45
INPUT_IMAGE_SIZE = 640
DEVICE = "cuda" if os.environ.get("USE_GPU", "1") == "1" else "cpu"

# ── Speckle Filter Settings ───────────────────────────────────────────────
SPECKLE_FILTER_TYPE = "lee"   # "lee" or "frost"
SPECKLE_WINDOW_SIZE = 7

# ── Tracker Settings ───────────────────────────────────────────────────────
TRACKER_TYPE = "bytetrack"
TRACK_HIGH_THRESH = 0.5
TRACK_LOW_THRESH = 0.1
TRACK_BUFFER = 30  # frames to keep lost tracks

# ── Threat Score Weights ───────────────────────────────────────────────────
THREAT_WEIGHTS = {
    "confidence": 0.3,
    "zone_proximity": 0.3,
    "speed": 0.2,
    "dwell_time": 0.2,
}
THREAT_LEVELS = {
    "LOW": (0, 33),
    "MEDIUM": (34, 66),
    "HIGH": (67, 100),
}

# ── Zone Alert Settings ───────────────────────────────────────────────────
ALERT_COOLDOWN_SECONDS = 30
MAX_ALERT_LOG_SIZE = 500

# ── Dark Vessel Detection ─────────────────────────────────────────────────
AIS_MATCH_RADIUS_METERS = 500

# ── Fleet Detection (DBSCAN) ──────────────────────────────────────────────
FLEET_DBSCAN_EPS = 50          # pixels
FLEET_DBSCAN_MIN_SAMPLES = 3

# ── Trajectory Prediction (Kalman) ────────────────────────────────────────
KALMAN_PREDICT_STEPS = 10
KALMAN_DT = 1.0  # time step in seconds

# ── Ship Classifier ───────────────────────────────────────────────────────
SHIP_CLASSES = ["cargo", "tanker", "fishing", "military"]
CLASSIFIER_INPUT_SIZE = 224

# ── Heatmap Settings ──────────────────────────────────────────────────────
HEATMAP_RESOLUTION = (640, 640)
HEATMAP_COLORMAP = "hot"
HEATMAP_GIF_FPS = 5

# ── Dashboard Settings ────────────────────────────────────────────────────
DASHBOARD_TITLE = "SAR Maritime Intelligence System"
DASHBOARD_ICON = "🛳️"
DASHBOARD_LAYOUT = "wide"
DASHBOARD_THEME = "dark"

# ── Training Settings ─────────────────────────────────────────────────────
TRAIN_EPOCHS = 50
TRAIN_BATCH_SIZE = 16
TRAIN_IMAGE_SIZE = 640
TRAIN_PATIENCE = 10

# Create output directories
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
