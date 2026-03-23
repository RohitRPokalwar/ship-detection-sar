"""
End-to-End Processing Pipeline.
Orchestrates detection, tracking, classification, analytics, and visualization.
"""
import numpy as np
import cv2, json, time, logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import asdict

logger = logging.getLogger(__name__)

from src.detection.detector import ShipDetector, Detection
from src.detection.tracker import ShipTracker
from src.detection.classifier import ShipClassifier
from src.analytics.threat_score import score_all_detections, load_zones
from src.analytics.zone_alerts import ZoneAlertSystem
from src.analytics.dark_vessel import DarkVesselDetector
from src.analytics.fleet_detect import FleetDetector
from src.analytics.trajectory import TrajectoryPredictor
from src.visualization.renderer import render_full_frame
from src.visualization.heatmap import TemporalHeatmap
from src.preprocessing.speckle_filter import apply_speckle_filter


class SARPipeline:
    """Full detection-to-visualization pipeline."""

    def __init__(self, config=None):
        from config.config import (YOLO_WEIGHTS, CLASSIFIER_WEIGHTS, CONFIDENCE_THRESHOLD,
            NMS_IOU_THRESHOLD, DEVICE, INPUT_IMAGE_SIZE, ZONES_PATH, MOCK_AIS_PATH,
            SPECKLE_FILTER_TYPE, SPECKLE_WINDOW_SIZE, ALERT_COOLDOWN_SECONDS,
            FLEET_DBSCAN_EPS, FLEET_DBSCAN_MIN_SAMPLES, KALMAN_DT, KALMAN_PREDICT_STEPS)

        w = str(YOLO_WEIGHTS) if YOLO_WEIGHTS.exists() else None
        self.detector = ShipDetector(weights_path=w, confidence=CONFIDENCE_THRESHOLD,
                                      iou_threshold=NMS_IOU_THRESHOLD, device=DEVICE,
                                      image_size=INPUT_IMAGE_SIZE)
        self.tracker = ShipTracker()
        cw = str(CLASSIFIER_WEIGHTS) if CLASSIFIER_WEIGHTS.exists() else None
        self.classifier = ShipClassifier(weights_path=cw, device=DEVICE)
        self.zone_alert_system = ZoneAlertSystem(str(ZONES_PATH) if ZONES_PATH.exists() else None,
                                                  cooldown_seconds=ALERT_COOLDOWN_SECONDS)
        self.zones = load_zones(str(ZONES_PATH)) if ZONES_PATH.exists() else []
        ais = str(MOCK_AIS_PATH) if MOCK_AIS_PATH.exists() else None
        self.dark_vessel_detector = DarkVesselDetector(ais_data_path=ais)
        self.fleet_detector = FleetDetector(eps=FLEET_DBSCAN_EPS, min_samples=FLEET_DBSCAN_MIN_SAMPLES)
        self.trajectory_predictor = TrajectoryPredictor(dt=KALMAN_DT, predict_steps=KALMAN_PREDICT_STEPS)
        self.heatmap = TemporalHeatmap()
        self.filter_type = SPECKLE_FILTER_TYPE
        self.filter_size = SPECKLE_WINDOW_SIZE
        self.frame_count = 0
        self.all_detections_log: List[Dict] = []
        logger.info("SARPipeline initialized")

    def process_frame(self, image: np.ndarray, apply_filter: bool = False,
                      timestamp: Optional[float] = None) -> Dict:
        ts = timestamp or time.time()
        self.frame_count += 1

        if apply_filter and len(image.shape) == 2:
            image = apply_speckle_filter(image, self.filter_type, self.filter_size).astype(np.uint8)

        # 1. Detect
        detections = self.detector.detect_with_tracking(image)
        # 2. Track
        tracks = self.tracker.update(detections, self.frame_count, ts)
        # 3. Classify
        detections = self.classifier.classify_batch(image, detections)
        # 4. Threat score
        detections = score_all_detections(detections, tracks, self.zones)
        # 5. Zone alerts
        new_alerts = self.zone_alert_system.check_violations(detections, ts)
        # 6. Dark vessel check
        if not self.dark_vessel_detector._ais_positions:
            self.dark_vessel_detector.generate_simulated_ais(image.shape, detections=detections)
        detections = self.dark_vessel_detector.detect_dark_vessels(detections)
        # 7. Fleet detection
        fleets = self.fleet_detector.detect_fleets(detections)
        detections = self.fleet_detector.annotate_detections(detections, fleets)
        # 8. Trajectory prediction
        predictions = self.trajectory_predictor.update_tracks(tracks)
        # 9. Heatmap
        self.heatmap.add_detections(detections, image.shape)
        # 10. Render
        rendered = render_full_frame(image, detections, predictions, fleets,
                                      self.zones, self.dark_vessel_detector.get_ais_positions())

        det_dicts = []
        for d in detections:
            det_dicts.append({"track_id": d.track_id, "bbox": d.bbox, "confidence": d.confidence,
                "ship_type": d.ship_type, "threat_score": d.threat_score, "threat_level": d.threat_level,
                "is_dark_vessel": d.is_dark_vessel, "center": d.center, "metadata": d.metadata})
        self.all_detections_log.extend(det_dicts)

        return {"rendered_image": rendered, "detections": det_dicts, "tracks": tracks,
                "fleets": fleets, "predictions": predictions, "new_alerts": new_alerts,
                "alert_log": self.zone_alert_system.get_alert_log(),
                "frame_count": self.frame_count}

    def process_image(self, image_path: str, apply_filter: bool = True) -> Dict:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE if image_path.lower().endswith('.tif') else cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Cannot read: {image_path}")
        return self.process_frame(image, apply_filter)

    def get_session_summary(self) -> Dict:
        dets = self.all_detections_log
        return {"total_frames": self.frame_count, "total_detections": len(dets),
                "unique_ships": len(set(d["track_id"] for d in dets)),
                "dark_vessels": sum(1 for d in dets if d.get("is_dark_vessel")),
                "high_threat": sum(1 for d in dets if d.get("threat_level") == "HIGH"),
                "total_alerts": len(self.zone_alert_system.get_alert_log())}

    def reset(self):
        self.tracker.reset()
        self.trajectory_predictor.reset()
        self.heatmap.reset()
        self.zone_alert_system.clear_log()
        self.all_detections_log.clear()
        self.frame_count = 0
