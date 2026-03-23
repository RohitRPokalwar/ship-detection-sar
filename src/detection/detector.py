"""
YOLOv8 Ship Detection Module.

Wraps the Ultralytics YOLOv8 model for SAR ship detection inference
with configurable confidence thresholds and NMS parameters.
"""

import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)

try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False
    logger.warning("ultralytics not installed. YOLOv8 detection will be unavailable.")


@dataclass
class Detection:
    """Single detection result."""
    bbox: List[int]            # [x1, y1, x2, y2] pixel coordinates
    confidence: float          # Detection confidence score
    class_id: int = 0          # Class index (0 = ship for single-class)
    class_name: str = "ship"   # Class label
    track_id: int = -1         # Assigned by tracker (-1 = untracked)
    ship_type: str = ""        # Assigned by classifier
    threat_score: float = 0.0  # Assigned by threat scorer
    threat_level: str = ""     # LOW / MEDIUM / HIGH
    is_dark_vessel: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def center(self) -> tuple:
        """Bounding box center (x, y)."""
        return (
            (self.bbox[0] + self.bbox[2]) // 2,
            (self.bbox[1] + self.bbox[3]) // 2
        )
    
    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]
    
    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]
    
    @property
    def area(self) -> int:
        return self.width * self.height


class ShipDetector:
    """
    YOLOv8-based ship detector for SAR imagery.
    
    Supports:
        - Single image and batch inference
        - Configurable confidence and NMS thresholds
        - Automatic device selection (CPU/GPU)
    """
    
    def __init__(
        self,
        weights_path: Optional[str] = None,
        confidence: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = "cpu",
        image_size: int = 640
    ):
        """
        Initialize the ship detector.
        
        Args:
            weights_path: Path to YOLOv8 weights (.pt file).
                         Uses yolov8n.pt if None.
            confidence: Minimum detection confidence.
            iou_threshold: NMS IoU threshold.
            device: "cuda" or "cpu".
            image_size: Input image size for inference.
        """
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.device = device
        self.image_size = image_size
        self.model = None
        
        if not HAS_ULTRALYTICS:
            logger.error("ultralytics package required for ShipDetector")
            return
        
        if weights_path and Path(weights_path).exists():
            logger.info(f"Loading model weights from: {weights_path}")
            self.model = YOLO(weights_path)
        else:
            logger.info("Loading pretrained YOLOv8n weights")
            self.model = YOLO("models/runs/yolov8n_sar3/weights/best.pt")
        
        logger.info(f"Detector initialized: conf={confidence}, iou={iou_threshold}, device={device}")
    
    def detect(
        self,
        image: np.ndarray,
        confidence: Optional[float] = None,
        iou_threshold: Optional[float] = None
    ) -> List[Detection]:
        """
        Run ship detection on a single image.
        
        Args:
            image: Input image (BGR or grayscale).
            confidence: Override default confidence threshold.
            iou_threshold: Override default NMS IoU threshold.
            
        Returns:
            List of Detection objects.
        """
        if self.model is None:
            logger.error("Model not loaded")
            return []
        
        conf = confidence or self.confidence
        iou = iou_threshold or self.iou_threshold
        
        # Convert grayscale to 3-channel if needed
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # Run inference
        results = self.model.predict(
            source=image,
            conf=conf,
            iou=iou,
            device=self.device,
            imgsz=self.image_size,
            verbose=False
        )
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            
            for i in range(len(boxes)):
                bbox = boxes.xyxy[i].cpu().numpy().astype(int).tolist()
                conf_score = float(boxes.conf[i].cpu().numpy())
                cls_id = int(boxes.cls[i].cpu().numpy())
                cls_name = result.names.get(cls_id, "ship")
                
                det = Detection(
                    bbox=bbox,
                    confidence=conf_score,
                    class_id=cls_id,
                    class_name=cls_name
                )
                detections.append(det)
        
        logger.debug(f"Detected {len(detections)} ships")
        return detections
    
    def detect_batch(
        self,
        images: List[np.ndarray],
        confidence: Optional[float] = None
    ) -> List[List[Detection]]:
        """
        Run detection on a batch of images.
        
        Args:
            images: List of input images.
            confidence: Override confidence threshold.
            
        Returns:
            List of detection lists (one per image).
        """
        all_detections = []
        for img in images:
            dets = self.detect(img, confidence=confidence)
            all_detections.append(dets)
        return all_detections
    
    def detect_with_tracking(
        self,
        image: np.ndarray,
        tracker_type: str = "bytetrack.yaml",
        persist: bool = True
    ) -> List[Detection]:
        """
        Run detection with built-in Ultralytics tracking.
        
        Args:
            image: Input image.
            tracker_type: Tracker config ("bytetrack.yaml" or "botsort.yaml").
            persist: Persist tracks across frames.
            
        Returns:
            List of Detection objects with track_id assigned.
        """
        if self.model is None:
            return []
        
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        results = self.model.track(
            source=image,
            conf=self.confidence,
            iou=self.iou_threshold,
            device="cpu",
            imgsz=self.image_size,
            tracker=tracker_type,
            persist=persist,
            verbose=False
        )
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            
            for i in range(len(boxes)):
                bbox = boxes.xyxy[i].cpu().numpy().astype(int).tolist()
                conf_score = float(boxes.conf[i].cpu().numpy())
                cls_id = int(boxes.cls[i].cpu().numpy())
                cls_name = result.names.get(cls_id, "ship")
                
                # Get track ID if available
                track_id = -1
                if boxes.id is not None:
                    track_id = int(boxes.id[i].cpu().numpy())
                
                det = Detection(
                    bbox=bbox,
                    confidence=conf_score,
                    class_id=cls_id,
                    class_name=cls_name,
                    track_id=track_id
                )
                detections.append(det)
        
        return detections
