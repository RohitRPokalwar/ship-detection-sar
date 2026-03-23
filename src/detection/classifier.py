"""
Ship Type Classifier Module.

Uses EfficientNet (via timm) as a secondary classification model.
For each YOLO-detected ship, crops the bounding box and classifies 
the ship type: cargo, tanker, fishing, or military.

This creates a two-model pipeline (YOLO detection + EfficientNet classification).
"""

import numpy as np
import cv2
from pathlib import Path
from typing import List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision import transforms
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False

from src.detection.detector import Detection

SHIP_CLASSES = ["cargo", "tanker", "fishing", "military"]


class ShipClassifier:
    """
    EfficientNet-based ship type classifier.
    
    Crops each YOLO detection, resizes to 224×224, and classifies 
    into one of the ship type categories.
    
    For MVP/demo: if no trained weights are available, uses a 
    confidence-based heuristic or random assignment with consistent 
    seeding based on detection features.
    """
    
    def __init__(
        self,
        weights_path: Optional[str] = None,
        model_name: str = "efficientnet_b0",
        input_size: int = 224,
        device: str = "cpu",
        num_classes: int = 4
    ):
        self.input_size = input_size
        self.device = device
        self.num_classes = num_classes
        self.classes = SHIP_CLASSES
        self.model = None
        self.transform = None
        self._use_heuristic = True
        
        if HAS_TORCH and HAS_TIMM:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            
            if weights_path and Path(weights_path).exists():
                try:
                    self.model = timm.create_model(
                        model_name, 
                        pretrained=False, 
                        num_classes=num_classes
                    )
                    state_dict = torch.load(weights_path, map_location=device)
                    self.model.load_state_dict(state_dict)
                    self.model.to(device)
                    self.model.eval()
                    self._use_heuristic = False
                    logger.info(f"Loaded classifier weights from {weights_path}")
                except Exception as e:
                    logger.warning(f"Failed to load classifier weights: {e}. Using heuristic.")
            else:
                logger.info("No classifier weights found. Using feature-based heuristic for demo.")
        else:
            logger.warning("torch/timm not available. Using heuristic classifier.")
    
    def classify(self, image: np.ndarray, detection: Detection) -> Tuple[str, float]:
        """
        Classify the ship type for a single detection.
        
        Args:
            image: Full frame image.
            detection: Detection with bounding box.
            
        Returns:
            Tuple of (ship_type, confidence).
        """
        # Crop the detection region
        x1, y1, x2, y2 = detection.bbox
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return "cargo", 0.5
        
        if not self._use_heuristic and self.model is not None:
            return self._classify_model(crop)
        else:
            return self._classify_heuristic(crop, detection)
    
    def _classify_model(self, crop: np.ndarray) -> Tuple[str, float]:
        """Classify using the trained EfficientNet model."""
        if len(crop.shape) == 2:
            crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
        elif crop.shape[2] == 4:
            crop = cv2.cvtColor(crop, cv2.COLOR_BGRA2RGB)
        else:
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        
        tensor = self.transform(crop).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(tensor)
            probs = F.softmax(output, dim=1)
            conf, pred = torch.max(probs, dim=1)
        
        ship_type = self.classes[pred.item()]
        confidence = conf.item()
        
        return ship_type, confidence
    
    def _classify_heuristic(self, crop: np.ndarray, detection: Detection) -> Tuple[str, float]:
        """
        Heuristic classification based on detection features.
        
        Uses aspect ratio, area, and intensity features to make a 
        deterministic classification. This provides a realistic demo
        without requiring a trained classifier.
        """
        h, w = crop.shape[:2]
        aspect_ratio = w / (h + 1e-6)
        area = detection.area
        
        # Use mean intensity as a feature
        mean_intensity = np.mean(crop)
        
        # Simple heuristic rules based on ship shape characteristics
        if aspect_ratio > 3.0 and area > 5000:
            # Long, large vessel → likely tanker
            ship_type = "tanker"
            conf = 0.72
        elif aspect_ratio > 2.5 and area > 3000:
            # Long, medium vessel → likely cargo
            ship_type = "cargo"
            conf = 0.68
        elif aspect_ratio < 1.5 and area < 2000:
            # Small, compact vessel → likely fishing
            ship_type = "fishing"
            conf = 0.65
        elif mean_intensity > 180 and area > 2000:
            # Bright radar return, medium size → military
            ship_type = "military"
            conf = 0.60
        elif aspect_ratio > 2.0:
            ship_type = "cargo"
            conf = 0.55
        else:
            # Default: deterministic based on hash of features
            feature_hash = hash((round(aspect_ratio, 1), area // 100, int(mean_intensity) // 10))
            ship_type = self.classes[feature_hash % len(self.classes)]
            conf = 0.50
        
        return ship_type, conf
    
    def classify_batch(
        self,
        image: np.ndarray,
        detections: List[Detection]
    ) -> List[Detection]:
        """
        Classify ship types for all detections in a frame.
        Updates each Detection's ship_type field in-place.
        
        Args:
            image: Full frame image.
            detections: List of detections.
            
        Returns:
            Updated detections with ship_type assigned.
        """
        for det in detections:
            ship_type, conf = self.classify(image, det)
            det.ship_type = ship_type
            det.metadata["classifier_confidence"] = conf
        
        return detections
