"""
Confidence-Based Threat Scoring Module.

Computes a 0–100 risk score for each detected ship using:
- Detection confidence
- Proximity to restricted zones
- Estimated speed
- Dwell time in sensitive areas

Ships are classified as LOW (green), MEDIUM (amber), or HIGH (red) threat.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from shapely.geometry import Point, Polygon
import json
import logging

logger = logging.getLogger(__name__)

from src.detection.detector import Detection
from src.detection.tracker import Track


# Default scoring weights
DEFAULT_WEIGHTS = {
    "confidence": 0.3,
    "zone_proximity": 0.3, 
    "speed": 0.2,
    "dwell_time": 0.2,
}

THREAT_COLORS = {
    "LOW": "#22C55E",      # Green
    "MEDIUM": "#F59E0B",   # Amber
    "HIGH": "#EF4444",     # Red
}


def compute_threat_score(
    detection: Detection,
    track: Optional[Track] = None,
    zones: Optional[List[Dict]] = None,
    weights: Optional[Dict[str, float]] = None,
    max_speed: float = 50.0,
    max_dwell: float = 300.0
) -> Tuple[float, str]:
    """
    Compute a threat score (0–100) for a single ship detection.
    
    Scoring components:
        1. Confidence (inverted): Lower detection confidence → higher suspicion
           since legitimate vessels typically have clear radar signatures.
        2. Zone proximity: Closer to restricted zones → higher threat.
        3. Speed: Unusual speeds (very fast or very slow) → higher threat.
        4. Dwell time: Longer time loitering near sensitive areas → higher threat.
    
    Args:
        detection: Current detection.
        track: Track history (if available).
        zones: List of zone polygons with pixel coordinates.
        weights: Scoring weights dict.
        max_speed: Maximum expected speed for normalization.
        max_dwell: Maximum dwell time for normalization.
        
    Returns:
        Tuple of (score 0-100, threat_level string).
    """
    w = weights or DEFAULT_WEIGHTS
    
    # ── Component 1: Detection confidence (inverted) ──
    # Lower confidence is more suspicious (could be partially submerged, stealth)
    conf_score = (1.0 - detection.confidence) * 100
    
    # ── Component 2: Zone proximity ──
    zone_score = 0.0
    if zones:
        cx, cy = detection.center
        point = Point(cx, cy)
        
        min_distance = float('inf')
        inside_zone = False
        
        for zone in zones:
            coords = zone.get("pixel_coordinates", [])
            if len(coords) < 3:
                continue
            
            poly = Polygon(coords)
            if poly.contains(point):
                inside_zone = True
                zone_score = 100.0
                break
            
            dist = poly.exterior.distance(point)
            min_distance = min(min_distance, dist)
        
        if not inside_zone and min_distance < float('inf'):
            # Normalize distance (closer = higher score)
            zone_score = max(0, 100 - min_distance * 0.5)
    
    # ── Component 3: Speed anomaly ──
    speed_score = 0.0
    if track and track.speed_pixels_per_sec > 0:
        speed = track.speed_pixels_per_sec
        normalized_speed = min(speed / max_speed, 1.0)
        
        # Both very fast and very slow are suspicious
        if normalized_speed > 0.8:
            speed_score = normalized_speed * 100  # Very fast
        elif normalized_speed < 0.1 and track.frames_seen > 5:
            speed_score = 70  # Loitering (very slow but persistent)
        else:
            speed_score = normalized_speed * 50  # Normal range
    
    # ── Component 4: Dwell time ──
    dwell_score = 0.0
    if track:
        dwell = track.dwell_time
        dwell_score = min(dwell / max_dwell, 1.0) * 100
    
    # ── Weighted combination ──
    total_score = (
        w["confidence"] * conf_score +
        w["zone_proximity"] * zone_score +
        w["speed"] * speed_score +
        w["dwell_time"] * dwell_score
    )
    
    total_score = np.clip(total_score, 0, 100)
    
    # ── Determine threat level ──
    if total_score >= 67:
        threat_level = "HIGH"
    elif total_score >= 34:
        threat_level = "MEDIUM"
    else:
        threat_level = "LOW"
    
    return float(total_score), threat_level


def score_all_detections(
    detections: List[Detection],
    tracks: Optional[Dict[int, Track]] = None,
    zones: Optional[List[Dict]] = None,
    weights: Optional[Dict[str, float]] = None
) -> List[Detection]:
    """
    Compute threat scores for all detections in a frame.
    Updates Detection objects in-place.
    
    Args:
        detections: List of detections.
        tracks: Active tracks dict keyed by track_id.
        zones: Zone definitions.
        weights: Scoring weights.
        
    Returns:
        Updated detections with threat_score and threat_level set.
    """
    for det in detections:
        track = None
        if tracks and det.track_id in tracks:
            track = tracks[det.track_id]
        
        score, level = compute_threat_score(
            detection=det,
            track=track,
            zones=zones,
            weights=weights
        )
        
        det.threat_score = score
        det.threat_level = level
        det.metadata["threat_color"] = THREAT_COLORS[level]
    
    return detections


def load_zones(zones_path: str) -> List[Dict]:
    """Load zone definitions from JSON file."""
    try:
        with open(zones_path, 'r') as f:
            data = json.load(f)
        return data.get("zones", [])
    except Exception as e:
        logger.error(f"Failed to load zones: {e}")
        return []
