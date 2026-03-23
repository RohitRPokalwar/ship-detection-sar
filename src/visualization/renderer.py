"""
Detection Visualization Renderer.

Draws bounding boxes, ship type labels, threat score badges,
Kalman prediction arrows, dark vessel highlights, and fleet
formation ellipses on SAR images.
"""

import numpy as np
import cv2
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

from src.detection.detector import Detection
from src.analytics.fleet_detect import Fleet

# ── Color definitions ──────────────────────────────────────────────────
THREAT_COLORS_BGR = {
    "LOW": (78, 197, 34),       # Green
    "MEDIUM": (11, 158, 245),   # Amber  
    "HIGH": (68, 68, 239),      # Red
}

DARK_VESSEL_COLOR = (0, 0, 200)     # Deep red
FLEET_COLOR = (255, 200, 0)         # Cyan-ish
PREDICTION_COLOR = (0, 255, 255)    # Yellow
AIS_COLOR = (0, 200, 0)            # Green

SHIP_TYPE_ICONS = {
    "cargo": "📦",
    "tanker": "🛢️",
    "fishing": "🎣",
    "military": "⚔️",
}


def draw_detections(
    image: np.ndarray,
    detections: List[Detection],
    show_labels: bool = True,
    show_threat_scores: bool = True,
    show_track_ids: bool = True,
    line_thickness: int = 2
) -> np.ndarray:
    """
    Draw bounding boxes and labels for all detections.
    
    Color-codes by threat level. Adds ship type, confidence,
    and threat score badges.
    """
    canvas = image.copy()
    if len(canvas.shape) == 2:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        
        # Choose color based on threat level or dark vessel status
        if det.is_dark_vessel:
            color = DARK_VESSEL_COLOR
        elif det.threat_level in THREAT_COLORS_BGR:
            color = THREAT_COLORS_BGR[det.threat_level]
        else:
            color = (0, 255, 0)
        
        # Draw bounding box
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, line_thickness)
        
        if show_labels:
            # Build label text
            parts = []
            
            if show_track_ids and det.track_id >= 0:
                parts.append(f"#{det.track_id}")
            
            if det.ship_type:
                parts.append(det.ship_type.upper())
            
            parts.append(f"{det.confidence:.2f}")
            
            if show_threat_scores and det.threat_score > 0:
                parts.append(f"T:{det.threat_score:.0f}")
            
            if det.is_dark_vessel:
                parts.append("DARK")
            
            label = " | ".join(parts)
            
            # Draw label background
            font_scale = 0.45
            font = cv2.FONT_HERSHEY_SIMPLEX
            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, 1)
            
            label_y = max(y1 - 8, text_h + 4)
            cv2.rectangle(
                canvas,
                (x1, label_y - text_h - 4),
                (x1 + text_w + 4, label_y + 4),
                color, -1
            )
            
            # Draw text
            text_color = (0, 0, 0) if det.threat_level != "HIGH" else (255, 255, 255)
            if det.is_dark_vessel:
                text_color = (255, 255, 255)
            
            cv2.putText(
                canvas, label,
                (x1 + 2, label_y),
                font, font_scale, text_color, 1, cv2.LINE_AA
            )
        
        # Draw threat score badge (small colored circle)
        if show_threat_scores and det.threat_score > 0:
            badge_x = x2 + 5
            badge_y = y1 + 10
            radius = 8
            cv2.circle(canvas, (badge_x, badge_y), radius, color, -1)
            cv2.circle(canvas, (badge_x, badge_y), radius, (255, 255, 255), 1)
    
    return canvas


def draw_predictions(
    image: np.ndarray,
    predictions: Dict[int, List[Tuple[float, float]]],
    detections: Optional[List[Detection]] = None,
    color: Tuple[int, int, int] = PREDICTION_COLOR,
    arrow_thickness: int = 1
) -> np.ndarray:
    """
    Draw trajectory prediction arrows (dashed forecast lines).
    """
    canvas = image.copy()
    if len(canvas.shape) == 2:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    
    for track_id, future_positions in predictions.items():
        if len(future_positions) < 2:
            continue
        
        # Find current position from detections
        start_pos = None
        if detections:
            for det in detections:
                if det.track_id == track_id:
                    start_pos = det.center
                    break
        
        # Draw dashed prediction line
        all_points = []
        if start_pos:
            all_points.append(start_pos)
        all_points.extend(future_positions)
        
        for i in range(len(all_points) - 1):
            pt1 = (int(all_points[i][0]), int(all_points[i][1]))
            pt2 = (int(all_points[i+1][0]), int(all_points[i+1][1]))
            
            # Dashed line effect: alternate drawing
            if i % 2 == 0:
                cv2.line(canvas, pt1, pt2, color, arrow_thickness, cv2.LINE_AA)
            
            # Draw small circle at each prediction point
            cv2.circle(canvas, pt2, 2, color, -1)
        
        # Draw arrowhead at the last prediction
        if len(all_points) >= 2:
            last = (int(all_points[-1][0]), int(all_points[-1][1]))
            prev = (int(all_points[-2][0]), int(all_points[-2][1]))
            cv2.arrowedLine(canvas, prev, last, color, arrow_thickness + 1, cv2.LINE_AA, tipLength=0.3)
    
    return canvas


def draw_fleets(
    image: np.ndarray,
    fleets: List[Fleet],
    color: Tuple[int, int, int] = FLEET_COLOR,
    thickness: int = 2
) -> np.ndarray:
    """
    Draw fleet formation bounding ellipses.
    """
    canvas = image.copy()
    if len(canvas.shape) == 2:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    
    for fleet in fleets:
        if len(fleet.ship_positions) < 3:
            continue
        
        # Draw ellipse around fleet
        points = np.array(fleet.ship_positions, dtype=np.int32)
        
        if len(points) >= 5:
            # Fit ellipse
            ellipse = cv2.fitEllipse(points)
            cv2.ellipse(canvas, ellipse, color, thickness, cv2.LINE_AA)
        else:
            # Draw convex hull
            hull = cv2.convexHull(points)
            cv2.drawContours(canvas, [hull], 0, color, thickness, cv2.LINE_AA)
        
        # Label fleet
        cx, cy = int(fleet.centroid[0]), int(fleet.centroid[1])
        label = f"Fleet #{fleet.fleet_id} ({fleet.num_ships} ships)"
        cv2.putText(
            canvas, label,
            (cx - 40, cy - int(fleet.radius) - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA
        )
    
    return canvas


def draw_zones(
    image: np.ndarray,
    zones: List[Dict],
    alpha: float = 0.2
) -> np.ndarray:
    """
    Draw semi-transparent zone overlays on image.
    """
    canvas = image.copy()
    if len(canvas.shape) == 2:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    
    overlay = canvas.copy()
    
    for zone in zones:
        coords = zone.get("pixel_coordinates", [])
        if len(coords) < 3:
            continue
        
        pts = np.array(coords, dtype=np.int32)
        
        # Parse zone color
        hex_color = zone.get("color", "#FFA500")
        b = int(hex_color[5:7], 16)
        g = int(hex_color[3:5], 16)
        r = int(hex_color[1:3], 16)
        color = (b, g, r)
        
        # Draw filled polygon on overlay
        cv2.fillPoly(overlay, [pts], color)
        
        # Draw border
        cv2.polylines(canvas, [pts], True, color, 2, cv2.LINE_AA)
        
        # Label zone
        cx = int(np.mean([c[0] for c in coords]))
        cy = int(np.mean([c[1] for c in coords]))
        cv2.putText(
            canvas, zone["name"],
            (cx - 30, cy),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA
        )
    
    # Blend overlay
    canvas = cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0)
    
    return canvas


def draw_ais_markers(
    image: np.ndarray,
    ais_positions: List[Tuple[float, float]],
    color: Tuple[int, int, int] = AIS_COLOR,
    marker_size: int = 6
) -> np.ndarray:
    """Draw AIS vessel markers as triangles."""
    canvas = image.copy()
    if len(canvas.shape) == 2:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    
    for (x, y) in ais_positions:
        x, y = int(x), int(y)
        # Draw triangle marker
        pts = np.array([
            [x, y - marker_size],
            [x - marker_size, y + marker_size],
            [x + marker_size, y + marker_size]
        ], dtype=np.int32)
        cv2.drawContours(canvas, [pts], 0, color, -1)
        cv2.drawContours(canvas, [pts], 0, (255, 255, 255), 1)
    
    return canvas


def render_full_frame(
    image: np.ndarray,
    detections: List[Detection],
    predictions: Optional[Dict[int, List[Tuple[float, float]]]] = None,
    fleets: Optional[List[Fleet]] = None,
    zones: Optional[List[Dict]] = None,
    ais_positions: Optional[List[Tuple[float, float]]] = None
) -> np.ndarray:
    """
    Render all visual elements on a single frame.
    
    Order: zones → AIS → fleets → detections → predictions
    """
    canvas = image.copy()
    if len(canvas.shape) == 2:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    
    if zones:
        canvas = draw_zones(canvas, zones)
    
    if ais_positions:
        canvas = draw_ais_markers(canvas, ais_positions)
    
    if fleets:
        canvas = draw_fleets(canvas, fleets)
    
    canvas = draw_detections(canvas, detections)
    
    if predictions:
        canvas = draw_predictions(canvas, predictions, detections)
    
    return canvas
