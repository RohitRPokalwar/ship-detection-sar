"""
ByteTrack Multi-Object Tracker Module.

Manages persistent track IDs across video frames, maintaining
track history for downstream analytics (trajectory prediction,
speed estimation, dwell time calculation).
"""

import numpy as np
from collections import defaultdict, deque
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import time
import logging

logger = logging.getLogger(__name__)

from src.detection.detector import Detection


@dataclass
class Track:
    """Persistent track for a single ship across frames."""
    track_id: int
    positions: List[Tuple[int, int]] = field(default_factory=list)  # (x, y) centers
    timestamps: List[float] = field(default_factory=list)
    bboxes: List[List[int]] = field(default_factory=list)           # historical bboxes
    confidences: List[float] = field(default_factory=list)
    frames_seen: int = 0
    last_seen_frame: int = 0
    is_active: bool = True
    ship_type: str = ""
    
    @property
    def current_position(self) -> Optional[Tuple[int, int]]:
        return self.positions[-1] if self.positions else None
    
    @property
    def speed_pixels_per_sec(self) -> float:
        """Estimate speed from last two positions and timestamps."""
        if len(self.positions) < 2 or len(self.timestamps) < 2:
            return 0.0
        
        p1, p2 = self.positions[-2], self.positions[-1]
        t1, t2 = self.timestamps[-2], self.timestamps[-1]
        dt = t2 - t1
        
        if dt <= 0:
            return 0.0
        
        dist = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        return dist / dt
    
    @property
    def heading(self) -> float:
        """Estimate heading in degrees from last two positions."""
        if len(self.positions) < 2:
            return 0.0
        
        p1, p2 = self.positions[-2], self.positions[-1]
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        
        angle = np.degrees(np.arctan2(-dy, dx))  # screen coords: y-axis inverted
        return angle % 360
    
    @property
    def dwell_time(self) -> float:
        """Total time this track has been active (seconds)."""
        if len(self.timestamps) < 2:
            return 0.0
        return self.timestamps[-1] - self.timestamps[0]
    
    @property
    def average_confidence(self) -> float:
        if not self.confidences:
            return 0.0
        return np.mean(self.confidences)


class ShipTracker:
    """
    Manages ship tracks across frames.
    
    Works with YOLOv8's built-in ByteTrack or as a standalone
    track manager when detections already have track_ids assigned.
    Maintains full history for each track for analytics.
    """
    
    def __init__(self, max_history: int = 100, lost_threshold: int = 30):
        """
        Args:
            max_history: Maximum position history to keep per track.
            lost_threshold: Frames before a lost track is deactivated.
        """
        self.tracks: Dict[int, Track] = {}
        self.max_history = max_history
        self.lost_threshold = lost_threshold
        self.frame_count = 0
        self._next_id = 1
    
    def update(
        self,
        detections: List[Detection],
        frame_index: Optional[int] = None,
        timestamp: Optional[float] = None
    ) -> Dict[int, Track]:
        """
        Update tracks with new detections from a frame.
        
        If detections have track_ids (from YOLOv8's built-in tracker),
        those IDs are used. Otherwise, simple IoU matching is applied.
        
        Args:
            detections: Detections from the current frame.
            frame_index: Current frame number.
            timestamp: Current timestamp (seconds).
            
        Returns:
            Dictionary of active tracks.
        """
        self.frame_count += 1
        frame_idx = frame_index or self.frame_count
        ts = timestamp or time.time()
        
        for det in detections:
            tid = det.track_id
            
            if tid == -1:
                # No track ID assigned — assign one via simple matching
                tid = self._match_or_create(det)
                det.track_id = tid
            
            if tid not in self.tracks:
                self.tracks[tid] = Track(track_id=tid)
            
            track = self.tracks[tid]
            track.positions.append(det.center)
            track.timestamps.append(ts)
            track.bboxes.append(det.bbox)
            track.confidences.append(det.confidence)
            track.frames_seen += 1
            track.last_seen_frame = frame_idx
            track.is_active = True
            
            # Trim history
            if len(track.positions) > self.max_history:
                track.positions = track.positions[-self.max_history:]
                track.timestamps = track.timestamps[-self.max_history:]
                track.bboxes = track.bboxes[-self.max_history:]
                track.confidences = track.confidences[-self.max_history:]
        
        # Deactivate lost tracks
        for tid, track in self.tracks.items():
            if frame_idx - track.last_seen_frame > self.lost_threshold:
                track.is_active = False
        
        return self.get_active_tracks()
    
    def _match_or_create(self, detection: Detection) -> int:
        """
        Match detection to existing track by IoU or proximity,
        or create a new track.
        """
        best_match = -1
        best_iou = 0.3  # Minimum IoU threshold for matching
        
        for tid, track in self.tracks.items():
            if not track.is_active or not track.bboxes:
                continue
            
            iou = self._compute_iou(detection.bbox, track.bboxes[-1])
            if iou > best_iou:
                best_iou = iou
                best_match = tid
        
        if best_match != -1:
            return best_match
        
        # Create new track
        new_id = self._next_id
        self._next_id += 1
        return new_id
    
    @staticmethod
    def _compute_iou(box1: List[int], box2: List[int]) -> float:
        """Compute IoU between two [x1, y1, x2, y2] boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        
        return inter / (union + 1e-6)
    
    def get_active_tracks(self) -> Dict[int, Track]:
        """Return only active tracks."""
        return {tid: t for tid, t in self.tracks.items() if t.is_active}
    
    def get_all_tracks(self) -> Dict[int, Track]:
        """Return all tracks (active + inactive)."""
        return self.tracks.copy()
    
    def get_track(self, track_id: int) -> Optional[Track]:
        """Get a specific track by ID."""
        return self.tracks.get(track_id)
    
    def reset(self):
        """Clear all tracks."""
        self.tracks.clear()
        self.frame_count = 0
        self._next_id = 1
