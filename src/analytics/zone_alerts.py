"""
Multi-Zone Alert System with Cooldown.

Named zones (EEZ, fishing ban, port exclusion) trigger alerts when ships
enter them. Cooldown timers prevent alert spam. Full timestamped logs
are maintained for dashboard display.
"""

import json
import time
from collections import deque
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from shapely.geometry import Point, Polygon
import logging

logger = logging.getLogger(__name__)

from src.detection.detector import Detection


@dataclass
class Alert:
    """Single zone violation alert."""
    timestamp: float
    zone_name: str
    zone_type: str
    alert_level: str          # HIGH / MEDIUM / LOW
    ship_track_id: int
    ship_position: Tuple[int, int]
    ship_confidence: float
    ship_type: str = ""
    message: str = ""
    
    @property
    def time_str(self) -> str:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.timestamp))
    
    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "time_str": self.time_str,
            "zone_name": self.zone_name,
            "zone_type": self.zone_type,
            "alert_level": self.alert_level,
            "ship_track_id": self.ship_track_id,
            "ship_position": self.ship_position,
            "ship_confidence": self.ship_confidence,
            "ship_type": self.ship_type,
            "message": self.message,
        }


class ZoneAlertSystem:
    """
    Multi-zone alert system with configurable cooldown.
    
    Features:
        - Named polygon zones loaded from JSON
        - Per-zone, per-ship cooldowns (no duplicate spam)
        - Timestamped alert log with configurable max size
        - Alert severity based on zone type
    """
    
    def __init__(
        self,
        zones_path: Optional[str] = None,
        cooldown_seconds: float = 30.0,
        max_log_size: int = 500
    ):
        self.zones: List[Dict] = []
        self.zone_polygons: Dict[str, Polygon] = {}
        self.cooldown_seconds = cooldown_seconds
        self.alert_log: deque = deque(maxlen=max_log_size)
        self._cooldowns: Dict[str, float] = {}  # key: "zone_name:track_id" → expiry timestamp
        
        if zones_path:
            self.load_zones(zones_path)
    
    def load_zones(self, zones_path: str):
        """Load zone definitions from JSON file."""
        try:
            with open(zones_path, 'r') as f:
                data = json.load(f)
            
            self.zones = data.get("zones", [])
            
            # Build Shapely polygons for pixel coordinates
            for zone in self.zones:
                name = zone["name"]
                coords = zone.get("pixel_coordinates", [])
                if len(coords) >= 3:
                    self.zone_polygons[name] = Polygon(coords)
            
            logger.info(f"Loaded {len(self.zones)} zones: {[z['name'] for z in self.zones]}")
        except Exception as e:
            logger.error(f"Failed to load zones from {zones_path}: {e}")
    
    def add_zone(self, name: str, zone_type: str, pixel_coords: List[List[int]], 
                 alert_level: str = "MEDIUM", color: str = "#FFA500"):
        """Add a zone dynamically."""
        zone = {
            "name": name,
            "type": zone_type,
            "alert_level": alert_level,
            "color": color,
            "pixel_coordinates": pixel_coords
        }
        self.zones.append(zone)
        if len(pixel_coords) >= 3:
            self.zone_polygons[name] = Polygon(pixel_coords)
        logger.info(f"Added zone: {name}")
    
    def check_violations(
        self,
        detections: List[Detection],
        timestamp: Optional[float] = None
    ) -> List[Alert]:
        """
        Check if any detections violate zone boundaries.
        
        Args:
            detections: Current frame detections.
            timestamp: Current timestamp (uses time.time() if None).
            
        Returns:
            List of new alerts (respecting cooldowns).
        """
        ts = timestamp or time.time()
        new_alerts = []
        
        for det in detections:
            cx, cy = det.center
            point = Point(cx, cy)
            
            for zone in self.zones:
                zone_name = zone["name"]
                
                if zone_name not in self.zone_polygons:
                    continue
                
                polygon = self.zone_polygons[zone_name]
                
                if polygon.contains(point):
                    # Check cooldown
                    cooldown_key = f"{zone_name}:{det.track_id}"
                    
                    if cooldown_key in self._cooldowns:
                        if ts < self._cooldowns[cooldown_key]:
                            continue  # Still in cooldown, skip
                    
                    # Fire alert
                    alert = Alert(
                        timestamp=ts,
                        zone_name=zone_name,
                        zone_type=zone["type"],
                        alert_level=zone.get("alert_level", "MEDIUM"),
                        ship_track_id=det.track_id,
                        ship_position=det.center,
                        ship_confidence=det.confidence,
                        ship_type=det.ship_type,
                        message=f"⚠️ Ship #{det.track_id} ({det.ship_type or 'unknown'}) "
                                f"entered {zone_name} [{zone['type']}]"
                    )
                    
                    new_alerts.append(alert)
                    self.alert_log.append(alert)
                    
                    # Set cooldown
                    self._cooldowns[cooldown_key] = ts + self.cooldown_seconds
                    
                    logger.info(f"ALERT: {alert.message}")
        
        # Cleanup expired cooldowns
        self._cleanup_cooldowns(ts)
        
        return new_alerts
    
    def _cleanup_cooldowns(self, current_time: float):
        """Remove expired cooldown entries."""
        expired = [k for k, v in self._cooldowns.items() if current_time > v]
        for k in expired:
            del self._cooldowns[k]
    
    def get_alert_log(self, n: Optional[int] = None) -> List[Dict]:
        """Get recent alerts as dictionaries."""
        alerts = list(self.alert_log)
        if n:
            alerts = alerts[-n:]
        return [a.to_dict() for a in alerts]
    
    def get_alerts_by_zone(self, zone_name: str) -> List[Dict]:
        """Get all alerts for a specific zone."""
        return [a.to_dict() for a in self.alert_log if a.zone_name == zone_name]
    
    def get_alerts_in_timerange(self, start_time: float, end_time: float) -> List[Dict]:
        """Get alerts within a time range."""
        return [
            a.to_dict() for a in self.alert_log
            if start_time <= a.timestamp <= end_time
        ]
    
    def get_zone_names(self) -> List[str]:
        """Get all zone names."""
        return [z["name"] for z in self.zones]
    
    def clear_log(self):
        """Clear the alert log."""
        self.alert_log.clear()
        self._cooldowns.clear()
