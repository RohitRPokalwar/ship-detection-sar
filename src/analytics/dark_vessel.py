"""
Dark Vessel Detection Module.

Ships with no AIS (Automatic Identification System) match in the 
detection zone are flagged as suspicious "dark vessels." This is the 
#1 real-world coast guard problem — vessels intentionally disabling 
their AIS transponders to avoid detection.

Uses spatial join between radar detections and AIS positions to 
identify unmatched detections.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from shapely.geometry import Point
import logging

logger = logging.getLogger(__name__)

from src.detection.detector import Detection


class DarkVesselDetector:
    """
    Detects ships with no AIS match — potential "dark vessels."
    
    How it works:
        1. Load AIS position data (real or simulated)
        2. For each radar detection, check if any AIS vessel is 
           within a configurable radius
        3. Unmatched detections → flagged as dark vessels
    """
    
    def __init__(
        self,
        ais_data_path: Optional[str] = None,
        match_radius_pixels: float = 50.0,
        match_radius_meters: float = 500.0
    ):
        """
        Args:
            ais_data_path: Path to AIS data CSV.
            match_radius_pixels: AIS match radius in pixel coordinates.
            match_radius_meters: AIS match radius in meters (for geo coords).
        """
        self.ais_data: Optional[pd.DataFrame] = None
        self.match_radius_px = match_radius_pixels
        self.match_radius_m = match_radius_meters
        self._ais_positions: List[Tuple[float, float]] = []
        self._ais_info: List[Dict] = []
        
        if ais_data_path:
            self.load_ais_data(ais_data_path)
    
    def load_ais_data(self, filepath: str):
        """
        Load AIS vessel data from CSV.
        
        Expected columns: mmsi, vessel_name, vessel_type, lat, lng, 
                         speed_knots, heading, timestamp
        """
        try:
            self.ais_data = pd.read_csv(filepath)
            logger.info(f"Loaded AIS data: {len(self.ais_data)} records")
            
            # Get latest position for each unique vessel (by MMSI)
            if 'mmsi' in self.ais_data.columns:
                latest = self.ais_data.sort_values('timestamp').groupby('mmsi').last().reset_index()
                
                self._ais_positions = list(zip(
                    latest.get('lng', latest.get('x', [])),
                    latest.get('lat', latest.get('y', []))
                ))
                
                self._ais_info = latest.to_dict('records')
                logger.info(f"Unique AIS vessels: {len(self._ais_positions)}")
        except Exception as e:
            logger.error(f"Failed to load AIS data: {e}")
    
    def set_ais_pixel_positions(self, positions: List[Tuple[float, float]], info: Optional[List[Dict]] = None):
        """
        Set AIS positions directly in pixel coordinates.
        Useful when positions have been projected from geo to pixel space.
        
        Args:
            positions: List of (x, y) pixel positions for AIS vessels.
            info: Optional vessel info dicts.
        """
        self._ais_positions = positions
        self._ais_info = info or [{} for _ in positions]
    
    def generate_simulated_ais(
        self,
        image_shape: Tuple[int, int],
        num_vessels: int = 10,
        detections: Optional[List[Detection]] = None
    ) -> List[Tuple[float, float]]:
        """
        Generate simulated AIS positions for demo purposes.
        
        Places some AIS markers near known detections (matching vessels) 
        and some randomly (non-matching), ensuring some detections are 
        left unmatched (dark vessels).
        
        Args:
            image_shape: (height, width) of the image.
            num_vessels: Number of simulated AIS vessels.
            detections: Known detections to partially match.
            
        Returns:
            List of simulated AIS pixel positions.
        """
        h, w = image_shape[:2]
        positions = []
        info = []
        
        vessel_types = ["cargo", "tanker", "fishing", "military"]
        
        if detections:
            # Match ~60% of detections with AIS
            num_matched = max(1, int(len(detections) * 0.6))
            matched_dets = detections[:num_matched]
            
            for i, det in enumerate(matched_dets):
                cx, cy = det.center
                # Add small offset to simulate imperfect AIS/radar alignment
                offset_x = np.random.normal(0, 10)
                offset_y = np.random.normal(0, 10)
                pos = (cx + offset_x, cy + offset_y)
                positions.append(pos)
                info.append({
                    "mmsi": 211234567 + i,
                    "vessel_name": f"AIS Vessel {i+1}",
                    "vessel_type": vessel_types[i % len(vessel_types)],
                    "matched": True
                })
        
        # Add random AIS vessels (not matching any detection)
        remaining = num_vessels - len(positions)
        for i in range(remaining):
            pos = (np.random.uniform(0, w), np.random.uniform(0, h))
            positions.append(pos)
            info.append({
                "mmsi": 211234600 + i,
                "vessel_name": f"Random Vessel {i+1}",
                "vessel_type": vessel_types[i % len(vessel_types)],
                "matched": False
            })
        
        self._ais_positions = positions
        self._ais_info = info
        
        logger.info(f"Generated {len(positions)} simulated AIS positions")
        return positions
    
    def detect_dark_vessels(
        self,
        detections: List[Detection],
        use_pixel_coords: bool = True
    ) -> List[Detection]:
        """
        Flag detections with no AIS match as dark vessels.
        
        For each radar detection, checks if any AIS vessel is within
        the match radius. If no match is found, the ship is flagged.
        
        Args:
            detections: Radar detections from current frame.
            use_pixel_coords: If True, use pixel distance; else geo distance.
            
        Returns:
            Updated detections with is_dark_vessel flag set.
        """
        if not self._ais_positions:
            logger.warning("No AIS data loaded. All vessels will be flagged as dark.")
            for det in detections:
                det.is_dark_vessel = True
                det.metadata["ais_status"] = "NO_DATA"
            return detections
        
        radius = self.match_radius_px if use_pixel_coords else self.match_radius_m
        
        for det in detections:
            cx, cy = det.center
            matched = False
            matched_vessel = None
            min_dist = float('inf')
            
            for i, (ax, ay) in enumerate(self._ais_positions):
                dist = np.sqrt((cx - ax)**2 + (cy - ay)**2)
                
                if dist < min_dist:
                    min_dist = dist
                
                if dist <= radius:
                    matched = True
                    matched_vessel = self._ais_info[i] if i < len(self._ais_info) else {}
                    break
            
            if matched:
                det.is_dark_vessel = False
                det.metadata["ais_status"] = "MATCHED"
                det.metadata["ais_vessel"] = matched_vessel
                det.metadata["ais_distance"] = min_dist
            else:
                det.is_dark_vessel = True
                det.metadata["ais_status"] = "DARK_VESSEL"
                det.metadata["ais_nearest_distance"] = min_dist
                logger.warning(
                    f"🔴 DARK VESSEL DETECTED: Track #{det.track_id} at "
                    f"({cx}, {cy}), nearest AIS: {min_dist:.0f}px"
                )
        
        dark_count = sum(1 for d in detections if d.is_dark_vessel)
        logger.info(f"Dark vessel scan: {dark_count}/{len(detections)} unmatched")
        
        return detections
    
    def get_ais_positions(self) -> List[Tuple[float, float]]:
        """Return current AIS positions."""
        return self._ais_positions
    
    def get_ais_info(self) -> List[Dict]:
        """Return AIS vessel information."""
        return self._ais_info
