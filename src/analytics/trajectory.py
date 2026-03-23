"""
Trajectory Prediction Module (Kalman Filter).

Predicts each ship's next position from tracked history using a 
constant-velocity Kalman filter. Draws a dashed forecast arrow 
showing predicted future positions.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

try:
    from filterpy.kalman import KalmanFilter as FilterPyKF
    HAS_FILTERPY = True
except ImportError:
    HAS_FILTERPY = False
    logger.warning("filterpy not installed. Using basic linear prediction.")

from src.detection.tracker import Track


class ShipKalmanFilter:
    """
    Kalman filter for a single ship's trajectory prediction.
    
    State vector: [x, y, vx, vy]
    Measurement: [x, y]
    
    Uses a constant-velocity model with process noise to handle
    acceleration and course changes.
    """
    
    def __init__(self, initial_position: Tuple[float, float], dt: float = 1.0):
        """
        Args:
            initial_position: Initial (x, y) position.
            dt: Time step between frames.
        """
        self.dt = dt
        
        if HAS_FILTERPY:
            self.kf = FilterPyKF(dim_x=4, dim_z=2)
            
            # State transition matrix (constant velocity model)
            self.kf.F = np.array([
                [1, 0, dt, 0],
                [0, 1, 0, dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
            
            # Measurement matrix (we only observe position)
            self.kf.H = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0]
            ])
            
            # Measurement noise
            self.kf.R *= 5.0
            
            # Process noise (accounts for acceleration/turns)
            q = 0.1
            self.kf.Q = np.array([
                [q*dt**3/3, 0, q*dt**2/2, 0],
                [0, q*dt**3/3, 0, q*dt**2/2],
                [q*dt**2/2, 0, q*dt, 0],
                [0, q*dt**2/2, 0, q*dt]
            ])
            
            # Initial state
            self.kf.x = np.array([initial_position[0], initial_position[1], 0, 0])
            
            # Initial uncertainty
            self.kf.P *= 100
        else:
            # Simple linear prediction fallback
            self._positions = [initial_position]
            self._velocity = (0.0, 0.0)
    
    def update(self, position: Tuple[float, float]):
        """Update the filter with a new measured position."""
        if HAS_FILTERPY:
            self.kf.predict()
            self.kf.update(np.array(position))
        else:
            if len(self._positions) >= 2:
                prev = self._positions[-1]
                self._velocity = (position[0] - prev[0], position[1] - prev[1])
            self._positions.append(position)
    
    def predict_future(self, n_steps: int = 10) -> List[Tuple[float, float]]:
        """
        Predict the next N positions.
        
        Args:
            n_steps: Number of future time steps to predict.
            
        Returns:
            List of (x, y) predicted positions.
        """
        predictions = []
        
        if HAS_FILTERPY:
            # Save current state
            x_save = self.kf.x.copy()
            P_save = self.kf.P.copy()
            
            for _ in range(n_steps):
                self.kf.predict()
                pred_x, pred_y = self.kf.x[0], self.kf.x[1]
                predictions.append((float(pred_x), float(pred_y)))
            
            # Restore state
            self.kf.x = x_save
            self.kf.P = P_save
        else:
            # Linear extrapolation
            if self._positions:
                last_pos = self._positions[-1]
                vx, vy = self._velocity
                for i in range(1, n_steps + 1):
                    pred_x = last_pos[0] + vx * i
                    pred_y = last_pos[1] + vy * i
                    predictions.append((float(pred_x), float(pred_y)))
        
        return predictions
    
    @property
    def current_state(self) -> Dict:
        """Get current filter state."""
        if HAS_FILTERPY:
            return {
                "x": float(self.kf.x[0]),
                "y": float(self.kf.x[1]),
                "vx": float(self.kf.x[2]),
                "vy": float(self.kf.x[3]),
            }
        else:
            pos = self._positions[-1] if self._positions else (0, 0)
            return {
                "x": pos[0],
                "y": pos[1],
                "vx": self._velocity[0],
                "vy": self._velocity[1],
            }


class TrajectoryPredictor:
    """
    Manages Kalman filters for all tracked ships and provides
    trajectory predictions for visualization.
    """
    
    def __init__(self, dt: float = 1.0, predict_steps: int = 10):
        """
        Args:
            dt: Time step between frames.
            predict_steps: Number of future positions to predict.
        """
        self.dt = dt
        self.predict_steps = predict_steps
        self.filters: Dict[int, ShipKalmanFilter] = {}
    
    def update_tracks(self, tracks: Dict[int, Track]) -> Dict[int, List[Tuple[float, float]]]:
        """
        Update filters with latest track positions and generate predictions.
        
        Args:
            tracks: Active tracks dict from ShipTracker.
            
        Returns:
            Dict mapping track_id → list of predicted future positions.
        """
        predictions = {}
        
        for track_id, track in tracks.items():
            if not track.current_position:
                continue
            
            pos = track.current_position
            
            if track_id not in self.filters:
                self.filters[track_id] = ShipKalmanFilter(pos, self.dt)
            
            self.filters[track_id].update(pos)
            predictions[track_id] = self.filters[track_id].predict_future(self.predict_steps)
        
        # Remove filters for inactive tracks
        active_ids = set(tracks.keys())
        removed = [tid for tid in self.filters if tid not in active_ids]
        for tid in removed:
            del self.filters[tid]
        
        return predictions
    
    def get_prediction(self, track_id: int) -> Optional[List[Tuple[float, float]]]:
        """Get predictions for a specific track."""
        if track_id in self.filters:
            return self.filters[track_id].predict_future(self.predict_steps)
        return None
    
    def get_all_predictions(self) -> Dict[int, List[Tuple[float, float]]]:
        """Get predictions for all tracked ships."""
        return {
            tid: kf.predict_future(self.predict_steps)
            for tid, kf in self.filters.items()
        }
    
    def reset(self):
        """Clear all filters."""
        self.filters.clear()
