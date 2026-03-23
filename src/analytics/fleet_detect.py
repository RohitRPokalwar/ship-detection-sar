"""
Fleet / Formation Detection Module.

Uses DBSCAN clustering on ship centroids to identify groups of ships
traveling together — potentially indicating convoys, fleet formations,
or coordinated illegal activity.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)

try:
    from sklearn.cluster import DBSCAN
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    logger.warning("scikit-learn not installed. Fleet detection will be unavailable.")

from src.detection.detector import Detection


@dataclass
class Fleet:
    """A detected fleet/formation of ships."""
    fleet_id: int
    ship_track_ids: List[int]
    ship_positions: List[Tuple[int, int]]
    centroid: Tuple[float, float]
    num_ships: int
    bounding_box: List[int]  # [x1, y1, x2, y2]
    radius: float
    
    @property
    def is_significant(self) -> bool:
        """Fleets with 3+ ships are considered significant."""
        return self.num_ships >= 3


class FleetDetector:
    """
    Detects fleet formations using DBSCAN clustering.
    
    DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
    groups nearby ship detections without requiring a predefined number 
    of clusters. Ships not belonging to any group are labeled as noise 
    (lone vessels).
    
    Parameters:
        eps: Maximum distance between two ships to be in the same cluster.
        min_samples: Minimum ships to form a cluster (fleet).
    """
    
    def __init__(
        self,
        eps: float = 50.0,
        min_samples: int = 3
    ):
        """
        Args:
            eps: Maximum distance between two points in a cluster (pixels).
            min_samples: Minimum number of points to form a cluster.
        """
        self.eps = eps
        self.min_samples = min_samples
    
    def detect_fleets(self, detections: List[Detection]) -> List[Fleet]:
        """
        Detect fleet formations from ship detections.
        
        Args:
            detections: List of detections from current frame.
            
        Returns:
            List of Fleet objects representing detected formations.
        """
        if len(detections) < self.min_samples:
            return []
        
        if not HAS_SKLEARN:
            logger.warning("scikit-learn required for fleet detection")
            return []
        
        # Extract centroids
        positions = np.array([det.center for det in detections])
        track_ids = [det.track_id for det in detections]
        
        # Run DBSCAN
        clustering = DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples,
            metric='euclidean'
        ).fit(positions)
        
        labels = clustering.labels_
        
        # Group by cluster label (ignore noise label = -1)
        fleets = []
        unique_labels = set(labels)
        unique_labels.discard(-1)  # Remove noise cluster
        
        for cluster_id in unique_labels:
            mask = labels == cluster_id
            cluster_positions = positions[mask]
            cluster_track_ids = [tid for tid, m in zip(track_ids, mask) if m]
            
            # Compute cluster properties
            centroid = tuple(cluster_positions.mean(axis=0))
            
            x_min, y_min = cluster_positions.min(axis=0)
            x_max, y_max = cluster_positions.max(axis=0)
            bounding_box = [int(x_min), int(y_min), int(x_max), int(y_max)]
            
            # Cluster radius (max distance from centroid)
            distances = np.sqrt(
                (cluster_positions[:, 0] - centroid[0])**2 + 
                (cluster_positions[:, 1] - centroid[1])**2
            )
            radius = float(distances.max())
            
            fleet = Fleet(
                fleet_id=int(cluster_id),
                ship_track_ids=cluster_track_ids,
                ship_positions=[tuple(p) for p in cluster_positions],
                centroid=centroid,
                num_ships=len(cluster_positions),
                bounding_box=bounding_box,
                radius=radius
            )
            fleets.append(fleet)
            
            logger.info(
                f"Fleet #{cluster_id}: {fleet.num_ships} ships, "
                f"centroid=({centroid[0]:.0f}, {centroid[1]:.0f}), "
                f"radius={radius:.0f}px"
            )
        
        # Mark noise points (solo vessels)
        noise_count = sum(1 for l in labels if l == -1)
        if noise_count > 0:
            logger.debug(f"Solo vessels (no fleet): {noise_count}")
        
        return fleets
    
    def annotate_detections(
        self,
        detections: List[Detection],
        fleets: List[Fleet]
    ) -> List[Detection]:
        """
        Annotate detections with their fleet membership.
        
        Adds 'fleet_id' to detection metadata.
        """
        # Build track_id → fleet_id mapping
        track_to_fleet = {}
        for fleet in fleets:
            for tid in fleet.ship_track_ids:
                track_to_fleet[tid] = fleet.fleet_id
        
        for det in detections:
            if det.track_id in track_to_fleet:
                det.metadata["fleet_id"] = track_to_fleet[det.track_id]
                det.metadata["in_formation"] = True
            else:
                det.metadata["fleet_id"] = -1
                det.metadata["in_formation"] = False
        
        return detections
    
    def get_fleet_summary(self, fleets: List[Fleet]) -> Dict:
        """Get summary statistics of detected fleets."""
        if not fleets:
            return {"num_fleets": 0, "total_ships_in_fleets": 0}
        
        return {
            "num_fleets": len(fleets),
            "total_ships_in_fleets": sum(f.num_ships for f in fleets),
            "largest_fleet": max(f.num_ships for f in fleets),
            "fleet_details": [
                {
                    "id": f.fleet_id,
                    "ships": f.num_ships,
                    "centroid": f.centroid,
                    "radius": f.radius
                }
                for f in fleets
            ]
        }
