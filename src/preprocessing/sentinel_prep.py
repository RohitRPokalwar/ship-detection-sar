"""
Sentinel-1 SAR Preprocessing Module.

Handles reading, calibrating, filtering, and tiling of Sentinel-1 
GRD (Ground Range Detected) products for model inference.
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, List, Optional, Dict
import logging

logger = logging.getLogger(__name__)

try:
    import rasterio
    from rasterio.transform import Affine
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    logger.warning("rasterio not installed. Sentinel-1 GeoTIFF reading will be unavailable.")

from src.preprocessing.speckle_filter import apply_speckle_filter


class SentinelPreprocessor:
    """
    Preprocesses Sentinel-1 GRD scenes for ship detection inference.
    
    Pipeline:
        1. Read GeoTIFF (VV or VH polarization)
        2. Radiometric calibration (DN → σ₀ backscatter in dB)
        3. Speckle filtering (Lee or Frost)
        4. Normalization to [0, 255]
        5. Tiling into model-compatible patches (640×640)
        6. Extract geotransform for coordinate back-projection
    """
    
    def __init__(
        self,
        tile_size: int = 640,
        overlap: int = 64,
        filter_type: str = "lee",
        filter_size: int = 7
    ):
        self.tile_size = tile_size
        self.overlap = overlap
        self.filter_type = filter_type
        self.filter_size = filter_size
        self.geotransform = None
        self.crs = None
        self.image_shape = None
    
    def read_geotiff(self, filepath: str) -> Tuple[np.ndarray, Optional[Dict]]:
        """
        Read a Sentinel-1 GeoTIFF file.
        
        Args:
            filepath: Path to the .tif/.tiff file.
            
        Returns:
            Tuple of (image array, geo_metadata dict).
        """
        filepath = Path(filepath)
        
        if HAS_RASTERIO:
            with rasterio.open(str(filepath)) as src:
                image = src.read(1).astype(np.float64)
                self.geotransform = src.transform
                self.crs = src.crs
                self.image_shape = image.shape
                
                geo_metadata = {
                    "transform": src.transform,
                    "crs": str(src.crs) if src.crs else None,
                    "width": src.width,
                    "height": src.height,
                    "bounds": src.bounds,
                }
                
                logger.info(f"Read GeoTIFF: {image.shape}, CRS: {src.crs}")
                return image, geo_metadata
        else:
            # Fallback: read as regular image
            image = cv2.imread(str(filepath), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise FileNotFoundError(f"Cannot read file: {filepath}")
            self.image_shape = image.shape
            logger.warning("Reading as regular image (no geospatial metadata).")
            return image.astype(np.float64), None
    
    def calibrate_sigma0(self, image: np.ndarray) -> np.ndarray:
        """
        Radiometric calibration: convert Digital Numbers to σ₀ backscatter (dB).
        
        For Sentinel-1 GRD products:
            σ₀ (linear) ≈ DN² / (calibration_factor)
            σ₀ (dB) = 10 * log10(σ₀_linear)
            
        Note: For precise calibration, the product annotation XML should be
        parsed for the actual calibration LUT. This is a simplified approximation.
        """
        # Avoid log of zero
        image = np.maximum(image, 1e-10)
        
        # Convert to σ₀ in dB (simplified)
        sigma0_db = 10.0 * np.log10(image ** 2 + 1e-10)
        
        logger.info(f"Calibrated to σ₀ dB: range [{sigma0_db.min():.1f}, {sigma0_db.max():.1f}]")
        return sigma0_db
    
    def normalize_to_uint8(self, image: np.ndarray, percentile_clip: Tuple[float, float] = (2, 98)) -> np.ndarray:
        """
        Normalize image to 0-255 uint8 using percentile clipping.
        
        Percentile clipping handles the wide dynamic range of SAR data
        by stretching contrast and removing extreme outliers.
        """
        p_low, p_high = np.percentile(image, percentile_clip)
        clipped = np.clip(image, p_low, p_high)
        
        # Normalize to [0, 255]
        if p_high - p_low > 0:
            normalized = ((clipped - p_low) / (p_high - p_low) * 255).astype(np.uint8)
        else:
            normalized = np.zeros_like(clipped, dtype=np.uint8)
        
        return normalized
    
    def create_tiles(
        self, image: np.ndarray
    ) -> List[Tuple[np.ndarray, Tuple[int, int]]]:
        """
        Split image into overlapping tiles for model inference.
        
        Args:
            image: Preprocessed image (H, W).
            
        Returns:
            List of (tile_image, (row_offset, col_offset)) tuples.
        """
        h, w = image.shape[:2]
        stride = self.tile_size - self.overlap
        tiles = []
        
        for y in range(0, h, stride):
            for x in range(0, w, stride):
                # Extract tile with padding at edges
                y_end = min(y + self.tile_size, h)
                x_end = min(x + self.tile_size, w)
                
                tile = np.zeros((self.tile_size, self.tile_size), dtype=image.dtype)
                tile_h = y_end - y
                tile_w = x_end - x
                tile[:tile_h, :tile_w] = image[y:y_end, x:x_end]
                
                # Convert grayscale to 3-channel for YOLO
                if len(tile.shape) == 2:
                    tile = cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR)
                
                tiles.append((tile, (y, x)))
        
        logger.info(f"Created {len(tiles)} tiles of size {self.tile_size}×{self.tile_size} with stride {stride}")
        return tiles
    
    def pixel_to_geo(self, px_x: int, px_y: int) -> Optional[Tuple[float, float]]:
        """
        Convert pixel coordinates to geographic coordinates using the geotransform.
        
        Args:
            px_x: Pixel x (column).
            px_y: Pixel y (row).
            
        Returns:
            Tuple of (longitude, latitude) or None if no geotransform available.
        """
        if self.geotransform is None:
            return None
        
        # Apply affine transform: geo = transform * pixel
        lng, lat = self.geotransform * (px_x, px_y)
        return (lng, lat)
    
    def tile_bbox_to_image_bbox(
        self,
        tile_offset: Tuple[int, int],
        bbox: Tuple[int, int, int, int]
    ) -> Tuple[int, int, int, int]:
        """
        Convert a bounding box from tile coordinates to full image coordinates.
        
        Args:
            tile_offset: (row_offset, col_offset) of the tile.
            bbox: (x1, y1, x2, y2) in tile coordinates.
            
        Returns:
            (x1, y1, x2, y2) in full image coordinates.
        """
        y_off, x_off = tile_offset
        return (bbox[0] + x_off, bbox[1] + y_off, bbox[2] + x_off, bbox[3] + y_off)
    
    def preprocess(self, filepath: str) -> Tuple[List[Tuple[np.ndarray, Tuple[int, int]]], np.ndarray, Optional[Dict]]:
        """
        Full preprocessing pipeline for a Sentinel-1 scene.
        
        Args:
            filepath: Path to Sentinel-1 GeoTIFF.
            
        Returns:
            Tuple of (tiles_list, full_preprocessed_image, geo_metadata).
        """
        logger.info(f"Preprocessing Sentinel-1 scene: {filepath}")
        
        # Step 1: Read GeoTIFF
        raw_image, geo_metadata = self.read_geotiff(filepath)
        
        # Step 2: Radiometric calibration
        calibrated = self.calibrate_sigma0(raw_image)
        
        # Step 3: Normalize to uint8
        normalized = self.normalize_to_uint8(calibrated)
        
        # Step 4: Speckle filtering
        filtered = apply_speckle_filter(
            normalized, 
            filter_type=self.filter_type, 
            size=self.filter_size
        )
        filtered = filtered.astype(np.uint8)
        
        # Step 5: Create tiles
        tiles = self.create_tiles(filtered)
        
        logger.info(f"Preprocessing complete: {len(tiles)} tiles generated")
        return tiles, filtered, geo_metadata
