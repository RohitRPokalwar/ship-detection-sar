"""
SAR Speckle Filtering Module.

Implements Lee and Frost filters for SAR image despeckling.
Speckle noise is a granular interference pattern inherent in SAR imagery
due to coherent radar wave scattering. These filters reduce speckle
while preserving ship target edges.
"""

import numpy as np
import cv2


def lee_filter(image: np.ndarray, size: int = 7) -> np.ndarray:
    """
    Apply Lee adaptive speckle filter to a SAR image.
    
    The Lee filter uses local statistics (mean and variance) within a 
    sliding window. It weights the center pixel against the local mean 
    based on the ratio of local variance to overall image variance.
    
    Where variance is high (edges/targets), the filter preserves the 
    original pixel. Where variance is low (homogeneous regions), it 
    smooths towards the local mean.
    
    Args:
        image: Input SAR image (grayscale, float32 or uint8).
        size: Window size for local statistics (must be odd).
        
    Returns:
        Filtered image as float32.
    """
    if size % 2 == 0:
        size += 1
        
    img = image.astype(np.float64)
    
    # Local mean using box filter
    local_mean = cv2.blur(img, (size, size))
    
    # Local variance: E[X^2] - (E[X])^2
    local_sq_mean = cv2.blur(img ** 2, (size, size))
    local_variance = local_sq_mean - local_mean ** 2
    local_variance = np.maximum(local_variance, 0)  # clamp negatives
    
    # Overall image variance (estimate of noise variance)
    overall_variance = np.var(img)
    
    if overall_variance == 0:
        return img.astype(np.float32)
    
    # Weighting factor: higher weight = more smoothing
    weight = local_variance / (local_variance + overall_variance + 1e-10)
    
    # Filtered output: weighted combination
    filtered = local_mean + weight * (img - local_mean)
    
    return filtered.astype(np.float32)


def frost_filter(image: np.ndarray, size: int = 7, damping_factor: float = 2.0) -> np.ndarray:
    """
    Apply Frost exponentially-weighted speckle filter to a SAR image.
    
    The Frost filter applies an exponential kernel whose decay rate 
    depends on the local coefficient of variation (CV = std/mean).
    In homogeneous regions (low CV), strong smoothing is applied.
    Near edges/targets (high CV), the filter preserves detail.
    
    Args:
        image: Input SAR image (grayscale, float32 or uint8).
        size: Window size (must be odd).
        damping_factor: Controls the decay rate of the exponential.
        
    Returns:
        Filtered image as float32.
    """
    if size % 2 == 0:
        size += 1
        
    img = image.astype(np.float64)
    half = size // 2
    rows, cols = img.shape[:2]
    
    # Pad image for border handling
    padded = cv2.copyMakeBorder(img, half, half, half, half, cv2.BORDER_REFLECT)
    filtered = np.zeros_like(img, dtype=np.float64)
    
    # Pre-compute distance matrix for the window
    y_dist, x_dist = np.mgrid[-half:half+1, -half:half+1]
    distance = np.sqrt(x_dist**2 + y_dist**2)
    
    for i in range(rows):
        for j in range(cols):
            # Extract local window
            window = padded[i:i+size, j:j+size]
            
            # Local statistics
            local_mean = np.mean(window)
            local_std = np.std(window)
            
            if local_mean == 0:
                filtered[i, j] = 0
                continue
            
            # Coefficient of variation
            cv = local_std / (local_mean + 1e-10)
            
            # Exponential kernel weighted by CV and distance
            kernel = np.exp(-damping_factor * cv * distance)
            kernel_sum = np.sum(kernel)
            
            if kernel_sum == 0:
                filtered[i, j] = img[i, j]
            else:
                filtered[i, j] = np.sum(kernel * window) / kernel_sum
    
    return filtered.astype(np.float32)


def frost_filter_fast(image: np.ndarray, size: int = 7, damping_factor: float = 2.0) -> np.ndarray:
    """
    Fast approximation of the Frost filter using pre-computed kernels.
    
    Uses local statistics to weight between smoothed and original pixels,
    avoiding the expensive per-pixel loop of the full Frost filter.
    
    Args:
        image: Input SAR image (grayscale, float32 or uint8).
        size: Window size (must be odd).
        damping_factor: Controls smoothing strength.
        
    Returns:
        Filtered image as float32.
    """
    if size % 2 == 0:
        size += 1
        
    img = image.astype(np.float64)
    
    # Local statistics
    local_mean = cv2.blur(img, (size, size))
    local_sq_mean = cv2.blur(img ** 2, (size, size))
    local_variance = np.maximum(local_sq_mean - local_mean ** 2, 0)
    local_std = np.sqrt(local_variance)
    
    # Coefficient of variation per pixel
    cv_map = local_std / (local_mean + 1e-10)
    
    # Adaptive weight: high CV → preserve original; low CV → smooth
    # Using exponential decay with CV as proxy
    weight = np.exp(-damping_factor * cv_map)
    
    # Gaussian smoothed version
    smoothed = cv2.GaussianBlur(img, (size, size), 0)
    
    # Blend: original vs. smoothed based on local CV
    filtered = weight * smoothed + (1 - weight) * img
    
    return filtered.astype(np.float32)


def apply_speckle_filter(
    image: np.ndarray,
    filter_type: str = "lee",
    size: int = 7,
    damping_factor: float = 2.0,
    fast_frost: bool = True
) -> np.ndarray:
    """
    Apply the specified speckle filter to a SAR image.
    
    Args:
        image: Input SAR image.
        filter_type: "lee" or "frost".
        size: Filter window size.
        damping_factor: Frost filter damping (ignored for Lee).
        fast_frost: Use fast Frost approximation if True.
        
    Returns:
        Filtered image.
    """
    if filter_type.lower() == "lee":
        return lee_filter(image, size)
    elif filter_type.lower() == "frost":
        if fast_frost:
            return frost_filter_fast(image, size, damping_factor)
        else:
            return frost_filter(image, size, damping_factor)
    else:
        raise ValueError(f"Unknown filter type: {filter_type}. Use 'lee' or 'frost'.")
