"""
Data Augmentation Pipeline for SAR Ship Detection.

Applies random transformations to SAR images and their bounding box
annotations to improve model generalization and robustness.
"""

import numpy as np
import cv2
import random
from typing import List, Tuple, Optional


def random_horizontal_flip(
    image: np.ndarray,
    bboxes: List[List[float]],
    prob: float = 0.5
) -> Tuple[np.ndarray, List[List[float]]]:
    """
    Randomly flip image and bounding boxes horizontally.
    
    Args:
        image: Input image (H, W, C) or (H, W).
        bboxes: List of [class_id, x_center, y_center, width, height] in YOLO format (normalized).
        prob: Probability of flipping.
        
    Returns:
        Flipped image and adjusted bounding boxes.
    """
    if random.random() < prob:
        image = cv2.flip(image, 1)  # horizontal flip
        bboxes = [[b[0], 1.0 - b[1], b[2], b[3], b[4]] for b in bboxes]
    return image, bboxes


def random_vertical_flip(
    image: np.ndarray,
    bboxes: List[List[float]],
    prob: float = 0.5
) -> Tuple[np.ndarray, List[List[float]]]:
    """
    Randomly flip image and bounding boxes vertically.
    """
    if random.random() < prob:
        image = cv2.flip(image, 0)  # vertical flip
        bboxes = [[b[0], b[1], 1.0 - b[2], b[3], b[4]] for b in bboxes]
    return image, bboxes


def random_rotation_90(
    image: np.ndarray,
    bboxes: List[List[float]],
    prob: float = 0.5
) -> Tuple[np.ndarray, List[List[float]]]:
    """
    Randomly rotate image by 90, 180, or 270 degrees.
    Only applied with probability `prob`.
    """
    if random.random() < prob:
        angle = random.choice([90, 180, 270])
        h, w = image.shape[:2]
        
        if angle == 90:
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            bboxes = [[b[0], b[2], 1.0 - b[1], b[4], b[3]] for b in bboxes]
        elif angle == 180:
            image = cv2.rotate(image, cv2.ROTATE_180)
            bboxes = [[b[0], 1.0 - b[1], 1.0 - b[2], b[3], b[4]] for b in bboxes]
        elif angle == 270:
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            bboxes = [[b[0], 1.0 - b[2], b[1], b[4], b[3]] for b in bboxes]
    
    return image, bboxes


def random_scale(
    image: np.ndarray,
    bboxes: List[List[float]],
    scale_range: Tuple[float, float] = (0.8, 1.2),
    prob: float = 0.5
) -> Tuple[np.ndarray, List[List[float]]]:
    """
    Randomly scale image. Bounding boxes remain in normalized coords
    so they stay valid after scaling.
    """
    if random.random() < prob:
        scale = random.uniform(*scale_range)
        h, w = image.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        image = cv2.resize(image, (new_w, new_h))
    
    return image, bboxes


def random_crop(
    image: np.ndarray,
    bboxes: List[List[float]],
    crop_ratio: Tuple[float, float] = (0.7, 0.9),
    prob: float = 0.5
) -> Tuple[np.ndarray, List[List[float]]]:
    """
    Randomly crop a portion of the image and adjust bounding boxes.
    Boxes that fall outside the crop are removed; partially visible
    boxes are clipped.
    """
    if random.random() >= prob:
        return image, bboxes
    
    h, w = image.shape[:2]
    ratio = random.uniform(*crop_ratio)
    crop_h, crop_w = int(h * ratio), int(w * ratio)
    
    y_start = random.randint(0, h - crop_h)
    x_start = random.randint(0, w - crop_w)
    
    image = image[y_start:y_start+crop_h, x_start:x_start+crop_w]
    
    # Adjust bounding boxes
    new_bboxes = []
    for b in bboxes:
        cls_id, xc, yc, bw, bh = b
        
        # Convert from normalized to pixel coords (original image)
        xc_px = xc * w
        yc_px = yc * h
        bw_px = bw * w
        bh_px = bh * h
        
        # Shift by crop offset
        xc_px -= x_start
        yc_px -= y_start
        
        # Compute box bounds in crop coords
        x1 = xc_px - bw_px / 2
        y1 = yc_px - bh_px / 2
        x2 = xc_px + bw_px / 2
        y2 = yc_px + bh_px / 2
        
        # Clip to crop boundaries
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(crop_w, x2)
        y2 = min(crop_h, y2)
        
        # Check if box is still valid (minimum area threshold)
        if x2 - x1 > 2 and y2 - y1 > 2:
            # Re-normalize to crop dimensions
            new_xc = ((x1 + x2) / 2) / crop_w
            new_yc = ((y1 + y2) / 2) / crop_h
            new_bw = (x2 - x1) / crop_w
            new_bh = (y2 - y1) / crop_h
            
            new_bboxes.append([cls_id, new_xc, new_yc, new_bw, new_bh])
    
    return image, new_bboxes


def add_gaussian_noise(
    image: np.ndarray,
    sigma: float = 10.0,
    prob: float = 0.3
) -> np.ndarray:
    """
    Add Gaussian noise to simulate varying SAR conditions.
    """
    if random.random() < prob:
        noise = np.random.normal(0, sigma, image.shape).astype(image.dtype)
        image = np.clip(image.astype(np.float32) + noise.astype(np.float32), 0, 255).astype(image.dtype)
    return image


def adjust_brightness(
    image: np.ndarray,
    factor_range: Tuple[float, float] = (0.7, 1.3),
    prob: float = 0.3
) -> np.ndarray:
    """
    Randomly adjust image brightness.
    """
    if random.random() < prob:
        factor = random.uniform(*factor_range)
        image = np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    return image


def augment_sample(
    image: np.ndarray,
    bboxes: List[List[float]],
    target_size: Optional[int] = 640
) -> Tuple[np.ndarray, List[List[float]]]:
    """
    Apply the full augmentation pipeline to a single image-bbox pair.
    
    Args:
        image: Input SAR image.
        bboxes: YOLO-format bounding boxes [class_id, xc, yc, w, h] normalized.
        target_size: Resize to this size after augmentation.
        
    Returns:
        Augmented image and adjusted bounding boxes.
    """
    # Geometric augmentations
    image, bboxes = random_horizontal_flip(image, bboxes)
    image, bboxes = random_vertical_flip(image, bboxes)
    image, bboxes = random_rotation_90(image, bboxes)
    image, bboxes = random_crop(image, bboxes)
    image, bboxes = random_scale(image, bboxes)
    
    # Photometric augmentations
    image = add_gaussian_noise(image)
    image = adjust_brightness(image)
    
    # Resize to target
    if target_size is not None:
        image = cv2.resize(image, (target_size, target_size))
    
    return image, bboxes
