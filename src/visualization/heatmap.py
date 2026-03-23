"""
Animated Temporal Heatmap Module.

Stacks N frames, animates ship density building over time,
and exports as a GIF. One image tells the whole story of 
maritime traffic patterns.
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False

from src.detection.detector import Detection


class TemporalHeatmap:
    """
    Builds and animates temporal ship density heatmaps.
    
    As frames are processed, detection positions accumulate 
    into a 2D density grid. The heatmap shows where ships 
    concentrate over time — revealing traffic lanes, anchorage 
    areas, and suspicious patrol patterns.
    """
    
    def __init__(
        self,
        resolution: Tuple[int, int] = (640, 640),
        colormap: str = "hot",
        decay: float = 0.95,
        sigma: int = 15
    ):
        """
        Args:
            resolution: Heatmap grid resolution (H, W).
            colormap: Matplotlib colormap name.
            decay: Temporal decay factor (0-1). Lower = faster fade.
            sigma: Gaussian blur kernel size for smoothing.
        """
        self.resolution = resolution
        self.colormap = colormap
        self.decay = decay
        self.sigma = sigma
        
        self.density = np.zeros(resolution, dtype=np.float64)
        self.frames: List[np.ndarray] = []
        self.frame_count = 0
    
    def add_detections(self, detections: List[Detection], image_shape: Optional[Tuple[int, int]] = None):
        """
        Add detection positions to the density accumulator.
        
        Args:
            detections: Detections from current frame.
            image_shape: Original image shape for coordinate scaling.
        """
        # Apply temporal decay to existing density
        self.density *= self.decay
        
        h, w = self.resolution
        scale_y, scale_x = 1.0, 1.0
        
        if image_shape:
            scale_y = h / image_shape[0]
            scale_x = w / image_shape[1]
        
        for det in detections:
            cx, cy = det.center
            # Scale to heatmap resolution
            hx = int(cx * scale_x)
            hy = int(cy * scale_y)
            
            if 0 <= hx < w and 0 <= hy < h:
                self.density[hy, hx] += 1.0
        
        # Smooth with Gaussian blur
        smoothed = cv2.GaussianBlur(self.density, (self.sigma, self.sigma), 0)
        
        # Store frame snapshot for animation
        self.frames.append(smoothed.copy())
        self.frame_count += 1
    
    def get_heatmap_image(self, background: Optional[np.ndarray] = None, alpha: float = 0.6) -> np.ndarray:
        """
        Get the current heatmap as a colored image.
        
        Args:
            background: Optional background image to overlay on.
            alpha: Overlay transparency.
            
        Returns:
            Colored heatmap image (BGR).
        """
        # Normalize density to [0, 255]
        if self.density.max() > 0:
            norm = (self.density / self.density.max() * 255).astype(np.uint8)
        else:
            norm = np.zeros(self.resolution, dtype=np.uint8)
        
        # Apply colormap
        heatmap = cv2.applyColorMap(norm, cv2.COLORMAP_HOT)
        
        if background is not None:
            bg = cv2.resize(background, (self.resolution[1], self.resolution[0]))
            if len(bg.shape) == 2:
                bg = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)
            
            # Only overlay where there's heat
            mask = norm > 10
            mask_3ch = np.stack([mask] * 3, axis=-1)
            result = bg.copy()
            result[mask_3ch] = cv2.addWeighted(heatmap, alpha, bg, 1 - alpha, 0)[mask_3ch]
            return result
        
        return heatmap
    
    def export_gif(
        self,
        output_path: str,
        fps: int = 5,
        background: Optional[np.ndarray] = None,
        max_frames: int = 50
    ) -> str:
        """
        Export the temporal heatmap as an animated GIF.
        
        Args:
            output_path: Path for the output GIF file.
            fps: Frames per second.
            background: Optional background image.
            max_frames: Maximum frames to include.
            
        Returns:
            Path to the saved GIF.
        """
        if not HAS_IMAGEIO:
            logger.error("imageio required for GIF export")
            return ""
        
        frames_to_use = self.frames[-max_frames:] if len(self.frames) > max_frames else self.frames
        
        if not frames_to_use:
            logger.warning("No frames to export")
            return ""
        
        gif_frames = []
        
        for frame_density in frames_to_use:
            # Normalize
            if frame_density.max() > 0:
                norm = (frame_density / max(self.density.max(), 1e-6) * 255).astype(np.uint8)
            else:
                norm = np.zeros(self.resolution, dtype=np.uint8)
            
            heatmap = cv2.applyColorMap(norm, cv2.COLORMAP_HOT)
            
            if background is not None:
                bg = cv2.resize(background, (self.resolution[1], self.resolution[0]))
                if len(bg.shape) == 2:
                    bg = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)
                heatmap = cv2.addWeighted(heatmap, 0.6, bg, 0.4, 0)
            
            # Convert BGR to RGB for imageio
            heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            gif_frames.append(heatmap_rgb)
        
        output_path = str(Path(output_path))
        imageio.mimsave(output_path, gif_frames, fps=fps, loop=0)
        logger.info(f"Heatmap GIF saved: {output_path} ({len(gif_frames)} frames)")
        
        return output_path
    
    def export_matplotlib_animation(
        self,
        output_path: str,
        fps: int = 5,
        title: str = "Ship Density Over Time"
    ) -> str:
        """
        Export using matplotlib for higher quality animation.
        """
        if not HAS_MATPLOTLIB:
            logger.error("matplotlib required for animation export")
            return ""
        
        frames_to_use = self.frames if self.frames else [self.density]
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        fig.patch.set_facecolor('#1a1a2e')
        ax.set_facecolor('#1a1a2e')
        
        im = ax.imshow(frames_to_use[0], cmap='hot', vmin=0, vmax=max(self.density.max(), 1))
        ax.set_title(title, color='white', fontsize=14, pad=10)
        ax.axis('off')
        
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Ship Density', color='white')
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
        
        frame_text = ax.text(
            0.02, 0.98, '', transform=ax.transAxes,
            color='white', fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.5)
        )
        
        def update(frame_idx):
            im.set_data(frames_to_use[frame_idx])
            frame_text.set_text(f'Frame: {frame_idx + 1}/{len(frames_to_use)}')
            return [im, frame_text]
        
        ani = animation.FuncAnimation(
            fig, update, frames=len(frames_to_use),
            interval=1000/fps, blit=True
        )
        
        output_path = str(Path(output_path))
        ani.save(output_path, writer='pillow', fps=fps)
        plt.close(fig)
        
        logger.info(f"Matplotlib animation saved: {output_path}")
        return output_path
    
    def reset(self):
        """Clear accumulated density data."""
        self.density = np.zeros(self.resolution, dtype=np.float64)
        self.frames.clear()
        self.frame_count = 0
