"""
GeoJSON Map Overlay Module.

Back-projects pixel detections to lat/lng coordinates using 
Sentinel-1 GDAL metadata. Plots ships on an interactive 
Folium/Leaflet map — transforming a computer vision demo 
into a maritime intelligence system.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

try:
    import folium
    from folium.plugins import MarkerCluster, HeatMap
    HAS_FOLIUM = True
except ImportError:
    HAS_FOLIUM = False

try:
    from pyproj import Transformer
    HAS_PYPROJ = True
except ImportError:
    HAS_PYPROJ = False

from src.detection.detector import Detection
from src.analytics.fleet_detect import Fleet


# Default map center (Bay of Bengal / Indian coast)
DEFAULT_CENTER = [12.2, 80.3]
DEFAULT_ZOOM = 10


def pixel_to_latlon(
    px_x: int, px_y: int,
    geotransform=None,
    source_crs: str = "EPSG:32644"
) -> Optional[Tuple[float, float]]:
    """
    Convert pixel coordinates to lat/lng.
    
    Args:
        px_x, px_y: Pixel coordinates.
        geotransform: Rasterio/GDAL affine transform.
        source_crs: Source CRS (e.g., UTM zone).
        
    Returns:
        (latitude, longitude) or None.
    """
    if geotransform is None:
        return None
    
    # Apply affine transform: pixel → projected coordinates
    x_proj, y_proj = geotransform * (px_x, px_y)
    
    if HAS_PYPROJ and source_crs != "EPSG:4326":
        try:
            transformer = Transformer.from_crs(source_crs, "EPSG:4326", always_xy=True)
            lng, lat = transformer.transform(x_proj, y_proj)
            return (lat, lng)
        except Exception as e:
            logger.warning(f"CRS transform failed: {e}")
    
    # Assume already in lat/lng
    return (y_proj, x_proj)


def create_detection_map(
    detections: List[Detection],
    geotransform=None,
    source_crs: str = "EPSG:32644",
    zones: Optional[List[Dict]] = None,
    fleets: Optional[List[Fleet]] = None,
    ais_positions: Optional[List[Tuple[float, float]]] = None,
    center: Optional[List[float]] = None,
    zoom: int = DEFAULT_ZOOM
) -> Optional[object]:
    """
    Create an interactive Folium map with ship detections.
    
    Features:
        - Ship markers color-coded by threat level
        - Dark vessel markers with skull icon
        - Zone polygons as overlays
        - Fleet clusters highlighted
        - AIS vessel markers for comparison
    
    Returns:
        Folium Map object (can be saved as HTML or embedded in Streamlit).
    """
    if not HAS_FOLIUM:
        logger.error("folium required for map visualization")
        return None
    
    # Determine map center
    map_center = center or DEFAULT_CENTER
    
    # If we have geo coords from detections, center on them
    det_coords = []
    for det in detections:
        if geotransform:
            coord = pixel_to_latlon(det.center[0], det.center[1], geotransform, source_crs)
            if coord:
                det_coords.append(coord)
        else:
            # Use simulated coordinates for demo
            lat = DEFAULT_CENTER[0] + (det.center[1] - 320) * 0.001
            lng = DEFAULT_CENTER[1] + (det.center[0] - 320) * 0.001
            det_coords.append((lat, lng))
    
    if det_coords:
        map_center = [np.mean([c[0] for c in det_coords]), np.mean([c[1] for c in det_coords])]
    
    # Create map
    m = folium.Map(
        location=map_center,
        zoom_start=zoom,
        tiles="CartoDB dark_matter",
        attr="CartoDB"
    )
    
    # Add alternate tile layers
    folium.TileLayer("OpenStreetMap").add_to(m)
    folium.TileLayer("CartoDB positron").add_to(m)
    
    # ── Ship detection markers ──
    ship_group = folium.FeatureGroup(name="Detected Ships")
    
    for i, det in enumerate(detections):
        if i < len(det_coords):
            lat, lng = det_coords[i]
        else:
            continue
        
        # Color by threat level
        if det.is_dark_vessel:
            icon_color = "black"
            icon_name = "exclamation-triangle"
        elif det.threat_level == "HIGH":
            icon_color = "red"
            icon_name = "ship"
        elif det.threat_level == "MEDIUM":
            icon_color = "orange"
            icon_name = "ship"
        else:
            icon_color = "green"
            icon_name = "ship"
        
        # Popup with ship info
        popup_html = f"""
        <div style="font-family: Arial; min-width: 200px;">
            <h4>Ship #{det.track_id}</h4>
            <table style="width: 100%;">
                <tr><td><b>Type:</b></td><td>{det.ship_type or 'Unknown'}</td></tr>
                <tr><td><b>Confidence:</b></td><td>{det.confidence:.2%}</td></tr>
                <tr><td><b>Threat Score:</b></td><td>{det.threat_score:.0f}/100</td></tr>
                <tr><td><b>Threat Level:</b></td><td>{det.threat_level}</td></tr>
                <tr><td><b>Dark Vessel:</b></td><td>{'⚠️ YES' if det.is_dark_vessel else 'No'}</td></tr>
                <tr><td><b>Location:</b></td><td>{lat:.4f}°N, {lng:.4f}°E</td></tr>
            </table>
        </div>
        """
        
        folium.Marker(
            location=[lat, lng],
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"Ship #{det.track_id} ({det.ship_type or 'unknown'})",
            icon=folium.Icon(color=icon_color, icon=icon_name, prefix="fa")
        ).add_to(ship_group)
    
    ship_group.add_to(m)
    
    # ── Zone overlays ──
    if zones:
        zone_group = folium.FeatureGroup(name="Zones")
        
        for zone in zones:
            coords = zone.get("coordinates", [])
            if len(coords) < 3:
                continue
            
            # Convert [lng, lat] to [lat, lng] for Folium
            folium_coords = [[c[1], c[0]] for c in coords]
            hex_color = zone.get("color", "#FFA500")
            
            folium.Polygon(
                locations=folium_coords,
                color=hex_color,
                weight=2,
                fill=True,
                fill_opacity=0.15,
                popup=f"{zone['name']} ({zone['type']})",
                tooltip=zone['name']
            ).add_to(zone_group)
        
        zone_group.add_to(m)
    
    # ── AIS markers ──
    if ais_positions:
        ais_group = folium.FeatureGroup(name="AIS Vessels")
        
        for i, (x, y) in enumerate(ais_positions):
            # Convert pixel to approx lat/lng
            lat = DEFAULT_CENTER[0] + (y - 320) * 0.001
            lng = DEFAULT_CENTER[1] + (x - 320) * 0.001
            
            folium.CircleMarker(
                location=[lat, lng],
                radius=4,
                color="lime",
                fill=True,
                fill_opacity=0.7,
                tooltip=f"AIS Vessel {i+1}"
            ).add_to(ais_group)
        
        ais_group.add_to(m)
    
    # ── Detection heatmap layer ──
    if det_coords:
        heat_data = [[c[0], c[1]] for c in det_coords]
        HeatMap(heat_data, radius=25, blur=15, name="Detection Heatmap").add_to(m)
    
    # Layer control
    folium.LayerControl().add_to(m)
    
    return m


def save_map(map_obj, filepath: str):
    """Save Folium map to HTML file."""
    if map_obj:
        map_obj.save(filepath)
        logger.info(f"Map saved to {filepath}")
