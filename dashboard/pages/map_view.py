"""Map View Page — Folium map with ship detections and zones."""
import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def show_map_page():
    st.markdown("## 🗺️ GeoJSON Map Overlay")

    result = st.session_state.get("last_result")
    if not result:
        st.warning("Run detection first on the Detection page to see ships on the map.")
        st.info("The map will show detected ships plotted at real (or simulated) coordinates with zone overlays.")
        return

    try:
        from src.visualization.geo_overlay import create_detection_map, save_map
        from src.detection.detector import Detection
        from src.analytics.threat_score import load_zones
        from config.config import ZONES_PATH
        import streamlit.components.v1 as components

        # Rebuild Detection objects
        detections = []
        for d in result["detections"]:
            det = Detection(
                bbox=d["bbox"], confidence=d["confidence"], track_id=d["track_id"],
                ship_type=d.get("ship_type", ""), threat_score=d.get("threat_score", 0),
                threat_level=d.get("threat_level", ""), is_dark_vessel=d.get("is_dark_vessel", False)
            )
            detections.append(det)

        zones = load_zones(str(ZONES_PATH)) if ZONES_PATH.exists() else []

        m = create_detection_map(detections=detections, zones=zones, zoom=11)
        if m:
            map_html = m._repr_html_()
            components.html(map_html, height=600, scrolling=True)

            # Stats
            col1, col2, col3 = st.columns(3)
            col1.metric("Ships on Map", len(detections))
            dark = sum(1 for d in detections if d.is_dark_vessel)
            col2.metric("Dark Vessels", dark)
            col3.metric("Zones Active", len(zones))

            # Save map button
            if st.button("💾 Save Map as HTML"):
                out = Path(__file__).resolve().parent.parent.parent / "outputs" / "detection_map.html"
                out.parent.mkdir(exist_ok=True)
                save_map(m, str(out))
                st.success(f"Map saved to {out}")
        else:
            st.error("Failed to create map. Install folium: pip install folium")
    except Exception as e:
        st.error(f"Map error: {e}")
        import traceback
        st.code(traceback.format_exc())
