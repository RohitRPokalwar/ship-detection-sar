"""Analytics Page — Fleet detection, trajectory, and heatmap views."""
import streamlit as st
import cv2, sys, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def show_analytics_page():
    st.markdown("## 📊 Analytics Dashboard")

    result = st.session_state.get("last_result")
    image = st.session_state.get("last_image")
    if not result:
        st.warning("Run detection first to see analytics.")
        return

    tab1, tab2, tab3 = st.tabs(["⚓ Fleet Detection", "📍 Trajectory Prediction", "🔥 Heatmap"])

    with tab1:
        _show_fleets(result)

    with tab2:
        _show_trajectories(result, image)

    with tab3:
        _show_heatmap(result, image)


def _show_fleets(result):
    st.markdown("### Fleet / Formation Detection (DBSCAN)")
    fleets = result.get("fleets", [])
    if not fleets:
        st.info("No fleet formations detected. Need 3+ ships in proximity.")
        return

    for f in fleets:
        with st.expander(f"Fleet #{f.fleet_id} — {f.num_ships} ships", expanded=True):
            c1, c2, c3 = st.columns(3)
            c1.metric("Ships", f.num_ships)
            c2.metric("Radius", f"{f.radius:.0f}px")
            c3.metric("Centroid", f"({f.centroid[0]:.0f}, {f.centroid[1]:.0f})")
            st.write(f"**Member tracks:** {f.ship_track_ids}")


def _show_trajectories(result, image):
    st.markdown("### Trajectory Prediction (Kalman Filter)")
    preds = result.get("predictions", {})
    if not preds:
        st.info("No trajectory predictions yet. Need multiple frames for tracking.")
        return

    for tid, future in preds.items():
        st.markdown(f"**Track #{tid}:** {len(future)} predicted positions")
        if future:
            st.write(f"  Next position → ({future[0][0]:.0f}, {future[0][1]:.0f})")

    if image is not None:
        from src.visualization.renderer import draw_predictions, draw_detections
        from src.detection.detector import Detection
        dets = [Detection(bbox=d["bbox"], confidence=d["confidence"], track_id=d["track_id"])
                for d in result["detections"]]
        vis = draw_detections(image, dets)
        vis = draw_predictions(vis, preds, dets)
        st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), caption="Trajectory Predictions", use_container_width=True)


def _show_heatmap(result, image):
    st.markdown("### Ship Density Heatmap")

    pipeline = st.session_state.get("pipeline")
    if not pipeline:
        st.info("Run detection to generate heatmap data.")
        return

    heatmap_img = pipeline.heatmap.get_heatmap_image(background=image)
    st.image(cv2.cvtColor(heatmap_img, cv2.COLOR_BGR2RGB), caption="Cumulative Ship Density",
             use_container_width=True)

    if st.button("📥 Export Heatmap GIF"):
        out = Path(__file__).resolve().parent.parent.parent / "outputs" / "heatmap.gif"
        out.parent.mkdir(exist_ok=True)
        path = pipeline.heatmap.export_gif(str(out), fps=5, background=image)
        if path:
            st.success(f"GIF saved to {path}")
            with open(path, "rb") as f:
                st.download_button("Download GIF", f.read(), "heatmap.gif", "image/gif")
