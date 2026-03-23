"""Detection Page — Upload SAR images and run ship detection."""
import streamlit as st
import cv2, numpy as np, time, sys
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def get_pipeline():
    if "pipeline" not in st.session_state:
        from src.pipeline import SARPipeline
        st.session_state["pipeline"] = SARPipeline()
    return st.session_state["pipeline"]


def show_detection_page():
    st.markdown("## 🔍 Ship Detection")

    upload_tab, sample_tab = st.tabs(["📤 Upload Image", "📁 Sample Images"])

    with upload_tab:
        uploaded = st.file_uploader("Upload a SAR image", type=["png", "jpg", "jpeg", "tif", "tiff"])
        if uploaded:
            file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if image is None:
                st.error("Failed to read image.")
                return
            _run_detection(image)

    with sample_tab:
        st.info("Place sample SAR images in `data/samples/` to see them here.")
        samples_dir = Path(__file__).resolve().parent.parent.parent / "data" / "samples"
        if samples_dir.exists():
            files = list(samples_dir.glob("*.[pjt]*"))
            if files:
                sel = st.selectbox("Select sample", [f.name for f in files])
                path = samples_dir / sel
                image = cv2.imread(str(path))
                if image is not None:
                    _run_detection(image)
        else:
            st.markdown("No sample directory found. Create `data/samples/` and add SAR images.")


def _run_detection(image: np.ndarray):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Original Image")
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)

    apply_filter = st.session_state.get("filter_type", "none") != "none"

    if st.button("🚀 Run Detection", type="primary", use_container_width=True):
        with st.spinner("Running detection pipeline..."):
            pipeline = get_pipeline()
            start = time.time()
            result = pipeline.process_frame(image, apply_filter=apply_filter)
            elapsed = time.time() - start

        with col2:
            st.markdown("### Detection Result")
            rendered = result["rendered_image"]
            st.image(cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB), use_container_width=True)

        # Metrics row
        dets = result["detections"]
        dark = sum(1 for d in dets if d.get("is_dark_vessel"))
        high = sum(1 for d in dets if d.get("threat_level") == "HIGH")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Ships Detected", len(dets))
        m2.metric("Dark Vessels", dark)
        m3.metric("High Threat", high)
        m4.metric("Inference Time", f"{elapsed:.2f}s")

        # Detection table
        if dets:
            st.markdown("### Detection Details")
            import pandas as pd
            df = pd.DataFrame([{
                "Track ID": d["track_id"], "Type": d.get("ship_type", "?"),
                "Confidence": f"{d['confidence']:.2%}",
                "Threat": f"{d['threat_score']:.0f} ({d['threat_level']})",
                "Dark Vessel": "⚠️ YES" if d.get("is_dark_vessel") else "✓ No",
            } for d in dets])
            st.dataframe(df, use_container_width=True)

        # Store results for other pages
        st.session_state["last_result"] = result
        st.session_state["last_image"] = image
