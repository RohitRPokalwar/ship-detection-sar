"""Reports Page — PDF/HTML report generation and download."""
import streamlit as st
import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "outputs"


def show_reports_page():
    st.markdown("## 📄 Report Generation")

    result = st.session_state.get("last_result")
    pipeline = st.session_state.get("pipeline")

    if not result:
        st.warning("Run detection first to generate reports.")
        return

    detections = result.get("detections", [])
    alerts = result.get("alert_log", [])

    # Session summary
    st.markdown("### 📊 Session Summary")
    if pipeline:
        summary = pipeline.get_session_summary()
    else:
        dark = sum(1 for d in detections if d.get("is_dark_vessel"))
        high = sum(1 for d in detections if d.get("threat_level") == "HIGH")
        summary = {"total_detections": len(detections), "dark_vessels": dark,
                    "high_threat": high, "total_alerts": len(alerts)}

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Ships", summary.get("total_detections", 0))
    c2.metric("Dark Vessels", summary.get("dark_vessels", 0))
    c3.metric("High Threat", summary.get("high_threat", 0))
    c4.metric("Alerts", summary.get("total_alerts", 0))

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("📄 Generate PDF Report", type="primary", use_container_width=True):
            OUTPUT_DIR.mkdir(exist_ok=True)
            pdf_path = OUTPUT_DIR / f"report_{int(time.time())}.pdf"
            from src.reporting.report_gen import generate_pdf_report
            path = generate_pdf_report(
                str(pdf_path), detections, alerts,
                fleets=result.get("fleets"), session_info=summary
            )
            if path and Path(path).exists():
                st.success(f"Report generated: {Path(path).name}")
                with open(path, "rb") as f:
                    st.download_button("📥 Download Report", f.read(),
                                       Path(path).name,
                                       "application/pdf" if path.endswith('.pdf') else "text/html")

    with col2:
        if st.button("🌐 Generate HTML Report", use_container_width=True):
            OUTPUT_DIR.mkdir(exist_ok=True)
            html_path = OUTPUT_DIR / f"report_{int(time.time())}.html"
            from src.reporting.report_gen import _generate_html_report
            path = _generate_html_report(
                str(html_path), detections, alerts,
                fleets=result.get("fleets"), heatmap_path=None, session_info=summary
            )
            if path and Path(path).exists():
                st.success(f"HTML report generated!")
                with open(path, "rb") as f:
                    st.download_button("📥 Download HTML", f.read(),
                                       Path(path).name, "text/html")
