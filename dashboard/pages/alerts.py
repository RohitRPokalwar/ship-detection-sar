"""Alerts Page — Zone alerts, dark vessel log, and NL query interface."""
import streamlit as st
import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def show_alerts_page():
    st.markdown("## ⚠️ Alerts & Intelligence")

    tab1, tab2, tab3 = st.tabs(["🚨 Zone Alerts", "🔴 Dark Vessels", "💬 NL Query"])

    result = st.session_state.get("last_result")
    detections = result["detections"] if result else []
    alerts = result.get("alert_log", []) if result else []

    with tab1:
        _show_zone_alerts(alerts)

    with tab2:
        _show_dark_vessels(detections)

    with tab3:
        _show_nl_query(detections, alerts)


def _show_zone_alerts(alerts):
    st.markdown("### Zone Violation Alerts")
    if not alerts:
        st.info("No zone alerts triggered. Ships need to enter defined zones.")
        return

    for a in reversed(alerts[-20:]):
        level = a.get("alert_level", "MEDIUM")
        css = {"HIGH": "threat-high", "MEDIUM": "threat-medium", "LOW": "threat-low"}.get(level, "threat-medium")
        st.markdown(
            f'<span class="{css}">{level}</span> '
            f'**{a.get("time_str", "")}** — {a.get("message", "")}',
            unsafe_allow_html=True
        )


def _show_dark_vessels(detections):
    st.markdown("### Dark Vessel Detection")
    st.markdown("Ships with **no AIS match** — potential illegal or evasive activity.")

    dark = [d for d in detections if d.get("is_dark_vessel")]
    if not dark:
        st.success("✅ No dark vessels detected. All ships have AIS matches.")
        return

    st.error(f"🔴 {len(dark)} DARK VESSEL(S) DETECTED")
    for d in dark:
        with st.expander(f"⚠️ Ship #{d['track_id']} — DARK VESSEL", expanded=True):
            c1, c2, c3 = st.columns(3)
            c1.metric("Confidence", f"{d['confidence']:.2%}")
            c2.metric("Threat Score", f"{d['threat_score']:.0f}")
            c3.metric("Type", d.get("ship_type", "Unknown"))


def _show_nl_query(detections, alerts):
    st.markdown("### 💬 Natural Language Query")
    st.markdown("Ask questions like: *Show dark vessels*, *How many ships?*, *High threat ships*")

    query = st.chat_input("Ask about the detection session...")
    if query:
        from src.reporting.nl_query import QueryExecutor
        executor = QueryExecutor()
        result = executor.execute(query, detections=detections, alerts=alerts)

        st.chat_message("user").write(query)
        st.chat_message("assistant").write(result["message"])

        if result["count"] > 0 and isinstance(result["results"], list) and result["results"]:
            import pandas as pd
            try:
                df = pd.DataFrame(result["results"][:20])
                st.dataframe(df, use_container_width=True)
            except Exception:
                st.json(result["results"][:10])
