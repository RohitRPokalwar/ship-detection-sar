"""
🛳️ SAR Maritime Intelligence System — Streamlit Dashboard
Main entry point with multi-page navigation.
Run: streamlit run dashboard/app.py
"""
import streamlit as st
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(
    page_title="SAR Maritime Intelligence",
    page_icon="🛳️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Dark Theme CSS ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

.stApp {
    font-family: 'Inter', sans-serif;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a0a1a 0%, #1a1a2e 100%);
}

/* Metric cards */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #16213e 0%, #1a1a2e 100%);
    border: 1px solid #2d3436;
    border-radius: 12px;
    padding: 16px;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
}

[data-testid="stMetricValue"] {
    font-size: 2rem !important;
    font-weight: 700 !important;
}

/* Custom alert badges */
.threat-high { background: #EF4444; color: white; padding: 4px 12px; border-radius: 20px; font-weight: 600; }
.threat-medium { background: #F59E0B; color: black; padding: 4px 12px; border-radius: 20px; font-weight: 600; }
.threat-low { background: #22C55E; color: white; padding: 4px 12px; border-radius: 20px; font-weight: 600; }
.dark-vessel { background: #DC2626; color: white; padding: 4px 12px; border-radius: 20px; font-weight: 600; animation: pulse 2s infinite; }

@keyframes pulse {
    0% { box-shadow: 0 0 0 0 rgba(220, 38, 38, 0.4); }
    70% { box-shadow: 0 0 0 10px rgba(220, 38, 38, 0); }
    100% { box-shadow: 0 0 0 0 rgba(220, 38, 38, 0); }
}

/* Header gradient */
.main-header {
    background: linear-gradient(135deg, #00d4ff 0%, #7b2ff7 50%, #ff6b6b 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.5rem;
    font-weight: 800;
    text-align: center;
    margin-bottom: 0;
}

.sub-header {
    text-align: center;
    color: #888;
    font-size: 1rem;
    margin-top: -10px;
    margin-bottom: 30px;
}

/* Status indicator */
.status-active { color: #22C55E; font-weight: 600; }
.status-idle { color: #888; }

/* Button styling */
.stButton > button {
    border-radius: 8px;
    font-weight: 600;
    transition: all 0.3s ease;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #1a1a2e; }
::-webkit-scrollbar-thumb { background: #00d4ff; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


def main():
    # ── Sidebar Navigation ──
    with st.sidebar:
        st.markdown("### 🛳️ Navigation")
        page = st.radio(
            "Select Module",
            ["🏠 Home", "🔍 Detection", "🗺️ Map View", "📊 Analytics", 
             "⚠️ Alerts", "📄 Reports"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.markdown("### ⚙️ Settings")
        
        confidence = st.slider("Detection Confidence", 0.1, 0.9, 0.25, 0.05)
        filter_type = st.selectbox("Speckle Filter", ["lee", "frost", "none"])
        
        st.markdown("---")
        st.markdown(
            '<p style="color:#888;font-size:0.8rem;">SAR Maritime Intelligence v1.0<br>'
            'Ship Detection in SAR Imagery</p>',
            unsafe_allow_html=True
        )
    
    # ── Store settings in session ──
    st.session_state["confidence"] = confidence
    st.session_state["filter_type"] = filter_type
    
    # ── Route to pages ──
    if page == "🏠 Home":
        show_home()
    elif page == "🔍 Detection":
        from dashboard.pages.detection import show_detection_page
        show_detection_page()
    elif page == "🗺️ Map View":
        from dashboard.pages.map_view import show_map_page
        show_map_page()
    elif page == "📊 Analytics":
        from dashboard.pages.analytics import show_analytics_page
        show_analytics_page()
    elif page == "⚠️ Alerts":
        from dashboard.pages.alerts import show_alerts_page
        show_alerts_page()
    elif page == "📄 Reports":
        from dashboard.pages.reports import show_reports_page
        show_reports_page()


def show_home():
    st.markdown('<h1 class="main-header">SAR Maritime Intelligence System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Ship Detection in SAR Imagery Using Lightweight Models</p>', unsafe_allow_html=True)
    
    # Feature cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🔍 Detection", "YOLOv8n", "ByteTrack")
    with col2:
        st.metric("🎯 Classification", "4 Types", "EfficientNet")
    with col3:
        st.metric("⚠️ Threat Scoring", "0-100", "Multi-factor")
    with col4:
        st.metric("🔴 Dark Vessels", "AIS Match", "Spatial Join")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🚀 Core Features")
        st.markdown("""
        - **YOLOv8n + ByteTrack** — Real-time ship detection & tracking
        - **Threat Scoring** — 0-100 risk score (confidence, zone, speed, dwell)
        - **Multi-Zone Alerts** — Named zones with cooldown timers
        - **SAR Speckle Filter** — Lee/Frost despeckling
        """)
        
        st.markdown("### 🌟 Unique Features")
        st.markdown("""
        - **Dark Vessel Detection** — No-AIS-match flagging
        - **Ship Type Classifier** — Cargo/Tanker/Fishing/Military
        - **Fleet Detection** — DBSCAN convoy clustering
        - **Trajectory Prediction** — Kalman filter forecasts
        """)
    
    with col2:
        st.markdown("### 🏆 Wow-Factor Features")
        st.markdown("""
        - **GeoJSON Map Overlay** — Real lat/lng on Folium maps
        - **NL Query Interface** — "Show ships in zone B"
        - **Temporal Heatmap** — Animated density GIF export
        - **PDF/HTML Reports** — One-click session summary
        """)
        
        st.markdown("### 📋 Quick Start")
        st.markdown("""
        1. Upload a SAR image in **🔍 Detection**
        2. View ship locations on **🗺️ Map View**
        3. Explore analytics in **📊 Analytics**
        4. Monitor alerts in **⚠️ Alerts**
        5. Generate reports in **📄 Reports**
        """)


if __name__ == "__main__":
    main()
