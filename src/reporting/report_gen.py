"""
Automated PDF/HTML Report Generator.
One-click session summary: ship count, alert log, heatmap, top detections.
"""
import os, time
from pathlib import Path
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False


def generate_pdf_report(
    output_path: str,
    detections: List[Dict],
    alerts: List[Dict],
    fleets: List = None,
    heatmap_path: str = None,
    session_info: Dict = None
) -> str:
    if not HAS_REPORTLAB:
        logger.error("reportlab required for PDF generation")
        return _generate_html_report(output_path.replace('.pdf', '.html'),
                                      detections, alerts, fleets, heatmap_path, session_info)

    doc = SimpleDocTemplate(str(output_path), pagesize=A4)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('CustomTitle', parent=styles['Title'], fontSize=20,
                                  textColor=colors.HexColor('#1a1a2e'), spaceAfter=20)
    elements = []

    # Title
    elements.append(Paragraph("🛳️ SAR Maritime Intelligence Report", title_style))
    elements.append(Paragraph(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    elements.append(Spacer(1, 20))

    # Summary stats
    dark_count = sum(1 for d in detections if d.get("is_dark_vessel"))
    high_threat = sum(1 for d in detections if d.get("threat_level") == "HIGH")
    summary_data = [
        ["Metric", "Value"],
        ["Total Ships Detected", str(len(detections))],
        ["Dark Vessels", str(dark_count)],
        ["High Threat Ships", str(high_threat)],
        ["Total Alerts", str(len(alerts))],
        ["Fleet Formations", str(len(fleets) if fleets else 0)],
    ]
    t = Table(summary_data, colWidths=[3*inch, 2*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a1a2e')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
    ]))
    elements.append(Paragraph("Summary", styles['Heading2']))
    elements.append(t)
    elements.append(Spacer(1, 20))

    # Top detections table
    if detections:
        elements.append(Paragraph("Top Detections", styles['Heading2']))
        det_data = [["Track ID", "Type", "Confidence", "Threat", "Dark?"]]
        for d in detections[:15]:
            det_data.append([
                str(d.get("track_id", "-")),
                d.get("ship_type", "unknown"),
                f"{d.get('confidence', 0):.2%}",
                f"{d.get('threat_score', 0):.0f} ({d.get('threat_level', '-')})",
                "⚠️" if d.get("is_dark_vessel") else "✓"
            ])
        dt = Table(det_data, colWidths=[1*inch, 1.2*inch, 1.2*inch, 1.5*inch, 0.8*inch])
        dt.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2d3436')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
        ]))
        elements.append(dt)
        elements.append(Spacer(1, 20))

    # Heatmap image
    if heatmap_path and os.path.exists(heatmap_path):
        elements.append(Paragraph("Ship Density Heatmap", styles['Heading2']))
        elements.append(RLImage(heatmap_path, width=5*inch, height=5*inch))

    # Alert log
    if alerts:
        elements.append(Spacer(1, 20))
        elements.append(Paragraph("Alert Log (Recent)", styles['Heading2']))
        alert_data = [["Time", "Zone", "Ship", "Level"]]
        for a in alerts[-10:]:
            alert_data.append([
                a.get("time_str", ""),
                a.get("zone_name", ""),
                str(a.get("ship_track_id", "")),
                a.get("alert_level", ""),
            ])
        at = Table(alert_data, colWidths=[2*inch, 1.5*inch, 1*inch, 1*inch])
        at.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#c0392b')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
        ]))
        elements.append(at)

    doc.build(elements)
    logger.info(f"PDF report saved: {output_path}")
    return str(output_path)


def _generate_html_report(output_path, detections, alerts, fleets, heatmap_path, session_info):
    dark_count = sum(1 for d in detections if d.get("is_dark_vessel"))
    high_threat = sum(1 for d in detections if d.get("threat_level") == "HIGH")
    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>SAR Maritime Intelligence Report</title>
<style>
body{{font-family:'Segoe UI',sans-serif;background:#0f0f1a;color:#e0e0e0;padding:40px;}}
h1{{color:#00d4ff;}} h2{{color:#ff6b6b;border-bottom:1px solid #333;padding-bottom:8px;}}
table{{border-collapse:collapse;width:100%;margin:16px 0;}}
th{{background:#1a1a2e;color:#00d4ff;padding:10px;}} td{{padding:8px;border:1px solid #333;}}
tr:nth-child(even){{background:#1a1a2e;}} .stat{{font-size:2em;color:#00d4ff;font-weight:bold;}}
.card{{background:#16213e;border-radius:12px;padding:20px;margin:16px 0;}}
</style></head><body>
<h1>🛳️ SAR Maritime Intelligence Report</h1>
<p>Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
<div class="card"><h2>Summary</h2>
<table><tr><th>Metric</th><th>Value</th></tr>
<tr><td>Total Ships</td><td class="stat">{len(detections)}</td></tr>
<tr><td>Dark Vessels</td><td>{dark_count}</td></tr>
<tr><td>High Threat</td><td>{high_threat}</td></tr>
<tr><td>Alerts</td><td>{len(alerts)}</td></tr>
<tr><td>Fleets</td><td>{len(fleets) if fleets else 0}</td></tr>
</table></div>
<div class="card"><h2>Detections</h2><table>
<tr><th>ID</th><th>Type</th><th>Conf</th><th>Threat</th><th>Dark</th></tr>"""
    for d in detections[:15]:
        html += f"<tr><td>{d.get('track_id','-')}</td><td>{d.get('ship_type','?')}</td>"
        html += f"<td>{d.get('confidence',0):.2%}</td><td>{d.get('threat_score',0):.0f}</td>"
        html += f"<td>{'⚠️' if d.get('is_dark_vessel') else '✓'}</td></tr>"
    html += "</table></div></body></html>"

    Path(output_path).write_text(html, encoding='utf-8')
    logger.info(f"HTML report saved: {output_path}")
    return str(output_path)
