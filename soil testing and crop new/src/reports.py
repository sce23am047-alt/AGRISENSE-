"""
src/reports.py
Generates a downloadable PDF soil-analysis report using reportlab.
"""

import sys
from datetime import datetime
from io import BytesIO
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)

sys.path.append(str(Path(__file__).resolve().parent.parent))


FOREST_GREEN = colors.HexColor("#1B4332")
AMBER = colors.HexColor("#B45309")


def build_report(soil_values: dict, soil_analysis: dict, crop_result: dict,
                  fertilizer_result: dict, ai_summary: str, location: str = "") -> bytes:
    """Returns PDF bytes for a complete soil-analysis report."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        topMargin=1.5 * cm, bottomMargin=1.5 * cm,
        leftMargin=1.8 * cm, rightMargin=1.8 * cm,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleGreen", parent=styles["Title"], textColor=FOREST_GREEN)
    heading_style = ParagraphStyle("HeadingGreen", parent=styles["Heading2"], textColor=FOREST_GREEN,
                                    spaceBefore=10, spaceAfter=6)
    body = styles["Normal"]

    story = []
    story.append(Paragraph("AgriSense AI", title_style))
    story.append(Paragraph("AI-Based Soil Testing and Crop Recommendation Report", styles["Heading3"]))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%d %b %Y, %H:%M')}", body))
    if location:
        story.append(Paragraph(f"Location: {location}", body))
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", color=FOREST_GREEN, thickness=1))

    # Soil parameters
    story.append(Paragraph("Soil Parameters", heading_style))
    param_rows = [["Parameter", "Value", "Status"]]
    labels = {"N": "Nitrogen (N)", "P": "Phosphorus (P)", "K": "Potassium (K)",
              "temperature": "Temperature (°C)", "moisture": "Moisture (%)", "ph": "pH"}
    for key, label in labels.items():
        val = soil_values.get(key, "-")
        status = soil_analysis.get("status", {}).get(key, "-")
        param_rows.append([label, str(val), status])
    t = Table(param_rows, colWidths=[6 * cm, 4 * cm, 4 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), FOREST_GREEN),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F1F5F1")]),
    ]))
    story.append(t)

    # Soil health
    story.append(Paragraph("Soil Health Indicator (AI/Rule-Based, not laboratory-certified)", heading_style))
    story.append(Paragraph(
        f"Score: <b>{soil_analysis.get('score', '-')}/100</b> — {soil_analysis.get('label', '-')}", body))

    # Crop recommendation
    story.append(Paragraph("AI Crop Recommendation", heading_style))
    story.append(Paragraph(f"Recommended Crop: <b>{crop_result.get('primary', '-')}</b>", body))
    if crop_result.get("confidence") is not None:
        story.append(Paragraph(f"Confidence: <b>{crop_result['confidence']}%</b>", body))
    alts = crop_result.get("alternatives", [])
    if alts:
        alt_text = ", ".join(f"{a['crop']} ({a['match']}%)" for a in alts)
        story.append(Paragraph(f"Alternative Crops: {alt_text}", body))

    # Fertilizer
    story.append(Paragraph("Precision Fertilizer Recommendation", heading_style))
    story.append(Paragraph(f"Status: <b>{fertilizer_result.get('status', '-')}</b>", body))
    story.append(Paragraph(f"Recommended Fertilizer: <b>{fertilizer_result.get('recommended_fertilizer', '-')}</b>", body))
    story.append(Paragraph(f"Application Stage: {fertilizer_result.get('application_stage', '-')}", body))
    story.append(Paragraph(fertilizer_result.get("guidance", ""), body))

    # AI decision summary
    story.append(Paragraph("AI Decision Summary", heading_style))
    story.append(Paragraph(ai_summary, body))

    story.append(Spacer(1, 14))
    story.append(HRFlowable(width="100%", color=colors.grey, thickness=0.5))
    story.append(Paragraph(
        "<i>Important: These recommendations are decision-support estimates generated "
        "from a trained machine-learning model and a transparent rule-based engine. "
        "They are not a substitute for a certified soil laboratory report. Final crop "
        "and fertilizer decisions should also consider soil type, crop variety, local "
        "agricultural guidance, and field conditions.</i>", styles["Italic"]))

    doc.build(story)
    return buffer.getvalue()
