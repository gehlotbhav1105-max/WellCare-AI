"""
WellCare AI - ReportLab PDF Report Generator
============================================
Generates comprehensive, beautifully styled PDF wellness reports.
Includes demographics, overall score, category badges, pillar tables,
factor ratings, strengths, improvement areas, recommendations, and
mandatory educational healthcare disclaimers.

DISCLAIMER:
WellCare AI is an educational wellness-support system. Its results are general
wellness indicators and should not be considered a medical diagnosis or a
substitute for professional healthcare advice.
"""

import os
from datetime import datetime
from typing import Dict, Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from recommendations import generate_recommendations
from scoring import calculate_wellness_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPORTS_DIR = os.path.join(BASE_DIR, "reports")


def ensure_reports_directory() -> None:
    """Ensures the reports directory exists."""
    os.makedirs(REPORTS_DIR, exist_ok=True)


def generate_pdf_report(user_data: Dict[str, Any], prediction_result: Dict[str, Any] = None) -> str:
    """
    Generates a professional PDF report for a user and saves it to reports/ directory.
    Returns the absolute path to the generated PDF.
    """
    ensure_reports_directory()
    user_id = str(user_data.get("User_ID", "WC000")).strip()
    pdf_filename = f"WellCare_Report_{user_id}.pdf"
    pdf_path = os.path.join(REPORTS_DIR, pdf_filename)

    # Compute scoring & recommendations
    scoring = calculate_wellness_score(user_data)
    recommendations = generate_recommendations(user_data, scoring)

    score = scoring["Wellness_Score"]
    category = scoring["Wellness_Category"]
    bmi = scoring["BMI"]
    bmi_cat = scoring["BMI_Category"]

    # PDF Document setup
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )
    styles = getSampleStyleSheet()

    # Custom Styles
    primary_color = colors.HexColor("#0D9488")    # Teal primary
    secondary_color = colors.HexColor("#0284C7")  # Sky secondary
    dark_text = colors.HexColor("#0F172A")        # Slate 900
    muted_text = colors.HexColor("#475569")       # Slate 600
    card_bg = colors.HexColor("#F8FAFC")          # Slate 50
    accent_green = colors.HexColor("#10B981")
    accent_amber = colors.HexColor("#F59E0B")
    accent_rose = colors.HexColor("#F43F5E")

    category_color = accent_green if "Excellent" in category else (
        primary_color if "Good" in category else (
            accent_amber if "Moderate" in category else accent_rose
        )
    )

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=22,
        leading=26,
        textColor=primary_color,
        spaceAfter=2,
    )

    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=muted_text,
        spaceAfter=12,
    )

    section_heading = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        textColor=dark_text,
        spaceBefore=10,
        spaceAfter=6,
    )

    body_style = ParagraphStyle(
        "ReportBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13,
        textColor=dark_text,
    )

    bold_body = ParagraphStyle(
        "BoldBody",
        parent=body_style,
        fontName="Helvetica-Bold",
    )

    disclaimer_style = ParagraphStyle(
        "DisclaimerText",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#64748B"),
    )

    elements = []

    # 1. Header Banner
    header_data = [
        [
            Paragraph("<b>WELLCARE AI</b><br/><font size=9 color='#0284C7'>Intelligent Personal Wellness Monitoring System</font>", title_style),
            Paragraph(f"<b>Report ID:</b> {user_id}<br/><b>Date:</b> {datetime.now().strftime('%d %b %Y, %H:%M')}<br/><b>Status:</b> Verified Assessment", subtitle_style),
        ]
    ]
    header_table = Table(header_data, colWidths=[3.8 * inch, 3.2 * inch])
    header_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ALIGN", (1, 0), (1, 0), "RIGHT"),
    ]))
    elements.append(header_table)
    elements.append(HRFlowable(width="100%", thickness=1.5, color=primary_color, spaceBefore=4, spaceAfter=10))

    # 2. User Information & Score Summary Banner
    user_info_content = [
        [Paragraph(f"<b>Name:</b> {user_data.get('Name', 'Anonymous')}", body_style), Paragraph(f"<b>Age:</b> {user_data.get('Age', '-')} yrs", body_style)],
        [Paragraph(f"<b>Gender:</b> {user_data.get('Gender', '-')}", body_style), Paragraph(f"<b>Height:</b> {user_data.get('Height', '-')} cm", body_style)],
        [Paragraph(f"<b>Weight:</b> {user_data.get('Weight', '-')} kg", body_style), Paragraph(f"<b>BMI:</b> {bmi} ({bmi_cat})", body_style)],
        [Paragraph(f"<b>Steps/Day:</b> {user_data.get('Steps', '-')}", body_style), Paragraph(f"<b>Sleep:</b> {user_data.get('Sleep_Hours', '-')} hrs/night", body_style)],
    ]
    info_table = Table(user_info_content, colWidths=[2.2 * inch, 2.0 * inch])
    info_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F1F5F9")),
        ("PADDING", (0, 0), (-1, -1), 4),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))

    score_badge_content = [
        [Paragraph("<font size=8 color='#64748B'>OVERALL WELLNESS SCORE</font>", ParagraphStyle("CenteredMuted", alignment=1))],
        [Paragraph(f"<font size=26 color='{category_color.hexval()}'><b>{score}/100</b></font>", ParagraphStyle("CenteredScore", alignment=1, leading=30))],
        [Paragraph(f"<b>{category.upper()}</b>", ParagraphStyle("CenteredCat", alignment=1, textColor=category_color, fontName="Helvetica-Bold", fontSize=11))],
    ]
    score_table = Table(score_badge_content, colWidths=[2.8 * inch])
    score_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#E2E8F0")),
        ("PADDING", (0, 0), (-1, -1), 6),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOX", (0, 0), (-1, -1), 1, category_color),
    ]))

    top_summary = Table([[info_table, score_table]], colWidths=[4.2 * inch, 2.8 * inch])
    top_summary.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    elements.append(top_summary)
    elements.append(Spacer(1, 10))

    # 3. Wellness Pillars Breakdown Table
    elements.append(Paragraph("1. Core Wellness Pillars Analysis", section_heading))
    pillars = scoring["Pillars"]
    sub_scores = scoring["Sub_Scores"]

    pillar_rows = [
        [Paragraph("<b>Wellness Dimension</b>", bold_body), Paragraph("<b>Sub-Score</b>", bold_body), Paragraph("<b>Score %</b>", bold_body), Paragraph("<b>Assessment Level</b>", bold_body)],
        [Paragraph("Physical Wellness (Steps, Exercise, BMI)", body_style), Paragraph(f"{sub_scores['Physical_Score']}/25", body_style), Paragraph(f"{pillars['Physical']}%", body_style), Paragraph("Optimal" if pillars['Physical'] >= 70 else ("Moderate" if pillars['Physical'] >= 50 else "Needs Attention"), body_style)],
        [Paragraph("Sleep & Circadian Restoration", body_style), Paragraph(f"{sub_scores['Sleep_Score']}/20", body_style), Paragraph(f"{pillars['Sleep']}%", body_style), Paragraph("Optimal" if pillars['Sleep'] >= 75 else ("Moderate" if pillars['Sleep'] >= 50 else "Needs Attention"), body_style)],
        [Paragraph("Hydration & Work-Life Balance", body_style), Paragraph(f"{sub_scores['Lifestyle_Score']}/20", body_style), Paragraph(f"{pillars['Lifestyle']}%", body_style), Paragraph("Optimal" if pillars['Lifestyle'] >= 70 else ("Moderate" if pillars['Lifestyle'] >= 50 else "Needs Attention"), body_style)],
        [Paragraph("Mental & Emotional Wellness", body_style), Paragraph(f"{sub_scores['Mental_Score']}/20", body_style), Paragraph(f"{pillars['Mental']}%", body_style), Paragraph("Optimal" if pillars['Mental'] >= 70 else ("Moderate" if pillars['Mental'] >= 50 else "Needs Attention"), body_style)],
        [Paragraph("Dietary Rhythm & Habit Moderation", body_style), Paragraph(f"{sub_scores['Habit_Score']}/15", body_style), Paragraph(f"{pillars['Habits']}%", body_style), Paragraph("Optimal" if pillars['Habits'] >= 70 else ("Moderate" if pillars['Habits'] >= 50 else "Needs Attention"), body_style)],
    ]
    pillar_table = Table(pillar_rows, colWidths=[3.2 * inch, 1.2 * inch, 1.2 * inch, 1.4 * inch])
    pillar_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0D9488")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("PADDING", (0, 0), (-1, -1), 4),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#F8FAFC"), colors.HexColor("#FFFFFF")]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    elements.append(pillar_table)
    elements.append(Spacer(1, 10))

    # 4. Strengths vs Areas Needing Improvement
    elements.append(Paragraph("2. Key Strengths & Focus Areas", section_heading))
    strengths_text = "<br/>".join([f"• <b>{s}</b>: Maintained within healthy wellness benchmarks." for s in scoring["Strengths"]])
    improvements_text = "<br/>".join([f"• <b>{i}</b>: Opportunity for targeted lifestyle optimization." for i in scoring["Improvements"]])

    strength_imp_data = [
        [
            Paragraph("<b>Identified Key Strengths:</b><br/>" + strengths_text, body_style),
            Paragraph("<b>Priority Focus Areas:</b><br/>" + improvements_text, body_style),
        ]
    ]
    strength_imp_table = Table(strength_imp_data, colWidths=[3.5 * inch, 3.5 * inch])
    strength_imp_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, 0), colors.HexColor("#ECFDF5")),
        ("BACKGROUND", (1, 0), (1, 0), colors.HexColor("#FFF1F2")),
        ("BOX", (0, 0), (0, 0), 1, colors.HexColor("#6EE7B7")),
        ("BOX", (1, 0), (1, 0), 1, colors.HexColor("#FDA4AF")),
        ("PADDING", (0, 0), (-1, -1), 6),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    elements.append(strength_imp_table)
    elements.append(Spacer(1, 10))

    # 5. Actionable Recommendations
    elements.append(Paragraph("3. Personalized Wellness Action Plan", section_heading))
    rec_items = []
    for act in recommendations["priority_actions"][:3]:
        rec_items.append(Paragraph(f"• <b>[{act['category']}] {act['title']}</b>: {act['action']}", body_style))
    for adv in recommendations["general_advice"][:2]:
        rec_items.append(Paragraph(f"• <b>[{adv['category']}] {adv['title']}</b>: {adv['action']}", body_style))

    rec_table_data = [[item] for item in rec_items]
    rec_table = Table(rec_table_data, colWidths=[7.0 * inch])
    rec_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
        ("PADDING", (0, 0), (-1, -1), 4),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
    ]))
    elements.append(rec_table)
    elements.append(Spacer(1, 12))

    # 6. Medical Disclaimer Box
    disclaimer_box_data = [
        [
            Paragraph(
                "<b>IMPORTANT EDUCATIONAL & GENERAL WELLNESS DISCLAIMER:</b><br/>"
                "WellCare AI is an academic educational wellness-support tool designed for lifestyle awareness. "
                "The wellness score, risk categories, and personalized suggestions are general algorithmic indicators "
                "and do NOT constitute medical diagnosis, clinical evaluation, treatment plans, or prescription. "
                "Always consult a qualified healthcare professional or physician for any medical concerns.",
                disclaimer_style,
            )
        ]
    ]
    disclaimer_table = Table(disclaimer_box_data, colWidths=[7.0 * inch])
    disclaimer_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FEF3C7")),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#F59E0B")),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    elements.append(disclaimer_table)

    # Build PDF
    doc.build(elements)
    return pdf_path
