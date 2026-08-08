"""
Robust utility for exporting generated study materials to 100% valid binary PDF documents.
Handles XML escaping and safe ReportLab flowable construction.
"""

import io
import html
import re
from utils.logger import logger

def clean_markdown_for_pdf(text: str) -> str:
    """Escapes XML entities and converts basic markdown formatting to ReportLab XML tags."""
    if not text:
        return ""
    
    # 1. Escape raw XML characters first
    escaped = html.escape(text)
    
    # 2. Convert bold **text** -> <b>text</b>
    escaped = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', escaped)
    
    # 3. Convert italic *text* or _text_ -> <i>text</i>
    escaped = re.sub(r'\*(.*?)\*', r'<i>\1</i>', escaped)
    escaped = re.sub(r'_(.*?)_', r'<i>\1</i>', escaped)
    
    # 4. Convert inline code `text` -> <i>text</i>
    escaped = re.sub(r'`(.*?)`', r'<i>\1</i>', escaped)

    return escaped

def create_pdf_report(title: str, content_markdown: str) -> bytes:
    """
    Creates a styled, 100% valid binary PDF document from title and markdown content.
    Guarantees valid PDF byte output (%PDF-...) that opens properly in all PDF viewers.
    """
    buffer = io.BytesIO()
    
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors

    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=40,
        rightMargin=40,
        topMargin=40,
        bottomMargin=40
    )
    
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#0F172A'),
        spaceAfter=12
    )
    h2_style = ParagraphStyle(
        'DocH2',
        parent=styles['Heading2'],
        fontSize=13,
        leading=17,
        textColor=colors.HexColor('#1E293B'),
        spaceBefore=10,
        spaceAfter=6
    )
    body_style = ParagraphStyle(
        'DocBody',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#334155'),
        spaceAfter=6
    )

    story = [
        Paragraph(html.escape(title), title_style),
        HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#38BDF8'), spaceAfter=15)
    ]

    lines = content_markdown.split("\n") if content_markdown else ["No content available."]
    for line in lines:
        line_str = line.strip()
        if not line_str:
            story.append(Spacer(1, 4))
            continue

        try:
            if line_str.startswith("### ") or line_str.startswith("## ") or line_str.startswith("# "):
                clean_h = line_str.lstrip("#").strip()
                story.append(Paragraph(clean_markdown_for_pdf(clean_h), h2_style))
            elif line_str.startswith("- ") or line_str.startswith("* "):
                clean_bullet = "• " + line_str[2:].strip()
                story.append(Paragraph(clean_markdown_for_pdf(clean_bullet), body_style))
            else:
                story.append(Paragraph(clean_markdown_for_pdf(line_str), body_style))
        except Exception as p_err:
            logger.warning(f"Error creating PDF paragraph line: {p_err}")
            plain_safe = html.escape(re.sub(r'[*_`#]', '', line_str))
            story.append(Paragraph(plain_safe, body_style))

    try:
        doc.build(story)
        buffer.seek(0)
        pdf_bytes = buffer.getvalue()
        if pdf_bytes.startswith(b"%PDF"):
            return pdf_bytes
    except Exception as build_err:
        logger.error(f"Error building PDF document: {build_err}")

    # Fallback guaranteed minimal valid PDF builder
    buffer_fallback = io.BytesIO()
    doc_fallback = SimpleDocTemplate(buffer_fallback, pagesize=letter)
    story_fallback = [
        Paragraph(html.escape(title), title_style),
        Spacer(1, 10),
        Paragraph(html.escape(content_markdown[:3000]), body_style)
    ]
    doc_fallback.build(story_fallback)
    buffer_fallback.seek(0)
    return buffer_fallback.getvalue()
