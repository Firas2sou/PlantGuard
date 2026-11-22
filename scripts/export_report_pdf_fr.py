#!/usr/bin/env python
"""Export French progress report 2 to a paginated PDF using ReportLab."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn
from docx.table import Table as DocxTable
from docx.text.paragraph import Paragraph
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (BaseDocTemplate, Frame, Image, PageTemplate,
                                Paragraph as PdfParagraph, Preformatted, Spacer,
                                Table, TableStyle)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reports" / "Rapport_Avancement_2_FR.docx"
OUTPUT = ROOT / "reports" / "Rapport_Avancement_2_FR.pdf"
BLUE = colors.HexColor("#1F4E79")
GREEN = colors.HexColor("#527A55")
INK = colors.HexColor("#24313B")
MUTED = colors.HexColor("#65717C")
PALE = colors.HexColor("#EFF4F7")


def styles():
    base = getSampleStyleSheet()
    return {
        "Title": ParagraphStyle("ReportTitle", parent=base["Title"], fontName="Helvetica-Bold",
                                fontSize=22, leading=27, textColor=BLUE, alignment=TA_LEFT,
                                spaceAfter=9),
        "Subtitle": ParagraphStyle("Subtitle", parent=base["Normal"], fontName="Helvetica",
                                    fontSize=13, leading=17, textColor=INK, spaceAfter=8),
        "Heading 1": ParagraphStyle("Section", parent=base["Heading1"], fontName="Helvetica-Bold",
                                     fontSize=14, leading=17, textColor=BLUE, spaceBefore=12,
                                     spaceAfter=6, keepWithNext=True),
        "Heading 2": ParagraphStyle("Subsection", parent=base["Heading2"], fontName="Helvetica-Bold",
                                     fontSize=11, leading=14, textColor=GREEN, spaceBefore=8,
                                     spaceAfter=4, keepWithNext=True),
        "Normal": ParagraphStyle("Body", parent=base["BodyText"], fontName="Helvetica",
                                  fontSize=9.2, leading=13, textColor=INK, spaceAfter=6),
        "List Bullet": ParagraphStyle("Bullet", parent=base["BodyText"], fontName="Helvetica",
                                       fontSize=9.2, leading=13, textColor=INK, leftIndent=13,
                                       firstLineIndent=-8, bulletIndent=1, spaceAfter=4),
        "Caption": ParagraphStyle("Caption", parent=base["Normal"], fontName="Helvetica-Oblique",
                                  fontSize=8.5, leading=11, textColor=MUTED, alignment=TA_CENTER,
                                  spaceAfter=6),
    }


def xml_text(value: str) -> str:
    return (value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace("\n", "<br/>"))


def on_page(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#D9E0E5"))
    canvas.setLineWidth(0.5)
    canvas.line(20 * mm, 15 * mm, A4[0] - 20 * mm, 15 * mm)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(20 * mm, 10 * mm, "PlantGuard - Rapport d'avancement 2")
    canvas.drawRightString(A4[0] - 20 * mm, 10 * mm, f"Page {doc.page}")
    canvas.restoreState()


def render_table(word_table: DocxTable, available_width: float, style_map):
    values = [[cell.text.strip() for cell in row.cells] for row in word_table.rows]
    if not values:
        return Spacer(1, 4)
    ncols = len(values[0])
    widest = [max(1, max(len(row[i]) for row in values)) for i in range(ncols)]
    weights = [min(34, max(9, n ** 0.55)) for n in widest]
    total = sum(weights)
    widths = [available_width * w / total for w in weights]
    cells = []
    for row_idx, row in enumerate(values):
        cell_style = style_map["TableHead"] if row_idx == 0 else style_map["TableCell"]
        cells.append([PdfParagraph(xml_text(value), cell_style) for value in row])
    t = Table(cells, colWidths=widths, repeatRows=1, hAlign="LEFT", splitByRow=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), BLUE),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("BACKGROUND", (0, 1), (-1, -1), colors.white),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE]),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D5DEE5")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return t


def main():
    word = Document(SOURCE)
    page_w, _ = A4
    margin = 20 * mm
    avail_w = page_w - 2 * margin
    st = styles()
    st["TableCell"] = ParagraphStyle("TableCell", fontName="Helvetica", fontSize=7.4,
                                     leading=9.6, textColor=INK, alignment=TA_LEFT,
                                     splitLongWords=True)
    st["TableHead"] = ParagraphStyle("TableHead", parent=st["TableCell"],
                                     fontName="Helvetica-Bold", textColor=colors.white)
    report = BaseDocTemplate(str(OUTPUT), pagesize=A4, leftMargin=margin, rightMargin=margin,
                             topMargin=18 * mm, bottomMargin=22 * mm,
                             title=word.core_properties.title or "Rapport d'avancement 2")
    frame = Frame(margin, 22 * mm, avail_w, A4[1] - 40 * mm, id="body")
    report.addPageTemplates(PageTemplate(id="main", frames=[frame], onPage=on_page))

    flow = []
    for block in word.element.body.iterchildren():
        if block.tag == qn("w:p"):
            para = Paragraph(block, word)
            text = para.text.strip()
            style_name = para.style.name if para.style else "Normal"
            # Place embedded figures at their position in the source document.
            for blip in block.iter(qn("a:blip")):
                rid = blip.get(qn("r:embed"))
                part = word.part.related_parts.get(rid)
                if part:
                    pic = Image(BytesIO(part.blob), width=avail_w * 0.83, height=65 * mm,
                                kind="proportional", hAlign="CENTER")
                    flow.append(Spacer(1, 4))
                    flow.append(pic)
            if not text:
                continue
            if style_name == "Title":
                flow.append(PdfParagraph(xml_text(text), st["Title"]))
            elif style_name == "Subtitle":
                flow.append(PdfParagraph(xml_text(text), st["Subtitle"]))
            elif style_name.startswith("Heading 1"):
                flow.append(PdfParagraph(xml_text(text), st["Heading 1"]))
            elif style_name.startswith("Heading 2"):
                flow.append(PdfParagraph(xml_text(text), st["Heading 2"]))
            elif style_name == "Caption":
                flow.append(PdfParagraph(xml_text(text), st["Caption"]))
            elif style_name == "List Bullet":
                flow.append(PdfParagraph("• " + xml_text(text), st["List Bullet"]))
            elif para.runs and para.runs[0].font.name == "Consolas":
                flow.append(Preformatted(text, ParagraphStyle("Code", fontName="Courier",
                                                              fontSize=7.2, leading=9,
                                                              backColor=PALE, borderPadding=5,
                                                              leftIndent=5, rightIndent=5,
                                                              spaceAfter=7)))
            else:
                flow.append(PdfParagraph(xml_text(text), st["Normal"]))
        elif block.tag == qn("w:tbl"):
            flow.append(render_table(DocxTable(block, word), avail_w, st))
            flow.append(Spacer(1, 5))

    report.build(flow)
    print("saved", OUTPUT)


if __name__ == "__main__":
    main()
