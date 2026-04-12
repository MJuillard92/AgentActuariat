"""
TOOL CONTRACT — build_pdf.assemble_sections
════════════════════════════════════════════════════════════════

IDENTITY
--------
name          : build_pdf.assemble_sections
domain        : mortality_experience
version       : 1.0.0
author        : Marc Juillard
last_updated  : 2026-04-07

DESCRIPTION
-----------
Assemble un rapport PDF de certification depuis le dict section_outputs
produit par le WriterAgent. Lit les sections dans l'ordre du
processing_sequence, intègre texte narratif, tableaux ReportLab et
images PNG, et génère un PDF final.

WHEN TO USE
-----------
Appeler en dernier dans le pipeline WriterAgent, après que toutes
les sections ont été rédigées et rendues.

PREREQUISITES
-------------
required_data_store_keys:
  - section_outputs  (dict — {section_id: {text, tables, graphs, status}})
  - template_context (dict — placeholders résolus, pour le titre etc.)

INPUTS
------
params:
  output_path:
    type    : string
    default : /tmp/rapport_writer.pdf
  title:
    type    : string
    default : "Rapport de Certification — Table de Mortalité"
  portfolio_info:
    type    : string
    default : ""

OUTPUTS
-------
data_store_keys_written: []
return_payload:
  succes      : bool
  output_path : str
  nb_sections : int
  warning     : str

CATALOGUE METADATA
------------------
display_name      : Assemblage PDF depuis sections
short_description : Génère le PDF final depuis section_outputs du WriterAgent.
domain            : mortality_experience
capability_group  : reporting
depends_on        : [build_pdf.load_yaml_template, build_pdf.table_renderer, graphs.graph_from_spec]
required_by       : []
client_visible    : false
"""
from __future__ import annotations

import logging
import os
from datetime import datetime
from io import BytesIO
from pathlib import Path

log = logging.getLogger(__name__)

_SECTION_ORDER = [
    "preamble", "data_submission", "construction", "analysis", "conclusion", "annex"
]

_SECTION_LABELS = {
    "preamble":        "Préambule",
    "data_submission": "Données soumises et prétraitement",
    "construction":    "Méthodologie de construction",
    "analysis":        "Analyse et validation",
    "conclusion":      "Conclusion",
    "annex":           "Annexe — Table de mortalité",
}


def run(data: dict | None = None, params: dict | None = None) -> dict:
    data   = data   or {}
    params = params or {}

    output_path    = params.get("output_path", "/tmp/rapport_writer.pdf")
    title          = params.get("title", "Rapport de Certification — Table de Mortalité d'Expérience")
    portfolio_info = params.get("portfolio_info", "")

    section_outputs  = data.get("section_outputs") or {}
    template_context = data.get("template_context") or {}

    # Fallback title from context
    if not title or title == "Rapport de Certification — Table de Mortalité d'Expérience":
        obj = template_context.get("study_objective", "")
        if obj:
            title = f"Table de mortalité d'expérience — {obj}"

    # Patch MD5 (Python/OpenSSL compat)
    import hashlib as _hashlib
    _orig_md5 = _hashlib.md5
    def _md5_compat(*a, **kw):
        kw.pop("usedforsecurity", None)
        return _orig_md5(*a, **kw)
    _hashlib.md5 = _md5_compat

    try:
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
            PageBreak, Image as RLImage, HRFlowable,
        )
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
    except ImportError as exc:
        _hashlib.md5 = _orig_md5
        return {"succes": False, "output_path": "", "nb_sections": 0,
                "warning": f"ReportLab non disponible : {exc}"}

    # ── Styles ────────────────────────────────────────────────────────────────
    styles = getSampleStyleSheet()
    BLUE   = colors.HexColor("#1A3668")
    LBLUE  = colors.HexColor("#2C5F8A")
    LIGHT  = colors.HexColor("#EAF0F7")
    GREY   = colors.HexColor("#6B6B6B")

    title_s = ParagraphStyle("WT",  parent=styles["Title"],   fontSize=16, textColor=BLUE,
                              alignment=TA_CENTER, spaceAfter=6)
    sub_s   = ParagraphStyle("WSu", parent=styles["Normal"],  fontSize=10, textColor=GREY,
                              alignment=TA_CENTER, spaceAfter=12)
    h1_s    = ParagraphStyle("WH1", parent=styles["Heading1"], fontSize=12, textColor=BLUE,
                              spaceBefore=14, spaceAfter=6)
    h2_s    = ParagraphStyle("WH2", parent=styles["Heading2"], fontSize=10, textColor=LBLUE,
                              spaceBefore=8,  spaceAfter=4)
    body_s  = ParagraphStyle("WB",  parent=styles["Normal"],  fontSize=9, leading=13,
                              spaceAfter=4, alignment=TA_JUSTIFY)
    small_s = ParagraphStyle("WS",  parent=styles["Normal"],  fontSize=7.5, textColor=GREY, leading=10)

    def _embed_image(path: str, width_cm: float = 16.0, height_cm: float = 7.0):
        if not path or not os.path.exists(path):
            return None
        try:
            img = RLImage(path, width=width_cm * cm, height=height_cm * cm)
            img.hAlign = "CENTER"
            return img
        except Exception as exc:
            log.warning("[assemble_sections] Impossible d'intégrer %s : %s", path, exc)
            return None

    # Running header/footer
    date_str = datetime.now().strftime("%d/%m/%Y")

    def _on_page(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(GREY)
        # Header
        canvas.drawString(2 * cm, A4[1] - 1.2 * cm, title[:80])
        canvas.drawRightString(A4[0] - 2 * cm, A4[1] - 1.2 * cm, f"Page {doc.page}")
        # Footer
        canvas.drawString(2 * cm, 1.0 * cm, f"Généré le {date_str} — Confidentiel")
        canvas.restoreState()

    # ── Build story ───────────────────────────────────────────────────────────
    story = []

    # Title page
    story.append(Spacer(1, 2 * cm))
    story.append(Paragraph(title, title_s))
    story.append(Spacer(1, 0.3 * cm))
    if portfolio_info:
        story.append(Paragraph(portfolio_info, sub_s))
    story.append(Paragraph(f"Date de génération : {date_str}", sub_s))
    story.append(HRFlowable(width="100%", thickness=1, color=BLUE, spaceAfter=20))
    story.append(PageBreak())

    # Table of contents placeholder
    story.append(Paragraph("Table des matières", h1_s))
    ordered_ids = [sid for sid in _SECTION_ORDER if sid in section_outputs]
    for sid in ordered_ids:
        label = _SECTION_LABELS.get(sid, sid)
        story.append(Paragraph(f"• {label}", body_s))
    story.append(PageBreak())

    # ── Sections ──────────────────────────────────────────────────────────────
    nb_sections = 0
    for sec_id in ordered_ids:
        sec = section_outputs.get(sec_id)
        if not sec:
            continue

        status = sec.get("status", "unknown")
        if status == "skipped":
            continue

        label = _SECTION_LABELS.get(sec_id, sec_id)
        story.append(Paragraph(label, h1_s))
        story.append(HRFlowable(width="100%", thickness=0.5, color=LBLUE, spaceAfter=8))

        # Narrative text
        text = sec.get("text") or ""
        if text:
            for para in text.split("\n\n"):
                para = para.strip()
                if para:
                    # Escape ampersands and angle brackets for ReportLab
                    para_safe = para.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                    story.append(Paragraph(para_safe, body_s))
            story.append(Spacer(1, 0.3 * cm))

        # Subsection texts
        subsection_texts = sec.get("subsection_texts") or {}
        for sub_label, sub_text in subsection_texts.items():
            if sub_text:
                story.append(Paragraph(sub_label, h2_s))
                for para in sub_text.split("\n\n"):
                    para = para.strip()
                    if para:
                        para_safe = para.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                        story.append(Paragraph(para_safe, body_s))
                story.append(Spacer(1, 0.2 * cm))

        # Tables (list of reportlab Table objects OR raw 2D list)
        tables = sec.get("tables") or []
        table_captions = sec.get("table_captions") or []
        for i, tbl in enumerate(tables):
            caption = table_captions[i] if i < len(table_captions) else f"Tableau {i+1}"
            story.append(Paragraph(caption, small_s))
            if tbl is None:
                story.append(Paragraph("(Tableau non disponible — données manquantes)", small_s))
                continue
            # If raw list-of-lists, build a Table
            if isinstance(tbl, list):
                if tbl:
                    n_cols = max(len(r) for r in tbl)
                    col_w  = [16 * cm / n_cols] * n_cols
                    rl_tbl = Table(tbl, colWidths=col_w, repeatRows=1)
                    rl_tbl.setStyle(TableStyle([
                        ("BACKGROUND",  (0, 0), (-1, 0), BLUE),
                        ("TEXTCOLOR",   (0, 0), (-1, 0), colors.white),
                        ("FONTSIZE",    (0, 0), (-1, -1), 8),
                        ("GRID",        (0, 0), (-1, -1), 0.3, GREY),
                        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [LIGHT, colors.white]),
                        ("TOPPADDING",  (0, 0), (-1, -1), 3),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                    ]))
                    story.append(rl_tbl)
            else:
                # Assume it's already a ReportLab Table object
                story.append(tbl)
            story.append(Spacer(1, 0.3 * cm))

        # Graphs (list of PNG paths)
        graphs = sec.get("graphs") or []
        graph_captions = sec.get("graph_captions") or []
        for i, graph_path in enumerate(graphs):
            caption = graph_captions[i] if i < len(graph_captions) else f"Graphique {i+1}"
            img = _embed_image(graph_path)
            if img:
                story.append(img)
                story.append(Paragraph(caption, small_s))
                story.append(Spacer(1, 0.3 * cm))
            else:
                story.append(Paragraph(f"({caption} — non disponible)", small_s))

        story.append(PageBreak())
        nb_sections += 1

    if not story:
        return {"succes": False, "output_path": "", "nb_sections": 0,
                "warning": "section_outputs vide — aucune section à assembler."}

    # ── Build PDF ─────────────────────────────────────────────────────────────
    try:
        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            leftMargin=2 * cm, rightMargin=2 * cm,
            topMargin=2.5 * cm, bottomMargin=2 * cm,
        )
        doc.build(story, onFirstPage=_on_page, onLaterPages=_on_page)
        _hashlib.md5 = _orig_md5
        return {
            "succes":      True,
            "output_path": output_path,
            "nb_sections": nb_sections,
            "warning":     "",
        }
    except Exception as exc:
        _hashlib.md5 = _orig_md5
        log.error("[assemble_sections] Erreur PDF : %s", exc)
        return {"succes": False, "output_path": "", "nb_sections": nb_sections,
                "warning": f"Erreur ReportLab : {exc}"}
