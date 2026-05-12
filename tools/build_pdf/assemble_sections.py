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
    "preamble",
    # Sections Design 3 (mortality_template.yaml actuel)
    "data_preprocessing",
    "data_analysis_unisex",
    "data_analysis_by_sex",
    "table_construction",    # taux bruts q_x = D_x / E_x (raw_rates, full_report)
    # Sections à venir (full_report)
    "smoothing", "validation", "benchmarking",
    # Sections legacy (rétro-compat) — restent en queue
    "data_submission", "construction", "analysis", "conclusion", "annex",
]

_SECTION_LABELS = {
    "preamble":              "Préambule",
    "data_preprocessing":    "Données et prétraitement",
    "data_analysis_unisex":  "Analyse descriptive du portefeuille",
    "data_analysis_by_sex":  "Analyse descriptive H/F",
    "table_construction":    "Construction de la table — taux bruts par âge",
    "smoothing":             "Lissage des taux bruts",
    "validation":            "Validation — observés vs prédits",
    "benchmarking":          "Benchmarking — tables réglementaires",
    # Legacy
    "data_submission":       "Données soumises et prétraitement",
    "construction":          "Méthodologie de construction",
    "analysis":              "Analyse et validation",
    "conclusion":            "Conclusion",
    "annex":                 "Annexe — Table de mortalité",
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

    # ── Styles — depuis report_styles.py (source unique) ─────────────────────
    from tools.build_pdf.report_styles import get_styles, COLORS, make_table
    S   = get_styles()
    C   = COLORS
    BLUE  = C["BLUE"]
    LBLUE = C["LBLUE"]
    GREY  = C["GREY"]
    LIGHT = C["LIGHT"]

    title_s = S.title
    sub_s   = S.subtitle
    h1_s    = S.h1
    h2_s    = S.h2
    body_s  = S.body
    small_s = S.small
    caption_s = S.caption

    # ── Rendu de texte avec formules LaTeX + markup structuré ────────────────
    from tools.build_pdf.math_renderer import split_math, has_math, render_formula
    import re as _re

    def _md_inline(text: str) -> str:
        """
        Convertit le markup inline en XML ReportLab :
          **gras**  → <b>gras</b>
          __gras__  → <b>gras</b>  (alternative)
        N'applique PAS _italic_ pour éviter les conflits avec les indices q_x, D_x.
        """
        text = _re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
        text = _re.sub(r'__(.+?)__',     r'<b>\1</b>', text)
        return text

    def _para_with_math(raw: str, style) -> list:
        """
        Transforme un fragment de texte en flowables ReportLab.
        Pipeline : échapper & → appliquer _md_inline → splitter les formules
          - $...$ inline  → PNG inline via <img> tag
          - $$...$$ block → RLImage centrée
        """
        raw = raw.strip()
        if not raw:
            return []

        if not has_math(raw):
            # Chemin rapide : escape & seulement, puis markup inline
            safe = raw.replace("&", "&amp;")
            safe = _md_inline(safe)
            try:
                return [Paragraph(safe, style)]
            except Exception:
                return [Paragraph(raw.replace("&", "&amp;"), style)]

        segments = split_math(raw)
        flowables = []
        inline_parts: list[str] = []

        def _flush_inline():
            if inline_parts:
                xml = "".join(inline_parts)
                try:
                    flowables.append(Paragraph(xml, style))
                except Exception:
                    # Fallback : retirer les tags <img> et réessayer
                    stripped = _re.sub(r'<img[^/]*/>', '', xml)
                    flowables.append(Paragraph(stripped, style))
                inline_parts.clear()

        for content, is_formula, is_display in segments:
            if not is_formula:
                safe = content.replace("&", "&amp;")
                safe = _md_inline(safe)
                inline_parts.append(safe)

            elif is_display:
                _flush_inline()
                png = render_formula(content, display=True, fontsize=11)
                if png:
                    img = RLImage(png)
                    img.hAlign = "CENTER"
                    flowables.append(Spacer(1, 0.2 * cm))
                    flowables.append(img)
                    flowables.append(Spacer(1, 0.2 * cm))
                else:
                    flowables.append(Paragraph(f"[{content}]", style))

            else:
                png = render_formula(content, display=False, fontsize=9.5)
                if png:
                    try:
                        import PIL.Image as _PIL
                        with _PIL.open(png) as im:
                            w_px, h_px = im.size
                        dpi   = 200
                        w_pt  = w_px / dpi * 72
                        h_pt  = h_px / dpi * 72
                        if h_pt > 14:
                            scale = 14 / h_pt
                            w_pt *= scale
                            h_pt  = 14
                        inline_parts.append(
                            f'<img src="{png}" width="{w_pt:.1f}" height="{h_pt:.1f}" valign="middle"/>'
                        )
                    except Exception:
                        inline_parts.append(f"[{content}]")
                else:
                    inline_parts.append(f"[{content}]")

        _flush_inline()
        return flowables

    # ── Parser de texte structuré (markup LLM → flowables) ────────────────────
    #
    # Convention de markup reconnue :
    #   ## Titre           → h2  (sous-section dans une section)
    #   ### Titre          → h3  (paragraphe titré)
    #   - item / * item    → liste à puces (• indenté)
    #   > note             → texte de note (petit, gris, indenté)
    #   **gras**           → gras inline
    #   $formule$          → formule inline LaTeX
    #   $$formule$$        → formule en bloc LaTeX
    #   Texte normal       → paragraphe body justifié
    #   Ligne vide         → séparation de paragraphes

    def _parse_structured_text(text: str) -> list:
        """
        Parse le texte structuré produit par le LLM et retourne une liste de flowables.
        Gère : titres ##/###, listes, notes >, gras inline, formules LaTeX.
        """
        if not text:
            return []

        flowables = []
        lines     = text.splitlines()
        i         = 0
        pending_bullets: list[str] = []

        def _flush_bullets():
            if not pending_bullets:
                return
            # Style liste : body avec retrait + puce
            bullet_style = ParagraphStyle(
                "RT_bullet", parent=S.body,
                leftIndent=14, firstLineIndent=-10,
                spaceBefore=1, spaceAfter=1,
            )
            for b in pending_bullets:
                flowables.extend(_para_with_math("• " + b, bullet_style))
            flowables.append(Spacer(1, 0.15 * cm))
            pending_bullets.clear()

        while i < len(lines):
            line    = lines[i]
            stripped = line.strip()

            # Ligne vide → flush bullets + petit espace
            if not stripped:
                _flush_bullets()
                i += 1
                continue

            # Titres ##
            if stripped.startswith("### "):
                _flush_bullets()
                flowables.append(Spacer(1, 0.1 * cm))
                flowables.append(Paragraph(stripped[4:].strip(), S.h3))
                i += 1
                continue

            if stripped.startswith("## "):
                _flush_bullets()
                flowables.append(Spacer(1, 0.2 * cm))
                flowables.append(Paragraph(stripped[3:].strip(), S.h2))
                flowables.append(HRFlowable(width="80%", thickness=0.4,
                                            color=C["HRULE"], spaceAfter=4))
                i += 1
                continue

            # Liste à puces
            if stripped.startswith(("- ", "* ", "• ")):
                pending_bullets.append(stripped[2:].strip())
                i += 1
                continue

            # Note / avertissement
            if stripped.startswith("> "):
                _flush_bullets()
                note_style = ParagraphStyle(
                    "RT_note", parent=S.small,
                    leftIndent=12,
                    borderPadding=(3, 6, 3, 6),
                    backColor=C["LIGHT2"],
                )
                flowables.extend(_para_with_math(stripped[2:].strip(), note_style))
                i += 1
                continue

            # Paragraphe normal — agrège les lignes consécutives non-spéciales
            _flush_bullets()
            para_lines = [stripped]
            while i + 1 < len(lines):
                nxt = lines[i + 1].strip()
                if (not nxt or
                        nxt.startswith(("## ", "### ", "- ", "* ", "• ", "> "))):
                    break
                i += 1
                para_lines.append(nxt)

            para = " ".join(para_lines)
            flowables.extend(_para_with_math(para, S.body))
            flowables.append(Spacer(1, 0.05 * cm))
            i += 1

        _flush_bullets()
        return flowables

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

        # Narrative text — parser structuré (## h2, ### h3, - listes, > notes, **gras**, $math$)
        text = sec.get("text") or ""
        if text:
            story.extend(_parse_structured_text(text))
            story.append(Spacer(1, 0.25 * cm))

        # Subsection texts (legacy — si le rapport vient d'un ancien pipeline)
        subsection_texts = sec.get("subsection_texts") or {}
        for sub_label, sub_text in subsection_texts.items():
            if sub_text:
                story.append(Paragraph(sub_label, S.h2))
                story.extend(_parse_structured_text(sub_text))
                story.append(Spacer(1, 0.2 * cm))

        # Tables (list of reportlab Table objects OR raw 2D list)
        tables = sec.get("tables") or []
        table_captions = sec.get("table_captions") or []
        for i, tbl in enumerate(tables):
            caption = table_captions[i] if i < len(table_captions) else f"Tableau {i+1}"
            story.append(Paragraph(caption, caption_s))
            if tbl is None:
                story.append(Paragraph("(Tableau non disponible — données manquantes)", small_s))
                continue
            # If raw list-of-lists, build a Table
            if isinstance(tbl, list):
                if tbl:
                    n_cols = max(len(r) for r in tbl)
                    col_w  = [16 * cm / n_cols] * n_cols
                    rl_tbl = Table(tbl, colWidths=col_w, repeatRows=1)
                    # Style enrichi : en-tête gras + padding généreux,
                    # corps centré verticalement, lignes alternées discrètes,
                    # bordures verticales fines plutôt qu'une grille complète.
                    rl_tbl.setStyle(TableStyle([
                        # En-tête (ligne 0)
                        ("BACKGROUND",     (0, 0), (-1, 0), BLUE),
                        ("TEXTCOLOR",      (0, 0), (-1, 0), colors.white),
                        ("FONTNAME",       (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("FONTSIZE",       (0, 0), (-1, 0), 9),
                        ("ALIGN",          (0, 0), (-1, 0), "CENTER"),
                        ("VALIGN",         (0, 0), (-1, 0), "MIDDLE"),
                        ("TOPPADDING",     (0, 0), (-1, 0), 6),
                        ("BOTTOMPADDING",  (0, 0), (-1, 0), 6),
                        # Corps (lignes 1+)
                        ("FONTSIZE",       (0, 1), (-1, -1), 8.5),
                        ("VALIGN",         (0, 1), (-1, -1), "MIDDLE"),
                        ("TOPPADDING",     (0, 1), (-1, -1), 4),
                        ("BOTTOMPADDING",  (0, 1), (-1, -1), 4),
                        ("LEFTPADDING",    (0, 0), (-1, -1), 6),
                        ("RIGHTPADDING",   (0, 0), (-1, -1), 6),
                        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
                        # Bordures discrètes : haut/bas en gras, intérieur fin
                        ("LINEABOVE",      (0, 0), (-1, 0),  1.0, BLUE),
                        ("LINEBELOW",      (0, 0), (-1, 0),  0.8, BLUE),
                        ("LINEBELOW",      (0, -1), (-1, -1), 0.6, BLUE),
                        ("INNERGRID",      (0, 0), (-1, -1), 0.2, GREY),
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
                story.append(Paragraph(caption, caption_s))
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
