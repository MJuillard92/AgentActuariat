"""
report_generator.py
Génération des outputs finaux : PDF (reportlab), trace de raisonnement (Markdown), notebook Jupyter (nbformat).
"""
from __future__ import annotations

import base64
import hashlib
import io
import re
from datetime import datetime
from pathlib import Path
from typing import Any

# ── Python 3.8 compatibility : reportlab 4.4+ uses md5(usedforsecurity=False)
# which is only supported from Python 3.9+.  Patch before any reportlab import.
_orig_md5 = hashlib.md5
def _md5_compat(*args, **kwargs):
    kwargs.pop("usedforsecurity", None)
    return _orig_md5(*args, **kwargs)
hashlib.md5 = _md5_compat  # type: ignore[assignment]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers communs
# ─────────────────────────────────────────────────────────────────────────────

def _escape_xml(text: str) -> str:
    """Échappe les caractères XML spéciaux pour reportlab."""
    return (
        text
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _md_inline(text: str) -> str:
    """Convertit **bold** et *italic* en balises reportlab (après escape XML)."""
    text = _escape_xml(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"\*(.+?)\*", r"<i>\1</i>", text)
    return text


def _md_to_story(text: str, body_style: Any, subsection_style: Any, cm: float) -> list:
    """Convertit du texte markdown simple en liste de flowables reportlab.

    Gère : titres (##/###), bullets (- /*), paragraphes, lignes vides.
    """
    from reportlab.platypus import Paragraph, Spacer

    elements: list = []
    for line in text.split("\n"):
        line = line.rstrip()
        if line.startswith("### ") or line.startswith("## "):
            content = re.sub(r"^#{2,3}\s+\d*\.?\s*", "", line).strip()
            if content:
                elements.append(Paragraph(_md_inline(content), subsection_style))
        elif line.startswith("- ") or line.startswith("* "):
            elements.append(Paragraph(f"• {_md_inline(line[2:])}", body_style))
        elif line.strip():
            elements.append(Paragraph(_md_inline(line.strip()), body_style))
        else:
            elements.append(Spacer(1, 0.15 * cm))
    return elements


def _clean_user_message(msg: str) -> str:
    """Extrait uniquement la demande utilisateur, sans les paramètres kernel."""
    for marker in ["Paramètres déjà définis", "FILE_PATH =", "\n- FILE_PATH"]:
        idx = msg.find(marker)
        if idx > 10:
            return msg[:idx].strip()
    return msg[:400].strip()


# Mots-clés identifiant des steps de debug/troubleshooting sans valeur documentaire
_DEBUG_KEYWORDS = [
    "vérification des résultats",
    "vérification des clés",
    "vérification des colonnes",
    "affichage des résultats",
    "affichage du tableau",
    "affichage du dataframe",
    "création d'un dataframe vide",
    "pour vérifier son contenu",
    "pour identifier les",
    "aperçu du fichier",
]


def _is_meaningful_step(step: dict) -> bool:
    """Retourne True si le step apporte de la valeur documentaire."""
    desc = (step.get("description") or step.get("content", "")).lower()
    if any(kw in desc for kw in _DEBUG_KEYWORDS):
        return False
    output = step.get("output", "")
    has_output = len(output) > 50 and "empty dataframe" not in output.lower()
    has_figures = bool(step.get("figures"))
    has_tables = bool(step.get("display_outputs"))
    return has_output or has_figures or has_tables


# Patterns d'extraction des chiffres clés depuis le summary texte
_KEY_PATTERNS: list[tuple[str, str]] = [
    ("Lignes finales",      r"Lignes finales\s*[:\-]\s*([\d\s,]+)"),
    ("Lignes chargées",     r"(\d[\d\s,]+)\s*lignes"),
    ("Total décès",         r"Total des décès\s*[:\-]\s*([\d\s,]+)"),
    ("Exposition totale",   r"Exposition totale\s*[:\-]\s*([\d\s,.]+)\s*années"),
    ("Taux bruts (âges)",   r"(\d+)\s*âges avec des taux valides"),
    ("SMR global",          r"SMR global\s*[:\-]\s*([\d.,]+)"),
    ("Test χ² (stat)",      r"[Cc]hi2\s*=\s*([\d.,]+)"),
    ("p-valeur χ²",         r"p\s*=\s*([\d.,]+)"),
    ("Lambda lissage",      r"[Ll]ambda\s*[=:]\s*([\d.,]+)"),
]


def _extract_key_results(summary: str) -> list[tuple[str, str]]:
    """Extrait les indicateurs clés du summary sous forme de liste (label, valeur)."""
    results = []
    seen_labels: set[str] = set()
    for label, pattern in _KEY_PATTERNS:
        m = re.search(pattern, summary, re.IGNORECASE)
        if m and label not in seen_labels:
            val = m.group(1).strip().replace("\n", " ")
            # Limiter la longueur
            if len(val) < 50:
                results.append((label, val))
                seen_labels.add(label)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# PDF Report (reportlab)
# ─────────────────────────────────────────────────────────────────────────────

def generate_pdf_report(
    steps: list[dict],
    summary: str,
    user_message: str,
    domain_label: str,
    output_path: str,
    study_ref: str = "",
) -> str:
    """Génère un rapport PDF professionnel à partir des steps de l'agent.

    Args:
        steps:        Liste de dicts step (description, output, figures, display_outputs, ...).
        summary:      Synthèse finale de l'agent (markdown).
        user_message: Message original de l'utilisateur.
        domain_label: Libellé du domaine (ex: "mortality").
        output_path:  Chemin de sortie du fichier PDF.
        study_ref:    Référence de l'étude pour l'en-tête (ex: "Analyse 20260327_080020").

    Returns:
        Chemin absolu du fichier PDF créé.
    """
    try:
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Table, TableStyle, Image,
            Spacer, PageBreak, HRFlowable,
        )
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    except ImportError as e:
        raise ImportError(f"reportlab est requis pour générer le PDF : {e}") from e

    try:
        from PIL import Image as PILImage
        _pil_available = True
    except ImportError:
        _pil_available = False

    _w, _h = A4
    max_content_width = _w - 4 * cm
    domain_display = domain_label.replace("_", " ").title() if domain_label else "Actuariat"
    _study_ref = study_ref or f"Analyse actuarielle — {datetime.now().strftime('%Y%m%d')}"

    # ── Styles ────────────────────────────────────────────────────────────────
    styles = getSampleStyleSheet()
    BLUE = colors.HexColor("#1A3A5C")
    BLUE_LIGHT = colors.HexColor("#2D5986")
    GREY = colors.HexColor("#555555")
    GREY_LIGHT = colors.HexColor("#888888")

    title_style = ParagraphStyle(
        "TitleStyle", parent=styles["Title"],
        fontName="Helvetica-Bold", fontSize=20, leading=26,
        textColor=BLUE, spaceAfter=12, alignment=TA_CENTER,
    )
    subtitle_style = ParagraphStyle(
        "SubtitleStyle", parent=styles["Normal"],
        fontName="Helvetica", fontSize=11, textColor=GREY,
        alignment=TA_CENTER, spaceAfter=6,
    )
    section_style = ParagraphStyle(
        "SectionStyle", parent=styles["Heading1"],
        fontName="Helvetica-Bold", fontSize=13, textColor=BLUE,
        spaceBefore=14, spaceAfter=6,
    )
    subsection_style = ParagraphStyle(
        "SubsectionStyle", parent=styles["Heading2"],
        fontName="Helvetica-Bold", fontSize=11, textColor=BLUE_LIGHT,
        spaceBefore=10, spaceAfter=4,
    )
    step_title_style = ParagraphStyle(
        "StepTitleStyle", parent=styles["Heading2"],
        fontName="Helvetica-Bold", fontSize=10, textColor=BLUE_LIGHT,
        spaceBefore=8, spaceAfter=3,
    )
    body_style = ParagraphStyle(
        "BodyStyle", parent=styles["Normal"],
        fontName="Helvetica", fontSize=10, leading=14, spaceAfter=5,
    )
    small_style = ParagraphStyle(
        "SmallStyle", parent=styles["Normal"],
        fontName="Helvetica", fontSize=8, leading=11, textColor=GREY_LIGHT,
    )
    italic_style = ParagraphStyle(
        "ItalicStyle", parent=styles["Normal"],
        fontName="Helvetica-Oblique", fontSize=9, textColor=GREY, spaceAfter=4,
    )

    # ── En-tête / pied de page ────────────────────────────────────────────────
    def _on_page(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(GREY_LIGHT)
        # En-tête gauche : référence
        canvas.drawString(2 * cm, _h - 1.3 * cm, _study_ref)
        # En-tête droite : CONFIDENTIEL
        canvas.drawRightString(_w - 2 * cm, _h - 1.3 * cm, "CONFIDENTIEL")
        # Ligne de séparation en-tête
        canvas.setStrokeColor(colors.HexColor("#DDDDDD"))
        canvas.setLineWidth(0.3)
        canvas.line(2 * cm, _h - 1.5 * cm, _w - 2 * cm, _h - 1.5 * cm)
        # Pied de page gauche : domaine
        canvas.drawString(2 * cm, 0.8 * cm, f"Généré par l'agent actuariel — {domain_display}")
        # Pied de page droit : numéro de page
        canvas.drawRightString(_w - 2 * cm, 0.8 * cm, f"Page {doc.page}")
        canvas.restoreState()

    def _on_first_page(canvas, doc):
        # Pas d'en-tête sur la page de couverture
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(GREY_LIGHT)
        canvas.drawRightString(_w - 2 * cm, 0.8 * cm, f"Page {doc.page}")
        canvas.restoreState()

    # ── Document setup ────────────────────────────────────────────────────────
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2.2 * cm,
        bottomMargin=1.8 * cm,
    )

    story: list[Any] = []

    # ─────────────────────────────────────────────────────────────────────────
    # 1. PAGE DE COUVERTURE
    # ─────────────────────────────────────────────────────────────────────────
    story.append(Spacer(1, 3 * cm))
    story.append(Paragraph("Rapport de Construction", title_style))
    story.append(Paragraph("Table de Mortalité d'Expérience", title_style))
    story.append(Spacer(1, 0.5 * cm))
    story.append(HRFlowable(width="80%", thickness=1, color=BLUE,
                             spaceAfter=12, lineCap="round"))
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph(f"Domaine : {domain_display}", subtitle_style))
    story.append(Paragraph(f"Date : {datetime.now().strftime('%d/%m/%Y')}", subtitle_style))
    story.append(Spacer(1, 2 * cm))
    # Demande utilisateur (nettoyée)
    user_req = _clean_user_message(user_message)
    if user_req:
        story.append(Paragraph(
            f"<i>Objet : {_escape_xml(user_req[:200])}</i>", italic_style
        ))
    story.append(Spacer(1, 1 * cm))
    story.append(Paragraph(
        "<i>Rapport généré automatiquement par l'agent actuariel.</i>", italic_style
    ))
    story.append(PageBreak())

    # ─────────────────────────────────────────────────────────────────────────
    # 2. RÉSUMÉ EXÉCUTIF
    # ─────────────────────────────────────────────────────────────────────────
    story.append(Paragraph("1. Résumé exécutif", section_style))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#CCCCCC"),
                             spaceAfter=8))

    # Tableau chiffres clés extrait du summary
    key_results = _extract_key_results(summary) if summary else []
    if key_results:
        story.append(Paragraph("<b>Chiffres clés</b>", subsection_style))
        tbl_data = [["Indicateur", "Valeur"]] + [[k, v] for k, v in key_results]
        tbl = Table(tbl_data, colWidths=[max_content_width * 0.65, max_content_width * 0.35])
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), BLUE),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#F0F4FA"), colors.white]),
            ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#CCCCCC")),
            ("ALIGN", (1, 0), (1, -1), "RIGHT"),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        story.append(tbl)
        story.append(Spacer(1, 0.4 * cm))

    # Synthèse narrative (markdown rendu)
    if summary:
        story.append(Paragraph("<b>Synthèse</b>", subsection_style))
        story.extend(_md_to_story(summary, body_style, subsection_style, cm))

    story.append(PageBreak())

    # ─────────────────────────────────────────────────────────────────────────
    # Filtrage des steps
    # ─────────────────────────────────────────────────────────────────────────
    all_steps_raw = [s for s in steps if (
        s.get("role") == "agent_step"
        or ("description" in s and "output" in s)
    )]
    meaningful_steps = [s for s in all_steps_raw if _is_meaningful_step(s)]
    # Les 5 premiers steps significatifs → Méthodologie ; les suivants → Résultats
    split_idx = min(5, len(meaningful_steps))
    methodology_steps = meaningful_steps[:split_idx]
    results_steps = meaningful_steps[split_idx:]

    # ─────────────────────────────────────────────────────────────────────────
    # 3. MÉTHODOLOGIE
    # ─────────────────────────────────────────────────────────────────────────
    story.append(Paragraph("2. Méthodologie de construction", section_style))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#CCCCCC"),
                             spaceAfter=8))

    if methodology_steps:
        for step in methodology_steps:
            _append_step_content(story, step, step_title_style, body_style, small_style,
                                 max_content_width, cm, _pil_available)
    else:
        story.append(Paragraph(
            "La méthodologie est décrite dans la synthèse ci-dessus.", italic_style
        ))

    story.append(PageBreak())

    # ─────────────────────────────────────────────────────────────────────────
    # 4. RÉSULTATS ET VALIDATION
    # ─────────────────────────────────────────────────────────────────────────
    story.append(Paragraph("3. Résultats et validation", section_style))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#CCCCCC"),
                             spaceAfter=8))

    if results_steps:
        for step in results_steps:
            _append_step_content(story, step, step_title_style, body_style, small_style,
                                 max_content_width, cm, _pil_available)
    else:
        story.append(Paragraph("Les résultats sont présentés dans la synthèse.", italic_style))

    story.append(PageBreak())

    # ─────────────────────────────────────────────────────────────────────────
    # 5. CONCLUSION ET RECOMMANDATIONS
    # ─────────────────────────────────────────────────────────────────────────
    story.append(Paragraph("4. Conclusion et recommandations", section_style))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#CCCCCC"),
                             spaceAfter=8))

    # Jugement de prudence basé sur SMR global
    smr_val = None
    m_smr = re.search(r"SMR global\s*[:\-]\s*([\d.,]+)", summary or "", re.IGNORECASE)
    if m_smr:
        try:
            smr_val = float(m_smr.group(1).replace(",", "."))
        except ValueError:
            pass

    if smr_val is not None:
        if smr_val < 0.85:
            prudence_text = (
                f"Le SMR global de {smr_val:.3f} indique que le nombre de décès observés est "
                f"significativement inférieur aux décès prédits par la table de référence. "
                f"La table ainsi construite présente un niveau de prudence élevé."
            )
        elif smr_val < 1.0:
            prudence_text = (
                f"Le SMR global de {smr_val:.3f} indique que la table est prudente : "
                f"elle prédit davantage de décès que ceux effectivement observés."
            )
        else:
            prudence_text = (
                f"Le SMR global de {smr_val:.3f} indique que la table est en ligne avec "
                f"la mortalité observée. Une revue de la marge de prudence est recommandée."
            )
        story.append(Paragraph(prudence_text, body_style))
        story.append(Spacer(1, 0.3 * cm))

    story.append(Paragraph("<b>Domaine de validité</b>", subsection_style))
    story.append(Paragraph(
        f"La table construite s'applique au périmètre des données analysées "
        f"({domain_display}). Son utilisation doit être limitée aux contrats "
        f"comparables à ceux de l'observation.",
        body_style,
    ))
    story.append(Spacer(1, 0.3 * cm))

    story.append(Paragraph("<b>Recommandations de suivi</b>", subsection_style))
    for rec in [
        "Comparaison annuelle des décès observés vs. prédits par classes d'âge de 5 ans.",
        "Positionnement des décès observés dans l'intervalle de confiance à 95 % autour de la valeur prédite.",
        "Suivi du SMR global et par décennie d'âge.",
        "Reconstruction de la table au terme de 5 ans ou en cas de dérive significative.",
    ]:
        story.append(Paragraph(f"• {rec}", body_style))

    story.append(Spacer(1, 0.5 * cm))
    story.append(Paragraph(
        f"<i>Généré le {datetime.now().strftime('%d/%m/%Y à %H:%M')} "
        f"par l'agent actuariel — {domain_display}.</i>",
        small_style,
    ))
    story.append(PageBreak())

    # ─────────────────────────────────────────────────────────────────────────
    # ANNEXE — Trace d'exécution complète (steps filtrés des purs debug)
    # ─────────────────────────────────────────────────────────────────────────
    if all_steps_raw:
        story.append(Paragraph("Annexe — Trace d'exécution", section_style))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#CCCCCC"),
                                 spaceAfter=8))
        story.append(Paragraph(
            f"{len(all_steps_raw)} étape(s) au total · {len(meaningful_steps)} retenues "
            f"({len(all_steps_raw) - len(meaningful_steps)} étapes de diagnostic exclues).",
            italic_style,
        ))
        story.append(Spacer(1, 0.3 * cm))

        for step_num, step in enumerate(meaningful_steps, start=1):
            description = step.get("description") or step.get("content", "")
            story.append(Paragraph(
                f"Étape {step_num} : {_escape_xml(description[:150])}",
                step_title_style,
            ))
            output = step.get("output", "")
            if output and len(output) > 50:
                _render_text_as_table(story, output, body_style, small_style, max_content_width)
            story.append(Spacer(1, 0.2 * cm))

    # ── Build ──────────────────────────────────────────────────────────────
    doc.build(story, onFirstPage=_on_first_page, onLaterPages=_on_page)
    return output_path


def _append_step_content(
    story: list,
    step: dict,
    step_title_style: Any,
    body_style: Any,
    small_style: Any,
    max_content_width: float,
    cm: float,
    pil_available: bool,
) -> None:
    """Ajoute le contenu d'un step (titre + tables + figures) à la story."""
    from reportlab.platypus import Image, Spacer, Paragraph

    description = step.get("description") or step.get("content", "")
    figures_b64 = step.get("figures", [])
    display_outputs = step.get("display_outputs", [])
    output = step.get("output", "")

    story.append(Paragraph(_escape_xml(description[:200]), step_title_style))

    # Display outputs (DataFrames)
    for do in display_outputs:
        text_content = do.get("text", "")
        if text_content and len(text_content) > 20:
            _render_text_as_table(story, text_content, body_style, small_style, max_content_width)

    # Output texte si pas de display_output
    if output and not display_outputs and len(output) > 50:
        _render_text_as_table(story, output, body_style, small_style, max_content_width)

    # Figures
    for b64 in figures_b64:
        if not b64:
            continue
        try:
            img_bytes = base64.b64decode(b64)
            img_io = io.BytesIO(img_bytes)
            if pil_available:
                from PIL import Image as PILImage
                pil_img = PILImage.open(io.BytesIO(img_bytes))
                orig_w, orig_h = pil_img.size
                dpi = 96
                img_w_cm = orig_w / dpi * 2.54
                img_h_cm = orig_h / dpi * 2.54
            else:
                img_w_cm, img_h_cm = 14.0, 10.0

            max_w_cm = 14.0
            if img_w_cm > max_w_cm:
                ratio = max_w_cm / img_w_cm
                img_w_cm = max_w_cm
                img_h_cm *= ratio

            rl_img = Image(img_io, width=img_w_cm * cm, height=img_h_cm * cm)
            story.append(rl_img)
            story.append(Spacer(1, 0.3 * cm))
        except Exception:
            pass

    story.append(Spacer(1, 0.3 * cm))


def _render_text_as_table(
    story: list,
    text_content: str,
    body_style: Any,
    small_style: Any,
    max_content_width: float,
) -> None:
    """Tente de parser du texte tabulaire et le rend en Table reportlab."""
    try:
        from reportlab.platypus import Table, TableStyle, Paragraph, Spacer
        from reportlab.lib import colors
        from reportlab.lib.units import cm as _cm

        lines = [ln for ln in text_content.strip().split("\n") if ln.strip()]
        if not lines:
            return

        rows = []
        for line in lines[:51]:
            if "\t" in line:
                cells = line.split("\t")
            else:
                cells = re.split(r"  +", line.strip())
            rows.append([str(c).strip()[:60] for c in cells])

        if not rows:
            return

        # Tronquer à 10 colonnes
        max_cols = 10
        if rows and len(rows[0]) > max_cols:
            rows = [r[:max_cols] + ["…"] for r in rows]

        # Normaliser la longueur des lignes
        max_len = max((len(r) for r in rows), default=0)
        if max_len == 0:
            return
        rows = [r + [""] * (max_len - len(r)) for r in rows]

        # Si une seule colonne, afficher comme texte
        if max_len == 1:
            story.append(Paragraph(
                f"<font name='Courier' size='7'>{_escape_xml(text_content[:600])}</font>",
                small_style,
            ))
            return

        n_cols = max_len
        col_width = max_content_width / n_cols

        tbl = Table(rows, colWidths=[col_width] * n_cols, repeatRows=1)
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#3A3A3A")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 7),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#F5F5F5"), colors.white]),
            ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#CCCCCC")),
            ("LEFTPADDING", (0, 0), (-1, -1), 3),
            ("RIGHTPADDING", (0, 0), (-1, -1), 3),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ]))
        story.append(tbl)
        story.append(Spacer(1, 0.2 * _cm))
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Reasoning Trace (Markdown)
# ─────────────────────────────────────────────────────────────────────────────

def generate_reasoning_trace(
    steps: list[dict],
    summary: str,
    user_message: str,
    approved_plan: list[dict],
    output_path: str,
) -> str:
    """Génère un fichier Markdown de trace de raisonnement."""
    lines: list[str] = []

    lines.append("# Trace de raisonnement actuariel")
    lines.append("")
    lines.append(f"**Date :** {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    lines.append(f"**Demande :** {_clean_user_message(user_message)}")
    lines.append("")

    if approved_plan:
        lines.append("## Plan approuvé")
        lines.append("")
        for s in approved_plan:
            step_id = s.get("id", "?")
            titre = s.get("titre", "")
            description = s.get("description", "")
            lines.append(f"{step_id}. **{titre}** — {description}")
        lines.append("")

    lines.append("## Étapes d'exécution")
    lines.append("")

    agent_steps = [s for s in steps if s.get("role") == "agent_step" or (
        "description" in s and "output" in s
    )]

    for i, step in enumerate(agent_steps, start=1):
        description = step.get("description") or step.get("content", "")
        output = step.get("output", "")
        success = step.get("success", True)
        if not success and output:
            success = not output.startswith("❌")

        lines.append(f"### Étape {i} : {description[:120]}")
        lines.append("")
        lines.append(f"**Raisonnement :** {description}")
        output_display = output[:500] + ("…" if len(output) > 500 else "")
        lines.append(f"**Résultat :** {output_display}")
        lines.append("✅ Succès" if success else "❌ Erreur")
        lines.append("")

    lines.append("## Synthèse finale")
    lines.append("")
    lines.append(summary if summary else "_(Aucune synthèse disponible)_")
    lines.append("")

    Path(output_path).write_text("\n".join(lines), encoding="utf-8")
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# Jupyter Notebook (nbformat)
# ─────────────────────────────────────────────────────────────────────────────

def generate_final_notebook(
    steps: list[dict],
    user_message: str,
    output_path: str,
) -> str:
    """Génère un notebook Jupyter .ipynb reproduisant les étapes de l'agent."""
    try:
        import nbformat
    except ImportError as e:
        raise ImportError(f"nbformat est requis pour générer le notebook : {e}") from e

    nb = nbformat.v4.new_notebook()
    cells = []

    title_md = (
        f"# Analyse actuarielle — Notebook généré automatiquement\n\n"
        f"**Date :** {datetime.now().strftime('%d/%m/%Y %H:%M')}  \n"
        f"**Demande :** {_clean_user_message(user_message)}\n"
    )
    cells.append(nbformat.v4.new_markdown_cell(title_md))

    import_code = (
        "# Imports standard — à adapter selon votre environnement\n"
        "import pandas as pd\n"
        "import numpy as np\n"
        "import matplotlib\n"
        "matplotlib.use('Agg')\n"
        "import matplotlib.pyplot as plt\n"
    )
    cells.append(nbformat.v4.new_code_cell(import_code))

    agent_steps = [s for s in steps if s.get("role") == "agent_step" or (
        "description" in s and "output" in s
    )]

    for i, step in enumerate(agent_steps, start=1):
        description = step.get("description") or step.get("content", "")
        code = step.get("code", "")

        step_md = f"## Étape {i} : {description[:200]}\n\n{description}"
        cells.append(nbformat.v4.new_markdown_cell(step_md))

        if code and code.strip():
            cells.append(nbformat.v4.new_code_cell(code))

    nb["cells"] = cells

    with open(output_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    return output_path
