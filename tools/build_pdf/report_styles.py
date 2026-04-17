"""
tools/build_pdf/report_styles.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Source unique de vérité pour tous les styles visuels du rapport PDF.

Importer depuis n'importe quel module build_pdf :
    from tools.build_pdf.report_styles import get_styles, make_table, COLORS

API publique :
    COLORS              — dict de couleurs ReportLab (BLUE, LBLUE, LIGHT, ...)
    get_styles()        — retourne un StyleBundle avec tous les ParagraphStyle
    make_table(rows, style="default", col_widths=None) → ReportLab Table
    make_header_table(rows, col_widths=None)           → Table en-tête bleue
    make_data_table(rows, col_widths=None)             → Table données tabulaire
    make_kpi_table(kpis: list[tuple[str,str]])         → Table KPI 2 colonnes
    tbl_style(name="default")                          → TableStyle
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# ── Imports ReportLab ─────────────────────────────────────────────────────────
try:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import Table, TableStyle
    _RL_AVAILABLE = True
except ImportError:
    _RL_AVAILABLE = False


# ── Palette ───────────────────────────────────────────────────────────────────

def _hex(h: str):
    return colors.HexColor(h) if _RL_AVAILABLE else h

COLORS: dict[str, Any] = {
    # Primaires
    "BLUE":       _hex("#1A3668"),   # Bleu foncé — en-têtes, titres
    "LBLUE":      _hex("#2C5F8A"),   # Bleu moyen — sous-titres, h2
    "ACCENT":     _hex("#1F6BB5"),   # Bleu vif — KPI highlights
    # Fond
    "LIGHT":      _hex("#EAF0F7"),   # Bleu très pâle — ligne paire des tables
    "LIGHT2":     _hex("#F5F8FC"),   # Encore plus pâle — alternance douce
    "WHITE":      _hex("#FFFFFF"),
    # Texte
    "GREY":       _hex("#555555"),   # Corps de texte secondaire
    "DARK":       _hex("#222222"),   # Corps principal
    "CAPTION":    _hex("#7A8A9A"),   # Captions, notes de bas
    # Sémantiques
    "GREEN":      _hex("#2E7D32"),   # OK / conforme
    "ORANGE":     _hex("#E65100"),   # Attention
    "RED":        _hex("#B71C1C"),   # Anomalie
    "GOLD":       _hex("#F9A825"),   # Highlight neutre
    # Séparateurs
    "BORDER":     _hex("#BECFDE"),   # Grille légère
    "HRULE":      _hex("#D0DCE8"),   # Ligne horizontale fine
}

C = COLORS  # alias court


# ── StyleBundle ───────────────────────────────────────────────────────────────

@dataclass
class StyleBundle:
    """Tous les ParagraphStyle du rapport — accès par attribut."""
    title:    Any
    subtitle: Any
    h1:       Any
    h2:       Any
    h3:       Any
    body:     Any
    body_c:   Any   # body centré
    bold:     Any
    small:    Any
    caption:  Any
    code:     Any
    kpi_val:  Any   # valeur KPI (grand, bleu)
    kpi_lab:  Any   # label KPI (petit, gris)


def get_styles() -> StyleBundle:
    """Construit et retourne tous les styles du rapport."""
    if not _RL_AVAILABLE:
        raise ImportError("ReportLab non disponible")

    base = getSampleStyleSheet()

    def _style(name, parent_name="Normal", **kw):
        return ParagraphStyle(name, parent=base[parent_name], **kw)

    return StyleBundle(
        title = _style("RT_title", "Title",
            fontSize=18, textColor=C["BLUE"],
            alignment=TA_CENTER, spaceAfter=6, spaceBefore=0,
            fontName="Helvetica-Bold",
        ),
        subtitle = _style("RT_sub", "Normal",
            fontSize=10, textColor=C["GREY"],
            alignment=TA_CENTER, spaceAfter=12,
        ),
        h1 = _style("RT_h1", "Heading1",
            fontSize=13, textColor=C["BLUE"],
            spaceBefore=16, spaceAfter=6,
            fontName="Helvetica-Bold",
            borderPadding=(0, 0, 4, 0),
        ),
        h2 = _style("RT_h2", "Heading2",
            fontSize=11, textColor=C["LBLUE"],
            spaceBefore=10, spaceAfter=4,
            fontName="Helvetica-Bold",
        ),
        h3 = _style("RT_h3", "Heading3",
            fontSize=10, textColor=C["LBLUE"],
            spaceBefore=6, spaceAfter=3,
            fontName="Helvetica-BoldOblique",
        ),
        body = _style("RT_body", "Normal",
            fontSize=9.5, leading=14,
            spaceAfter=5, alignment=TA_JUSTIFY,
            textColor=C["DARK"],
        ),
        body_c = _style("RT_bodyc", "Normal",
            fontSize=9.5, leading=14,
            spaceAfter=5, alignment=TA_CENTER,
            textColor=C["DARK"],
        ),
        bold = _style("RT_bold", "Normal",
            fontSize=9.5, leading=14,
            fontName="Helvetica-Bold",
            textColor=C["DARK"],
        ),
        small = _style("RT_small", "Normal",
            fontSize=8, leading=11, textColor=C["GREY"],
        ),
        caption = _style("RT_caption", "Normal",
            fontSize=8, leading=10, textColor=C["CAPTION"],
            alignment=TA_CENTER, spaceAfter=6, spaceBefore=2,
            fontName="Helvetica-Oblique",
        ),
        code = _style("RT_code", "Code",
            fontSize=8, leading=11,
            fontName="Courier",
            backColor=C["LIGHT2"],
            textColor=C["DARK"],
        ),
        kpi_val = _style("RT_kpiv", "Normal",
            fontSize=18, leading=22,
            fontName="Helvetica-Bold",
            textColor=C["ACCENT"],
            alignment=TA_CENTER,
        ),
        kpi_lab = _style("RT_kpil", "Normal",
            fontSize=8, leading=10,
            textColor=C["GREY"],
            alignment=TA_CENTER,
        ),
    )


# ── TableStyle presets ────────────────────────────────────────────────────────

def tbl_style(name: str = "default") -> "TableStyle":
    """
    Retourne un TableStyle préconfiguré.

    name :
      "default"   — en-tête bleu, lignes alternées, grille fine
      "compact"   — même mais padding réduit (pour tableaux denses par âge)
      "borderless"— fond blanc, séparateurs horizontaux seulement
      "kpi"       — pour tables KPI 2 colonnes (label | valeur)
    """
    if not _RL_AVAILABLE:
        raise ImportError("ReportLab non disponible")

    common = [
        # En-tête
        ("BACKGROUND",    (0, 0), (-1, 0),  C["BLUE"]),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  C["WHITE"]),
        ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, 0),  8.5),
        ("ALIGN",         (0, 0), (-1, 0),  "CENTER"),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        # Corps
        ("FONTNAME",      (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",      (0, 1), (-1, -1), 8),
        ("TEXTCOLOR",     (0, 1), (-1, -1), C["DARK"]),
        # Grille
        ("GRID",          (0, 0), (-1, -1), 0.3, C["BORDER"]),
        ("LINEBELOW",     (0, 0), (-1, 0),  1.0, C["BLUE"]),
        # Alternance lignes
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [C["LIGHT2"], C["WHITE"]]),
    ]

    paddings = {
        "default":    [("TOPPADDING",    (0, 0), (-1, -1), 5),
                       ("BOTTOMPADDING",(0, 0), (-1, -1), 5),
                       ("LEFTPADDING",  (0, 0), (-1, -1), 6),
                       ("RIGHTPADDING", (0, 0), (-1, -1), 6)],
        "compact":    [("TOPPADDING",    (0, 0), (-1, -1), 2),
                       ("BOTTOMPADDING",(0, 0), (-1, -1), 2),
                       ("LEFTPADDING",  (0, 0), (-1, -1), 4),
                       ("RIGHTPADDING", (0, 0), (-1, -1), 4)],
        "borderless": [("TOPPADDING",    (0, 0), (-1, -1), 4),
                       ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
                       ("LEFTPADDING",  (0, 0), (-1, -1), 6),
                       ("RIGHTPADDING", (0, 0), (-1, -1), 6)],
        "kpi":        [("TOPPADDING",    (0, 0), (-1, -1), 8),
                       ("BOTTOMPADDING",(0, 0), (-1, -1), 8),
                       ("LEFTPADDING",  (0, 0), (-1, -1), 10),
                       ("RIGHTPADDING", (0, 0), (-1, -1), 10)],
    }

    extras = {
        "default":    [],
        "compact":    [("FONTSIZE", (0, 1), (-1, -1), 7.5)],
        "borderless": [
            ("GRID",          (0, 0), (-1, -1), 0,   C["WHITE"]),
            ("LINEBELOW",     (0, 0), (-1, 0),  0.8, C["BLUE"]),
            ("LINEBELOW",     (0, 1), (-1, -2), 0.3, C["HRULE"]),
            ("BACKGROUND",    (0, 1), (-1, -1), C["WHITE"]),
            ("ROWBACKGROUNDS",(0, 1), (-1, -1), [C["WHITE"], C["WHITE"]]),
        ],
        "kpi":        [
            ("BACKGROUND",    (0, 0), (-1, 0),  C["WHITE"]),
            ("TEXTCOLOR",     (0, 0), (-1, 0),  C["GREY"]),
            ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica"),
            ("FONTSIZE",      (0, 0), (-1, 0),  8),
            ("BACKGROUND",    (0, 1), (-1, -1), C["WHITE"]),
            ("ROWBACKGROUNDS",(0, 1), (-1, -1), [C["WHITE"], C["WHITE"]]),
            ("BOX",           (0, 0), (-1, -1), 1.0, C["BORDER"]),
            ("GRID",          (0, 0), (-1, -1), 0,   C["WHITE"]),
        ],
    }

    style_name = name if name in paddings else "default"
    all_cmds = common + paddings[style_name] + extras[style_name]
    return TableStyle(all_cmds)


# ── Constructeurs de tables ───────────────────────────────────────────────────

def _auto_col_widths(rows: list, total_cm: float = 16.0) -> list:
    """Calcule des largeurs de colonnes proportionnelles au contenu."""
    if not rows:
        return []
    n_cols = max(len(r) for r in rows)
    if n_cols == 0:
        return []
    # Colonne 0 plus large si probable label textuel
    if n_cols >= 2:
        weights = [2.5] + [1.0] * (n_cols - 1)
    else:
        weights = [1.0]
    total_w = sum(weights)
    return [w / total_w * total_cm * cm for w in weights]


def make_table(
    rows:       list[list],
    style:      str  = "default",
    col_widths: list | None = None,
    repeat_header: bool = True,
) -> "Table":
    """
    Construit une Table ReportLab stylisée.

    Args:
        rows       : liste de listes (première ligne = en-tête)
        style      : "default" | "compact" | "borderless" | "kpi"
        col_widths : largeurs explicites en cm (ex: [3, 2, 2, 2])
        repeat_header : répéter l'en-tête sur chaque page

    Returns:
        Table ReportLab prête à insérer dans le story
    """
    if not _RL_AVAILABLE:
        raise ImportError("ReportLab non disponible")
    if not rows:
        return Table([[""]])

    col_w = ([w * cm for w in col_widths] if col_widths
             else _auto_col_widths(rows))

    tbl = Table(rows, colWidths=col_w, repeatRows=1 if repeat_header else 0)
    tbl.setStyle(tbl_style(style))
    return tbl


def make_data_table(rows: list[list], col_widths: list | None = None) -> "Table":
    """Table de données dense par âge (style compact, colonnes numériques)."""
    return make_table(rows, style="compact", col_widths=col_widths)


def make_kpi_table(kpis: list[tuple[str, str]], n_cols: int = 4) -> "Table":
    """
    Table KPI horizontale.

    Args:
        kpis   : liste de (label, valeur) — ex: [("SMR", "0.748"), ("χ² p", "0.312")]
        n_cols : nombre de KPIs par ligne (défaut 4)

    Returns:
        Table avec labels en petits gris, valeurs en grand bleu
    """
    if not _RL_AVAILABLE:
        raise ImportError("ReportLab non disponible")

    from reportlab.platypus import Paragraph

    styles = get_styles()

    # Grouper en lignes de n_cols
    cells = []
    for label, value in kpis:
        cells.append([
            Paragraph(value, styles.kpi_val),
            Paragraph(label, styles.kpi_lab),
        ])

    # Padding pour compléter la dernière ligne
    while len(cells) % n_cols != 0:
        cells.append([Paragraph("", styles.kpi_val), Paragraph("", styles.kpi_lab)])

    rows = []
    for i in range(0, len(cells), n_cols):
        val_row = [cells[j][0] for j in range(i, i + n_cols)]
        lab_row = [cells[j][1] for j in range(i, i + n_cols)]
        rows.append(val_row)
        rows.append(lab_row)

    col_w = [16.0 / n_cols * cm] * n_cols
    tbl = Table(rows, colWidths=col_w)
    tbl.setStyle(TableStyle([
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
        ("TOPPADDING",    (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("BOX",           (0, 0), (-1, -1), 1.0, C["BORDER"]),
        ("LINEBEFORE",    (1, 0), (-1, -1), 0.5, C["HRULE"]),
        ("BACKGROUND",    (0, 0), (-1, -1), C["WHITE"]),
    ]))
    return tbl
