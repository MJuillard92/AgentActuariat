"""
TOOL CONTRACT — build_pdf.descriptive_report
════════════════════════════════════════════════════════════════

IDENTITY
--------
name          : build_pdf.descriptive_report
domain        : descriptive
version       : 1.0.0
author        : Marc Juillard
last_updated  : 2026-03-31

DESCRIPTION
-----------
Génère un rapport PDF descriptif du portefeuille à partir des résultats
des outils statistical_analysis. Contient : résumé global, distribution
des âges, évolution annuelle, segmentation, et commentaires narratifs.
Au minimum, summary (portfolio_summary) est requis.

WHEN TO USE
-----------
Appeler après avoir exécuté les outils statistical_analysis souhaités et
après validation du plan du rapport avec le client. Permet de synthétiser
une analyse descriptive complète dans un document PDF professionnel.

WHEN NOT TO USE
---------------
Ne pas appeler si aucun résultat de statistical_analysis n'est disponible.
Ne pas appeler avant que le client ait validé le contenu à inclure.

PREREQUISITES
-------------
required_tools:
  - statistical_analysis.portfolio_summary → provides summary (REQUIS AU MINIMUM)
  - statistical_analysis.age_distribution → provides ages (optionnel)
  - statistical_analysis.time_series → provides series (optionnel)
  - statistical_analysis.segmentation → provides segmentation (optionnel)
required_data_store_keys:
  - summary (requis au minimum)
  - ages, series, segmentation (optionnels — sections conditionnelles)

INPUTS
------
params:
  output_path:
    type    : string
    values  : chemin de fichier
    default : /tmp/rapport_descriptif.pdf
    note    : L'interface gère le téléchargement. Ne pas exposer au client.
  title:
    type    : string
    values  : texte libre
    default : "Analyse descriptive du portefeuille"
    note    : Personnaliser avec le nom du portefeuille du client.

OUTPUTS
-------
data_store_keys_written: []
return_payload:
  succes            : bool
  output_path       : str
  nb_pages_estimees : int

QUALITY GATES
-------------
BLOCKING:
  - Aucune donnée statistical_analysis dans data_store → rapport vide ou erreur.
    Appeler au minimum statistical_analysis.portfolio_summary d'abord.
NON-BLOCKING:
  - Sections optionnelles absentes → rapport généré sans ces sections.

ERROR HANDLING
--------------
error: "ReportLab non disponible : ..."
  → cause  : Bibliothèque reportlab non installée.
  → action : Signaler l'erreur technique. Ne pas relancer sans résoudre.

AGENT GUIDANCE
--------------
reasoning_hint: >
  Proposer le rapport descriptif à la fin d'une analyse statistique.
  Inclure tous les résultats disponibles (summary, ages, series, segmentation).
  Ne jamais mentionner le chemin de fichier dans la réponse au client.
  Dire simplement : "Le rapport a été généré. Le téléchargement démarre automatiquement."
exemplar_query: >
  Quelles sections inclure dans un rapport descriptif d'un portefeuille prévoyance ?

CATALOGUE METADATA
------------------
display_name      : Rapport PDF descriptif
short_description : Génère un rapport PDF de l'analyse descriptive du portefeuille.
domain            : descriptive
capability_group  : reporting
depends_on        : [statistical_analysis.portfolio_summary, statistical_analysis.age_distribution, statistical_analysis.time_series, statistical_analysis.segmentation]
required_by       : []
client_visible    : true
"""
from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Any


def run(data: dict, params: dict | None = None) -> dict:
    """
    Génère un PDF descriptif avec les résultats d'analyse déjà calculés.

    data dict attendu :
      - "summary"     : résultat de portfolio_summary.run()
      - "ages"        : résultat de age_distribution.run() (optionnel)
      - "series"      : résultat de time_series.run() (optionnel)
      - "segmentation": résultat de segmentation.run() (optionnel)
      - "narrative"   : texte narratif rédigé par le WriterAgent (optionnel)
    """
    p = params or {}
    output_path = p.get("output_path", "/tmp/rapport_descriptif.pdf")
    title = p.get("title", "Analyse descriptive du portefeuille")

    import hashlib as _hashlib
    _orig_md5 = _hashlib.md5
    def _md5_compat(*a, **kw):
        kw.pop("usedforsecurity", None)
        return _orig_md5(*a, **kw)
    _hashlib.md5 = _md5_compat

    try:
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        )
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
    except ImportError as e:
        _hashlib.md5 = _orig_md5
        return {"erreur": f"ReportLab non disponible : {e}"}

    styles = getSampleStyleSheet()
    BLUE  = colors.HexColor("#1A3A5C")
    GREY  = colors.HexColor("#6B6B6B")
    LIGHT = colors.HexColor("#EAF0F7")

    title_s = ParagraphStyle("T", parent=styles["Title"],
                             fontSize=18, textColor=BLUE, alignment=TA_CENTER, spaceAfter=8)
    h1_s    = ParagraphStyle("H1", parent=styles["Heading1"],
                             fontSize=13, textColor=BLUE, spaceBefore=14, spaceAfter=6)
    body_s  = ParagraphStyle("B", parent=styles["Normal"],
                             fontSize=10, leading=14, spaceAfter=5, alignment=TA_JUSTIFY)
    small_s = ParagraphStyle("S", parent=styles["Normal"],
                             fontSize=8, textColor=GREY, leading=11)

    story: list[Any] = []

    # ── Titre ─────────────────────────────────────────────────────────────────
    story.append(Paragraph(title, title_s))
    story.append(Spacer(1, 0.4 * cm))

    # ── Résumé ────────────────────────────────────────────────────────────────
    summary = data.get("summary", {})
    if summary:
        story.append(Paragraph("1. Résumé du portefeuille", h1_s))
        rows = [["Indicateur", "Valeur"]]
        mapping = [
            ("nb_contrats",                  "Nombre de contrats"),
            ("nb_deces",                     "Nombre de décès"),
            ("taux_brut_deces_pour_1000_pa", "Taux brut (‰ PA)"),
            ("exposition_totale_pa",         "Exposition totale (P-A)"),
            ("age_min",                      "Âge minimum"),
            ("age_max",                      "Âge maximum"),
            ("age_moyen",                    "Âge moyen à l'entrée"),
            ("date_entree_min",              "Première entrée"),
            ("date_entree_max",              "Dernière entrée"),
            ("date_sortie_max",              "Dernière sortie"),
        ]
        for key, label in mapping:
            if key in summary and summary[key] is not None:
                val = summary[key]
                if isinstance(val, float):
                    val = f"{val:,.2f}".replace(",", " ")
                rows.append([label, str(val)])

        tbl = Table(rows, colWidths=[9 * cm, 7 * cm])
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), BLUE),
            ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
            ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",   (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
            ("GRID",       (0, 0), (-1, -1), 0.25, colors.HexColor("#C5BDB0")),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING",   (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
        ]))
        story.append(tbl)
        story.append(Spacer(1, 0.3 * cm))

    # ── Distribution des âges ─────────────────────────────────────────────────
    ages = data.get("ages", {})
    if ages and "distribution" in ages:
        story.append(Paragraph("2. Distribution des âges à l'entrée", h1_s))
        dist = ages["distribution"]
        rows = [["Tranche d'âge", "Nb contrats"]]
        for tranche, nb in dist.items():
            rows.append([tranche, str(nb)])
        tbl = Table(rows, colWidths=[8 * cm, 8 * cm])
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), BLUE),
            ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
            ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",   (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
            ("GRID",       (0, 0), (-1, -1), 0.25, colors.HexColor("#C5BDB0")),
            ("ALIGN",      (1, 0), (1, -1), "RIGHT"),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING",  (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 3),
        ]))
        story.append(tbl)
        story.append(Spacer(1, 0.3 * cm))

    # ── Série temporelle ──────────────────────────────────────────────────────
    series_data = data.get("series", {})
    if series_data and "serie" in series_data:
        story.append(Paragraph("3. Évolution annuelle", h1_s))
        rows = [["Année", "Entrées", "Décès", "Exposition (P-A)"]]
        for r in series_data["serie"]:
            rows.append([
                str(r.get("annee", "")),
                str(r.get("nb_entres", "")),
                str(r.get("nb_deces", "")),
                f"{r.get('exposition_pa', 0):,.1f}".replace(",", " "),
            ])
        tbl = Table(rows, colWidths=[3 * cm, 4 * cm, 4 * cm, 5 * cm])
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), BLUE),
            ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
            ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",   (0, 0), (-1, -1), 8),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
            ("GRID",       (0, 0), (-1, -1), 0.25, colors.HexColor("#C5BDB0")),
            ("ALIGN",      (1, 0), (-1, -1), "RIGHT"),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING",  (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 3),
        ]))
        story.append(tbl)
        if series_data.get("anomalies"):
            story.append(Spacer(1, 0.15 * cm))
            for a in series_data["anomalies"]:
                story.append(Paragraph(f"⚠ {a}", small_s))
        story.append(Spacer(1, 0.3 * cm))

    # ── Segmentation ─────────────────────────────────────────────────────────
    seg = data.get("segmentation", {})
    if seg and "segmentations" in seg:
        story.append(Paragraph("4. Segmentation", h1_s))
        for var, rows_data in seg["segmentations"].items():
            story.append(Paragraph(f"Par {var}", body_s))
            rows = [["Valeur", "Contrats", "Décès", "% Contrats"]]
            for r in rows_data:
                rows.append([
                    str(r.get("valeur", "")),
                    str(r.get("nb_contrats", "")),
                    str(r.get("nb_deces", "")),
                    f"{r.get('pct_contrats', 0):.1f}%",
                ])
            tbl = Table(rows, colWidths=[5 * cm, 3.5 * cm, 3.5 * cm, 4 * cm])
            tbl.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), BLUE),
                ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
                ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE",   (0, 0), (-1, -1), 9),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
                ("GRID",       (0, 0), (-1, -1), 0.25, colors.HexColor("#C5BDB0")),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING",  (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING",(0, 0), (-1, -1), 3),
            ]))
            story.append(tbl)
            story.append(Spacer(1, 0.2 * cm))

    # ── Narratif ──────────────────────────────────────────────────────────────
    narrative = data.get("narrative", "")
    if narrative:
        story.append(Paragraph("5. Analyse et commentaires", h1_s))
        for para in narrative.split("\n\n"):
            para = para.strip()
            if para:
                story.append(Paragraph(para.replace("\n", "<br/>"), body_s))

    # ── Construction du PDF ───────────────────────────────────────────────────
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=2.5 * cm, rightMargin=2.5 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm,
    )
    doc.build(story)
    _hashlib.md5 = _orig_md5  # restore

    return {
        "succes": True,
        "output_path": output_path,
        "nb_pages_estimees": max(1, len(story) // 25),
    }
