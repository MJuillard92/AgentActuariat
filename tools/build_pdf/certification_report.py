"""
TOOL CONTRACT — build_pdf.certification_report
════════════════════════════════════════════════════════════════

IDENTITY
--------
name          : build_pdf.certification_report
domain        : mortality_experience
version       : 1.0.0
author        : Marc Juillard
last_updated  : 2026-03-31

DESCRIPTION
-----------
Génère un rapport PDF de certification actuarielle complet de la table de
mortalité d'expérience. Contient : pipeline utilisé, table complète avec
taux bruts et lissés, diagnostics de crédibilité, intervalles de confiance,
facteurs d'abattement vs référence, SMR global, graphiques intégrés, et conclusion.

WHEN TO USE
-----------
Appeler après validation du plan du rapport avec le client. Exige au minimum
exposure_table. Les autres sections (smoothing, diagnostics, validation,
benchmarking) sont optionnelles mais fortement recommandées pour un rapport complet.

WHEN NOT TO USE
---------------
Ne pas appeler si exposure_table est absent (retourne erreur).
Ne pas appeler avant que le client ait validé le plan du rapport (voir step4).
Ne pas appeler si n_non_monotone > 0 (résoudre la monotonie d'abord).

PREREQUISITES
-------------
required_tools:
  - builder.exposure → provides exposure_table (REQUIS)
  - builder.crude_rates → provides qx_table (recommandé)
  - builder.smoothing → provides smoothed_table (recommandé)
  - builder.diagnostics → provides diagnostics (recommandé)
  - builder.validation → provides validation (recommandé)
  - builder.benchmarking → provides benchmarking (recommandé)
required_data_store_keys:
  - exposure_table (REQUIS)
  - smoothed_table (optionnel — section taux lissés)
  - diagnostics (optionnel — section crédibilité)
  - validation (optionnel — section validation)
  - benchmarking (optionnel — section abattement)

INPUTS
------
params:
  output_path:
    type    : string
    values  : chemin de fichier
    default : /tmp/rapport_certification.pdf
    note    : L'interface gère le téléchargement. Ne pas exposer au client.
  title:
    type    : string
    values  : texte libre
    default : "Table de mortalité d'expérience — Certification"
    note    : Personnaliser avec le nom du portefeuille (ex: "Portefeuille Retraite 2024").
  portfolio_info:
    type    : string
    values  : texte court
    default : ""
    note    : Description courte (ex: "45 231 lignes, 2010-2023, produit prévoyance").
  sexe:
    type    : string
    values  : H | F
    default : H
    note    : Doit correspondre au portefeuille analysé.
  commentary:
    type    : string
    default : ""
    note    : >
      Texte narratif rédigé par l'agent AVANT d'appeler ce tool.
      Structure attendue (5 paragraphes, 800-1200 mots au total) :
      §1 — Contexte et données (100-150 mots) : nature du portefeuille,
           période, effectifs, exposition totale, exclusions détectées.
      §2 — Méthode retenue et justification (150-200 mots) : pourquoi
           cette méthode de lissage pour ce portefeuille spécifique.
           Citer les caractéristiques qui ont motivé le choix.
           Ne pas nommer la méthode sans la justifier.
      §3 — Résultats et analyse (250-350 mots) : SMR global puis par
           décile d'âges. Zones de sur/sous-mortalité et interprétation.
           Croiser avec intervalles de confiance. Mentionner tout
           n_non_monotone résiduel et sa cause.
      §4 — Limites et précautions (150-200 mots) : âges peu crédibles,
           hypothèses d'extrapolation, données exclues et leur poids.
      §5 — Conclusion et recommandation d'usage (100-150 mots) :
           utilisabilité en tarification et/ou provisionnement.
           Horizon de révision si pertinent.
      Séparer les paragraphes par \n\n. Ne pas utiliser de markdown.

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
  - exposure_table absent → retourne erreur. Appeler builder.exposure d'abord.
  - n_non_monotone > 0 dans smoothed_table → bloquer la génération.
    Signaler au client : "La table lissée contient N violations de
    monotonie. Le rapport ne peut pas être généré dans cet état.
    Relancer builder.smoothing avec un lambda plus élevé ou changer
    de méthode avant de générer le rapport."
    Ne jamais produire un rapport de certification sur une table
    non monotone.
  - commentary absent ou vide → l'agent doit rédiger l'analyse
    complète (§1 à §5) avant d'appeler ce tool. Ne pas appeler
    avec commentary="" ou commentary non fourni.
NON-BLOCKING:
  - Sections optionnelles absentes → le rapport est généré sans ces sections
    (mention explicite dans le rapport que l'étape n'a pas été effectuée).

ERROR HANDLING
--------------
error: "exposure_table manquant dans data_store."
  → cause  : exposure_table absent. Pipeline builder incomplet.
  → action : Appeler builder.exposure → crude_rates → smoothing avant de générer.
error: "ReportLab non disponible : ..."
  → cause  : Bibliothèque reportlab non installée.
  → action : Signaler l'erreur technique. Ne pas relancer sans résoudre.

AGENT GUIDANCE
--------------
reasoning_hint: >
  Étape obligatoire avant d'appeler ce tool :
  1. Lire dans le data_store : smr_global (benchmarking),
     n_non_monotone (smoothed_table), ages peu crédibles (diagnostics),
     intervalles de confiance (validation).
  2. Croiser ces signaux :
     - SMR global hors [0.85, 1.15] → anomalie à expliquer en §3
     - n_non_monotone > 0 → BLOQUER (voir quality gates)
     - Plus de 20% des âges peu crédibles → renforcer §4 limites
     - IC larges sur âges extrêmes → mentionner en §4
  3. Rédiger le commentary (§1 à §5) en s'appuyant sur ces signaux.
  4. Choisir les graphiques à inclure :
     - smr_by_decile    : TOUJOURS si smr_global hors [0.85, 1.15]
     - confidence_bands : TOUJOURS si âges peu crédibles présents
     - credibility_heatmap : si plus de 20% des âges peu crédibles
     - abatement_curve  : si benchmarking disponible
     Chaque graphique inclus doit être référencé dans le commentary.
  5. Seulement après ces étapes : appeler ce tool.
exemplar_query: >
  Utiliser ces queries contre le corpus RAG exemplaires avant
  de rédiger chaque paragraphe du commentary :
  - §2 méthode : "justification choix lissage whittaker portefeuille [taille]"
  - §3 résultats : "interprétation SMR [valeur] rapport certification mortalité"
  - §4 limites : "formulation limites âges peu crédibles rapport actuariel"
  - §5 conclusion : "recommandation usage table mortalité expérience tarification"

CATALOGUE METADATA
------------------
display_name      : Rapport PDF de certification
short_description : Génère le rapport de certification de la table de mortalité d'expérience.
domain            : mortality_experience
capability_group  : reporting
depends_on        : [builder.exposure, builder.crude_rates, builder.smoothing, builder.diagnostics, builder.validation, builder.benchmarking]
required_by       : []
client_visible    : true
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path


def run(data: dict, params: dict | None = None) -> dict:
    data   = dict(data or {})  # copie pour normalisation
    params = params or {}

    output_path    = params.get("output_path", "/tmp/rapport_certification.pdf")
    title          = params.get("title", "Table de mortalité d'expérience — Certification")
    portfolio_info = params.get("portfolio_info", "")
    sexe           = params.get("sexe", "H")
    commentary     = params.get("commentary", "")

    # Découper le commentary en 5 sections narratives (§1 à §5, séparés par \n\n)
    _c_paras = [p.strip() for p in commentary.split("\n\n") if p.strip()] if commentary else []
    c1 = _c_paras[0] if len(_c_paras) > 0 else ""  # §1 Contexte et données
    c2 = _c_paras[1] if len(_c_paras) > 1 else ""  # §2 Méthode retenue et justification
    c3 = _c_paras[2] if len(_c_paras) > 2 else ""  # §3 Résultats et analyse
    c4 = _c_paras[3] if len(_c_paras) > 3 else ""  # §4 Limites et précautions
    c5 = _c_paras[4] if len(_c_paras) > 4 else ""  # §5 Conclusion et recommandation

    # ── Normalisation : data_store plat → structure attendue ──────────────────
    # Le data_store accumule les résultats à plat (ex: data["smr_global"]).
    # Le PDF attend des sous-dicts (ex: data["benchmarking"]["smr_global"]).
    if "benchmarking" not in data and ("smr_global" in data or "abatement_table" in data):
        data["benchmarking"] = {
            k: data.get(k)
            for k in ("smr_global", "abatement_table", "reference_name", "summary")
        }
    if "diagnostics" not in data and ("n_low" in data or "recommendation" in data):
        data["diagnostics"] = {
            k: data[k] for k in (
                "low_credibility_ages", "n_low", "pct_low", "zero_exposure_ages",
                "recommendation", "recommendation_reason", "overall_assessment",
            ) if k in data
        }
    if "validation" not in data and "ci_table" in data:
        data["validation"] = {k: data.get(k) for k in ("ci_table", "alpha")}
    if "smoothing" not in data and "method" in data:
        data["smoothing"] = {"method": data["method"]}

    # Vérification données minimales
    exposure_records = data.get("exposure_table")
    if not exposure_records:
        return {
            "erreur": (
                "exposure_table manquant dans data_store. "
                "Exécuter d'abord builder.exposure → crude_rates → smoothing."
            )
        }

    # Patch Python 3.8 / OpenSSL incompatibility with md5(usedforsecurity=False)
    import hashlib as _hashlib
    _orig_md5 = _hashlib.md5
    def _md5_compat(*a, **kw):
        kw.pop("usedforsecurity", None)
        return _orig_md5(*a, **kw)
    _hashlib.md5 = _md5_compat

    try:
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image as RLImage,
        )
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
    except ImportError as exc:
        _hashlib.md5 = _orig_md5
        return {"erreur": f"ReportLab non disponible : {exc}"}

    import io
    import pandas as pd

    # ── Helper : intégrer un graphique matplotlib (bytes PNG) dans le PDF ────
    def _embed_chart(png_bytes: bytes, width_cm: float = 16.0, height_cm: float = 7.5):
        """Convertit un PNG (bytes) en Image ReportLab centrée."""
        buf = io.BytesIO(png_bytes)
        img = RLImage(buf, width=width_cm * cm, height=height_cm * cm)
        img.hAlign = "CENTER"
        return img

    # ── Chargement du module de visualisation ─────────────────────────────────
    try:
        from tools.builder._nb_loader import load_nb as _load_nb
        _nb_viz = _load_nb("08_visualization")
    except Exception:
        _nb_viz = None

    # ── Styles ────────────────────────────────────────────────────────────────
    styles = getSampleStyleSheet()
    BLUE   = colors.HexColor("#1A3A5C")
    LBLUE  = colors.HexColor("#2C5F8A")
    LIGHT  = colors.HexColor("#EAF0F7")
    GREY   = colors.HexColor("#6B6B6B")
    GREEN  = colors.HexColor("#1A7A3C")
    ORANGE = colors.HexColor("#D35400")

    title_s = ParagraphStyle("T",  parent=styles["Title"],   fontSize=16, textColor=BLUE,
                              alignment=TA_CENTER, spaceAfter=6)
    sub_s   = ParagraphStyle("Su", parent=styles["Normal"],  fontSize=10, textColor=GREY,
                              alignment=TA_CENTER, spaceAfter=12)
    h1_s    = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=12, textColor=BLUE,
                              spaceBefore=14, spaceAfter=6)
    h2_s    = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=10, textColor=LBLUE,
                              spaceBefore=8,  spaceAfter=4)
    body_s  = ParagraphStyle("B",  parent=styles["Normal"],  fontSize=9,  leading=13,
                              spaceAfter=4, alignment=TA_JUSTIFY)
    small_s = ParagraphStyle("S",  parent=styles["Normal"],  fontSize=7.5, textColor=GREY, leading=10)

    def _tbl_style(header_color=BLUE, row_colors=(colors.white, LIGHT)):
        return TableStyle([
            ("BACKGROUND",    (0, 0), (-1, 0),  header_color),
            ("TEXTCOLOR",     (0, 0), (-1, 0),  colors.white),
            ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
            ("FONTSIZE",      (0, 0), (-1, -1), 8),
            ("ROWBACKGROUNDS",(0, 1), (-1, -1), row_colors),
            ("GRID",          (0, 0), (-1, -1), 0.25, colors.HexColor("#C5BDB0")),
            ("LEFTPADDING",   (0, 0), (-1, -1), 5),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 5),
            ("TOPPADDING",    (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("ALIGN",         (1, 0), (-1, -1), "RIGHT"),
        ])

    story = []

    # ── Page de titre ─────────────────────────────────────────────────────────
    story.append(Spacer(1, 1.5 * cm))
    story.append(Paragraph(title, title_s))
    sub_parts = [datetime.now().strftime("%d/%m/%Y")]
    if portfolio_info:
        sub_parts.append(portfolio_info)
    if sexe:
        sub_parts.append(f"Sexe : {'Hommes' if sexe == 'H' else 'Femmes'}")
    story.append(Paragraph(" · ".join(sub_parts), sub_s))
    story.append(Spacer(1, 0.5 * cm))

    # ── Section 1 : pipeline utilisé ──────────────────────────────────────────
    story.append(Paragraph("1. Pipeline de calcul", h1_s))
    if c1:
        story.append(Paragraph(c1, body_s))
        story.append(Spacer(1, 0.2 * cm))
    smoothed_records = data.get("smoothed_table", [])
    smoothed_method = "—"
    n_non_monotone = None
    if isinstance(data.get("smoothed_table"), list) and data.get("smoothed_table"):
        # La méthode est stockée dans data_store par writer_agent
        sm_raw = data.get("smoothing", {})
        smoothed_method = sm_raw.get("method", "whittaker") if isinstance(sm_raw, dict) else "whittaker"
    validation_raw  = data.get("validation", {})
    benchmarking_raw = data.get("benchmarking", {})
    reference_name = (
        benchmarking_raw.get("reference_name", "TH0002")
        if isinstance(benchmarking_raw, dict) else "TH0002"
    )

    pipeline_rows = [
        ["Étape", "Méthode / paramètre", "Résultat"],
        ["Exposition",  "Dates individuelles (centrale)",
         f"{len(exposure_records)} âges"],
        ["Taux bruts",  "Méthode centrale μ̂_x = D_x / E_x", ""],
        ["Lissage",     smoothed_method.capitalize(), f"{len(smoothed_records)} âges lissés"],
        ["Validation",  "Intervalles de confiance Poisson (α=5%)", ""],
        ["Benchmarking",f"Facteurs d'abattement vs {reference_name}", ""],
    ]
    tbl = Table(pipeline_rows, colWidths=[4 * cm, 9 * cm, 4 * cm])
    tbl.setStyle(_tbl_style())
    story.append(tbl)
    story.append(Spacer(1, 0.3 * cm))
    if c2:
        story.append(Paragraph(c2, body_s))
    else:
        story.append(Paragraph(
            f"La construction de la table suit la méthode actuarielle standard : exposition centrale "
            f"(personne-années) calculée à partir des dates individuelles, taux bruts estimés par "
            f"μ̂_x = D_x / E_x, lissage par la méthode {smoothed_method.capitalize()} pour régulariser "
            f"les fluctuations d'échantillonnage, puis validation statistique et benchmarking "
            f"par rapport à la table de référence {reference_name}.",
            body_s,
        ))

    # ── Section 2 : table de mortalité ────────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("2. Table de mortalité construite", h1_s))

    exp_df  = pd.DataFrame(exposure_records)
    smth_df = pd.DataFrame(smoothed_records) if smoothed_records else pd.DataFrame()

    # Fusionner
    if not smth_df.empty and "age" in smth_df.columns:
        qx_col = next((c for c in ("q_x_lisse", "qx") if c in smth_df.columns), None)
        if qx_col:
            exp_df = exp_df.merge(smth_df[["age", qx_col]], on="age", how="left")

    # Colonnes à afficher
    def _fmt(v, decimals=5):
        if v is None or (isinstance(v, float) and v != v):
            return "—"
        if isinstance(v, float):
            return f"{v:.{decimals}f}"
        return str(v)

    headers = ["Âge", "E_x (P-A)", "D_x", "q_x brut"]
    col_map = [
        ("age",      lambda v: str(int(v)) if v is not None else "—"),
        ("E_x",      lambda v: _fmt(v, 1)),
        ("D_x",      lambda v: str(int(v)) if v is not None else "—"),
        ("q_x_brut", lambda v: _fmt(v, 5)),
    ]
    if "q_x_lisse" in exp_df.columns:
        headers.append("q_x lissé")
        col_map.append(("q_x_lisse", lambda v: _fmt(v, 5)))
    elif "qx" in exp_df.columns:
        headers.append("q_x lissé")
        col_map.append(("qx", lambda v: _fmt(v, 5)))

    rows = [headers]
    for _, row in exp_df.iterrows():
        rows.append([fmt(row.get(col)) for col, fmt in col_map])

    # Limiter à 80 lignes (page max)
    if len(rows) > 81:
        rows_display = rows[:41] + [["…"] * len(headers)] + rows[-30:]
    else:
        rows_display = rows

    col_w = [1.5 * cm] + [3 * cm] * (len(headers) - 1)
    tbl = Table(rows_display, colWidths=col_w)
    tbl.setStyle(_tbl_style())
    story.append(tbl)
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph(
        f"Table complète : {len(exp_df)} âges. "
        + (f"Taux lissés : méthode {smoothed_method}." if smoothed_records else ""),
        small_s,
    ))

    # ── Graphique : taux bruts vs lissés (log scale) ──────────────────────────
    if _nb_viz is not None and smoothed_records:
        try:
            smoothed_dict = {}
            smth_plot = pd.DataFrame(smoothed_records)
            qx_col_plot = next((c for c in ("q_x_lisse", "qx") if c in smth_plot.columns), None)
            if qx_col_plot:
                smoothed_dict[smoothed_method] = {
                    "ages": smth_plot["age"].tolist(),
                    "qx_smoothed": smth_plot[qx_col_plot].tolist(),
                }
            png = _nb_viz.plot_crude_vs_smoothed(
                exp_df, smoothed_dict=smoothed_dict, sexe=sexe,
            )
            story.append(Spacer(1, 0.3 * cm))
            story.append(_embed_chart(png, width_cm=16.0, height_cm=7.0))
            story.append(Paragraph(
                "Figure 1 — Taux bruts observés (points), courbe lissée (rouge) et table de "
                f"référence {reference_name} (bleu tiretés), en échelle logarithmique.",
                small_s,
            ))
        except Exception:
            pass  # graphique optionnel

    # ── Section 3 : diagnostics ───────────────────────────────────────────────
    diagnostics = data.get("diagnostics", {})
    if diagnostics and isinstance(diagnostics, dict) and "erreur" not in diagnostics:
        story.append(PageBreak())
        story.append(Paragraph("3. Diagnostics de crédibilité", h1_s))

        cred_rows = [["Indicateur", "Valeur"]]
        for k, v in diagnostics.items():
            if isinstance(v, (int, float, str)) and not k.startswith("_"):
                cred_rows.append([k, _fmt(v, 2) if isinstance(v, float) else str(v)])

        if len(cred_rows) > 1:
            tbl = Table(cred_rows, colWidths=[9 * cm, 8 * cm])
            tbl.setStyle(_tbl_style())
            story.append(tbl)
            story.append(Spacer(1, 0.3 * cm))

        # ── Interprétation crédibilité ────────────────────────────────────
        pct_low = diagnostics.get("pct_low", 0) or 0
        n_low   = diagnostics.get("n_low", 0) or 0
        n_zero  = len(diagnostics.get("zero_exposure_ages", []) or [])
        reco    = diagnostics.get("recommendation", "")
        if c4:
            story.append(Paragraph(c4, body_s))
        elif pct_low > 0 or n_zero > 0:
            cred_text = (
                f"Sur les {len(exp_df)} âges observés, {n_zero} ont une exposition nulle "
                f"(aucun contrat) et {n_low} ont une exposition inférieure au seuil de "
                f"crédibilité statistique (E_x < 10 personne-années). "
                f"Ces âges représentent {pct_low:.1f}% du domaine total. "
                + (f"Recommandation : {reco}" if reco else "")
                + " Le lissage est appliqué pour régulariser ces taux peu fiables."
            )
            story.append(Paragraph(cred_text, body_s))

        # ── Graphique exposition ───────────────────────────────────────────
        if _nb_viz is not None:
            try:
                png_exp = _nb_viz.plot_exposure_by_age(pd.DataFrame(exposure_records))
                story.append(Spacer(1, 0.2 * cm))
                story.append(_embed_chart(png_exp, width_cm=16.0, height_cm=6.5))
                story.append(Paragraph(
                    "Figure 2 — Exposition centrale E_x (personne-années) et décès D_x observés par âge.",
                    small_s,
                ))
            except Exception:
                pass

        # Credibility detail si présent
        credibility_detail = diagnostics.get("credibility_by_age")
        if credibility_detail:
            story.append(Paragraph("Crédibilité par âge (extrait)", h2_s))
            det_df = pd.DataFrame(credibility_detail).head(20)
            det_headers = list(det_df.columns)
            det_rows = [det_headers] + [
                [_fmt(v, 2) if isinstance(v, float) else str(v) for v in row]
                for _, row in det_df.iterrows()
            ]
            det_tbl = Table(det_rows, colWidths=[2.5 * cm] * min(len(det_headers), 7))
            det_tbl.setStyle(_tbl_style())
            story.append(det_tbl)

    # ── Section 4 : validation statistique ───────────────────────────────────
    validation = data.get("validation", {})
    if validation and isinstance(validation, dict) and "erreur" not in validation:
        story.append(Paragraph("4. Validation statistique", h1_s))

        ci_records = validation.get("ci_table")
        if ci_records:
            story.append(Paragraph("Intervalles de confiance Poisson (α = 5%)", h2_s))
            ci_df = pd.DataFrame(ci_records)
            # Sélectionner colonnes principales
            ci_cols = [c for c in ("age", "q_x_lisse", "qx", "ci_lower", "ci_upper", "in_ci")
                       if c in ci_df.columns]
            if ci_cols:
                ci_display = ci_df[ci_cols].head(30)
                ci_headers = ci_cols
                ci_rows = [ci_headers] + [
                    [_fmt(v, 5) if isinstance(v, float) else str(v)
                     for v in row]
                    for _, row in ci_display.iterrows()
                ]
                tbl = Table(ci_rows, colWidths=[2 * cm] * len(ci_headers))
                tbl.setStyle(_tbl_style())
                story.append(tbl)
                story.append(Spacer(1, 0.2 * cm))
                in_ci = sum(1 for r in ci_records if r.get("in_ci") in (True, 1, "True"))
                story.append(Paragraph(
                    f"{in_ci} âges sur {len(ci_records)} dans l'intervalle de confiance "
                    f"({100 * in_ci // len(ci_records) if ci_records else 0}%).",
                    small_s,
                ))

        # Chi2
        chi2_stat = validation.get("chi2_statistic") or validation.get("chi2")
        chi2_pval = validation.get("p_value") or validation.get("p_val")
        if chi2_stat is not None:
            story.append(Paragraph("Test chi2 vs table de référence", h2_s))
            chi2_rows = [["Statistique", "Valeur"]]
            for k, v in validation.items():
                if k in ("chi2_statistic", "chi2", "p_value", "p_val",
                         "df", "degrees_of_freedom", "interpretation"):
                    if isinstance(v, (int, float, str)):
                        chi2_rows.append([k, _fmt(v, 4) if isinstance(v, float) else str(v)])
            tbl = Table(chi2_rows, colWidths=[9 * cm, 8 * cm])
            tbl.setStyle(_tbl_style())
            story.append(tbl)

    # ── Section 5 : benchmarking ──────────────────────────────────────────────
    smr_global   = None  # initialisé avant le bloc conditionnel
    benchmarking = data.get("benchmarking", {})
    if benchmarking and isinstance(benchmarking, dict) and "erreur" not in benchmarking:
        story.append(PageBreak())
        story.append(Paragraph(
            f"5. Facteurs d'abattement vs {reference_name}", h1_s
        ))

        smr_global = benchmarking.get("smr_global")
        if smr_global is not None:
            color = GREEN if smr_global < 1.0 else ORANGE
            smr_style = ParagraphStyle("smr", parent=styles["Normal"],
                                       fontSize=11, textColor=color,
                                       spaceBefore=4, spaceAfter=8)
            story.append(Paragraph(
                f"SMR global : {smr_global:.3f} "
                f"({'sous-mortalité' if smr_global < 1.0 else 'sur-mortalité'} vs {reference_name})",
                smr_style,
            ))

        abatement_records = benchmarking.get("abatement_table")
        if abatement_records:
            # ── Interprétation SMR ────────────────────────────────────────
            if c3:
                story.append(Paragraph(c3, body_s))
            elif smr_global is not None:
                pct_delta = abs(1.0 - smr_global) * 100
                direction = "sous-mortalité" if smr_global < 1.0 else "sur-mortalité"
                interp = (
                    f"Le facteur d'abattement moyen (SMR global = {smr_global:.3f}) indique une "
                    f"{direction} de {pct_delta:.1f}% par rapport à la table de référence {reference_name}. "
                )
                if smr_global < 0.7:
                    interp += (
                        "Cet écart significatif est typique d'un portefeuille avec un fort effet "
                        "de sélection (assurés ayant passé des formalités médicales, actifs en bonne santé). "
                    )
                elif smr_global > 1.3:
                    interp += (
                        "Cet écart suggère une population plus vulnérable que la référence. "
                        "Une prudence accrue dans la tarification est recommandée. "
                    )
                interp += (
                    "Les facteurs d'abattement par âge (graphique ci-dessous) permettent "
                    "d'identifier les tranches d'âge où l'écart est le plus marqué."
                )
                story.append(Paragraph(interp, body_s))

            ab_df = pd.DataFrame(abatement_records)
            ab_cols = [c for c in ab_df.columns if c not in ("_sa",)]
            ab_display = ab_df[ab_cols].head(50)
            ab_headers = list(ab_display.columns)
            ab_rows = [ab_headers] + [
                [_fmt(v, 4) if isinstance(v, float) else str(v) for v in row]
                for _, row in ab_display.iterrows()
            ]
            col_w_ab = [2 * cm] * len(ab_headers)
            tbl = Table(ab_rows, colWidths=col_w_ab)
            tbl.setStyle(_tbl_style())
            story.append(tbl)
            story.append(Spacer(1, 0.2 * cm))
            story.append(Paragraph(
                f"Facteurs d'abattement sur {len(abatement_records)} âges "
                f"vs table de référence {reference_name}.",
                small_s,
            ))

            # ── Graphique facteurs d'abattement ───────────────────────────
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                _BG_C  = "#FBF8F1"
                _GRID_C = "#E8E3D8"
                ab_plot = ab_df.dropna(subset=["abatement_factor"])
                _ages_ab   = ab_plot["age"].tolist()
                _factors   = ab_plot["abatement_factor"].tolist()
                _bar_colors = ["#2C5F8A" if f <= 1.0 else "#E67E22" for f in _factors]

                fig_ab, ax_ab = plt.subplots(figsize=(13, 4.5))
                fig_ab.patch.set_facecolor(_BG_C)
                ax_ab.set_facecolor(_BG_C)
                ax_ab.bar(_ages_ab, _factors, color=_bar_colors, alpha=0.8, width=0.75, edgecolor="none")
                ax_ab.axhline(y=1.0, color="#C0392B", linewidth=1.8, linestyle="--",
                              label="Référence (α = 1.0)")
                if smr_global is not None:
                    ax_ab.axhline(y=smr_global, color="#27AE60", linewidth=1.4, linestyle=":",
                                  label=f"SMR global = {smr_global:.3f}")
                ax_ab.set_xlabel("Âge", fontsize=10)
                ax_ab.set_ylabel("Facteur α", fontsize=10)
                ax_ab.set_title(
                    f"Facteurs d'abattement par âge — vs {reference_name} "
                    f"({'H' if sexe == 'H' else 'F'})",
                    fontsize=10, loc="left",
                )
                ax_ab.legend(facecolor=_BG_C, edgecolor=_GRID_C, fontsize=8, framealpha=0.9)
                ax_ab.grid(True, color=_GRID_C, linewidth=0.8, alpha=0.8)
                ax_ab.spines["top"].set_visible(False)
                ax_ab.spines["right"].set_visible(False)

                buf_ab = io.BytesIO()
                fig_ab.savefig(buf_ab, format="png", dpi=130, bbox_inches="tight",
                               facecolor=_BG_C)
                plt.close(fig_ab)
                buf_ab.seek(0)
                story.append(Spacer(1, 0.2 * cm))
                story.append(_embed_chart(buf_ab.read(), width_cm=16.0, height_cm=5.0))
                story.append(Paragraph(
                    f"Figure 3 — Facteurs d'abattement α_x = q_x_exp / q_x_ref par âge. "
                    f"Bleu = sous-mortalité (α ≤ 1), orange = sur-mortalité (α > 1).",
                    small_s,
                ))
            except Exception:
                pass

    # ── Conclusion ────────────────────────────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("Conclusion", h1_s))
    if c5:
        story.append(Paragraph(c5, body_s))
    else:
        n_ages  = len(exp_df)
        e_total = sum(r.get("E_x", 0) or 0 for r in exposure_records)
        d_total = sum(r.get("D_x", 0) or 0 for r in exposure_records)
        conclusion = (
            f"La table de mortalité d'expérience a été construite sur {n_ages} âges, "
            f"représentant {e_total:,.0f} personne-années d'exposition "
            f"et {int(d_total)} décès observés. "
            f"Le lissage a été effectué par la méthode {smoothed_method.capitalize()}. "
        )
        if smr_global is not None:
            pct_delta = abs(1.0 - smr_global) * 100
            direction = "sous-mortalité" if smr_global < 1.0 else "sur-mortalité"
            conclusion += (
                f"Le SMR global de {smr_global:.3f} confirme une {direction} "
                f"de {pct_delta:.1f}% par rapport à la table {reference_name}. "
                f"Cette table peut être utilisée comme base actuarielle pour "
                f"la tarification, le provisionnement ou la construction de tables de projection "
                f"internes au portefeuille."
            )
        story.append(Paragraph(conclusion, body_s))

    # ── Construction du PDF ───────────────────────────────────────────────────
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=2.2 * cm, rightMargin=2.2 * cm,
        topMargin=2 * cm,    bottomMargin=2 * cm,
    )
    doc.build(story)
    _hashlib.md5 = _orig_md5  # restore after build

    return {
        "succes":            True,
        "output_path":       output_path,
        "nb_pages_estimees": max(2, len(story) // 20),
    }
