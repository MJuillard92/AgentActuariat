"""
TOOL CONTRACT — graphs.builder_plots
════════════════════════════════════════════════════════════════

IDENTITY
--------
name          : graphs.builder_plots
domain        : mortality_experience
version       : 1.0.0
author        : Marc Juillard
last_updated  : 2026-03-31

DESCRIPTION
-----------
Génère les graphiques actuariels du pipeline de construction de table :
exposition par âge, taux bruts vs lissés, SMR par décennie, intervalles
de confiance, courbe de survie, et facteurs d'abattement. Retourne une
image PNG encodée en base64.

WHEN TO USE
-----------
Appeler après les étapes du pipeline builder pour visualiser les résultats.
Intégrer dans le rapport PDF ou afficher directement dans l'interface.
Recommandé après chaque étape clé (exposure, smoothing, benchmarking).

WHEN NOT TO USE
---------------
Ne pas appeler sans les données requises pour le chart choisi.
Ne pas appeler "smr" sans résultat de builder.diagnostics (function_name=smr).

PREREQUISITES
-------------
required_tools: [varies by chart]
  - exposure → builder.exposure (exposure_table)
  - crude_smoothed → builder.exposure + builder.smoothing
  - smr → builder.diagnostics (function_name=smr)
  - ci_bands → builder.exposure + builder.smoothing + builder.validation
  - survival_curve → builder.exposure + builder.smoothing
  - abatement_chart → builder.benchmarking (abatement_table)
required_data_store_keys: [varies by chart — see above]

INPUTS
------
params:
  chart:
    type    : string
    values  : exposure | crude_smoothed | smr | ci_bands | survival_curve | abatement_chart
    default : exposure
    note    : Choisir selon les données disponibles dans le data_store.
  sexe:
    type    : string
    values  : H | F
    default : H
    note    : Pour les graphiques incluant une courbe de référence TH/TF.
  title_suffix:
    type    : string
    values  : texte libre
    default : ""
    note    : Texte ajouté au titre du graphique (ex: "— Portefeuille retraite").

OUTPUTS
-------
data_store_keys_written: []
return_payload:
  chart     : str — type de graphique produit
  image_b64 : str — image PNG encodée en base64

QUALITY GATES
-------------
BLOCKING:
  - Données requises absentes → retourne erreur avec indication du tool à appeler.
NON-BLOCKING:
  - Graphique produit mais vide (ex: aucun abatement_factor) → avertir le client.

ERROR HANDLING
--------------
error: "exposure_table manquant. Appeler builder.exposure d'abord."
  → cause  : Données requises pour ce chart absentes du data_store.
  → action : Appeler le tool indiqué dans le message d'erreur.
error: "chart inconnu : '...'"
  → cause  : Valeur de chart incorrecte.
  → action : Utiliser uniquement : exposure, crude_smoothed, smr, ci_bands,
             survival_curve, abatement_chart.

AGENT GUIDANCE
--------------
reasoning_hint: >
  Après chaque étape du pipeline builder, proposer le graphique correspondant :
  exposure → chart="exposure", smoothing → chart="crude_smoothed",
  benchmarking → chart="abatement_chart". Les graphiques enrichissent le rapport
  PDF et aident le client à comprendre les résultats.
exemplar_query: >
  Quels graphiques inclure dans un rapport de certification de table de mortalité ?

CATALOGUE METADATA
------------------
display_name      : Graphiques de construction de table
short_description : Génère les visualisations actuarielles du pipeline builder (exposition, taux, SMR).
domain            : mortality_experience
capability_group  : graphs
depends_on        : [builder.exposure, builder.smoothing, builder.diagnostics, builder.benchmarking, builder.validation]
required_by       : []
client_visible    : true
"""
from __future__ import annotations

import base64
import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from tools.builder._nb_loader import load_nb

# ── Palette standardisée (cohérente avec le YAML generation_rules) ────────────
_BG     = "#FBF8F1"   # fond uniforme
_GRID   = "#E8E3D8"   # grille légère
_BLUE   = "#2C5F8A"   # males / ligne principale
_RED    = "#C0392B"   # females / modélisé / anomalie
_ORANGE = "#E67E22"   # accentuation secondaire
_GREEN  = "#27AE60"   # SMR global / ligne référence positive
_CI     = "#AED6F1"   # bande IC (light blue)
_CRUDE  = "#888888"   # taux bruts (gris)
_MALE   = _BLUE
_FEMALE = _RED


def _to_b64(png_bytes: bytes) -> str:
    return base64.b64encode(png_bytes).decode()


def _std_fig(figsize=(13, 5)):
    """Crée une figure matplotlib avec le style standard du projet."""
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(_BG)
    ax.set_facecolor(_BG)
    ax.grid(True, color=_GRID, linewidth=0.8, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig, ax


def _std_save(fig) -> str:
    """Sauvegarde la figure et retourne le base64 PNG."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor=_BG)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _ages_vals(d: dict) -> tuple:
    """Convertit un dict {age_str: val} en (ages_sorted, vals_sorted)."""
    if not d:
        return [], []
    pairs = []
    for k, v in d.items():
        try:
            pairs.append((int(k), float(v) if v is not None else 0.0))
        except (ValueError, TypeError):
            pass
    pairs.sort()
    return [p[0] for p in pairs], [p[1] for p in pairs]


def _ctx(data: dict) -> dict:
    """Retourne template_context si disponible, sinon data directement."""
    return data.get("template_context") or data


def run(data: dict | None, params: dict | None = None) -> dict:
    data = data or {}
    params = params or {}
    chart = params.get("chart", "exposure")
    title_suffix = params.get("title_suffix", "")
    sexe = params.get("sexe", "H")

    nb = load_nb("08_visualization")

    if chart == "exposure":
        exposure_records = data.get("exposure_table")
        if not exposure_records:
            return {"erreur": "exposure_table manquant. Appeler builder.exposure d'abord."}
        exposure_table = pd.DataFrame(exposure_records)
        png = nb.plot_exposure_by_age(exposure_table, title_suffix=title_suffix)
        return {"chart": "exposure", "image_b64": _to_b64(png)}

    elif chart == "crude_smoothed":
        exposure_records = data.get("exposure_table")
        if not exposure_records:
            return {"erreur": "exposure_table manquant. Appeler builder.exposure d'abord."}
        exposure_table = pd.DataFrame(exposure_records)

        # Construire smoothed_dict depuis data store
        smoothed_dict = {}
        smoothed_records = data.get("smoothed_table")
        if smoothed_records:
            smoothed_df = pd.DataFrame(smoothed_records)
            qx_col = next((c for c in ("q_x_lisse", "qx") if c in smoothed_df.columns), None)
            method = data.get("smoothing", {}).get("method", "Lissé") if isinstance(data.get("smoothing"), dict) else "Lissé"
            if qx_col:
                smoothed_dict[method] = {
                    "ages": smoothed_df["age"].tolist(),
                    "qx_smoothed": smoothed_df[qx_col].tolist(),
                }

        # Enrichir avec smoothers_dict si plusieurs méthodes disponibles
        extra_smoothers = data.get("smoothers_dict") or {}
        for m_name, m_data in extra_smoothers.items():
            if m_name not in smoothed_dict and m_data.get("ages") and m_data.get("qx_smoothed"):
                smoothed_dict[m_name] = {
                    "ages": m_data["ages"],
                    "qx_smoothed": m_data["qx_smoothed"],
                }

        # Mode multi-méthodes : construire le graphique manuellement avec inset table
        comparison = None
        diag = data.get("diagnostics")
        if isinstance(diag, dict):
            comparison = diag.get("comparison")  # list of {method, AIC_poisson, BIC_poisson, MSE_vs_crude, n_non_monotone}
        best_method = (diag or {}).get("best_method") if isinstance(diag, dict) else None

        if len(smoothed_dict) > 1:
            import io
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np

            _BG    = "#FBF8F1"
            _GRID  = "#E8E3D8"
            _CRUDE = "#888888"
            _PALETTE = ["#2C5F8A", "#E67E22", "#27AE60", "#8E44AD", "#C0392B", "#16A085"]

            has_table = comparison is not None and len(comparison) > 0
            fig_h = 5.5 if not has_table else 7.5
            fig, ax = plt.subplots(figsize=(13, fig_h))
            fig.patch.set_facecolor(_BG)
            ax.set_facecolor(_BG)

            # Taux bruts
            qx_brut_col = next((c for c in ("q_x_brut", "qx_brut", "qx") if c in exposure_table.columns), None)
            if qx_brut_col:
                mask = exposure_table[qx_brut_col] > 0
                ax.scatter(
                    exposure_table.loc[mask, "age"],
                    exposure_table.loc[mask, qx_brut_col],
                    s=14, color=_CRUDE, alpha=0.55, label="Taux bruts", zorder=2,
                )

            # Courbes lissées
            for i, (m_name, m_data) in enumerate(smoothed_dict.items()):
                ages_m = m_data["ages"]
                qx_m   = m_data["qx_smoothed"]
                is_best = (m_name == best_method)
                lw = 2.8 if is_best else 1.5
                label = f"★ {m_name}" if is_best else m_name
                ax.plot(ages_m, qx_m,
                        color=_PALETTE[i % len(_PALETTE)],
                        linewidth=lw, label=label, zorder=3 + i)

            ax.set_yscale("log")
            ax.set_xlabel("Âge", fontsize=10)
            ax.set_ylabel("qx (échelle log)", fontsize=10)
            title = f"Comparaison des modèles de lissage{' — ' + title_suffix if title_suffix else ''}"
            ax.set_title(title, fontsize=11, loc="left")
            ax.legend(facecolor=_BG, edgecolor=_GRID, fontsize=9, framealpha=0.9)
            ax.grid(True, color=_GRID, linewidth=0.8, alpha=0.8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            # Tableau de métriques sous le graphique
            if has_table:
                col_labels = ["Méthode", "AIC", "BIC", "MSE", "Non-monotone"]
                rows = []
                row_colors = []
                for row in comparison:
                    m = row.get("method", "?")
                    aic = f"{row.get('AIC_poisson', float('nan')):.1f}"
                    bic = f"{row.get('BIC_poisson', float('nan')):.1f}"
                    mse = f"{row.get('MSE_vs_crude', float('nan')):.5f}"
                    nmon = str(row.get("n_non_monotone", "?"))
                    label = f"★ {m}" if m == best_method else m
                    rows.append([label, aic, bic, mse, nmon])
                    is_best_row = (m == best_method)
                    row_colors.append(["#d4edda"] * 5 if is_best_row else [_BG] * 5)

                table = ax.table(
                    cellText=rows,
                    colLabels=col_labels,
                    cellColours=row_colors,
                    bbox=[0.0, -0.42, 1.0, 0.35],
                    loc="bottom",
                )
                table.auto_set_font_size(False)
                table.set_fontsize(8.5)
                for (r, c), cell in table.get_celld().items():
                    cell.set_edgecolor(_GRID)
                    if r == 0:
                        cell.set_facecolor("#dce3ec")
                        cell.set_text_props(fontweight="bold")
                plt.subplots_adjust(bottom=0.40)

            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor=_BG)
            plt.close(fig)
            buf.seek(0)
            import base64 as _b64
            return {"chart": "crude_smoothed", "image_b64": _b64.b64encode(buf.read()).decode()}

        # Mode single-method : déléguer au notebook (comportement inchangé)
        png = nb.plot_crude_vs_smoothed(
            exposure_table,
            smoothed_dict=smoothed_dict,
            sexe=sexe,
            title_suffix=title_suffix,
        )
        return {"chart": "crude_smoothed", "image_b64": _to_b64(png)}

    elif chart == "smr":
        smr_data = data.get("smr") or data.get("diagnostics", {})
        if isinstance(smr_data, dict) and "smr_by_decade" not in smr_data:
            return {"erreur": "Données SMR manquantes. Appeler builder.diagnostics (function_name=smr) d'abord."}

        # Reconvertir smr_by_decade en DataFrame si c'est une liste
        if isinstance(smr_data.get("smr_by_decade"), list):
            smr_data = dict(smr_data)
            smr_data["smr_by_decade"] = pd.DataFrame(smr_data["smr_by_decade"])

        png = nb.plot_smr_by_age(smr_data, title_suffix=title_suffix)
        return {"chart": "smr", "image_b64": _to_b64(png)}

    elif chart == "ci_bands":
        exposure_records = data.get("exposure_table")
        if not exposure_records:
            return {"erreur": "exposure_table manquant. Appeler builder.exposure d'abord."}
        exposure_table = pd.DataFrame(exposure_records)

        # Fusionner taux lissés
        smoothed_records = data.get("smoothed_table")
        if smoothed_records:
            smth_df = pd.DataFrame(smoothed_records)
            qx_col = next((c for c in ("q_x_lisse", "qx") if c in smth_df.columns), None)
            if qx_col and qx_col not in exposure_table.columns:
                exposure_table = exposure_table.merge(smth_df[["age", qx_col]], on="age", how="left")

        ci_records = data.get("ci_table")
        ci_df = pd.DataFrame(ci_records) if ci_records else None

        png = nb.plot_confidence_bands(
            exposure_table,
            ci_result=ci_df,
            sexe=sexe,
            title_suffix=title_suffix,
        )
        return {"chart": "ci_bands", "image_b64": _to_b64(png)}

    elif chart == "survival_curve":
        exposure_records = data.get("exposure_table")
        if not exposure_records:
            return {"erreur": "exposure_table manquant. Appeler builder.exposure d'abord."}
        exposure_table = pd.DataFrame(exposure_records)

        smoothed_records = data.get("smoothed_table")
        if smoothed_records:
            smth_df = pd.DataFrame(smoothed_records)
            qx_col = next((c for c in ("q_x_lisse", "qx") if c in smth_df.columns), None)
            if qx_col and qx_col not in exposure_table.columns:
                exposure_table = exposure_table.merge(smth_df[["age", qx_col]], on="age", how="left")

        png = nb.plot_survival_curve(exposure_table, sexe=sexe, title_suffix=title_suffix)
        return {"chart": "survival_curve", "image_b64": _to_b64(png)}

    elif chart == "abatement_chart":
        abatement_records = data.get("abatement_table")
        if not abatement_records:
            return {"erreur": "abatement_table manquant. Appeler builder.benchmarking d'abord."}

        import io
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        _BG    = "#FBF8F1"
        _GRID  = "#E8E3D8"
        _BLUE  = "#2C5F8A"
        _ORANGE = "#E67E22"
        _RED   = "#C0392B"
        _GREEN = "#27AE60"

        ab_df = pd.DataFrame(abatement_records).dropna(subset=["abatement_factor"])
        ages    = ab_df["age"].tolist()
        factors = ab_df["abatement_factor"].tolist()
        bar_colors = [_BLUE if f <= 1.0 else _ORANGE for f in factors]

        fig, ax = plt.subplots(figsize=(13, 5))
        fig.patch.set_facecolor(_BG)
        ax.set_facecolor(_BG)
        ax.bar(ages, factors, color=bar_colors, alpha=0.80, width=0.75, edgecolor="none")
        ax.axhline(y=1.0, color=_RED, linewidth=1.8, linestyle="--", label="Référence (α = 1.0)")
        smr_global = data.get("smr_global")
        if smr_global is not None:
            ax.axhline(y=smr_global, color=_GREEN, linewidth=1.4, linestyle=":",
                       label=f"SMR global = {smr_global:.3f}")
        ref = data.get("reference_name", "TH0002")
        ax.set_xlabel("Âge", fontsize=10)
        ax.set_ylabel("Facteur d'abattement (α)", fontsize=10)
        ax.set_title(
            f"Facteurs d'abattement vs {ref}{' — ' + title_suffix if title_suffix else ''}",
            fontsize=11, loc="left",
        )
        ax.legend(facecolor=_BG, edgecolor=_GRID, fontsize=9, framealpha=0.9)
        ax.grid(True, color=_GRID, linewidth=0.8, alpha=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor=_BG)
        plt.close(fig)
        buf.seek(0)
        import base64
        return {"chart": "abatement_chart", "image_b64": base64.b64encode(buf.read()).decode()}

    # ── deaths_by_age : décès observés par âge, séries H/F ───────────────────
    elif chart == "deaths_by_age":
        c = _ctx(data)
        male_d   = c.get("deaths_by_age_male")   or {}
        female_d = c.get("deaths_by_age_female") or {}

        # Fallback : colonne D_x de l'exposure_table (sans segmentation sexe)
        if not male_d and not female_d:
            records = data.get("exposure_table")
            if not records:
                return {"erreur": "exposure_table manquant. Appeler builder.exposure d'abord."}
            exp_df = pd.DataFrame(records)
            if "D_x" in exp_df.columns:
                male_d = {str(int(r["age"])): r["D_x"] for _, r in exp_df.iterrows()}

        if not male_d and not female_d:
            return {"erreur": "deaths_by_age_male / deaths_by_age_female manquants."}

        fig, ax = _std_fig()
        if male_d:
            ages, vals = _ages_vals(male_d)
            ax.plot(ages, vals, color=_MALE, linewidth=1.8, label="Hommes")
        if female_d:
            ages, vals = _ages_vals(female_d)
            ax.plot(ages, vals, color=_FEMALE, linewidth=1.8, label="Femmes")

        ax.set_xlabel("Âge", fontsize=10)
        ax.set_ylabel("Nombre de décès", fontsize=10)
        ax.set_title(
            f"Décès par âge{' — ' + title_suffix if title_suffix else ''}",
            fontsize=11, loc="left",
        )
        ax.legend(facecolor=_BG, edgecolor=_GRID, fontsize=9)
        return {"chart": "deaths_by_age", "image_b64": _std_save(fig)}

    # ── obs_vs_modeled : décès observés vs modélisés + bande IC + anomalies ──
    elif chart == "obs_vs_modeled":
        c = _ctx(data)
        obs     = c.get("observed_deaths_by_age")  or {}
        mod     = c.get("modeled_deaths_by_age")   or {}
        ci_low  = c.get("ci_lower_by_age")         or {}
        ci_high = c.get("ci_upper_by_age")         or {}

        # Fallback depuis validation.ci_table
        if not obs or not mod:
            val = data.get("validation") or {}
            ci_table = val.get("ci_table") if isinstance(val, dict) else []
            if ci_table:
                obs     = obs  or {str(r["age"]): r.get("D_x", 0)             for r in ci_table}
                mod     = mod  or {str(r["age"]): r.get("expected_deaths", 0)  for r in ci_table}
                ci_low  = ci_low  or {str(r["age"]): r.get("ci_lower", 0)     for r in ci_table}
                ci_high = ci_high or {str(r["age"]): r.get("ci_upper", 0)     for r in ci_table}

        if not obs or not mod:
            return {"erreur": "observed_deaths_by_age / modeled_deaths_by_age manquants. Appeler builder.validation d'abord."}

        ages_obs, vals_obs = _ages_vals(obs)
        ages_mod, vals_mod = _ages_vals(mod)

        fig, ax = _std_fig(figsize=(13, 6))

        # Bande IC
        if ci_low and ci_high:
            ages_ci, low_vals  = _ages_vals(ci_low)
            _,       high_vals = _ages_vals(ci_high)
            ax.fill_between(ages_ci, low_vals, high_vals,
                            alpha=0.25, color=_CI, label=f"IC {c.get('confidence_interval_level', 95)}%")

        ax.plot(ages_mod, vals_mod, color=_RED,    linewidth=2.0, label="Décès modélisés")
        ax.plot(ages_obs, vals_obs, color="#2C3E50", linewidth=1.5,
                linestyle="--", label="Décès observés")

        # Anomalies : points où observé > IC supérieur
        if ci_high:
            ages_ci_h, high_vals = _ages_vals(ci_high)
            ci_h_map = dict(zip(ages_ci_h, high_vals))
            anom_x = [a for a, v in zip(ages_obs, vals_obs) if v > ci_h_map.get(a, float("inf"))]
            anom_y = [obs[str(a)] for a in anom_x if str(a) in obs]
            if anom_x:
                ax.scatter(anom_x, anom_y, color=_RED, s=40, zorder=5,
                           marker="o", label="Anomalie (hors IC)")

        ax.set_xlabel("Âge", fontsize=10)
        ax.set_ylabel("Nombre de décès", fontsize=10)
        ax.set_title(
            f"Décès observés vs modélisés{' — ' + title_suffix if title_suffix else ''}",
            fontsize=11, loc="left",
        )
        ax.legend(facecolor=_BG, edgecolor=_GRID, fontsize=9)
        return {"chart": "obs_vs_modeled", "image_b64": _std_save(fig)}

    # ── rate_ratio : ratio taux courant / taux précédent par âge ─────────────
    elif chart == "rate_ratio":
        c = _ctx(data)
        ratios = c.get("rate_ratio_current_vs_prior") or {}

        # Fallback depuis precedent_comparison
        if not ratios:
            prec = data.get("precedent_comparison") or {}
            if isinstance(prec, dict):
                ct = prec.get("comparison_table") or []
                ratios = {str(r.get("age", "")): r.get("ratio", 1.0) for r in ct if r.get("age")}

        if not ratios:
            return {"erreur": "rate_ratio_current_vs_prior manquant. Appeler builder.precedent_comparison d'abord."}

        ages, vals = _ages_vals(ratios)
        fig, ax = _std_fig()
        ax.plot(ages, vals, color=_BLUE, linewidth=2.0, label="Ratio courant / précédent")
        ax.axhline(y=1.0, color=_CRUDE, linewidth=1.4, linestyle="--", label="Pas de changement (= 1.0)")

        # Signaler les déviations > 10%
        anom_x = [a for a, v in zip(ages, vals) if abs(v - 1.0) > 0.10]
        anom_y = [ratios[str(a)] for a in anom_x if str(a) in ratios]
        if anom_x:
            ax.scatter(anom_x, anom_y, color=_ORANGE, s=40, zorder=5,
                       marker="^", label="Écart > 10%")

        ax.set_xlabel("Âge", fontsize=10)
        ax.set_ylabel("Ratio taux d'expérience (courant / précédent)", fontsize=10)
        ax.set_title(
            f"Comparaison avec la table précédente{' — ' + title_suffix if title_suffix else ''}",
            fontsize=11, loc="left",
        )
        ax.legend(facecolor=_BG, edgecolor=_GRID, fontsize=9)
        return {"chart": "rate_ratio", "image_b64": _std_save(fig)}

    # ── discount_line : facteurs d'abattement en courbe (vs barres) ──────────
    elif chart == "discount_line":
        c = _ctx(data)
        discounts = c.get("discount_by_age") or {}

        # Fallback depuis benchmarking.abatement_table
        if not discounts:
            bm = data.get("benchmarking") or {}
            ab_table = bm.get("abatement_table") if isinstance(bm, dict) else None
            if ab_table:
                discounts = {str(r["age"]): r.get("abatement_factor", 0) for r in ab_table}

        if not discounts:
            return {"erreur": "discount_by_age manquant. Appeler builder.benchmarking d'abord."}

        ages, vals = _ages_vals(discounts)
        ref  = (data.get("benchmarking") or {}).get("reference_name") or \
               (data.get("template_context") or {}).get("baseline_regulatory_table") or "TH0002"
        smr_global = data.get("smr_global") or (data.get("benchmarking") or {}).get("smr_global")

        fig, ax = _std_fig()
        ax.plot(ages, vals, color=_BLUE, linewidth=2.0, label="Facteur d'abattement")
        ax.axhline(y=1.0, color=_CRUDE, linewidth=1.4, linestyle="--",
                   label=f"Référence (= {ref})")
        if smr_global is not None:
            ax.axhline(y=smr_global, color=_GREEN, linewidth=1.4, linestyle=":",
                       label=f"SMR global = {smr_global:.3f}")

        ax.set_xlabel("Âge", fontsize=10)
        ax.set_ylabel("Facteur d'abattement (α)", fontsize=10)
        ax.set_title(
            f"Abattements vs {ref}{' — ' + title_suffix if title_suffix else ''}",
            fontsize=11, loc="left",
        )
        ax.legend(facecolor=_BG, edgecolor=_GRID, fontsize=9)
        return {"chart": "discount_line", "image_b64": _std_save(fig)}

    else:
        return {"erreur": (
            f"chart inconnu : '{chart}'. Valeurs : "
            "exposure, crude_smoothed, smr, ci_bands, survival_curve, abatement_chart, "
            "deaths_by_age, obs_vs_modeled, rate_ratio, discount_line"
        )}
