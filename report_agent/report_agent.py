"""
report_agent/report_agent.py
Agent rédacteur LLM — génère un rapport actuariel narratif à partir des résultats
de l'agent de calcul (steps, summary, figures).

Générique : fonctionne pour tables de mortalité, provisionnement non-vie, VIF, etc.
Le domaine est paramétré via domain_label + kb_path (fichier JSON de guidelines).

Architecture :
    _run_agent_in_thread (canvas_app.py)
        └─ generate_narrative_report()   ← point d'entrée principal
               ├─ _load_kb_guidelines()  ← charge le JSON de guidelines
               ├─ _build_section_context() ← prépare les données par section
               ├─ _write_section()       ← appel LLM (gpt-4o-mini) par section
               ├─ _generate_report_figures() ← graphiques produits par le rédacteur
               └─ _assemble_pdf()        ← reportlab : narrative + figures + tables

Séparation des rôles :
    - Agent de calcul : produit les données (DataFrames, métriques, tableaux).
                        Il peut générer des graphiques techniques (debug/annexe).
    - Agent rédacteur (ce module) : produit les graphiques de présentation
                        (mise en forme professionnelle, couleurs, légendes soignées)
                        en re-exploitant les données brutes des steps.
"""
from __future__ import annotations

import base64
import io
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from report_agent.prompts import SYSTEM_PROMPT, SECTION_PROMPTS


# ═════════════════════════════════════════════════════════════════════════════
# GRAPHIQUES PROFESSIONNELS — produits par le rédacteur, pas l'agent de calcul
# ═════════════════════════════════════════════════════════════════════════════

# Palette professionnelle (inspirée du rapport de référence AF8796)
_C_BLUE    = "#1A3A5C"   # bleu marine — courbe lissée, titres
_C_GREY    = "#888888"   # gris — taux bruts, éléments secondaires
_C_RED     = "#C0392B"   # rouge — alertes, SMR > 1
_C_GREEN   = "#27AE60"   # vert — SMR ≤ 1, adéquation
_C_LBLUE   = "#4A90D9"   # bleu clair — référence externe
_C_ORANGE  = "#E67E22"   # orange — IC, bandes d'incertitude


def render_equation(latex_str: str, fontsize: int = 13,
                    fig_w: float = 7.0, fig_h: float = 0.75) -> bytes:
    """Rend une expression LaTeX en PNG via matplotlib mathtext.

    Args:
        latex_str: Expression LaTeX (sans les $ délimiteurs — ils sont ajoutés).
                   Ex : r"\\hat{q}_x = \\frac{D_x}{E_x}"
        fontsize:  Taille de la police (pt).
        fig_w/h:   Dimensions de la figure en pouces (ajustées automatiquement).

    Returns:
        Bytes PNG transparents, prêts à intégrer dans ReportLab via Image().
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.text(
        0.5, 0.5, f"${latex_str}$",
        ha="center", va="center",
        fontsize=fontsize,
        transform=ax.transAxes,
        fontfamily="DejaVu Serif",
    )
    ax.axis("off")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight",
                transparent=True, pad_inches=0.05)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ─── Extraction des données depuis les steps de l'agent de calcul ─────────────

def _parse_df_from_text(text: str):
    """Tente de parser un DataFrame pandas depuis un texte tabulaire.

    Gère : pandas to_string() / to_string(index=False) (fixed-width), TSV.
    Retourne None si l'analyse échoue ou si le résultat est trop petit.
    """
    try:
        import pandas as pd
    except ImportError:
        return None

    if not text or len(text.strip()) < 20:
        return None
    lines = [ln for ln in text.strip().split("\n") if ln.strip()]
    if len(lines) < 2:
        return None

    # Essai 1 : TSV
    if "\t" in lines[0]:
        try:
            df = pd.read_csv(io.StringIO(text), sep="\t")
            if df.shape[0] >= 1 and df.shape[1] >= 2:
                return _drop_index_col(df)
        except Exception:
            pass

    # Essai 2 : fixed-width (pd.to_string() — format le plus courant en sortie agent)
    try:
        df = pd.read_fwf(io.StringIO(text))
        if df.shape[0] >= 1 and df.shape[1] >= 2:
            df.columns = [str(c).strip() for c in df.columns]
            return _drop_index_col(df)
    except Exception:
        pass

    # Essai 3 : espaces multiples (séparateur ≥ 2 espaces)
    try:
        # Normaliser en retirant les espaces initiaux de chaque ligne
        cleaned = "\n".join(ln.lstrip() for ln in text.strip().split("\n"))
        df = pd.read_csv(io.StringIO(cleaned), sep=r"\s{2,}", engine="python")
        if df.shape[0] >= 1 and df.shape[1] >= 2:
            df.columns = [str(c).strip() for c in df.columns]
            return _drop_index_col(df)
    except Exception:
        pass

    # Essai 4 : délimiteur espace simple
    try:
        df = pd.read_csv(io.StringIO(text), delim_whitespace=True)
        if df.shape[0] >= 1 and df.shape[1] >= 2:
            return _drop_index_col(df)
    except Exception:
        pass

    return None


def _drop_index_col(df):
    """Supprime la première colonne si c'est un index numérique (0, 1, 2 …)."""
    if df.empty or df.shape[1] < 2:
        return df
    first = str(df.columns[0]).strip()
    # Colonne sans nom ou dont TOUTES les valeurs sont des entiers croissants
    if first in ("", "Unnamed: 0"):
        return df.iloc[:, 1:]
    try:
        vals = df.iloc[:, 0].astype(float)
        if (vals == range(len(vals))).all() or (vals == range(1, len(vals) + 1)).all():
            return df.iloc[:, 1:]
    except Exception:
        pass
    return df


def _find_exposure_table(steps: list[dict]):
    """Cherche le tableau d'exposition (age, E_x, D_x, qx_brut) dans les steps."""
    _REQUIRED = {"age"}
    _RATE_COLS = {"e_x", "ex", "exposition", "exposure"}
    for step in steps:
        for do in step.get("display_outputs", []):
            df = _parse_df_from_text(do.get("text", ""))
            if df is None:
                continue
            cols_low = {c.lower().strip() for c in df.columns}
            if _REQUIRED.issubset(cols_low) and _RATE_COLS & cols_low:
                # Normalise les noms de colonnes
                rename = {}
                for c in df.columns:
                    cl = c.lower().strip()
                    if cl in ("e_x", "ex", "exposition", "exposure"):
                        rename[c] = "E_x"
                    elif cl in ("d_x", "dx", "deces", "deaths"):
                        rename[c] = "D_x"
                    elif cl in ("q_x_brut", "qx_brut", "mu_x", "crude_rate", "q_brut"):
                        rename[c] = "q_brut"
                    elif cl in ("qx_lisse", "q_lisse", "qx_smooth", "smoothed", "qx"):
                        rename[c] = "q_lisse"
                    elif cl in ("ic_inf", "ci_lower", "ci_inf", "lower"):
                        rename[c] = "IC_inf"
                    elif cl in ("ic_sup", "ci_upper", "ci_sup", "upper"):
                        rename[c] = "IC_sup"
                    elif cl in ("d_exp", "dexp", "expected"):
                        rename[c] = "D_exp"
                    elif cl in ("rapport_oa", "o/a", "oa", "ratio"):
                        rename[c] = "OA"
                df = df.rename(columns=rename)
                try:
                    df["age"] = df["age"].astype(float)
                    df = df.sort_values("age").reset_index(drop=True)
                except Exception:
                    pass
                return df
    return None


def _find_smr_data(steps: list[dict]):
    """Cherche les données SMR (smr, IC) dans les steps."""
    for step in steps:
        for do in step.get("display_outputs", []):
            df = _parse_df_from_text(do.get("text", ""))
            if df is None:
                continue
            cols_low = {c.lower().strip() for c in df.columns}
            if "smr" in cols_low and (
                {"ic_inf", "ci_lower", "lower", "ic_sup", "ci_upper", "upper"} & cols_low
            ):
                rename = {}
                for c in df.columns:
                    cl = c.lower().strip()
                    if cl == "smr":
                        rename[c] = "SMR"
                    elif cl in ("ic_inf", "ci_lower", "lower", "ci_inf"):
                        rename[c] = "IC_inf"
                    elif cl in ("ic_sup", "ci_upper", "upper", "ci_sup"):
                        rename[c] = "IC_sup"
                    elif cl in ("decade", "decennie", "tranche"):
                        rename[c] = "tranche"
                    elif cl == "age":
                        rename[c] = "age"
                df = df.rename(columns=rename)
                return df
    return None


def _find_reference_data(steps: list[dict]):
    """Cherche les données de comparaison (qx_exp, qx_ref) dans les steps."""
    for step in steps:
        for do in step.get("display_outputs", []):
            df = _parse_df_from_text(do.get("text", ""))
            if df is None:
                continue
            cols_low = {c.lower().strip() for c in df.columns}
            if "age" in cols_low and (
                {"qx_ref", "q_ref"} & cols_low
            ) and (
                {"qx_lisse", "q_lisse", "qx_exp", "q_exp", "qx"} & cols_low
            ):
                rename = {}
                for c in df.columns:
                    cl = c.lower().strip()
                    if cl in ("qx_ref", "q_ref"):
                        rename[c] = "q_ref"
                    elif cl in ("qx_exp", "q_exp", "qx_lisse", "q_lisse", "qx"):
                        rename[c] = "q_exp"
                df = df.rename(columns=rename)
                return df
    return None


# ─── Fonctions de génération des graphiques ───────────────────────────────────

def _apply_report_style(ax, title: str = "") -> None:
    """Style professionnel commun à tous les graphiques du rapport."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CCCCCC")
    ax.spines["bottom"].set_color("#CCCCCC")
    ax.grid(True, axis="y", alpha=0.35, linestyle="--", color="#AAAAAA")
    ax.tick_params(colors="#444444", labelsize=9)
    ax.xaxis.label.set_color("#444444")
    ax.yaxis.label.set_color("#444444")
    if title:
        ax.set_title(title, fontsize=11, color=_C_BLUE, fontweight="bold", pad=8)


def chart_crude_vs_smoothed(exposure_df) -> bytes | None:
    """Graphique : taux bruts (scatter) vs taux lissés (ligne) par âge (en ‰)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    if exposure_df is None or "age" not in exposure_df.columns:
        return None
    has_crude  = "q_brut" in exposure_df.columns
    has_smooth = "q_lisse" in exposure_df.columns
    if not has_crude and not has_smooth:
        return None

    fig, ax = plt.subplots(figsize=(9, 4.5))
    age = exposure_df["age"].values

    if has_crude:
        crude = exposure_df["q_brut"].values * 1000
        ax.scatter(age, crude, color=_C_GREY, s=20, alpha=0.65,
                   label="Taux bruts q̂ₓ (‰)", zorder=3)

    if has_smooth:
        smooth = exposure_df["q_lisse"].values * 1000
        ax.plot(age, smooth, color=_C_BLUE, linewidth=2.2,
                label="Taux lissés qₓ (‰)", zorder=4)

    # IC si disponibles
    if "IC_inf" in exposure_df.columns and "IC_sup" in exposure_df.columns:
        ci_lo = exposure_df["IC_inf"].values * 1000
        ci_hi = exposure_df["IC_sup"].values * 1000
        ax.fill_between(age, ci_lo, ci_hi, alpha=0.15, color=_C_BLUE,
                        label="IC 95 %")

    ax.set_xlabel("Âge (années révolues)", fontsize=10)
    ax.set_ylabel("Taux de mortalité (‰)", fontsize=10)
    ax.legend(framealpha=0.9, fontsize=9, loc="upper left")
    _apply_report_style(ax, "Taux bruts et taux lissés par âge")

    plt.tight_layout(pad=0.6)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def chart_smr_by_group(smr_df, exposure_df=None) -> bytes | None:
    """Graphique : SMR par décennie d'âge avec intervalles de confiance."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    if smr_df is None or "SMR" not in smr_df.columns:
        return None

    # Identifier la colonne de groupement (tranche ou âge)
    group_col = "tranche" if "tranche" in smr_df.columns else (
        "age" if "age" in smr_df.columns else None
    )
    if group_col is None:
        return None

    groups = smr_df[group_col].astype(str).values
    smr    = smr_df["SMR"].values.astype(float)
    has_ci = "IC_inf" in smr_df.columns and "IC_sup" in smr_df.columns
    ci_lo  = smr_df["IC_inf"].values.astype(float) if has_ci else smr - 0.05
    ci_hi  = smr_df["IC_sup"].values.astype(float) if has_ci else smr + 0.05

    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(groups))

    colors = [_C_RED if v > 1 else _C_GREEN for v in smr]
    bars = ax.bar(x, smr, color=colors, alpha=0.75, width=0.55, zorder=3)

    # Barres d'erreur pour IC
    err_lo = smr - ci_lo
    err_hi = ci_hi - smr
    ax.errorbar(x, smr, yerr=[err_lo, err_hi], fmt="none",
                color="#333333", capsize=4, linewidth=1.2, zorder=4)

    # Ligne SMR = 1
    ax.axhline(1.0, color=_C_BLUE, linewidth=1.5, linestyle="--",
               label="SMR = 1 (référence)")

    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("SMR (rapport d'expérience)", fontsize=10)
    ax.legend(fontsize=9)
    _apply_report_style(ax, "SMR par groupe d'âge avec IC 95 %")

    plt.tight_layout(pad=0.6)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def chart_observed_vs_expected(exposure_df) -> bytes | None:
    """Graphique : décès observés vs modélisés par décennie d'âge."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    if exposure_df is None or "D_x" not in exposure_df.columns:
        return None
    has_expected = "D_exp" in exposure_df.columns
    if not has_expected:
        return None

    df = exposure_df.copy()
    try:
        df["decade"] = (df["age"] // 10 * 10).astype(int)
    except Exception:
        return None

    agg = df.groupby("decade").agg(
        D_obs=("D_x", "sum"),
        D_exp=("D_exp", "sum"),
    ).reset_index()

    x     = np.arange(len(agg))
    w     = 0.35
    labs  = [f"{int(d)}-{int(d)+9}" for d in agg["decade"].values]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x - w/2, agg["D_obs"], w, color=_C_BLUE,  alpha=0.8, label="Décès observés", zorder=3)
    ax.bar(x + w/2, agg["D_exp"], w, color=_C_ORANGE, alpha=0.8, label="Décès attendus", zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(labs, fontsize=9)
    ax.set_ylabel("Nombre de décès", fontsize=10)
    ax.legend(fontsize=9)
    _apply_report_style(ax, "Décès observés vs modélisés par décennie d'âge")

    plt.tight_layout(pad=0.6)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def chart_exposure_by_age(exposure_df) -> bytes | None:
    """Graphique : exposition centrale Eₓ par âge."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if exposure_df is None or "E_x" not in exposure_df.columns:
        return None

    fig, ax = plt.subplots(figsize=(9, 3.5))
    age = exposure_df["age"].values
    ex  = exposure_df["E_x"].values

    ax.bar(age, ex, width=0.8, color=_C_BLUE, alpha=0.7, zorder=3)
    ax.set_xlabel("Âge (années révolues)", fontsize=10)
    ax.set_ylabel("Exposition Eₓ (années-personnes)", fontsize=10)
    _apply_report_style(ax, "Exposition centrale par âge")

    plt.tight_layout(pad=0.6)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def chart_comparison_with_reference(exposure_df, ref_df) -> bytes | None:
    """Graphique : comparaison table d'expérience vs table de référence."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if exposure_df is None or ref_df is None:
        return None
    if "q_lisse" not in exposure_df.columns or "q_ref" not in ref_df.columns:
        return None
    if "age" not in ref_df.columns:
        return None

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(exposure_df["age"], exposure_df["q_lisse"] * 1000,
            color=_C_BLUE, linewidth=2.2, label="Table d'expérience qₓ (‰)", zorder=4)
    ax.plot(ref_df["age"], ref_df["q_ref"] * 1000,
            color=_C_LBLUE, linewidth=1.8, linestyle="--",
            label="Table de référence qₓʳᵉᶠ (‰)", zorder=3)

    ax.set_xlabel("Âge (années révolues)", fontsize=10)
    ax.set_ylabel("Taux de mortalité (‰)", fontsize=10)
    ax.legend(fontsize=9)
    _apply_report_style(ax, "Positionnement vs table de référence")

    plt.tight_layout(pad=0.6)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _generate_demo_charts(methodology: dict | None = None) -> dict[str, bytes]:
    """Génère 5 graphiques à partir de données numpy synthétiques réalistes.

    Utilisé quand _generate_report_figures() ne trouve pas de données exploitables
    (simulation encoder, pipeline sans display_outputs parsables).

    Args:
        methodology: Dict optionnel issu du template (age_min, age_max, etc.).

    Returns:
        Dict clé → bytes PNG. Mêmes clés que _generate_report_figures().
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    rng = np.random.default_rng(42)
    meth = methodology or {}
    age_min = int(meth.get("age_min", 25))
    age_max = int(meth.get("age_max", 85))
    ages = np.arange(age_min, age_max + 1, dtype=float)
    n = len(ages)

    # Taux de mortalité Gompertz-Makeham réaliste
    A, B, c, x0 = 0.0003, 0.000025, 0.10, 50.0
    q_ref = np.clip(A + B * np.exp(c * (ages - x0)), 5e-5, 0.9)

    # Exposition décroissante avec l'âge
    E_x = rng.integers(300, 3500, size=n).astype(float)
    E_x *= np.linspace(1.0, 0.25, n)
    E_x = np.maximum(E_x, 20.0)

    # Décès observés (Poisson)
    D_x = rng.poisson(q_ref * E_x).astype(float)
    q_brut = np.where(E_x > 0, D_x / E_x, q_ref)

    # Lissage (moyenne mobile gaussienne)
    kernel = np.exp(-0.5 * ((np.arange(-4, 5, dtype=float)) / 2.0) ** 2)
    kernel /= kernel.sum()
    q_lisse = np.convolve(q_brut, kernel, mode="same")
    q_lisse = np.maximum(q_lisse, 5e-5)

    # IC 95 %
    IC_inf = np.maximum(0.0, q_lisse - 1.96 * np.sqrt(q_lisse / E_x))
    IC_sup = q_lisse + 1.96 * np.sqrt(q_lisse / E_x)

    # Tranches décennales
    decades = [d for d in range(20, 91, 10) if d >= age_min and d < age_max]
    if not decades:
        decades = [age_min]
    dec_labels = [f"{d}-{d+9}" for d in decades]

    figs: dict[str, bytes] = {}

    def _save(fig) -> bytes:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return buf.read()

    # Figure 1 — Exposition par âge
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(ages, E_x / 1000, color=_C_BLUE, alpha=0.72, width=0.8, zorder=3)
    ax.set_xlabel("Age (annees revolues)", fontsize=10)
    ax.set_ylabel("Exposition (milliers d'annees-personnes)", fontsize=10)
    _apply_report_style(ax, "Exposition centrale par age")
    plt.tight_layout(pad=0.6)
    figs["exposure"] = _save(fig)

    # Figure 2 — Taux bruts vs lissés
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.scatter(ages, q_brut * 1000, color=_C_GREY, s=18, alpha=0.65,
               label="Taux bruts (‰)", zorder=3)
    ax.plot(ages, q_lisse * 1000, color=_C_BLUE, linewidth=2.2,
            label="Taux lisses (‰)", zorder=4)
    ax.fill_between(ages, IC_inf * 1000, IC_sup * 1000,
                    alpha=0.15, color=_C_BLUE, label="IC 95 %")
    ax.set_xlabel("Age (annees revolues)", fontsize=10)
    ax.set_ylabel("Taux de mortalite (‰)", fontsize=10)
    ax.legend(framealpha=0.9, fontsize=9, loc="upper left")
    _apply_report_style(ax, "Taux bruts et taux lisses par age")
    plt.tight_layout(pad=0.6)
    figs["rates"] = _save(fig)

    # Figure 3 — SMR par décennie
    smr_vals, ci_lo, ci_hi = [], [], []
    for d in decades:
        mask = (ages >= d) & (ages < d + 10)
        d_obs = D_x[mask].sum()
        d_exp = (q_ref[mask] * E_x[mask]).sum()
        smr = d_obs / max(d_exp, 1.0)
        ci = 1.96 / np.sqrt(max(d_exp, 1.0))
        smr_vals.append(smr)
        ci_lo.append(max(0.0, smr - ci * smr))
        ci_hi.append(smr + ci * smr)
    smr_arr = np.array(smr_vals)
    err_lo = smr_arr - np.array(ci_lo)
    err_hi = np.array(ci_hi) - smr_arr

    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(decades))
    bar_colors = [_C_RED if v > 1 else _C_GREEN for v in smr_vals]
    ax.bar(x, smr_arr, color=bar_colors, alpha=0.75, width=0.55, zorder=3)
    ax.errorbar(x, smr_arr, yerr=[err_lo, err_hi],
                fmt="none", color="#333333", capsize=4, linewidth=1.2, zorder=4)
    ax.axhline(1.0, color=_C_BLUE, linewidth=1.5, linestyle="--",
               label="SMR = 1 (reference)")
    ax.set_xticks(x)
    ax.set_xticklabels(dec_labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("SMR", fontsize=10)
    ax.legend(fontsize=9)
    _apply_report_style(ax, "SMR par groupe d'age avec IC 95 %")
    plt.tight_layout(pad=0.6)
    figs["smr"] = _save(fig)

    # Figure 4 — Observés vs Attendus par décennie
    d_obs_dec = np.array([D_x[(ages >= d) & (ages < d+10)].sum() for d in decades])
    d_exp_dec = np.array([(q_ref * E_x)[(ages >= d) & (ages < d+10)].sum() for d in decades])
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(decades))
    w = 0.35
    ax.bar(x - w/2, d_obs_dec, width=w, color=_C_BLUE, alpha=0.75, label="Observes")
    ax.bar(x + w/2, d_exp_dec, width=w, color=_C_GREY, alpha=0.75, label="Attendus (modele)")
    ax.set_xticks(x)
    ax.set_xticklabels(dec_labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Nombre de deces", fontsize=10)
    ax.legend(fontsize=9)
    _apply_report_style(ax, "Deces observes vs modélisés par décennie")
    plt.tight_layout(pad=0.6)
    figs["oa"] = _save(fig)

    # Figure 5 — Comparaison table construite / référence
    abattement = q_lisse / np.maximum(q_ref, 1e-10)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(ages, q_ref * 1000, color=_C_LBLUE, linewidth=1.8, linestyle="--",
            label="Reference externe (‰)")
    ax.plot(ages, q_lisse * 1000, color=_C_BLUE, linewidth=2.2,
            label="Table construite (‰)")
    ax2 = ax.twinx()
    ax2.plot(ages, abattement, color=_C_ORANGE, linewidth=1.5, alpha=0.7,
             linestyle=":", label="Abattement")
    ax2.set_ylabel("Abattement", fontsize=9, color=_C_ORANGE)
    ax2.tick_params(axis="y", colors=_C_ORANGE)
    ax2.set_ylim(0.0, 2.0)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper left")
    ax.set_xlabel("Age (annees revolues)", fontsize=10)
    ax.set_ylabel("Taux de mortalite (‰)", fontsize=10)
    _apply_report_style(ax, "Table d'experience vs table de reference")
    plt.tight_layout(pad=0.6)
    figs["comparison"] = _save(fig)

    return figs


def _generate_report_figures(steps: list[dict]) -> dict[str, bytes]:
    """Génère l'ensemble des graphiques du rapport à partir des données des steps.

    Returns:
        Dict clé → bytes PNG. Clés possibles :
            "exposure"    — exposition par âge
            "rates"       — taux bruts vs lissés
            "smr"         — SMR par groupe
            "oa"          — observés vs attendus
            "comparison"  — comparaison avec référence
    """
    exposure_df = _find_exposure_table(steps)
    smr_df      = _find_smr_data(steps)
    ref_df      = _find_reference_data(steps)

    figs: dict[str, bytes] = {}

    for key, fn, *args in [
        ("exposure",   chart_exposure_by_age,         exposure_df),
        ("rates",      chart_crude_vs_smoothed,        exposure_df),
        ("smr",        chart_smr_by_group,             smr_df, exposure_df),
        ("oa",         chart_observed_vs_expected,     exposure_df),
        ("comparison", chart_comparison_with_reference, exposure_df, ref_df),
    ]:
        try:
            result = fn(*args)
            if result is not None:
                figs[key] = result
        except Exception:
            pass

    return figs


# ── Modèle LLM utilisé pour la rédaction ──────────────────────────────────────
_WRITER_MODEL = "gpt-4o"
_TEMPERATURE = 0.4          # légère créativité narrative, mais rester factuel

# Tokens par section — doublés pour permettre des développements complets
_MAX_TOKENS_BY_SECTION: dict[str, int] = {
    "contexte":       1200,
    "donnees":        1400,
    "methodologie":  2800,   # formules complètes, paramètres, conditions d'application
    "resultats":     2400,   # tableaux de validation, IC, backtesting
    "positionnement": 1800,
    "conclusion":     1400,
}

# ── Sections du rapport dans l'ordre ─────────────────────────────────────────
_SECTION_ORDER = [
    ("contexte",       "1. Contexte et objet de l'analyse"),
    ("donnees",        "2. Données et statistiques descriptives"),
    ("methodologie",   "3. Méthodologie de construction"),
    ("resultats",      "4. Résultats et validation statistique"),
    ("positionnement", "5. Positionnement et comparaison"),
    ("conclusion",     "6. Conclusion et recommandations de suivi"),
]

# ─────────────────────────────────────────────────────────────────────────────
# Mapping sections template → clés hardcodées (pour placement figures/tableaux)
# ─────────────────────────────────────────────────────────────────────────────

_SECTION_KEY_PATTERNS = {
    "contexte":       ["contexte", "objet", "introduction", "présentation", "périmètre"],
    "donnees":        ["données", "statistiques", "descriptives", "source", "portefeuille", "effectif"],
    "methodologie":   ["méthod", "construction", "lissage", "estimat", "modèle", "approche"],
    "resultats":      ["résult", "validation", "test", "indicateur", "backtesting", "mesure"],
    "positionnement": ["positionnement", "comparaison", "benchmark", "référence", "abattement"],
    "conclusion":     ["conclusion", "recommandation", "suivi", "synthèse finale"],
}


def _map_template_sections(template_sections: list[dict]) -> list[tuple[str, str, str]]:
    """Mappe les sections du template vers les clés hardcodées pour placement figures/tableaux.

    Returns:
        Liste de (sec_id, sec_title, mapped_hardcoded_key).
    """
    result = []
    for s in template_sections:
        sid = s.get("id", "")
        title = s.get("title", sid)
        t_lower = title.lower()
        mapped = "resultats"  # défaut
        for key, patterns in _SECTION_KEY_PATTERNS.items():
            if any(p in t_lower for p in patterns):
                mapped = key
                break
        result.append((sid, title, mapped))
    return result


def _build_data_injection(steps: list[dict], summary: str) -> str:
    """Agrège toutes les données disponibles en un bloc texte unique pour l'injection."""
    parts = []

    seen = set()
    for s in steps:
        for do in s.get("display_outputs", []):
            txt = do.get("text", "") if isinstance(do, dict) else str(do)
            lbl = do.get("label", "") if isinstance(do, dict) else ""
            key = txt[:80]
            if txt and len(txt.strip()) > 30 and key not in seen:
                seen.add(key)
                header = f"[{lbl}]" if lbl else "[Tableau de résultats]"
                # Tronquer les tableaux trop longs
                truncated = txt[:1200] + ("\n…(tronqué)" if len(txt) > 1200 else "")
                parts.append(f"{header}\n{truncated}")

    if parts:
        tables_block = "TABLEAUX ET DONNÉES CALCULÉS :\n" + "\n\n".join(parts)
    else:
        tables_block = "(aucun tableau disponible — rédiger en accord avec les instructions)"

    summary_block = ""
    if summary and summary.strip():
        s = summary[:2500]
        summary_block = f"\nSYNTHÈSE GLOBALE DE L'ANALYSE :\n{s}"

    return tables_block + summary_block


_PROFESSIONAL_STYLE_ADDENDUM = """\

══════════════════════════════════════════════════════════════
RAPPEL CRITIQUE — STYLE DE RÉDACTION PROFESSIONNEL
══════════════════════════════════════════════════════════════
Tu rédiges un rapport de certification actuariel de niveau cabinet senior.

COMMENCE par une phrase FACTUELLE et SPÉCIFIQUE — jamais une généralité.
✓ "L'exposition totale sur la période s'élève à 780 411 années-personnes, réparties \
entre 253 067 assurés actifs au 31/12/2011."
✗ "Dans un contexte où la gestion du risque est cruciale pour les acteurs du secteur..."
✗ "L'analyse actuelle vise à certifier..."
✗ "Il est important de noter que..."

FORMULES : notation Unicode UNIQUEMENT. JAMAIS LaTeX, JAMAIS underscores.
  ✓ q̂ₓ = Dₓ / Eₓ     ✓ SMR = Σ Dₓᵒᵇˢ / Σ Dₓᵉˣᵖ     ✓ IC₉₅%
  ✗ q_x = D_x / E_x   ✗ \\hat{{q}}_x   ✗ $SMR$   ✗ \\frac{{D_x}}{{E_x}}

STRUCTURE :
– Sous-sections numérotées (N.1, N.2...) si plusieurs thèmes distincts
– Listes avec tirets (–), jamais puces (•)
– Référencer les tableaux : "cf. Tableau X", "comme l'indique le Tableau X"
– Référencer les figures : "La Figure X illustre..."

LONGUEUR MINIMALE : 300 mots. Cible : 400–500 mots.
Toute section < 250 mots est insuffisante et sera rejetée.
"""


def _sanitize_formulas(text: str) -> str:
    """Nettoie les notations LaTeX/underscore résiduelles produites par le LLM.

    Appliqué sur chaque section rédigée avant insertion dans le PDF.
    """
    # Supprimer délimiteurs LaTeX inline
    text = re.sub(r'\\\(\s*', '', text)
    text = re.sub(r'\s*\\\)', '', text)
    text = re.sub(r'\\\[\s*', ' ', text)
    text = re.sub(r'\s*\\\]', ' ', text)
    # Supprimer délimiteurs $...$ (non inline code)
    text = re.sub(r'\$\$(.+?)\$\$', r'\1', text, flags=re.DOTALL)
    text = re.sub(r'(?<!\w)\$([^$\n]+?)\$(?!\w)', r'\1', text)
    # Remplacer subscripts actuariels courants (_x, _i, etc.)
    _SUBS = [
        (r'(?<=[A-Za-zμσλΔ])_\{?x\}?', 'ₓ'),
        (r'(?<=[A-Za-zμσλΔ])_\{?i\}?', 'ᵢ'),
        (r'(?<=[A-Za-zμσλΔ])_\{?j\}?', 'ⱼ'),
        (r'(?<=[A-Za-zμσλΔ])_\{?n\}?', 'ₙ'),
        (r'(?<=[A-Za-zμσλΔ])_\{?k\}?', 'ₖ'),
        (r'(?<=[A-Za-zμσλΔ])_\{?0\}?', '₀'),
        (r'(?<=[A-Za-zμσλΔ])_\{?1\}?', '₁'),
        (r'(?<=[A-Za-zμσλΔ])_\{?2\}?', '₂'),
        (r'(?<=[A-Za-zμσλΔ])_\{?obs\}?', 'ᵒᵇˢ'),
        (r'(?<=[A-Za-zμσλΔ])_\{?exp\}?', 'ᵉˣᵖ'),
        (r'(?<=[A-Za-zμσλΔ])_\{?ref\}?', ' ref'),
        (r'\^\{?entry\}?', '⁺'),
        (r'\^\{?exit\}?', '⁻'),
        (r'(?<=[A-Za-zμσλΔ0-9])\^\{?2\}?', '²'),
        (r'(?<=[A-Za-zμσλΔ0-9])\^\{?3\}?', '³'),
    ]
    for pattern, replacement in _SUBS:
        text = re.sub(pattern, replacement, text)
    # Supprimer accolades LaTeX isolées restantes
    text = re.sub(r'\\hat\{([^}]+)\}', r'̂\1', text)
    text = re.sub(r'\\tilde\{([^}]+)\}', r'̃\1', text)
    text = re.sub(r'\\bar\{([^}]+)\}', r'\1', text)
    text = re.sub(r'\{([^{}\n]{1,40})\}', r'\1', text)
    return text


def _write_section_with_writer_prompt(
    client: Any,
    section_title: str,
    section_idx: int,
    data_block: str,
    figures_note: str,
    writer_prompt: str,
) -> str:
    """Rédige une section via gpt-4o avec le writer_prompt comme system.

    Message utilisateur = injection pure de données.
    Instructions éditoriales = writer_prompt + _PROFESSIONAL_STYLE_ADDENDUM.
    """
    figs_part = ""
    if figures_note and "aucun" not in figures_note.lower():
        figs_part = f"\n\nGRAPHIQUES INTÉGRÉS DANS CETTE SECTION :\n{figures_note}"

    user_content = (
        f"SECTION À RÉDIGER : {section_idx}. {section_title}\n\n"
        f"{data_block}"
        f"{figs_part}\n\n"
        "Rédige uniquement le texte narratif de cette section (sans son titre ni numéro), "
        "en français formel actuariel professionnel. "
        "Longueur minimale : 300 mots. Cible : 400–500 mots."
    )

    combined_system = writer_prompt.strip() + _PROFESSIONAL_STYLE_ADDENDUM

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": combined_system},
                {"role": "user", "content": user_content},
            ],
            max_tokens=2200,
            temperature=0.2,
        )
        raw = (response.choices[0].message.content or "").strip()
        return _sanitize_formulas(raw)
    except Exception as e:
        return f"(erreur rédaction section {section_title} : {e})"


# Mots-clés identifiant les steps sans valeur documentaire
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


# ─────────────────────────────────────────────────────────────────────────────
# Point d'entrée principal
# ─────────────────────────────────────────────────────────────────────────────

def generate_narrative_report(
    steps: list[dict],
    summary: str,
    user_message: str,
    domain_label: str,
    output_path: str,
    study_ref: str = "",
    kb_path: str | None = None,
    writer_prompt: str | None = None,
    template_sections: list[dict] | None = None,
    prebuilt_figures: dict[str, bytes] | None = None,
    methodology: dict | None = None,
) -> str:
    """Génère un rapport PDF narratif via LLM + reportlab.

    Args:
        steps:             Steps de l'agent (description, output, figures, display_outputs).
        summary:           Synthèse finale de l'agent (markdown).
        user_message:      Demande originale de l'utilisateur.
        domain_label:      Libellé du domaine (ex: "mortality", "nonlife_reserving").
        output_path:       Chemin du PDF à créer.
        study_ref:         Référence de l'étude pour l'en-tête.
        kb_path:           Chemin vers le JSON de guidelines. Si None, utilise le défaut.
        writer_prompt:     Prompt rédacteur généré par l'encodeur (agent_system_prompt du
                           template). Si fourni, sert de system prompt exclusif (chemin A).
        template_sections: Liste de dicts {id, title, description} issus du template encodeur.
                           Si fourni avec writer_prompt, ces sections remplacent _SECTION_ORDER.

    Returns:
        Chemin absolu du PDF créé.
    """
    # Charger les guidelines KB
    guidelines = _load_kb_guidelines(kb_path, domain_label)

    # Nettoyer le message utilisateur
    user_req = _clean_user_message(user_message)

    # Préparer les steps filtrés
    meaningful_steps = [s for s in steps if _is_meaningful_step(s)]

    # Collecter les figures de l'agent de calcul (pour l'annexe uniquement)
    agent_figures_b64: list[str] = []
    for s in meaningful_steps:
        agent_figures_b64.extend(s.get("figures", []))

    # Prompt système effectif : celui de l'encodeur s'il est fourni, sinon SYSTEM_PROMPT par défaut
    effective_system_prompt = writer_prompt.strip() if writer_prompt else SYSTEM_PROMPT

    # Générer les graphiques professionnels du rédacteur
    if prebuilt_figures:
        report_figures = prebuilt_figures
    else:
        report_figures = _generate_report_figures(meaningful_steps)
        # Si aucun graphique extrait des steps, générer des charts de démo
        if not report_figures:
            report_figures = _generate_demo_charts(methodology)

    # ── Construire l'ordre effectif des sections ──────────────────────────────
    if template_sections and writer_prompt:
        effective_sections = _map_template_sections(template_sections)
    else:
        effective_sections = [(k, t, k) for k, t in _SECTION_ORDER]

    # ── Préparer le bloc de données (réutilisé pour toutes les sections) ──────
    data_block = _build_data_injection(meaningful_steps, summary)

    # ── Générer les textes de section ─────────────────────────────────────────
    from agent import _get_client
    client = _get_client()
    section_texts: dict[str, str] = {}

    _SECTION_FIGURES = {
        "donnees":        ["exposure"],
        "methodologie":   ["rates"],
        "resultats":      ["smr", "oa"],
        "positionnement": ["comparison"],
    }
    _FIG_LABELS = {
        "exposure":   "Graphique — Exposition par âge/durée",
        "rates":      "Graphique — Taux bruts et taux lissés",
        "smr":        "Graphique — Indicateur de validation par groupe",
        "oa":         "Graphique — Observé vs modélisé",
        "comparison": "Graphique — Comparaison avec la référence",
    }

    for idx, (sec_id, sec_title, mapped_key) in enumerate(effective_sections, 1):
        # Figures note pour cette section
        fig_parts = [
            f"  \u2713 {_FIG_LABELS.get(fk, fk)}"
            for fk in _SECTION_FIGURES.get(mapped_key, [])
            if fk in report_figures
        ]
        figures_note = (
            "Graphiques intégrés dans cette section :\n" + "\n".join(fig_parts)
            if fig_parts else "(aucun graphique pour cette section)"
        )

        if writer_prompt:
            # Chemin A : writer_prompt comme system, injection pure de données comme user
            text = _write_section_with_writer_prompt(
                client, sec_title, idx, data_block, figures_note, effective_system_prompt
            )
        else:
            # Chemin B : SYSTEM_PROMPT générique + SECTION_PROMPTS avec contexte structuré
            context = _build_section_context(
                mapped_key, meaningful_steps, summary, user_req, guidelines,
                domain_label, report_figures,
            )
            text = _write_section(client, mapped_key, context, effective_system_prompt)

        section_texts[sec_id] = text

    # Assembler le PDF
    return _assemble_pdf(
        section_texts=section_texts,
        report_figures=report_figures,
        agent_figures_b64=agent_figures_b64,
        steps=meaningful_steps,
        summary=summary,
        user_req=user_req,
        domain_label=domain_label,
        output_path=output_path,
        study_ref=study_ref or f"Analyse {datetime.now().strftime('%Y%m%d')}",
        effective_sections=effective_sections,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Chargement des guidelines KB
# ─────────────────────────────────────────────────────────────────────────────

def _load_kb_guidelines(kb_path: str | None, domain_label: str) -> dict[str, str]:
    """Charge les chunks guideline_narrative depuis le JSON de la KB.

    Retourne un dict {section_id: contenu_guideline}.
    """
    if kb_path is None:
        # Chercher le JSON de guidelines dans Knowledge Base/
        default_kb = Path(__file__).parent.parent / "Knowledge Base" / "rapport_professionnel_TD.json"
        if default_kb.exists():
            kb_path = str(default_kb)

    if not kb_path or not Path(kb_path).exists():
        return {}

    try:
        chunks = json.loads(Path(kb_path).read_text(encoding="utf-8"))
    except Exception:
        return {}

    # Mapper les chunks vers les clés de section
    _SECTION_TAG_MAP = {
        "contexte":        ["structure_rapport", "context"],
        "donnees":         ["statistiques_descriptives", "donnees_initiales"],
        "methodologie":    ["construction_table", "methodologie", "lissage"],
        "resultats":       ["deces_observes_modelises", "validation", "SMR"],
        "positionnement":  ["positionnement_reglementaire", "abattement"],
        "conclusion":      ["suivi_indicateurs", "domaine_validite", "suivi"],
    }

    guidelines: dict[str, str] = {}
    for section_key, target_tags in _SECTION_TAG_MAP.items():
        relevant = []
        for chunk in chunks:
            chunk_tags = chunk.get("tags", [])
            chunk_section = chunk.get("section", "")
            if any(t in chunk_tags or t in chunk_section for t in target_tags):
                relevant.append(chunk.get("contenu", ""))
        if relevant:
            guidelines[section_key] = "\n\n".join(relevant)

    return guidelines


# ─────────────────────────────────────────────────────────────────────────────
# Construction du contexte par section
# ─────────────────────────────────────────────────────────────────────────────

# Quels steps (par position dans la liste filtrée) sont pertinents par section
_SECTION_STEP_RANGES = {
    "contexte":       None,   # pas de steps — contexte pur
    "donnees":        (0, 3), # premiers steps : chargement, nettoyage
    "methodologie":   (2, 7), # calcul expositions, taux bruts, lissage
    "resultats":      (5, None),  # validation, SMR, chi2, tableaux
    "positionnement": (5, None),  # SMR, comparaison référence
    "conclusion":     None,   # synthèse uniquement
}


def _build_section_context(
    section_key: str,
    steps: list[dict],
    summary: str,
    user_req: str,
    guidelines: dict[str, str],
    domain_label: str,
    report_figures: dict[str, bytes] | None = None,
) -> dict[str, str]:
    """Construit le dict de contexte à injecter dans le prompt de section."""
    step_range = _SECTION_STEP_RANGES.get(section_key)
    if step_range is None:
        relevant_steps = []
    else:
        start, end = step_range
        relevant_steps = steps[start:end]

    steps_context = _format_steps_for_llm(relevant_steps)

    # Informer le LLM des graphiques disponibles pour cette section
    _SECTION_FIGURES = {
        "donnees":        ["exposure"],
        "methodologie":   ["rates"],
        "resultats":      ["smr", "oa"],
        "positionnement": ["comparison"],
    }
    available_figs = report_figures or {}
    fig_note_parts = []
    for fig_key in _SECTION_FIGURES.get(section_key, []):
        if fig_key in available_figs:
            _FIG_LABELS = {
                "exposure":   "Graphique — Exposition Eₓ par âge",
                "rates":      "Graphique — Taux bruts q̂ₓ et taux lissés qₓ par âge",
                "smr":        "Graphique — SMR par groupe d'âge avec IC 95 %",
                "oa":         "Graphique — Décès observés vs modélisés par décennie",
                "comparison": "Graphique — Comparaison table d'expérience / référence",
            }
            fig_note_parts.append(f"  ✓ {_FIG_LABELS.get(fig_key, fig_key)}")
    figures_note = (
        "Graphiques intégrés dans cette section :\n" + "\n".join(fig_note_parts)
        if fig_note_parts
        else "(aucun graphique généré pour cette section)"
    )

    return {
        "domain_label":  domain_label.replace("_", " ").title(),
        "user_request":  user_req,
        "summary":       summary[:4000] if summary else "(pas de synthèse disponible)",
        "steps_context": steps_context or "(aucune donnée spécifique pour cette section)",
        "guidelines":    guidelines.get(section_key, "(aucune guideline disponible pour ce domaine)"),
        "figures_note":  figures_note,
    }


def _format_steps_for_llm(steps: list[dict]) -> str:
    """Formate les steps pertinents en texte lisible pour le LLM."""
    parts = []
    for i, s in enumerate(steps, 1):
        desc = s.get("description") or s.get("content", "")
        output = s.get("output", "")
        # Tronquer l'output
        if len(output) > 900:
            output = output[:900] + "…"
        parts.append(f"[Étape {i}] {desc}\nRésultat : {output}")
    return "\n\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Appel LLM par section
# ─────────────────────────────────────────────────────────────────────────────

def _write_section(
    client: Any,
    section_key: str,
    context: dict[str, str],
    system_prompt: str | None = None,
) -> str:
    """Appelle gpt-4o-mini pour rédiger une section du rapport.

    Args:
        system_prompt: Prompt système effectif. Si None, utilise SYSTEM_PROMPT de prompts.py.
    """
    prompt_template = SECTION_PROMPTS.get(section_key, "")
    if not prompt_template:
        return "(section non disponible)"

    try:
        user_prompt = prompt_template.format(**context)
    except KeyError:
        user_prompt = prompt_template

    effective_sp = system_prompt if system_prompt else SYSTEM_PROMPT

    try:
        max_tok = _MAX_TOKENS_BY_SECTION.get(section_key, 800)
        response = client.chat.completions.create(
            model=_WRITER_MODEL,
            messages=[
                {"role": "system", "content": effective_sp},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tok,
            temperature=_TEMPERATURE,
        )
        raw = response.choices[0].message.content or ""
        return _sanitize_formulas(raw)
    except Exception as e:
        return f"(erreur rédaction section {section_key} : {e})"


# ─────────────────────────────────────────────────────────────────────────────
# Assemblage PDF (reportlab)
# ─────────────────────────────────────────────────────────────────────────────

def _assemble_pdf(
    section_texts: dict[str, str],
    report_figures: dict[str, bytes],
    agent_figures_b64: list[str],
    steps: list[dict],
    summary: str,
    user_req: str,
    domain_label: str,
    output_path: str,
    study_ref: str,
    effective_sections: list[tuple[str, str, str]] | None = None,
) -> str:
    """Assemble le PDF final avec reportlab.

    Les graphiques du rapport (report_figures) sont placés dans le corps des sections.
    Les graphiques de l'agent de calcul (agent_figures_b64) vont uniquement en annexe.
    """
    try:
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Table, TableStyle, Image,
            Spacer, PageBreak, HRFlowable, KeepTogether,
        )
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
    except ImportError as e:
        raise ImportError(f"reportlab requis : {e}") from e

    try:
        from PIL import Image as PILImage
        _pil = True
    except ImportError:
        _pil = False

    _w, _h = A4
    max_w = _w - 4 * cm
    domain_display = domain_label.replace("_", " ").title() if domain_label else "Actuariat"

    # ── Polices Unicode (DejaVu bundlé avec matplotlib) ───────────────────────
    # Nécessaire pour rendre les indices/exposants Unicode (ₓ, Σ, ², etc.)
    try:
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        import matplotlib as _mpl
        _fdir = Path(_mpl.get_data_path()) / "fonts" / "ttf"
        _already = set(pdfmetrics.getRegisteredFontNames())
        for _fn, _ff in [
            ("DejaVuSans",        "DejaVuSans.ttf"),
            ("DejaVuSans-Bold",   "DejaVuSans-Bold.ttf"),
            ("DejaVuSans-Oblique","DejaVuSans-Oblique.ttf"),
        ]:
            if _fn not in _already and (_fdir / _ff).exists():
                pdfmetrics.registerFont(TTFont(_fn, str(_fdir / _ff)))
        _F  = "DejaVuSans"
        _FB = "DejaVuSans-Bold"
        _FI = "DejaVuSans-Oblique"
    except Exception:
        _F  = "Helvetica"
        _FB = "Helvetica-Bold"
        _FI = "Helvetica-Oblique"

    # ── Styles ────────────────────────────────────────────────────────────────
    styles = getSampleStyleSheet()
    BLUE  = colors.HexColor("#1A3A5C")
    GREY  = colors.HexColor("#555555")
    GREY2 = colors.HexColor("#888888")
    LBLUE = colors.HexColor("#E8EFF7")

    title_s  = ParagraphStyle("T",  parent=styles["Title"],   fontName=_FB,
                               fontSize=20, textColor=BLUE, alignment=TA_CENTER, spaceAfter=10)
    sub_s    = ParagraphStyle("S",  parent=styles["Normal"],  fontName=_F,
                               fontSize=11, textColor=GREY, alignment=TA_CENTER, spaceAfter=6)
    sec_s    = ParagraphStyle("H1", parent=styles["Heading1"],fontName=_FB,
                               fontSize=13, textColor=BLUE, spaceBefore=14, spaceAfter=6)
    body_s   = ParagraphStyle("B",  parent=styles["Normal"],  fontName=_F,
                               fontSize=10, leading=15, spaceAfter=6, alignment=TA_JUSTIFY)
    caption_s= ParagraphStyle("Cap",parent=styles["Normal"],  fontName=_FI,
                               fontSize=8, textColor=GREY, alignment=TA_CENTER, spaceAfter=4)
    small_s  = ParagraphStyle("Sm", parent=styles["Normal"],  fontName=_F,
                               fontSize=8, textColor=GREY2, leading=11)
    mono_s   = ParagraphStyle("Mo", parent=styles["Normal"],  fontName="Courier",
                               fontSize=8, textColor=GREY, leading=12)

    # ── En-têtes / pieds de page ───────────────────────────────────────────────
    def _on_cover(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(GREY2)
        canvas.drawRightString(_w - 2*cm, 0.8*cm, f"Page {doc.page}")
        canvas.restoreState()

    def _on_page(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(GREY2)
        canvas.drawString(2*cm, _h - 1.3*cm, study_ref)
        canvas.drawRightString(_w - 2*cm, _h - 1.3*cm, "CONFIDENTIEL")
        canvas.setStrokeColor(colors.HexColor("#DDDDDD"))
        canvas.setLineWidth(0.3)
        canvas.line(2*cm, _h - 1.5*cm, _w - 2*cm, _h - 1.5*cm)
        canvas.drawString(2*cm, 0.8*cm, f"Agent actuariel — {domain_display}")
        canvas.drawRightString(_w - 2*cm, 0.8*cm, f"Page {doc.page}")
        canvas.restoreState()

    ref_s    = ParagraphStyle("Ref", parent=styles["Normal"], fontName=_FB,
                               fontSize=11, textColor=BLUE, alignment=TA_CENTER, spaceAfter=4)
    toc_s    = ParagraphStyle("Toc", parent=styles["Normal"], fontName=_F,
                               fontSize=10, textColor=GREY, spaceAfter=3, leftIndent=0)
    pre_s    = ParagraphStyle("Pre", parent=styles["Normal"], fontName=_F,
                               fontSize=10, leading=15, spaceAfter=6, alignment=TA_JUSTIFY,
                               textColor=GREY)

    doc = SimpleDocTemplate(output_path, pagesize=A4,
                            leftMargin=2*cm, rightMargin=2*cm,
                            topMargin=2.2*cm, bottomMargin=1.8*cm)
    story: list[Any] = []

    # ── Page de couverture professionnelle ────────────────────────────────────
    story.append(Spacer(1, 2.5*cm))
    # Titre principal du rapport (issu du study_ref ou générique)
    report_display_title = user_req[:120] if user_req else "Rapport d'Analyse Actuarielle"
    story.append(Paragraph(_esc(report_display_title), title_s))
    story.append(Spacer(1, 0.5*cm))
    story.append(HRFlowable(width="80%", thickness=2, color=BLUE, spaceAfter=12, lineCap="butt"))
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph(_esc(f"Étude Actuarielle — {study_ref}"), ref_s))
    story.append(Spacer(1, 0.4*cm))
    story.append(Paragraph(f"Domaine : {_esc(domain_display)}", sub_s))
    story.append(Paragraph(f"Date : {datetime.now().strftime('%d %B %Y').capitalize()}", sub_s))
    story.append(Spacer(1, 2*cm))
    story.append(HRFlowable(width="40%", thickness=0.5, color=GREY2, spaceAfter=8, lineCap="butt"))
    story.append(Paragraph(
        "<i>Rapport généré par l'agent actuariel IA</i>", small_s
    ))
    story.append(PageBreak())

    # ── Préambule ──────────────────────────────────────────────────────────────
    if summary and len(summary.strip()) > 100:
        story.append(Paragraph("PRÉAMBULE", sec_s))
        story.append(HRFlowable(width="100%", thickness=0.5,
                                color=colors.HexColor("#CCCCCC"), spaceAfter=8))
        # Nettoyer le résumé et le tronquer à ~500 mots
        preambule_text = summary.strip()
        # Supprimer les lignes qui ressemblent à du code Python ou des commandes
        preambule_lines = [
            ln for ln in preambule_text.split("\n")
            if not ln.strip().startswith(("#", ">>>", "import ", "from ", "def ", "```"))
        ]
        preambule_clean = " ".join(preambule_lines)[:2000]
        story.extend(_md_to_story(preambule_clean, pre_s, sec_s, small_s,
                                   max_w_pt=max_w, BLUE=BLUE))
        story.append(Spacer(1, 0.5*cm))
        story.append(PageBreak())

    # ── Sommaire ───────────────────────────────────────────────────────────────
    _eff_secs_for_toc = effective_sections or [(k, t, k) for k, t in _SECTION_ORDER]
    story.append(Paragraph("SOMMAIRE", sec_s))
    story.append(HRFlowable(width="100%", thickness=0.5,
                            color=colors.HexColor("#CCCCCC"), spaceAfter=8))
    toc_data = []
    for toc_idx, (_, toc_title, _) in enumerate(_eff_secs_for_toc, 1):
        toc_data.append([
            Paragraph(f"<b>{toc_idx}.</b>", toc_s),
            Paragraph(_esc(toc_title), toc_s),
        ])
    toc_data.append([Paragraph("", toc_s), Paragraph("", toc_s)])
    toc_data.append([
        Paragraph("<b>Annexe</b>", toc_s),
        Paragraph("Trace d'exécution et données de calcul", toc_s),
    ])
    toc_table = Table(toc_data, colWidths=[1.2*cm, max_w - 1.2*cm])
    toc_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 2),
        ("RIGHTPADDING", (0, 0), (-1, -1), 2),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LINEBELOW", (0, -2), (-1, -2), 0.3, colors.HexColor("#DDDDDD")),
    ]))
    story.append(toc_table)
    story.append(PageBreak())

    # ── Associer les tableaux (display_outputs) à leur section ───────────────────
    _STEP_SECTION_KEYWORDS = {
        "donnees":        ["chargement", "nettoyage", "clean", "load", "données", "âges",
                           "compute_ages", "distribution", "anomalie"],
        "methodologie":   ["exposition", "exposure", "taux bruts", "crude_rates", "kaplan",
                           "lissage", "smooth", "whittaker", "gompertz", "makeham",
                           "crédibilité", "diagnose", "sélection", "auto_select"],
        "resultats":      ["validation", "chi", "intervalle", "confiance", "smr", "backtesting",
                           "observés", "modélisés", "rapport o/a", "observed", "expected",
                           "export", "table finale", "plot"],
        "positionnement": ["benchmark", "référence", "abattement", "abatt", "positionnement",
                           "th00", "tf00", "td88"],
    }

    def _classify_step(step: dict) -> str:
        desc = (step.get("description") or "").lower()
        for sec, kws in _STEP_SECTION_KEYWORDS.items():
            if any(kw in desc for kw in kws):
                return sec
        return "resultats"

    tables_by_section: dict[str, list[dict]] = {
        k: [] for k in ["donnees", "methodologie", "resultats", "positionnement"]
    }
    for step in steps:
        sec = _classify_step(step)
        for do in step.get("display_outputs", []):
            tables_by_section[sec].append(do)

    # Graphiques du rédacteur par section (report_figures, format bytes PNG)
    _SECTION_REPORT_FIGS: dict[str, list[tuple[str, str]]] = {
        "donnees":        [("exposure", "Exposition centrale Eₓ par âge")],
        "methodologie":   [("rates",    "Taux bruts q̂ₓ et taux lissés qₓ par âge")],
        "resultats":      [("smr",      "SMR par groupe d'âge avec IC 95 %"),
                           ("oa",       "Décès observés vs modélisés par décennie d'âge")],
        "positionnement": [("comparison", "Comparaison table d'expérience / table de référence")],
    }

    # ── Sections narratives + visuels ─────────────────────────────────────────
    fig_counter = [0]
    tbl_counter = [0]

    def _add_section_visuals(sec_key: str) -> None:
        """Insère tableaux (computation agent) puis graphiques (report_agent)."""
        # Tableaux display_outputs
        for do in tables_by_section.get(sec_key, []):
            tbl_counter[0] += 1
            label = do.get("label", f"Tableau {tbl_counter[0]}")
            story.append(Paragraph(f"<b>Tableau {tbl_counter[0]} — {_esc(label)}</b>", small_s))
            txt = do.get("text", "")
            if txt and len(txt) > 10:
                _add_text_table(story, txt, small_s, mono_s, max_w)
            story.append(Spacer(1, 0.2*cm))

        # Graphiques du rédacteur (report_figures, bytes PNG)
        for fig_key, fig_label in _SECTION_REPORT_FIGS.get(sec_key, []):
            fig_bytes = report_figures.get(fig_key)
            if fig_bytes:
                fig_counter[0] += 1
                _add_figure_bytes(story, fig_bytes, max_w, cm)
                story.append(Paragraph(
                    f"Figure {fig_counter[0]} — {_esc(fig_label)}", caption_s
                ))

    _eff_secs = effective_sections or [(k, t, k) for k, t in _SECTION_ORDER]
    for sec_id, sec_title, mapped_key in _eff_secs:
        narrative = section_texts.get(sec_id, "")
        if not narrative:
            continue

        story.append(Paragraph(sec_title, sec_s))
        story.append(HRFlowable(width="100%", thickness=0.5,
                                color=colors.HexColor("#CCCCCC"), spaceAfter=8))

        # Texte narratif avec rendu markdown minimal
        story.extend(_md_to_story(narrative, body_s, sec_s, small_s,
                                   max_w_pt=max_w, BLUE=BLUE))

        # Tableaux + figures de cette section (basé sur mapped_key pour les visuels)
        _add_section_visuals(mapped_key)

        story.append(Spacer(1, 0.4*cm))

    # ── Annexe — Trace d'exécution + graphiques techniques de l'agent ────────────
    story.append(PageBreak())
    story.append(Paragraph("Annexe — Trace d'exécution", sec_s))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#CCCCCC"),
                             spaceAfter=8))
    story.append(Paragraph(
        f"{len(steps)} étape(s) significative(s) retenues pour ce rapport.",
        small_s,
    ))
    story.append(Spacer(1, 0.3*cm))
    for i, s in enumerate(steps, 1):
        desc = s.get("description") or s.get("content", "")
        story.append(Paragraph(f"<b>Étape {i}</b> — {_esc(desc[:180])}", small_s))
        out = s.get("output", "")
        if out and len(out) > 30:
            _add_text_table(story, out[:600], small_s, mono_s, max_w)
        story.append(Spacer(1, 0.15*cm))

    # Graphiques techniques produits par l'agent de calcul
    if agent_figures_b64:
        story.append(Spacer(1, 0.4*cm))
        story.append(Paragraph("<b>Graphiques de l'agent de calcul</b>", small_s))
        story.append(Spacer(1, 0.2*cm))
        for b64 in agent_figures_b64:
            _add_figure(story, b64, max_w, cm, _pil)
            story.append(Spacer(1, 0.2*cm))

    story.append(Spacer(1, 0.5*cm))
    story.append(Paragraph(
        f"<i>Généré le {datetime.now().strftime('%d/%m/%Y à %H:%M')} — {domain_display}.</i>",
        small_s,
    ))

    doc.build(story, onFirstPage=_on_cover, onLaterPages=_on_page)
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# Helpers PDF
# ─────────────────────────────────────────────────────────────────────────────

def _esc(text: str) -> str:
    return (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;"))


def _md_inline(text: str) -> str:
    """Convertit **bold** et *italic* en balises ReportLab (après _esc)."""
    text = _esc(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"\*(.+?)\*",     r"<i>\1</i>", text)
    return text


def _render_md_table(table_lines: list[str], max_w_pt: float, BLUE: Any, small_s: Any) -> list:
    """Rend des lignes de tableau Markdown en Table ReportLab avec style professionnel."""
    try:
        from reportlab.platypus import Table, TableStyle, Spacer
        from reportlab.lib import colors
        from reportlab.lib.units import cm as _cm
    except ImportError:
        return []

    rows: list[list[str]] = []
    has_separator = False
    for ln in table_lines:
        stripped = ln.strip()
        # Ligne séparateur (|---|---|)
        if re.match(r'^\|[\s:|-]+\|', stripped):
            has_separator = True
            continue
        # Ligne de données
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if cells:
            rows.append(cells)

    if len(rows) < 2:
        return []

    n_cols = max(len(r) for r in rows)
    if n_cols < 1:
        return []
    rows = [r + [""] * (n_cols - len(r)) for r in rows]
    # Tronquer les cellules trop longues
    rows = [[c[:80] for c in r] for r in rows]

    col_w = max_w_pt / n_cols

    from reportlab.platypus import Table, TableStyle, Spacer
    from reportlab.lib import colors
    from reportlab.lib.units import cm as _cm

    tbl = Table(rows, colWidths=[col_w] * n_cols, repeatRows=1 if has_separator else 0)
    style_cmds = [
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#CCCCCC")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#F5F2E7"), colors.white]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]
    if has_separator and rows:
        style_cmds += [
            ("BACKGROUND", (0, 0), (-1, 0), BLUE),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("LINEBELOW", (0, 0), (-1, 0), 1, BLUE),
        ]
    tbl.setStyle(TableStyle(style_cmds))
    return [tbl, Spacer(1, 0.3 * _cm)]


def _md_to_story(text: str, body_s: Any, sec_s: Any, small_s: Any,
                 max_w_pt: float = 450, BLUE: Any = None) -> list:
    """Convertit du markdown en flowables ReportLab.

    Gère : titres ##/###, listes -, tableaux |col|, **bold**, *italic*, blocs de code.
    """
    try:
        from reportlab.platypus import Paragraph, Spacer
        from reportlab.lib.units import cm as _cm
        from reportlab.lib import colors as _colors
    except ImportError:
        return []

    if BLUE is None:
        BLUE = _colors.HexColor("#1A3A5C")

    elements: list[Any] = []
    lines = text.split("\n")
    in_code = False
    in_table = False
    code_lines: list[str] = []
    table_lines: list[str] = []

    def flush_code():
        if not code_lines:
            return
        block = "\n".join(code_lines)
        elements.append(Paragraph(_esc(block).replace("\n", "<br/>"), small_s))
        elements.append(Spacer(1, 0.15 * _cm))
        code_lines.clear()

    def flush_table():
        if not table_lines:
            return
        tbl_elements = _render_md_table(table_lines, max_w_pt, BLUE, small_s)
        elements.extend(tbl_elements)
        table_lines.clear()

    for line in lines:
        stripped = line.rstrip()

        # Bloc de code fenced (```)
        if stripped.startswith("```"):
            flush_table()
            if in_code:
                flush_code()
                in_code = False
            else:
                in_code = True
            continue
        if in_code:
            code_lines.append(stripped)
            continue

        # Tableau markdown (ligne commençant par |)
        if stripped.startswith("|") and stripped.endswith("|") and len(stripped) > 2:
            flush_code()
            in_table = True
            table_lines.append(stripped)
            continue
        elif in_table:
            flush_table()
            in_table = False

        # Bloc de code indenté (4 espaces)
        if stripped.startswith("    ") and stripped.strip():
            code_lines.append(stripped.strip())
            continue
        elif code_lines:
            flush_code()

        # Titre ### (sous-section N.M)
        if stripped.startswith("### "):
            subsec_s = ParagraphStyle(
                "SS", parent=body_s,
                fontSize=10, spaceBefore=8, spaceAfter=4,
            )
            content = stripped[4:].strip()
            elements.append(Spacer(1, 0.15 * _cm))
            elements.append(Paragraph(f"<b>{_md_inline(content)}</b>", subsec_s))
            continue

        # Titre ## (sous-section principale)
        if stripped.startswith("## "):
            content = stripped[3:].strip()
            elements.append(Spacer(1, 0.2 * _cm))
            elements.append(Paragraph(_md_inline(content), sec_s))
            continue

        # Liste avec tiret – ou - ou *
        if re.match(r"^\s*[–\-\*]\s+", stripped):
            content = re.sub(r"^\s*[–\-\*]\s+", "", stripped)
            elements.append(Paragraph(f"– {_md_inline(content)}", body_s))
            continue

        # Liste numérotée
        if re.match(r"^\s*\d+[.)]\s+", stripped):
            content = re.sub(r"^\s*\d+[.)]\s+", "", stripped)
            elements.append(Paragraph(f"   {_md_inline(content)}", body_s))
            continue

        # Ligne vide → espace
        if not stripped:
            elements.append(Spacer(1, 0.15 * _cm))
            continue

        # Paragraphe normal
        elements.append(Paragraph(_md_inline(stripped), body_s))

    flush_code()
    flush_table()
    return elements


def _add_figure_bytes(story: list, img_bytes: bytes, max_w: float, cm: float) -> None:
    """Insère une figure depuis des bytes PNG (graphique rédacteur)."""
    if not img_bytes:
        return
    try:
        from reportlab.platypus import Image, Spacer
        try:
            from PIL import Image as PILImage
            pil_img = PILImage.open(io.BytesIO(img_bytes))
            ow, oh = pil_img.size
            dpi = 150
            iw = ow / dpi * 2.54
            ih = oh / dpi * 2.54
        except Exception:
            iw, ih = 14.0, 7.0
        max_w_cm = 14.0
        if iw > max_w_cm:
            ih *= max_w_cm / iw
            iw = max_w_cm
        story.append(Image(io.BytesIO(img_bytes), width=iw * cm, height=ih * cm))
        story.append(Spacer(1, 0.3 * cm))
    except Exception:
        pass


def _add_figure(story: list, b64: str, max_w: float, cm: float, pil: bool) -> None:
    """Décode et insère une figure base64 dans la story."""
    if not b64:
        return
    try:
        from reportlab.platypus import Image, Spacer
        img_bytes = base64.b64decode(b64)
        img_io = io.BytesIO(img_bytes)
        if pil:
            from PIL import Image as PILImage
            pil_img = PILImage.open(io.BytesIO(img_bytes))
            ow, oh = pil_img.size
            dpi = 96
            iw = ow / dpi * 2.54
            ih = oh / dpi * 2.54
        else:
            iw, ih = 14.0, 9.0
        max_w_cm = 14.0
        if iw > max_w_cm:
            ih *= max_w_cm / iw
            iw = max_w_cm
        story.append(Image(img_io, width=iw * cm, height=ih * cm))
        story.append(Spacer(1, 0.3 * cm))
    except Exception:
        pass


def _add_step_tables(
    story: list, steps: list[dict], section_key: str,
    body_s: Any, small_s: Any, mono_s: Any, max_w: float,
) -> None:
    """Insère les tables display_outputs des steps pertinents pour la section."""
    from reportlab.platypus import Spacer
    from reportlab.lib.units import cm as _cm
    step_slice = {"donnees": steps[:3], "resultats": steps[3:]}.get(section_key, [])
    for s in step_slice:
        for do in s.get("display_outputs", []):
            txt = do.get("text", "")
            if txt and len(txt) > 20:
                _add_text_table(story, txt, small_s, mono_s, max_w)


def _add_text_table(story: list, text: str, body_s: Any, small_s: Any, max_w: float) -> None:
    """Parse du texte tabulaire et le rend en Table reportlab."""
    try:
        import re as _re
        from reportlab.platypus import Table, TableStyle, Spacer
        from reportlab.lib import colors
        from reportlab.lib.units import cm as _cm

        lines = [ln for ln in text.strip().split("\n") if ln.strip()]
        if not lines:
            return
        rows = []
        for ln in lines[:51]:
            cells = ln.split("\t") if "\t" in ln else _re.split(r"  +", ln.strip())
            rows.append([str(c).strip()[:60] for c in cells])
        max_len = max((len(r) for r in rows), default=0)
        if max_len < 2:
            story.append(Spacer(1, 0.05 * _cm))
            return
        rows = [r + [""] * (max_len - len(r)) for r in rows]
        if max_len > 10:
            rows = [r[:10] + ["…"] for r in rows]
            max_len = 11
        col_w = max_w / max_len
        tbl = Table(rows, colWidths=[col_w] * max_len, repeatRows=1)
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#3A3A3A")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 7),
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
# Helpers steps
# ─────────────────────────────────────────────────────────────────────────────

def _is_meaningful_step(step: dict) -> bool:
    desc = (step.get("description") or step.get("content", "")).lower()
    if any(kw in desc for kw in _DEBUG_KEYWORDS):
        return False
    output = step.get("output", "")
    return (
        (len(output) > 50 and "empty dataframe" not in output.lower())
        or bool(step.get("figures"))
        or bool(step.get("display_outputs"))
    )


def _clean_user_message(msg: str) -> str:
    for marker in ["Paramètres déjà définis", "FILE_PATH =", "\n- FILE_PATH"]:
        idx = msg.find(marker)
        if idx > 10:
            return msg[:idx].strip()
    return msg[:400].strip()
