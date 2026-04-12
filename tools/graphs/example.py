#!/usr/bin/env python3
"""
report_agent/tools/graphs/example.py
─────────────────────────────────────────────────────────────────────────────
Galerie complète de tous les graphiques disponibles, avec données synthétiques.

But : montrer à l'actuaire les rendus disponibles AVANT l'analyse réelle,
      et tester que les fonctions graphiques fonctionnent correctement.

Usage (depuis la racine du projet) :
    python report_agent/tools/graphs/example.py
    → Sauvegarde les PNG dans /tmp/graphs_example/
    → Génère /tmp/graphs_example/index.html  (ouvrir dans le navigateur)

Personnalisation :
    PALETTE  — dict de couleurs HEX (ex: {"_BLUE": "#003366"})
    SEXE     — "H" ou "F" pour la table de référence
    AGE_MIN, AGE_MAX — plage d'âge simulée
"""
from __future__ import annotations

import base64
import io
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# ─── Ajouter la racine au PYTHONPATH ─────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ─── Palette (modifiable) ────────────────────────────────────────────────────
PALETTE = {
    "_BG":     "#FBF8F1",
    "_GRID":   "#E8E3D8",
    "_BLUE":   "#2C5F8A",
    "_RED":    "#C0392B",
    "_GREEN":  "#27AE60",
    "_ORANGE": "#E67E22",
    "_PURPLE": "#8E44AD",
    "_GREY":   "#7F8C8D",
}
SEXE    = "H"
AGE_MIN = 40
AGE_MAX = 85
OUT_DIR = Path("/tmp/graphs_example")

# ─── Données synthétiques ────────────────────────────────────────────────────

def _synthetic_data() -> dict:
    """Génère une table de mortalité synthétique réaliste (Makeham)."""
    ages  = np.arange(AGE_MIN, AGE_MAX + 1)
    n     = len(ages)
    rng   = np.random.default_rng(42)

    # Paramètres Makeham : qx = 1 - exp(-A - B*c^x)
    A, B, c = 0.0007, 0.00005, 1.10
    mu_true = A + B * c ** ages
    qx_true = 1.0 - np.exp(-mu_true)

    # Exposition décroissante avec l'âge (simulée)
    exposure = 5000 * np.exp(-0.04 * (ages - AGE_MIN)) + rng.uniform(-200, 200, n)
    exposure = np.maximum(exposure, 50.0)

    # Décès observés (Poisson)
    deaths = rng.poisson(mu_true * exposure)
    mu_obs = deaths / exposure
    qx_brut = 1.0 - np.exp(-mu_obs)

    # Taux lissés (légère régularisation Whittaker simulée)
    from scipy.ndimage import uniform_filter1d
    qx_lisse = uniform_filter1d(qx_brut, size=5, mode="nearest")
    qx_lisse = np.maximum(qx_lisse, 1e-6)

    # IC Poisson 95%
    alpha = 0.05
    from scipy.stats import chi2
    ci_lower = chi2.ppf(alpha / 2, 2 * deaths) / (2 * exposure)
    ci_upper = chi2.ppf(1 - alpha / 2, 2 * (deaths + 1)) / (2 * exposure)

    # Table de référence TH0002 (approximation Makeham)
    A_ref, B_ref, c_ref = 0.0008, 0.00003, 1.115
    qx_ref = 1.0 - np.exp(-(A_ref + B_ref * c_ref ** ages))

    # Facteurs d'abattement
    alpha_factors = np.where(qx_ref > 0, qx_lisse / qx_ref, np.nan)
    smr_global = np.nanmean(alpha_factors)

    # Survie
    S_exp = np.cumprod(1 - qx_lisse)
    S_ref = np.cumprod(1 - qx_ref)

    exposure_df = pd.DataFrame({
        "age":     ages,
        "E_x":     exposure,
        "D_x":     deaths,
        "mu_x":    mu_obs,
        "q_x_brut": qx_brut,
        "q_x_lisse": qx_lisse,
    })
    ci_df = pd.DataFrame({
        "age":      ages,
        "qx":       qx_lisse,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    })
    abatement_df = pd.DataFrame({
        "age":              ages,
        "qx_exp":           qx_lisse,
        "qx_ref":           qx_ref,
        "abatement_factor": alpha_factors,
    })

    return {
        "ages":         ages,
        "exposure_df":  exposure_df,
        "ci_df":        ci_df,
        "abatement_df": abatement_df,
        "qx_lisse":     qx_lisse,
        "qx_ref":       qx_ref,
        "S_exp":        S_exp,
        "S_ref":        S_ref,
        "smr_global":   smr_global,
    }


# ─── Utilitaires matplotlib ───────────────────────────────────────────────────

def _theme(fig, axes):
    fig.patch.set_facecolor(PALETTE["_BG"])
    for ax in axes:
        ax.set_facecolor(PALETTE["_BG"])
        ax.grid(True, color=PALETTE["_GRID"], linewidth=0.8, alpha=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=9)


def _save(fig, name: str) -> str:
    """Sauvegarde la figure et retourne le chemin."""
    path = OUT_DIR / f"{name}.png"
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor=PALETTE["_BG"])
    plt.close(fig)
    return str(path)


def _fig_to_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


# ─── Fonctions graphiques ─────────────────────────────────────────────────────

def chart_exposure(d: dict) -> str:
    """Exposition centrale E_x et décès D_x par âge."""
    df = d["exposure_df"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7), sharex=True)

    ax1.bar(df["age"], df["E_x"], color=PALETTE["_BLUE"], alpha=0.8, width=0.75, edgecolor="none")
    ax1.set_ylabel("Exposition E_x (P-A)", fontsize=10)
    ax1.set_title("Exposition par âge (personne-années et décès observés)", fontsize=11, loc="left")

    ax2.bar(df["age"], df["D_x"], color=PALETTE["_RED"], alpha=0.8, width=0.75, edgecolor="none")
    ax2.set_ylabel("Décès observés D_x", fontsize=10)
    ax2.set_xlabel("Âge", fontsize=10)

    _theme(fig, [ax1, ax2])
    fig.tight_layout()
    return _save(fig, "01_exposure")


def chart_crude_smoothed(d: dict) -> str:
    """Taux bruts vs lissés en échelle logarithmique + courbe de référence."""
    df   = d["exposure_df"]
    ages = d["ages"]
    valid = df["q_x_brut"].notna() & (df["q_x_brut"] > 0)

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.scatter(df.loc[valid, "age"], df.loc[valid, "q_x_brut"],
               s=18, color=PALETTE["_GREY"], alpha=0.6, label="q_x brut", zorder=2)
    ax.plot(ages, d["qx_lisse"],
            color=PALETTE["_RED"], linewidth=2.5, label="q_x lissé (Whittaker)", zorder=3)
    ax.plot(ages, d["qx_ref"],
            color=PALETTE["_BLUE"], linewidth=1.5, linestyle="--", alpha=0.8,
            label=f"Référence TH0002 ({SEXE})", zorder=4)
    ax.set_yscale("log")
    ax.set_xlabel("Âge", fontsize=10)
    ax.set_ylabel("q_x (échelle log)", fontsize=10)
    ax.set_title("Taux bruts vs lissés vs référence (échelle logarithmique)", fontsize=11, loc="left")
    ax.legend(facecolor=PALETTE["_BG"], edgecolor=PALETTE["_GRID"], fontsize=9)
    _theme(fig, [ax])
    return _save(fig, "02_crude_smoothed")


def chart_ci_bands(d: dict) -> str:
    """Courbe lissée avec intervalles de confiance Poisson 95%."""
    ages  = d["ages"]
    ci_df = d["ci_df"]

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.fill_between(ages, ci_df["ci_lower"], ci_df["ci_upper"],
                    alpha=0.25, color=PALETTE["_BLUE"], label="IC 95% Poisson")
    ax.plot(ages, d["qx_lisse"],
            color=PALETTE["_RED"], linewidth=2.5, label="q_x lissé", zorder=3)
    ax.plot(ages, d["qx_ref"],
            color=PALETTE["_BLUE"], linewidth=1.5, linestyle="--", alpha=0.8,
            label=f"Référence TH0002 ({SEXE})", zorder=4)
    ax.set_yscale("log")
    ax.set_xlabel("Âge", fontsize=10)
    ax.set_ylabel("q_x (échelle log)", fontsize=10)
    ax.set_title("Taux lissés avec intervalles de confiance Poisson 95%", fontsize=11, loc="left")
    ax.legend(facecolor=PALETTE["_BG"], edgecolor=PALETTE["_GRID"], fontsize=9)
    _theme(fig, [ax])
    return _save(fig, "03_ci_bands")


def chart_survival(d: dict) -> str:
    """Courbe de survie expérience vs référence."""
    ages = d["ages"]
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(ages, d["S_exp"], color=PALETTE["_RED"], linewidth=2.5, label="Expérience (lissé)")
    ax.plot(ages, d["S_ref"], color=PALETTE["_BLUE"], linewidth=1.8, linestyle="--",
            alpha=0.8, label=f"Référence TH0002 ({SEXE})")
    ax.fill_between(ages, d["S_exp"], d["S_ref"],
                    where=(d["S_exp"] > d["S_ref"]),
                    alpha=0.15, color=PALETTE["_GREEN"], label="Zone sur-survie expérience")
    ax.set_xlabel("Âge", fontsize=10)
    ax.set_ylabel("Probabilité de survie S(x)", fontsize=10)
    ax.set_title("Courbe de survie — expérience vs référence", fontsize=11, loc="left")
    ax.legend(facecolor=PALETTE["_BG"], edgecolor=PALETTE["_GRID"], fontsize=9)
    _theme(fig, [ax])
    return _save(fig, "04_survival")


def chart_abatement(d: dict) -> str:
    """Facteurs d'abattement par âge avec référence à 1.0."""
    ab_df  = d["abatement_df"].dropna(subset=["abatement_factor"])
    ages   = ab_df["age"].tolist()
    factors = ab_df["abatement_factor"].tolist()
    colors = [PALETTE["_BLUE"] if f <= 1.0 else PALETTE["_ORANGE"] for f in factors]

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(ages, factors, color=colors, alpha=0.8, width=0.75, edgecolor="none")
    ax.axhline(y=1.0, color=PALETTE["_RED"], linewidth=2, linestyle="--", label="Référence (α = 1.0)")
    smr = d["smr_global"]
    ax.axhline(y=smr, color=PALETTE["_GREEN"], linewidth=1.5, linestyle=":",
               label=f"SMR global = {smr:.3f}")
    patch_u = mpatches.Patch(color=PALETTE["_BLUE"],   alpha=0.8, label="Sous-mortalité (α ≤ 1)")
    patch_s = mpatches.Patch(color=PALETTE["_ORANGE"], alpha=0.8, label="Sur-mortalité (α > 1)")
    ax.legend(handles=[patch_u, patch_s,
                        plt.Line2D([0], [0], color=PALETTE["_RED"],   linewidth=2, linestyle="--"),
                        plt.Line2D([0], [0], color=PALETTE["_GREEN"], linewidth=1.5, linestyle=":")],
              labels=["Sous-mortalité (α ≤ 1)", "Sur-mortalité (α > 1)",
                      "Référence (α = 1.0)", f"SMR global = {smr:.3f}"],
              facecolor=PALETTE["_BG"], edgecolor=PALETTE["_GRID"], fontsize=9)
    ax.set_xlabel("Âge", fontsize=10)
    ax.set_ylabel("Facteur d'abattement α", fontsize=10)
    ax.set_title("Facteurs d'abattement vs TH0002 par âge", fontsize=11, loc="left")
    _theme(fig, [ax])
    return _save(fig, "05_abatement")


def chart_smr_bars(d: dict) -> str:
    """SMR par décennie d'âge."""
    ab_df = d["abatement_df"].dropna(subset=["abatement_factor"])
    ab_df["decade"] = (ab_df["age"] // 10) * 10
    smr_by_decade = ab_df.groupby("decade")["abatement_factor"].mean()

    fig, ax = plt.subplots(figsize=(9, 5))
    bar_colors = [PALETTE["_BLUE"] if v <= 1.0 else PALETTE["_ORANGE"]
                  for v in smr_by_decade.values]
    bars = ax.bar([str(d) + "-" + str(d + 9) for d in smr_by_decade.index],
                  smr_by_decade.values, color=bar_colors, alpha=0.8, edgecolor="none")
    ax.axhline(y=1.0, color=PALETTE["_RED"], linewidth=1.8, linestyle="--")
    for bar, val in zip(bars, smr_by_decade.values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.005, f"{val:.3f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("Décennie d'âge", fontsize=10)
    ax.set_ylabel("SMR moyen", fontsize=10)
    ax.set_title("SMR moyen par décennie", fontsize=11, loc="left")
    _theme(fig, [ax])
    fig.tight_layout()
    return _save(fig, "06_smr_bars")


def chart_age_pyramid(d: dict) -> str:
    """Pyramide des effectifs (exposition) par tranche d'âge quinquennale."""
    df = d["exposure_df"].copy()
    df["tranche"] = (df["age"] // 5) * 5
    grouped = df.groupby("tranche")["E_x"].sum()

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.barh([f"{t}–{t+4}" for t in grouped.index], grouped.values,
            color=PALETTE["_BLUE"], alpha=0.8, edgecolor="none")
    ax.set_xlabel("Exposition (P-A)", fontsize=10)
    ax.set_ylabel("Tranche d'âge", fontsize=10)
    ax.set_title("Pyramide d'exposition par tranche de 5 ans", fontsize=11, loc="left")
    _theme(fig, [ax])
    fig.tight_layout()
    return _save(fig, "07_age_pyramid")


def chart_mortality_heatmap(d: dict) -> str:
    """Heatmap qx lissé (variation relative vs référence) par âge — encodage couleur."""
    ages   = d["ages"]
    ratio  = d["qx_lisse"] / d["qx_ref"]   # valeur relative vs référence

    fig, ax = plt.subplots(figsize=(14, 2.5))
    im = ax.imshow(ratio.reshape(1, -1), aspect="auto", cmap="RdYlGn_r",
                   vmin=0.0, vmax=2.0, extent=[ages[0] - 0.5, ages[-1] + 0.5, -0.5, 0.5])
    plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.35,
                 label="Ratio expérience / référence  (< 1 = sous-mortalité)")
    ax.set_xlabel("Âge", fontsize=10)
    ax.set_yticks([])
    ax.set_title("Heatmap de mortalité relative — expérience vs TH0002", fontsize=11, loc="left")
    fig.patch.set_facecolor(PALETTE["_BG"])
    ax.set_facecolor(PALETTE["_BG"])
    return _save(fig, "08_heatmap")


# ─── Catalogue des graphiques ─────────────────────────────────────────────────

CHARTS = [
    {
        "id":          "01_exposure",
        "title":       "Exposition par âge",
        "description": "Barres d'exposition centrale E_x (personne-années) et décès D_x observés "
                       "par âge. Permet de visualiser la densité de données et d'identifier "
                       "les âges avec peu d'observations (faible crédibilité).",
        "category":    "builder",
        "fn":          chart_exposure,
    },
    {
        "id":          "02_crude_smoothed",
        "title":       "Taux bruts vs lissés (log)",
        "description": "Nuage des taux bruts (points gris), courbe lissée (rouge) et table de "
                       "référence (bleu tiretés) en échelle logarithmique. Graphique central de "
                       "tout rapport de construction de loi.",
        "category":    "builder",
        "fn":          chart_crude_smoothed,
    },
    {
        "id":          "03_ci_bands",
        "title":       "Intervalles de confiance Poisson 95%",
        "description": "Courbe lissée entourée du bandeau IC 95% (Poisson exact) et courbe de "
                       "référence. Montre la précision statistique de la table à chaque âge. "
                       "Des IC larges indiquent une faible exposition.",
        "category":    "builder",
        "fn":          chart_ci_bands,
    },
    {
        "id":          "04_survival",
        "title":       "Courbe de survie",
        "description": "Fonction de survie S(x) = ∏(1 - q_t) de l'âge minimal à x. "
                       "Compare la survie expérience et la référence. Intuitive pour "
                       "les assureurs et les actuaires de tarification.",
        "category":    "builder",
        "fn":          chart_survival,
    },
    {
        "id":          "05_abatement",
        "title":       "Facteurs d'abattement par âge",
        "description": "α_x = q_x_expérience / q_x_référence par âge. Bleu = sous-mortalité "
                       "(α ≤ 1), orange = sur-mortalité (α > 1). Ligne verte = SMR global. "
                       "Graphique clé pour la certification.",
        "category":    "builder",
        "fn":          chart_abatement,
    },
    {
        "id":          "06_smr_bars",
        "title":       "SMR par décennie",
        "description": "SMR moyen par tranche de 10 ans. Permet de détecter si la sous-mortalité "
                       "est homogène ou concentrée sur certaines tranches d'âge.",
        "category":    "builder",
        "fn":          chart_smr_bars,
    },
    {
        "id":          "07_age_pyramid",
        "title":       "Pyramide d'exposition",
        "description": "Distribution de l'exposition centrale (P-A) par tranches d'âge de 5 ans. "
                       "Visualise la structure démographique du portefeuille.",
        "category":    "descriptive",
        "fn":          chart_age_pyramid,
    },
    {
        "id":          "08_heatmap",
        "title":       "Heatmap mortalité relative",
        "description": "Encodage couleur du ratio expérience/référence par âge. "
                       "Vert = sous-mortalité, rouge = sur-mortalité. "
                       "Lecture rapide du profil global de mortalité.",
        "category":    "builder",
        "fn":          chart_mortality_heatmap,
    },
]


# ─── Génération HTML ──────────────────────────────────────────────────────────

def _generate_html(chart_paths: list[dict]) -> str:
    cards = ""
    for c in chart_paths:
        b64 = _fig_to_b64(c["path"])
        cards += f"""
        <div class="card">
          <div class="badge {'builder' if c['category'] == 'builder' else 'desc'}">
            {c['category']}
          </div>
          <h3>{c['title']}</h3>
          <p class="desc">{c['description']}</p>
          <img src="data:image/png;base64,{b64}" alt="{c['title']}">
          <div class="path">{c['path']}</div>
        </div>"""

    return f"""<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <title>Galerie de graphiques — Agent Actuariat</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
             background: #F5F3EE; margin: 0; padding: 24px; color: #333; }}
    h1   {{ color: #1A3A5C; border-bottom: 2px solid #2C5F8A; padding-bottom: 8px; }}
    h2   {{ color: #555; font-size: 14px; font-weight: normal; margin-bottom: 32px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(640px, 1fr));
              gap: 24px; }}
    .card {{ background: #FBF8F1; border-radius: 10px; padding: 20px;
              box-shadow: 0 2px 8px rgba(0,0,0,0.08); position: relative; }}
    .card h3 {{ margin: 8px 0 4px; color: #1A3A5C; font-size: 15px; }}
    .card .desc {{ color: #666; font-size: 13px; margin-bottom: 12px; }}
    .card img {{ width: 100%; border-radius: 6px; border: 1px solid #E8E3D8; }}
    .card .path {{ font-size: 11px; color: #aaa; margin-top: 6px; }}
    .badge {{ position: absolute; top: 16px; right: 16px; font-size: 10px;
               font-weight: 600; padding: 2px 8px; border-radius: 20px; text-transform: uppercase; }}
    .badge.builder {{ background: #EAF0F7; color: #2C5F8A; }}
    .badge.desc     {{ background: #EAF7EE; color: #27AE60; }}
  </style>
</head>
<body>
  <h1>🔢 Galerie de graphiques — Agent Actuariat</h1>
  <h2>Données synthétiques (Makeham, âges {AGE_MIN}–{AGE_MAX}) · Palette modifiable dans example.py</h2>
  <div class="grid">{cards}</div>
</body>
</html>"""


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        from scipy.stats import chi2
        from scipy.ndimage import uniform_filter1d
    except ImportError:
        print("⚠  scipy non disponible — installer : pip install scipy")
        sys.exit(1)

    print(f"Génération des données synthétiques (âges {AGE_MIN}–{AGE_MAX})…")
    data = _synthetic_data()

    chart_paths = []
    for chart_info in CHARTS:
        fn_name = chart_info["id"]
        print(f"  [{fn_name}] {chart_info['title']}…", end=" ")
        try:
            path = chart_info["fn"](data)
            chart_paths.append({**chart_info, "path": path})
            print("✓")
        except Exception as exc:
            print(f"✗ {exc}")

    # Générer l'index HTML
    html = _generate_html(chart_paths)
    html_path = OUT_DIR / "index.html"
    html_path.write_text(html, encoding="utf-8")

    print(f"\n{'─'*60}")
    print(f"✓  {len(chart_paths)} graphiques générés dans : {OUT_DIR}/")
    print(f"✓  Galerie HTML : file://{html_path}")
    print(f"{'─'*60}")
    print("\nPour personnaliser les couleurs, modifier PALETTE en tête de ce fichier.")
    print(f"Pour changer la plage d'âge : AGE_MIN = {AGE_MIN}, AGE_MAX = {AGE_MAX}")


if __name__ == "__main__":
    main()
