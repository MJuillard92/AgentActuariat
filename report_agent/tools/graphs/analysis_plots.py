"""
report_agent/tools/graphs/analysis_plots.py
Graphiques descriptifs du portefeuille (analyse statistique).

════════════════════════════════════════════════════════════════
INPUTS
════════════════════════════════════════════════════════════════
Colonnes du DataFrame (détectées automatiquement via column_schema) :

  Paramètres (params dict) :
    chart        : str   — type de graphique à produire :
                           "age_pyramid"   — pyramide des âges (H/F)
                           "time_series"   — décès et exposition par année
                           "segmentation"  — camemberts par variable catégorielle
    title_suffix : str   — texte ajouté au titre (défaut : "")
    by_sex       : bool  — pyramide H/F si true (défaut : false)

  Données statistiques pré-calculées (dans data store, optionnel) :
    data["ages"]        — sortie de statistical_analysis.age_distribution
    data["series"]      — sortie de statistical_analysis.time_series
    data["segmentation"]— sortie de statistical_analysis.segmentation

════════════════════════════════════════════════════════════════
OUTPUT  (dict)
════════════════════════════════════════════════════════════════
    chart        : str   — type de graphique produit
    image_b64    : str   — image PNG encodée en base64
    erreur       : str   (si données manquantes ou chart inconnu)
════════════════════════════════════════════════════════════════

Interface : run(data, params) -> dict
"""
from __future__ import annotations

import base64
import io
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_BG = "#FBF8F1"
_GRID = "#E8E3D8"
_BLUE = "#2C5F8A"
_RED = "#C0392B"
_GREEN = "#27AE60"
_ORANGE = "#E67E22"


def _to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor=_BG)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _theme(fig, axes):
    fig.patch.set_facecolor(_BG)
    for ax in axes:
        ax.set_facecolor(_BG)
        ax.grid(True, color=_GRID, linewidth=0.8, alpha=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)


def _age_pyramid(data: dict, params: dict) -> dict:
    ages_data = data.get("ages") or {}
    by_sex = params.get("by_sex", False)
    title_suffix = params.get("title_suffix", "")

    def _dist_to_arrays(dist_obj):
        """Convertit distribution (dict ou list) en (bands, counts)."""
        if isinstance(dist_obj, dict):
            return list(dist_obj.keys()), list(dist_obj.values())
        if isinstance(dist_obj, list):
            df = pd.DataFrame(dist_obj)
            band_col = next((c for c in ("tranche", "band", "age_band") if c in df.columns), df.columns[0])
            count_col = next((c for c in ("count", "nb", "n") if c in df.columns), df.columns[1])
            return df[band_col].astype(str).tolist(), df[count_col].tolist()
        return [], []

    if by_sex and "distribution_H" in ages_data and "distribution_F" in ages_data:
        bands_h, counts_h = _dist_to_arrays(ages_data["distribution_H"])
        bands_f, counts_f = _dist_to_arrays(ages_data["distribution_F"])
        bands = bands_h or bands_f
        fig, ax = plt.subplots(figsize=(11, 6))
        x = np.arange(len(bands))
        width = 0.35
        ax.bar(x - width / 2, counts_h, width, color=_BLUE, alpha=0.8, label="Hommes")
        ax.bar(x + width / 2, counts_f, width, color=_RED, alpha=0.8, label="Femmes")
        ax.set_xticks(x)
        ax.set_xticklabels(bands, rotation=45, ha="right", fontsize=8)
        ax.legend(facecolor=_BG, edgecolor=_GRID)
    elif "distribution" in ages_data:
        bands, counts = _dist_to_arrays(ages_data["distribution"])
        fig, ax = plt.subplots(figsize=(11, 5))
        x = np.arange(len(bands))
        ax.bar(x, counts, color=_BLUE, alpha=0.85, edgecolor="none")
        ax.set_xticks(x)
        ax.set_xticklabels(bands, rotation=45, ha="right", fontsize=8)
    else:
        return {"erreur": "Données ages manquantes dans data store. Appeler statistical_analysis.age_distribution d'abord."}

    title = f"Distribution des âges{' — ' + title_suffix if title_suffix else ''}"
    ax.set_title(title, fontsize=11, loc="left")
    ax.set_ylabel("Nombre de contrats")
    _theme(fig, [ax])
    return {"chart": "age_pyramid", "image_b64": _to_b64(fig)}


def _time_series(data: dict, params: dict) -> dict:
    series_data = data.get("series") or {}
    title_suffix = params.get("title_suffix", "")
    serie = series_data.get("serie")
    if not serie:
        return {"erreur": "Données series manquantes dans data store. Appeler statistical_analysis.time_series d'abord."}

    df = pd.DataFrame(serie)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax1.fill_between(df["annee"], df["exposition_pa"], alpha=0.4, color=_BLUE)
    ax1.plot(df["annee"], df["exposition_pa"], color=_BLUE, linewidth=2)
    ax1.set_ylabel("Exposition (P-A)")
    ax1.set_title(f"Exposition et décès par année{' — ' + title_suffix if title_suffix else ''}", fontsize=11, loc="left")

    ax2.bar(df["annee"], df["nb_deces"], color=_RED, alpha=0.8, edgecolor="none")
    ax2.set_ylabel("Décès")
    ax2.set_xlabel("Année")

    _theme(fig, [ax1, ax2])
    fig.tight_layout()
    return {"chart": "time_series", "image_b64": _to_b64(fig)}


def _segmentation(data: dict, params: dict) -> dict:
    seg_data = data.get("segmentation") or {}
    title_suffix = params.get("title_suffix", "")
    segs = seg_data.get("segmentations")
    if not segs:
        return {"erreur": "Données segmentation manquantes dans data store. Appeler statistical_analysis.segmentation d'abord."}

    n = len(segs)
    if n == 0:
        return {"erreur": "Aucune segmentation disponible."}

    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    if n == 1:
        axes = [axes]
    elif rows == 1:
        axes = list(axes)
    else:
        axes = [ax for row in axes for ax in row]

    colors = [_BLUE, _RED, _GREEN, _ORANGE, "#8E44AD"]
    for i, (var_name, records) in enumerate(segs.items()):
        ax = axes[i]
        df = pd.DataFrame(records)
        labels = df["valeur"].astype(str).values
        sizes = df["nb_contrats"].values
        ax.pie(sizes, labels=labels, autopct="%1.1f%%",
               colors=[colors[j % len(colors)] for j in range(len(labels))],
               startangle=90)
        ax.set_title(var_name, fontsize=10)

    # Masquer les axes vides
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    title = f"Répartition{' — ' + title_suffix if title_suffix else ''}"
    fig.suptitle(title, fontsize=11, y=1.02)
    _theme(fig, [])
    fig.patch.set_facecolor(_BG)
    return {"chart": "segmentation", "image_b64": _to_b64(fig)}


_CHARTS = {
    "age_pyramid":  _age_pyramid,
    "time_series":  _time_series,
    "segmentation": _segmentation,
}


def run(data: dict | None, params: dict | None = None) -> dict:
    data = data or {}
    params = params or {}
    chart = params.get("chart", "age_pyramid")
    fn = _CHARTS.get(chart)
    if fn is None:
        return {"erreur": f"chart inconnu : '{chart}'. Valeurs : {list(_CHARTS)}"}
    result = fn(data, params)
    if "chart" not in result:
        result["chart"] = chart
    return result
