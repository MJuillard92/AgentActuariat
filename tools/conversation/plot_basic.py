"""
TOOL CONTRACT — conversation.plot_basic
════════════════════════════════════════════════════════════════

IDENTITY
--------
name          : conversation.plot_basic
domain        : conversation
version       : 1.0.0
author        : Marc Juillard
last_updated  : 2026-05-13

DESCRIPTION
-----------
Plots matplotlib courants — histogram, time_series, bar, scatter.
Produit un PNG dans tmp/conversation_plots/ et retourne son path.

WHEN TO USE
-----------
Phase conversationnelle, plot standard sur 1-2 colonnes.
Pour un plot complexe (multi-séries, KM, heatmap) utiliser eval_pandas.

INPUTS
------
params:
  function_name:
    type    : string
    values  : histogram | time_series | bar | scatter
    default : histogram
  column:
    type    : string
    note    : Colonne principale (histogram, bar).
  x_col, y_col:
    type    : string
    note    : Pour scatter / time_series.
  bins:
    type    : int
    default : 30
    note    : Nb de bins pour histogram.
  agg:
    type    : string
    default : sum
    values  : sum | mean | count
    note    : Agrégation pour time_series par année.

OUTPUTS
-------
return_payload:
  function_name : str
  png_path      : str — chemin relatif du PNG généré
  title         : str — titre du plot

CATALOGUE METADATA
------------------
display_name      : Plots rapides
short_description : Histogram / time-series / bar / scatter via matplotlib.
domain            : conversation
capability_group  : data_exploration
client_visible    : true
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd


def _out_path(prefix: str) -> Path:
    out_dir = Path("tmp/conversation_plots")
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{prefix}_{int(time.time() * 1000)}.png"


def _save_and_close(fig, path: Path) -> str:
    fig.savefig(path, dpi=100, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return str(path)


def run(df: pd.DataFrame, params: dict | None = None) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    params = params or {}
    fn = params.get("function_name", "histogram")

    if df is None or len(df) == 0:
        return {"erreur": "DataFrame vide."}

    if fn == "histogram":
        column = params.get("column")
        if not column or column not in df.columns:
            return {"erreur": f"Colonne '{column}' absente."}
        bins = int(params.get("bins", 30))
        fig, ax = plt.subplots(figsize=(8, 5))
        df[column].dropna().hist(bins=bins, ax=ax, color="#1A3668", edgecolor="white")
        title = f"Histogramme — {column}"
        ax.set_title(title)
        ax.set_xlabel(column)
        ax.set_ylabel("Effectif")
        return {"function_name": fn, "title": title,
                "png_path": _save_and_close(fig, _out_path("histogram"))}

    if fn == "bar":
        column = params.get("column")
        if not column or column not in df.columns:
            return {"erreur": f"Colonne '{column}' absente."}
        n = int(params.get("n", 15))
        counts = df[column].value_counts(dropna=False).head(n)
        fig, ax = plt.subplots(figsize=(8, 5))
        counts.plot.bar(ax=ax, color="#1A3668")
        title = f"Distribution — {column}"
        ax.set_title(title)
        ax.set_xlabel(column)
        ax.set_ylabel("Effectif")
        fig.autofmt_xdate()
        return {"function_name": fn, "title": title,
                "png_path": _save_and_close(fig, _out_path("bar"))}

    if fn == "scatter":
        x_col = params.get("x_col")
        y_col = params.get("y_col")
        if not x_col or x_col not in df.columns:
            return {"erreur": f"Colonne x '{x_col}' absente."}
        if not y_col or y_col not in df.columns:
            return {"erreur": f"Colonne y '{y_col}' absente."}
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(df[x_col], df[y_col], s=8, alpha=0.5, color="#1A3668")
        title = f"{y_col} vs {x_col}"
        ax.set_title(title)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        return {"function_name": fn, "title": title,
                "png_path": _save_and_close(fig, _out_path("scatter"))}

    if fn == "time_series":
        x_col = params.get("x_col")
        y_col = params.get("y_col")
        agg = params.get("agg", "sum")
        if not x_col or x_col not in df.columns:
            return {"erreur": f"Colonne x '{x_col}' absente."}
        if not y_col or y_col not in df.columns:
            return {"erreur": f"Colonne y '{y_col}' absente."}
        # x_col supposée date — on agrège par année
        x_dates = pd.to_datetime(df[x_col], format="mixed", dayfirst=True, errors="coerce")
        valid_mask = x_dates.notna()
        if not valid_mask.any():
            return {"erreur": f"Colonne '{x_col}' non parsable en date."}
        grouped = df[valid_mask].groupby(x_dates[valid_mask].dt.year)[y_col]
        ts = getattr(grouped, agg)() if agg in ("sum", "mean", "count") else grouped.sum()
        fig, ax = plt.subplots(figsize=(8, 5))
        ts.plot(ax=ax, color="#1A3668", marker="o")
        title = f"{y_col} par année ({agg})"
        ax.set_title(title)
        ax.set_xlabel("Année")
        ax.set_ylabel(y_col)
        return {"function_name": fn, "title": title,
                "png_path": _save_and_close(fig, _out_path("time_series"))}

    return {"erreur": f"function_name inconnu : '{fn}'. "
                       f"Valeurs : histogram | bar | scatter | time_series"}
