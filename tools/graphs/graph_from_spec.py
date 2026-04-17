"""
TOOL CONTRACT — graphs.graph_from_spec
════════════════════════════════════════════════════════════════

IDENTITY
--------
name          : graphs.graph_from_spec
domain        : mortality_experience
version       : 1.0.0
author        : Marc Juillard
last_updated  : 2026-04-07

DESCRIPTION
-----------
Generates a PNG graph from a YAML spec + resolved context dict.
Dispatches to existing builder_plots tools when spec.id matches a known
chart type; otherwise builds a generic matplotlib figure.
Returns the path to the saved PNG file.

WHEN TO USE
-----------
Called by the WriterAgent for each graph element in the processing_sequence.

INPUTS
------
params:
  spec:
    type : dict
    note : Graph spec from the YAML template section.
  context:
    type : dict
    note : Resolved context dict from load_yaml_template.

OUTPUTS
-------
return_payload:
  path    : str  — absolute path to PNG file
  warning : str  — populated if graph could not be generated

CATALOGUE METADATA
------------------
display_name      : Graphique depuis spec YAML
short_description : Génère un PNG depuis un spec YAML (dispatch ou matplotlib générique).
domain            : mortality_experience
capability_group  : reporting
depends_on        : [build_pdf.load_yaml_template]
required_by       : [build_pdf.assemble_sections]
client_visible    : false
"""
from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# ── Dispatch table : YAML graph id → builder_plots chart name ─────────────────
_DISPATCH: dict[str, str] = {
    # Exposition
    "graph_exposure_by_age":         "exposure",
    "graph_exposure_distribution":   "exposure",
    # Décès
    "graph_deaths_by_age":           "deaths_by_age",
    "graph_deaths_distribution":     "deaths_by_age",
    # Observé vs modélisé + IC
    "graph_comparison":              "obs_vs_modeled",
    "graph_obs_vs_modeled_by_age":   "obs_vs_modeled",
    # Comparaison avec table antérieure
    "graph_prior_comparison":        "rate_ratio",
    "graph_rate_ratio":              "rate_ratio",
    # Abattements / remises
    "graph_discounts":               "discount_line",
    "graph_discount_factors":        "discount_line",
    # Taux bruts + lissés
    "graph_crude_smoothed":          "crude_smoothed",
    # SMR
    "graph_smr":                     "smr",
    # Courbe de survie
    "graph_survival_curve":          "survival_curve",
}


def _flatten_for_builder_plots(data: dict) -> dict:
    """
    builder_plots expects keys at root level (e.g. data["abatement_table"]).
    The data_store stores them nested (data["benchmarking"]["abatement_table"]).
    This function flattens the known nested structures.
    """
    flat = dict(data)

    # Flatten benchmarking
    bm = data.get("benchmarking")
    if isinstance(bm, dict):
        for k in ("abatement_table", "smr_global", "reference_name", "smr_by_decade", "summary"):
            if k in bm and k not in flat:
                flat[k] = bm[k]

    # Flatten validation
    val = data.get("validation")
    if isinstance(val, dict):
        for k in ("ci_table", "p_value", "alpha"):
            if k in val and k not in flat:
                flat[k] = val[k]

    # Flatten diagnostics
    diag = data.get("diagnostics")
    if isinstance(diag, dict):
        for k in ("low_credibility_ages", "n_non_monotone", "n_low", "recommendation"):
            if k in diag and k not in flat:
                flat[k] = diag[k]

    return flat


def _call_builder_plots(chart_name: str, data: dict, params: dict) -> bytes | None:
    """Calls graphs.builder_plots.run() and returns PNG bytes."""
    try:
        from tools.graphs.builder_plots import run as bp_run
        flat_data = _flatten_for_builder_plots(data)
        result = bp_run(data=flat_data, params={"chart": chart_name, **params})
        if "erreur" in result:
            log.warning("[graph_from_spec] builder_plots.%s erreur: %s", chart_name, result["erreur"])
            return None
        img_b64 = result.get("image_b64") or result.get("chart_b64")
        if img_b64:
            import base64
            return base64.b64decode(img_b64)
    except Exception as exc:
        log.warning("[graph_from_spec] builder_plots.%s failed: %s", chart_name, exc)
    return None


def _generic_line_chart(
    title: str,
    series: dict[str, dict[str, float]],
    xlabel: str = "Âge",
    ylabel: str = "Valeur",
    output_path: str | None = None,
) -> str | None:
    """Build a generic line chart with matplotlib."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker

        fig, ax = plt.subplots(figsize=(10, 5))
        colors_cycle = ["#1A3668", "#E25B34", "#2CA02C", "#9467BD", "#8C564B"]

        for i, (label, pts) in enumerate(series.items()):
            if not pts:
                continue
            ages  = [int(a) for a in sorted(pts.keys(), key=lambda x: int(x))]
            vals  = [pts[str(a)] if str(a) in pts else pts.get(a, 0) for a in ages]
            ax.plot(ages, vals, label=label, color=colors_cycle[i % len(colors_cycle)], linewidth=1.5)

        ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
        plt.tight_layout()

        path = output_path or os.path.join(tempfile.gettempdir(), f"graph_{title[:20].replace(' ', '_')}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path
    except Exception as exc:
        log.error("[graph_from_spec] generic chart error: %s", exc)
        return None


def _generic_bar_chart(
    title: str,
    series: dict[str, float],
    xlabel: str = "Catégorie",
    ylabel: str = "Valeur",
    output_path: str | None = None,
) -> str | None:
    """Build a generic bar chart with matplotlib."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 5))
        labels = list(series.keys())
        vals   = [series[k] for k in labels]
        ax.bar(labels, vals, color="#1A3668", edgecolor="white")
        ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()

        path = output_path or os.path.join(tempfile.gettempdir(), f"bar_{title[:20].replace(' ', '_')}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path
    except Exception as exc:
        log.error("[graph_from_spec] bar chart error: %s", exc)
        return None


def generate_graph_from_spec(spec: dict, data: dict) -> str:
    """
    Main function: generates a PNG from spec + data.
    Returns absolute path to PNG, or "" if generation failed.
    """
    if not spec:
        return ""

    graph_id  = spec.get("id", "graph")
    title     = spec.get("title", graph_id)
    output_dir = tempfile.gettempdir()
    output_path = os.path.join(output_dir, f"{graph_id}.png")

    # ── Try dispatch to builder_plots ─────────────────────────────────────────
    dispatched = False
    chart_name = _DISPATCH.get(graph_id)
    if chart_name:
        dispatched = True
        extra_params = {k: v for k, v in spec.items() if k not in ("id", "title", "type")}
        png_bytes = _call_builder_plots(chart_name, data, extra_params)
        if png_bytes:
            with open(output_path, "wb") as f:
                f.write(png_bytes)
            return output_path

    # ── Generic fallback — only if explicit series keys are declared in spec ──
    # Do NOT auto-detect all age-indexed series (produces garbage charts).
    chart_type  = spec.get("type", "line")
    series_keys = spec.get("series", [])
    xlabel      = spec.get("xlabel", "Âge")
    ylabel      = spec.get("ylabel", "Valeur")

    if not series_keys:
        # Dispatch failed and no series specified → give up cleanly
        if dispatched:
            log.warning("[graph_from_spec] %s — builder_plots dispatch échoué, pas de series déclarée", graph_id)
        return ""

    if chart_type == "bar":
        key = series_keys[0] if isinstance(series_keys, list) else series_keys
        raw = data.get(key) or {}
        if isinstance(raw, dict) and raw:
            # Filter non-numeric values
            filtered = {}
            for k, v in raw.items():
                try:
                    filtered[str(k)] = float(v)
                except (TypeError, ValueError):
                    pass
            if filtered:
                path = _generic_bar_chart(title, filtered, xlabel, ylabel, output_path)
                return path or ""
        return ""

    # Line chart with explicit series keys
    series_dict: dict[str, dict] = {}
    for key in (series_keys if isinstance(series_keys, list) else [series_keys]):
        raw = data.get(key) or {}
        if isinstance(raw, dict) and raw:
            filtered = {}
            for k, v in raw.items():
                try:
                    filtered[str(k)] = float(v)
                except (TypeError, ValueError):
                    pass
            if filtered:
                series_dict[key] = filtered
    if series_dict:
        path = _generic_line_chart(title, series_dict, xlabel, ylabel, output_path)
        return path or ""

    log.warning("[graph_from_spec] %s — impossible de générer le graphique (données insuffisantes)", graph_id)
    return ""


# ── Tool entry point ──────────────────────────────────────────────────────────

def run(data: dict | None = None, params: dict | None = None) -> dict:
    """
    Entry point for tool registry.
    params: {spec: dict}
    context is pulled from data["template_context"] or data directly.
    """
    data    = data    or {}
    params  = params  or {}
    spec    = params.get("spec") or {}
    context = params.get("context") or data.get("template_context") or data

    path = generate_graph_from_spec(spec, context)

    result = {
        "path":    path,
        "success": bool(path),
        "warning": "" if path else f"Graphique {spec.get('id', '?')} non généré.",
    }
    data["_last_graph_path"] = path
    return result
