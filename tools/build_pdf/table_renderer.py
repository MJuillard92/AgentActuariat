"""
TOOL CONTRACT — build_pdf.table_renderer
════════════════════════════════════════════════════════════════

IDENTITY
--------
name          : build_pdf.table_renderer
domain        : mortality_experience
version       : 1.0.0
author        : Marc Juillard
last_updated  : 2026-04-07

DESCRIPTION
-----------
Renders data tables from YAML spec + resolved context dict.
Called by the WriterAgent for each table element in the processing_sequence.

Returns (reportlab_table_obj, html_string). If required data is missing,
returns (None, "") with a warning logged — non-blocking.

WHEN TO USE
-----------
After build_pdf.load_yaml_template to render each table in the section spec.

OUTPUTS
-------
return_payload:
  rl_table  : reportlab Table object (or None)
  html      : str — HTML table string for display fallback
  warning   : str — populated if data missing, empty if success

CATALOGUE METADATA
------------------
display_name      : Rendu table depuis spec YAML
short_description : Construit un objet ReportLab Table depuis un spec YAML.
domain            : mortality_experience
capability_group  : reporting
depends_on        : [build_pdf.load_yaml_template]
required_by       : [build_pdf.assemble_sections]
client_visible    : false
"""
from __future__ import annotations

import logging
import math
from typing import Any

log = logging.getLogger(__name__)

# ReportLab is imported lazily inside functions (to allow md5 patch to be applied first)
_RL_AVAILABLE: bool | None = None  # None = not yet checked
Table = TableStyle = colors = cm = None  # type: ignore
BLUE = LBLUE = LIGHT = GREY = None


def _ensure_reportlab():
    """Lazy import + patch MD5 for Python 3.8 compatibility."""
    global _RL_AVAILABLE, Table, TableStyle, colors, cm, BLUE, LBLUE, LIGHT, GREY

    if _RL_AVAILABLE is not None:
        return _RL_AVAILABLE

    # Patch hashlib.md5 before reportlab import (Python 3.8 / OpenSSL compat)
    import hashlib as _hl
    _orig = _hl.md5
    def _md5c(*a, **kw):
        kw.pop("usedforsecurity", None)
        return _orig(*a, **kw)
    _hl.md5 = _md5c

    try:
        from reportlab.platypus import Table as _T, TableStyle as _TS
        from reportlab.lib import colors as _c
        from reportlab.lib.units import cm as _cm
        Table = _T; TableStyle = _TS; colors = _c; cm = _cm
        BLUE  = _c.HexColor("#1A3668")
        LBLUE = _c.HexColor("#D6E4F7")
        LIGHT = _c.HexColor("#F5F8FF")
        GREY  = _c.HexColor("#888888")
        _RL_AVAILABLE = True
    except ImportError:
        _RL_AVAILABLE = False

    return _RL_AVAILABLE


def _fmt(val: Any, fmt: str = "") -> str:
    """Format a single cell value according to a format string."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "—"
    if fmt == "pct":
        return f"{float(val):.1%}"
    if fmt == "int":
        return f"{int(val):,}"
    if fmt == "float2":
        return f"{float(val):.2f}"
    if fmt == "float4":
        return f"{float(val):.4f}"
    if fmt == "sci":
        return f"{float(val):.3e}"
    return str(val)


def _build_age_rows(columns: list[dict], data: dict) -> list[list[str]]:
    """
    Build rows from age-indexed data vectors.
    columns: list of {key: placeholder_name, label: str, format: str}
    data: resolved context dict (placeholder_name → {age_str: value})
    """
    # Collect all ages across all series
    ages: set[int] = set()
    for col in columns:
        series = data.get(col["key"]) or {}
        if isinstance(series, dict):
            ages.update(int(a) for a in series.keys() if str(a).isdigit())

    if not ages:
        return []

    rows = []
    for age in sorted(ages):
        row = [str(age)]
        for col in columns:
            series = data.get(col["key"]) or {}
            val = series.get(str(age)) or series.get(age)
            row.append(_fmt(val, col.get("format", "")))
        rows.append(row)
    return rows


def _build_static_rows(rows_spec: list, data: dict, col_formats: list[str]) -> list[list[str]]:
    """Build rows from a static list spec."""
    result = []
    for row_spec in rows_spec:
        if isinstance(row_spec, dict):
            # Resolve any placeholder values
            row = []
            for i, val in enumerate(row_spec.values()):
                fmt = col_formats[i] if i < len(col_formats) else ""
                if isinstance(val, str) and val.startswith("{{") and val.endswith("}}"):
                    key = val[2:-2].strip()
                    resolved = data.get(key, val)
                    row.append(_fmt(resolved, fmt))
                else:
                    row.append(_fmt(val, fmt))
            result.append(row)
        elif isinstance(row_spec, list):
            row = [_fmt(v, col_formats[i] if i < len(col_formats) else "") for i, v in enumerate(row_spec)]
            result.append(row)
    return result


def render_table_from_spec(spec: dict, data: dict) -> tuple[Any, str, list]:
    """
    Returns (reportlab_Table | None, html_str).
    spec keys: id, name, columns, rows, highlight_rule, format
    """
    if not spec:
        return None, ""

    table_id   = spec.get("id", "table")
    table_name = spec.get("name", table_id)
    columns    = spec.get("columns", [])
    rows_spec  = spec.get("rows", "dynamic")
    col_formats = [c.get("format", "") for c in columns] if columns else []

    headers = [c.get("label", c.get("key", str(i))) for i, c in enumerate(columns)] if columns else []

    # Build rows
    if rows_spec == "dynamic" or rows_spec == "age_indexed":
        data_rows = _build_age_rows(columns[1:] if headers and headers[0].lower() == "âge" else columns, data)
        if headers and headers[0].lower() in ("âge", "age"):
            pass  # age column already prepended by _build_age_rows
    elif isinstance(rows_spec, list):
        data_rows = _build_static_rows(rows_spec, data, col_formats)
    else:
        data_rows = []

    if not data_rows:
        log.warning("[table_renderer] %s — aucune ligne générée (données manquantes ?)", table_id)
        return None, "", []

    all_rows = [headers] + data_rows if headers else data_rows

    # ── HTML fallback ─────────────────────────────────────────────────────────
    html_rows = []
    for i, row in enumerate(all_rows):
        tag = "th" if i == 0 else "td"
        cells = "".join(f"<{tag}>{v}</{tag}>" for v in row)
        html_rows.append(f"<tr>{cells}</tr>")
    html = f"<table border='1'><caption>{table_name}</caption>{''.join(html_rows)}</table>"

    if not _ensure_reportlab():
        return None, html

    # ── ReportLab Table ───────────────────────────────────────────────────────
    n_cols = max(len(r) for r in all_rows)
    col_width = 16 * cm / n_cols if n_cols else 2 * cm

    try:
        tbl = Table(all_rows, colWidths=[col_width] * n_cols, repeatRows=1)

        style_cmds = [
            ("BACKGROUND",  (0, 0), (-1, 0), BLUE),
            ("TEXTCOLOR",   (0, 0), (-1, 0), colors.white),
            ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",    (0, 0), (-1, -1), 8),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [LIGHT, colors.white]),
            ("GRID",        (0, 0), (-1, -1), 0.3, GREY),
            ("TOPPADDING",  (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("ALIGN",       (0, 0), (-1, -1), "CENTER"),
        ]

        # Totals row highlight
        highlight_rule = spec.get("highlight_rule", "")
        if highlight_rule == "totals_row" and len(all_rows) > 1:
            style_cmds.append(("BACKGROUND", (0, -1), (-1, -1), LBLUE))
            style_cmds.append(("FONTNAME",   (0, -1), (-1, -1), "Helvetica-Bold"))

        tbl.setStyle(TableStyle(style_cmds))
        return tbl, html, all_rows

    except Exception as exc:
        log.error("[table_renderer] Erreur ReportLab pour %s : %s", table_id, exc)
        return None, html, all_rows


def render_statistical_output(spec: dict, data: dict) -> tuple[Any, str, list]:
    """
    Renders a statistical summary table (Cox, logit, chi-squared, annual check).
    Returns (reportlab_Table | None, html_str).
    If required data is absent → returns (None, "") with warning.
    """
    if not spec:
        return None, "", []

    stat_type = spec.get("type", "")
    title     = spec.get("name", stat_type)

    rows: list[list[str]] = []

    # ── Cox regression ────────────────────────────────────────────────────────
    if stat_type == "cox_proportional_hazards":
        cox = data.get("cox_regression") or {}
        if not cox:
            log.warning("[table_renderer] cox_regression absent — section Cox ignorée")
            return None, "", []
        rows = [
            ["Paramètre", "Valeur"],
            ["Hazard Ratio (H/F)", _fmt(cox.get("hazard_ratio"), "float2")],
            ["IC 95%", f"[{_fmt(cox.get('ci_lower_95'), 'float2')}, {_fmt(cox.get('ci_upper_95'), 'float2')}]"],
            ["p-value", _fmt(cox.get("cox_pvalue"), "sci")],
            ["Décès hommes", _fmt(cox.get("deaths_male"), "int")],
            ["Décès femmes", _fmt(cox.get("deaths_female"), "int")],
            ["Taux brut hommes (‰)", _fmt(cox.get("crude_rate_male"), "float4")],
            ["Taux brut femmes (‰)", _fmt(cox.get("crude_rate_female"), "float4")],
        ]

    # ── Logit regression ──────────────────────────────────────────────────────
    elif stat_type == "logit_regression":
        logit = data.get("logit_regression") or {}
        if not logit:
            log.warning("[table_renderer] logit_regression absent — section logit ignorée")
            return None, "", []
        rows = [
            ["Paramètre", "Valeur"],
            ["Pente α", _fmt(logit.get("slope_alpha"), "float4")],
            ["Intercept β", _fmt(logit.get("intercept_beta"), "float4")],
            ["R²", _fmt(logit.get("r_squared"), "float4")],
            ["p-value", _fmt(logit.get("p_value"), "sci")],
            ["Nombre d'âges", _fmt(logit.get("n_ages"), "int")],
            ["Formule", logit.get("formula", "—")],
        ]

    # ── Annual cohort check ────────────────────────────────────────────────────
    elif stat_type == "annual_cohort_check":
        ratios = data.get("annual_prediction_ratio") or {}
        if not ratios:
            log.warning("[table_renderer] annual_prediction_ratio absent — section ignorée")
            return None, "", []
        rows = [["Année", "Ratio observé/modélisé"]]
        for yr in sorted(ratios.keys()):
            rows.append([str(yr), _fmt(ratios[yr], "float2")])

    # ── Chi-squared test ──────────────────────────────────────────────────────
    elif stat_type == "chi_squared":
        validation = data.get("validation") or {}
        p_val = validation.get("p_value") if isinstance(validation, dict) else data.get("chi_squared_p")
        if p_val is None:
            log.warning("[table_renderer] chi_squared_p absent")
            return None, "", []
        rows = [
            ["Test", "Résultat"],
            ["Chi² p-value", _fmt(p_val, "sci")],
            ["Significatif (α=5%)", "Oui" if float(p_val) < 0.05 else "Non"],
        ]

    else:
        log.warning("[table_renderer] type de stat inconnu : %s", stat_type)
        return None, "", []

    # ── Build table ───────────────────────────────────────────────────────────
    html_rows = []
    for i, row in enumerate(rows):
        tag = "th" if i == 0 else "td"
        cells = "".join(f"<{tag}>{v}</{tag}>" for v in row)
        html_rows.append(f"<tr>{cells}</tr>")
    html = f"<table border='1'><caption>{title}</caption>{''.join(html_rows)}</table>"

    if not _ensure_reportlab():
        return None, html, rows

    try:
        n_cols = max(len(r) for r in rows)
        col_w  = [8 * cm, 8 * cm] if n_cols == 2 else [16 * cm / n_cols] * n_cols

        tbl = Table(rows, colWidths=col_w, repeatRows=1)
        tbl.setStyle(TableStyle([
            ("BACKGROUND",  (0, 0), (-1, 0), BLUE),
            ("TEXTCOLOR",   (0, 0), (-1, 0), colors.white),
            ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",    (0, 0), (-1, -1), 8),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [LIGHT, colors.white]),
            ("GRID",        (0, 0), (-1, -1), 0.3, GREY),
            ("TOPPADDING",  (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("ALIGN",       (0, 1), (-1, -1), "CENTER"),
            ("ALIGN",       (0, 0), (0, -1), "LEFT"),
        ]))
        return tbl, html, rows

    except Exception as exc:
        log.error("[table_renderer] Erreur ReportLab stat (%s) : %s", stat_type, exc)
        return None, html, rows


# ── Tool entry point ──────────────────────────────────────────────────────────

def run(data: dict | None = None, params: dict | None = None) -> dict:
    """
    Entry point for tool registry.
    params: {spec: dict, render_type: "table" | "stat"}
    Writes result to data["_last_table_render"].
    """
    data   = data   or {}
    params = params or {}

    spec        = params.get("spec") or {}
    render_type = params.get("render_type", "table")
    context     = params.get("context") or data.get("template_context") or data

    if render_type == "stat":
        rl_tbl, html, rows = render_statistical_output(spec, context)
    else:
        rl_tbl, html, rows = render_table_from_spec(spec, context)

    result = {
        "rendered": rl_tbl is not None,
        "html":     html,
        "warning":  "" if rl_tbl is not None else "Données manquantes pour ce tableau.",
    }
    data["_last_table_render"] = result
    # Store raw rows (list-of-lists) for write_section to accumulate into section_outputs
    if rows:
        data["_last_table_rows"] = rows
    return result
