"""
agents/report/pipeline/04_redaction.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ÉTAPE 4 — Boucle Python déterministe + LLM par section (parallèle)

Reçoit le ReportPlan enrichi (étape 03).
Pour chaque section :
  1. Appelle les tools tableaux/graphiques (déterministe)
  2. Appelle GPT-4o avec le prompt de section enrichi (RAG inclus)
  3. Stocke le résultat dans section_outputs via write_section

Les sections sont traitées en parallèle via ThreadPoolExecutor :
  - Chaque thread reçoit un snapshot read-only du data_store
  - Pas d'écriture partagée pendant l'exécution parallèle
  - Les résultats sont écrits séquentiellement à la fin (ordre du plan préservé)
  - max_workers=5 pour éviter les 429 OpenAI TPM

Interface publique :
    redact_plan(plan, data_store) -> dict
        retourne data_store mis à jour avec section_outputs rempli
"""
from __future__ import annotations

import concurrent.futures
import logging
import math
from typing import Any

log = logging.getLogger(__name__)

# Cache lazy des formats globaux du YAML (chargés une fois par run)
_FORMATS_CACHE: dict | None = None


def _get_formats() -> dict:
    """Charge formats:{defaults, na_display, number_separator} depuis le YAML.
    Cache au niveau module pour ne pas relire le YAML à chaque cellule."""
    global _FORMATS_CACHE
    if _FORMATS_CACHE is None:
        try:
            from knowledge_base.report_template.template_loader import load_formats
            _FORMATS_CACHE = load_formats()
        except Exception:
            _FORMATS_CACHE = {
                "defaults": {}, "na_display": "—", "number_separator": " ",
            }
    return _FORMATS_CACHE


def _format_cell(value: Any, fmt: str, na_display: str = "—",
                  thousand_sep: str = " ") -> str:
    """Délègue à `tools.build_pdf.table_renderer._fmt` pour garder une
    SEULE source de vérité (sinon désynchros silencieuses entre PDF et
    contexte narrative — bug réel rencontré avec pct2 reconnu par _fmt
    mais pas par cette fonction → valeurs brutes affichées).

    `na_display` et `thousand_sep` sont conservés en signature pour
    rétro-compat mais n'ont plus d'effet : _fmt utilise par convention
    "—" pour les None/NaN et l'espace fine comme séparateur."""
    if value is None:
        return na_display
    if isinstance(value, float) and math.isnan(value):
        return na_display
    from tools.build_pdf.table_renderer import _fmt
    return _fmt(value, fmt)

_MAX_TOKENS_NARRATIVE = 1200
_TEMPERATURE          = 0.4   # Faible : style professionnel, peu créatif


# ── Hydratation des specs YAML avec données réelles ──────────────────────────

def _resolve_source(source: str | None, data_store: dict):
    """Résout une `source` de visual_spec qui peut être une clé directe
    (`segmentations`) ou un sub-path pointé (`segmentations.sexe`)."""
    if not source:
        return None
    if "." not in source:
        return data_store.get(source)
    root, *parts = source.split(".")
    cur = data_store.get(root)
    for p in parts:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(p)
    return cur


def _hydrate_visual_spec(spec: dict, data_store: dict) -> dict:
    """Design 3 : `source` pointe vers une clé data_store contenant la donnée,
    éventuellement via un sub-path pointé (ex. `segmentations.sexe`).
    Tableau → (headers, rows). Chart simple → (x_values, y_values).
    Chart multi_series → series_hydrated (liste de séries avec leurs propres
    sources, styles et valeurs)."""
    stype = spec.get("type")

    # ── Chart multi-séries : chaque série a sa propre source ──────────────
    # Pas de `source` global ; le spec déclare une liste `series:` où
    # chaque entrée pointe vers sa propre source/keys.
    if stype == "chart" and spec.get("chart_type") == "multi_series":
        series_specs = spec.get("series") or []
        series_hydrated: list[dict] = []
        for s in series_specs:
            src = s.get("source")
            d = _resolve_source(src, data_store) or []
            if not isinstance(d, list):
                # Sources comme dict (rare) → skip silencieusement
                continue
            key_x = s.get("key_x", "age")
            entry = {
                "style":  s.get("style", "line"),
                "label":  s.get("label", src or ""),
                "color":  s.get("color"),
                "alpha":  s.get("alpha", 1.0),
                "xs":     [row.get(key_x) for row in d],
            }
            if s.get("style") == "area":
                # Aire : deux y nécessaires (lower + upper)
                entry["ys_lower"] = [row.get(s.get("key_y_lower", "ci_lower")) for row in d]
                entry["ys_upper"] = [row.get(s.get("key_y_upper", "ci_upper")) for row in d]
            else:
                key_y = s.get("key_y", "value")
                entry["ys"] = [row.get(key_y) for row in d]
            series_hydrated.append(entry)
        return {
            **spec,
            "series_hydrated": series_hydrated,
            "x_label":  (spec.get("x_axis") or {}).get("label", ""),
            "y_label":  (spec.get("y_axis") or {}).get("label", ""),
            "error":    None,
        }

    # ── Cas mono-source (tableaux + chart classique) ──────────────────────
    source = spec.get("source")
    data = _resolve_source(source, data_store)

    if data is None:
        return {**spec, "error": f"source '{source}' absente du data_store"}

    if stype == "table":
        columns = spec.get("columns", [])
        headers = [c.get("label", c.get("key", "")) for c in columns]
        # Application des formats : inline (col["format"]) > formats.defaults[key]
        fmt_cfg = _get_formats()
        defaults = fmt_cfg.get("defaults") or {}
        na = fmt_cfg.get("na_display", "—")
        sep = fmt_cfg.get("number_separator", " ")
        rows = []
        for row in (data or []):
            row_out = []
            for c in columns:
                key = c["key"]
                val = row.get(key)
                fmt = c.get("format") or defaults.get(key, "")
                row_out.append(_format_cell(val, fmt, na_display=na, thousand_sep=sep))
            rows.append(row_out)
        return {**spec, "headers": headers, "rows": rows, "error": None}

    if stype == "chart":
        x_key = (spec.get("x_axis") or {}).get("key")
        y_key = (spec.get("y_axis") or {}).get("key")
        return {
            **spec,
            "x_values": [row.get(x_key) for row in (data or [])],
            "y_values": [row.get(y_key) for row in (data or [])],
            "x_label":  (spec.get("x_axis") or {}).get("label", x_key),
            "y_label":  (spec.get("y_axis") or {}).get("label", y_key),
            "error":    None,
        }

    return {**spec, "error": f"type non supporté: {stype}"}


def _hydrate_table_spec_LEGACY_DESIGN1(spec: dict, context: dict) -> dict:
    """
    LEGACY Design 1 — conservé temporairement pour éviter les régressions sur
    le code historique. N'est plus appelé par le pipeline Design 3 (US-24).
    TODO(US-27): supprimer après cutover complet.
    """
    import copy
    spec = copy.deepcopy(spec)
    sid  = spec.get("id", "")

    if sid == "table_construction":
        n0 = context.get("initial_record_count")
        nx = context.get("total_exclusions", 0) or 0
        n1 = context.get("final_record_count") or (int(n0) - int(nx) if n0 else None)
        if n0 is not None:
            spec["rows"] = [
                ["Étape", "Effectif", "Notes"],
                ["Enregistrements initiaux", str(n0), ""],
                ["Exclusions", str(nx), "hors plage d'âge, exposition nulle"],
                ["Enregistrements finaux", str(n1 or ""), ""],
            ]

    elif sid == "exposure_stats":
        series = context.get("series") or {}
        serie  = series.get("serie", []) if isinstance(series, dict) else []
        if not serie:
            eby   = context.get("exposure_by_year") or {}
            serie = [{"annee": k, "exposition_pa": v} for k, v in eby.items()]
        if serie:
            rows = [["Année", "Exposition (a.-p.)", "Âge moyen", "Genre (H/F %)"]]
            for r in sorted(serie, key=lambda x: x.get("annee", 0)):
                rows.append([
                    str(r.get("annee", "")),
                    f"{r.get('exposition_pa', 0):,.0f}",
                    str(r.get("age_moyen", "")),
                    str(r.get("pct_by_sex", r.get("gender_split", ""))),
                ])
            spec["rows"] = rows

    elif sid == "death_stats":
        series = context.get("series") or {}
        serie  = series.get("serie", []) if isinstance(series, dict) else []
        if not serie:
            dby   = context.get("deaths_by_year") or {}
            serie = [{"annee": k, "nb_deces": v} for k, v in dby.items()]
        if serie:
            rows = [["Année", "Décès", "Taux (‰)", "Âge moyen au décès"]]
            for r in sorted(serie, key=lambda x: x.get("annee", 0)):
                tm = r.get("taux_mortalite") or r.get("death_rate")
                rows.append([
                    str(r.get("annee", "")),
                    str(r.get("nb_deces", "")),
                    f"{tm:.2f}" if tm else "",
                    str(r.get("age_moyen_deces", r.get("mean_age_death", ""))),
                ])
            spec["rows"] = rows

    elif "obs_vs_modeled" in sid or "analysis" in sid or "comparison" in sid:
        validation = context.get("validation") or {}
        ci_table   = (validation.get("ci_table") if isinstance(validation, dict) else []) or []
        exposure_t = context.get("exposure_table") or []
        exp_by_age = {r["age"]: r.get("E_x", 0) for r in exposure_t if "age" in r}
        total_exp  = sum(exp_by_age.values()) or 1

        if ci_table:
            # Agrégation par classes d'âges de 5 ans (pratique actuarielle Winter).
            import collections
            buckets = collections.OrderedDict()
            for r in ci_table:
                age = r.get("age")
                if age is None:
                    continue
                bucket_min = (int(age) // 5) * 5
                key = f"{bucket_min}-{bucket_min + 4}"
                b = buckets.setdefault(key, {
                    "exp": 0.0, "obs": 0, "mod": 0.0, "cl": [], "cu": [],
                })
                b["exp"] += exp_by_age.get(age, 0)
                b["obs"] += r.get("observed_deaths", r.get("D_x_obs", 0)) or 0
                b["mod"] += r.get("expected_deaths", r.get("modeled_deaths", 0)) or 0
                if r.get("ci_lower") is not None: b["cl"].append(r["ci_lower"])
                if r.get("ci_upper") is not None: b["cu"].append(r["ci_upper"])

            rows = [[
                "Classe d'âges", "Exposition", "Proportion",
                "Décès obs.", "Décès prédits", "Écart",
                "Diff/Prédits %", "IC bas (95 %)", "IC haut (95 %)",
            ]]
            for key, b in buckets.items():
                diff = b["obs"] - b["mod"]
                pct  = (diff / b["mod"] * 100) if b["mod"] else 0.0
                ic_lo = sum(b["cl"]) if b["cl"] else None
                ic_hi = sum(b["cu"]) if b["cu"] else None
                rows.append([
                    key,
                    f"{b['exp']:,.0f}".replace(",", " "),
                    f"{(b['exp'] / total_exp * 100):.1f} %",
                    f"{b['obs']:.0f}",
                    f"{b['mod']:.0f}",
                    f"{diff:+.0f}",
                    f"{pct:+.1f} %",
                    f"{ic_lo:.0f}" if ic_lo is not None else "",
                    f"{ic_hi:.0f}" if ic_hi is not None else "",
                ])
            spec["rows"] = rows

    elif "final_mortality" in sid or "mortality_table" in sid or sid == "annex":
        fmt = context.get("final_mortality_table_by_age") or {}
        if not fmt:
            smoothed = context.get("smoothed_table") or []
            if smoothed:
                qx_col = next(
                    (c for c in ("q_x_lisse", "qx", "q_x_brut")
                     if c in (smoothed[0] if smoothed else {})),
                    None,
                )
                if qx_col:
                    fmt = {str(r["age"]): r.get(qx_col, 0) for r in smoothed}
        if fmt:
            rows = [["Âge", "q_x (%)"]]
            for age_s, qx in sorted(
                fmt.items(),
                key=lambda x: int(x[0]) if str(x[0]).isdigit() else 999,
            ):
                rows.append([str(age_s), f"{float(qx) * 100:.4f}"])
            spec["rows"] = rows

    else:
        log.info("[04_redaction] _hydrate_table_spec: aucun hydratateur pour sid=%s", sid)

    return spec


def _enrich_graph_context(ctx: dict) -> dict:
    """
    Dérive les dicts {age: valeur} attendus par builder_plots depuis les
    artefacts déjà présents (validation.ci_table, exposure_table,
    benchmarking.abatement_table). Sans cela, seul le graphique `exposure`
    rend — les autres spécifications dispatchées échouent faute de clés.
    """
    validation = ctx.get("validation") or {}
    ci_table   = validation.get("ci_table") if isinstance(validation, dict) else None
    if ci_table and "observed_deaths_by_age" not in ctx:
        ctx["observed_deaths_by_age"] = {
            str(r["age"]): r.get("observed_deaths", r.get("D_x_obs"))
            for r in ci_table if "age" in r
        }
        ctx["modeled_deaths_by_age"] = {
            str(r["age"]): r.get("expected_deaths", r.get("modeled_deaths"))
            for r in ci_table if "age" in r
        }
        ctx["ci_lower_by_age"] = {
            str(r["age"]): r.get("ci_lower", r.get("ci_lower_95"))
            for r in ci_table
            if "age" in r and (r.get("ci_lower") is not None or r.get("ci_lower_95") is not None)
        }
        ctx["ci_upper_by_age"] = {
            str(r["age"]): r.get("ci_upper", r.get("ci_upper_95"))
            for r in ci_table
            if "age" in r and (r.get("ci_upper") is not None or r.get("ci_upper_95") is not None)
        }

    # deaths_by_age : si la cohorte est segmentée H/F, on remonte les deux
    # séries ; sinon on laisse les clés absentes pour que builder_plots.
    # deaths_by_age déclenche son propre fallback sur exposure_table.D_x.
    exp = ctx.get("exposure_table") or []
    if exp and "deaths_by_age_male" not in ctx and "deaths_by_age_female" not in ctx:
        has_split = any(isinstance(r, dict) and "D_x_male" in r for r in exp)
        if has_split:
            ctx["deaths_by_age_male"] = {
                str(r["age"]): r.get("D_x_male", 0) for r in exp if "age" in r
            }
            ctx["deaths_by_age_female"] = {
                str(r["age"]): r.get("D_x_female", 0) for r in exp if "age" in r
            }

    # Abattements : le BuilderAgent stocke `abatement_factor` en priorité ;
    # on accepte aussi `discount_pct` / `abatement` pour tolérance.
    bench = ctx.get("benchmarking") or {}
    abat  = bench.get("abatement_table") if isinstance(bench, dict) else None
    if abat and "discount_by_age" not in ctx:
        ctx["discount_by_age"] = {
            str(r["age"]): (
                r.get("abatement_factor")
                if r.get("abatement_factor") is not None
                else r.get("discount_pct", r.get("abatement"))
            )
            for r in abat if "age" in r
        }

    # Ratio courant/précédent : produit depuis precedent_comparison pour que
    # le chart rate_ratio puisse être dispatché sans que les clés soient
    # peuplées directement dans data_store.
    prec = ctx.get("precedent_comparison") or {}
    comp_table = prec.get("comparison_table") if isinstance(prec, dict) else None
    if comp_table and "rate_ratio_current_vs_prior" not in ctx:
        ctx["rate_ratio_current_vs_prior"] = {
            str(r["age"]): r.get("ratio", r.get("rate_ratio"))
            for r in comp_table
            if "age" in r and (r.get("ratio") is not None or r.get("rate_ratio") is not None)
        }

    return ctx


# ── Appels outils déterministes ───────────────────────────────────────────────

def _run_tables(section, data_store: dict) -> list[dict]:
    """
    Design 3 (US-24) : itère sur section.visual_specs et ne retient que les
    specs de type `table`. Hydratation directe via _hydrate_visual_spec.
    """
    results = []
    for spec in (section.visual_specs or []):
        if spec.get("type") != "table":
            continue
        try:
            hydrated = _hydrate_visual_spec(spec, data_store)
            if hydrated.get("error"):
                log.warning("[04_redaction] tableau '%s' : %s",
                            spec.get("id", "?"), hydrated["error"])
                continue
            headers = hydrated.get("headers", [])
            body    = hydrated.get("rows", [])
            rows    = ([headers] + body) if headers else body
            if rows:
                results.append({"spec": spec, "html": "", "rows": rows})
                data_store["_last_table_rows"] = rows
                log.info("[04_redaction] tableau '%s' rendu (%d lignes)",
                         spec.get("id", "?"), len(rows))
        except Exception as exc:
            log.warning("[04_redaction] tableau '%s' échoué : %s", spec.get("id", "?"), exc)

    return results


def _run_stats(section, data_store: dict) -> list[dict]:
    """
    Design 3 : plus de stat_specs au niveau preamble. No-op conservé pour
    compatibilité avec les appelants historiques.
    """
    return []


def _run_graphs(section, data_store: dict) -> list[str]:
    """
    Design 3 : itère sur visual_specs de type `chart` et rend chacun en PNG
    via matplotlib (renderer direct). Retourne la liste des chemins PNG.
    """
    paths: list[str] = []
    for spec in (section.visual_specs or []):
        if spec.get("type") != "chart":
            continue
        hydrated = _hydrate_visual_spec(spec, data_store)
        if hydrated.get("error"):
            log.warning("[04_redaction] graphique '%s' : %s",
                        spec.get("id", "?"), hydrated["error"])
            continue
        path = _render_chart_to_png(hydrated)
        if path:
            paths.append(path)
            log.info("[04_redaction] graphique '%s' rendu → %s",
                     spec.get("id", "?"), path)
        else:
            log.warning("[04_redaction] graphique '%s' : rendu échoué",
                        spec.get("id", "?"))
    return paths


def _render_chart_to_png(hydrated: dict) -> str:
    """Rend un spec chart hydraté (Design 3) en PNG via matplotlib.

    Spec attendu : {id, type: 'chart', chart_type: 'bar'|'line'|'scatter',
                    x_values, y_values, x_label, y_label, purpose, [name]}
    Retourne le path PNG, ou "" si échec.
    """
    import os, tempfile
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.error("[04_redaction] matplotlib indisponible")
        return ""

    chart_type = hydrated.get("chart_type", "bar")
    x_label    = hydrated.get("x_label", "")
    y_label    = hydrated.get("y_label", "")
    title      = hydrated.get("name") or hydrated.get("purpose") or hydrated.get("id", "Graphique")

    # Désactiver text.usetex explicitement (cf. fix LaTeX missing dans Lot 1)
    plt.rcParams["text.usetex"] = False

    # ── Cas multi-séries : on dispatch chaque série selon son style ──────
    if chart_type == "multi_series":
        series_hydrated = hydrated.get("series_hydrated") or []
        if not series_hydrated:
            log.warning("[04_redaction] multi_series '%s' : aucune série",
                        hydrated.get("id", "?"))
            return ""

        fig, ax = plt.subplots(figsize=(10, 5))
        DEFAULT_COLORS = ["#1A3668", "#E25B34", "#2CA02C", "#9467BD"]
        n_drawn = 0
        for i, ser in enumerate(series_hydrated):
            xs = ser.get("xs") or []
            style = ser.get("style", "line")
            color = ser.get("color") or DEFAULT_COLORS[i % len(DEFAULT_COLORS)]
            label = ser.get("label", "")

            if style == "area":
                ys_lower = ser.get("ys_lower") or []
                ys_upper = ser.get("ys_upper") or []
                triples = [(x, lo, hi) for x, lo, hi in zip(xs, ys_lower, ys_upper)
                           if x is not None and lo is not None and hi is not None]
                if not triples:
                    continue
                tx, tl, tu = zip(*triples)
                ax.fill_between(tx, tl, tu, color=color, alpha=ser.get("alpha", 0.15), label=label)
                n_drawn += 1
            else:
                ys = ser.get("ys") or []
                pairs = [(x, y) for x, y in zip(xs, ys)
                         if x is not None and y is not None]
                if not pairs:
                    continue
                px, py = zip(*pairs)
                if style == "point":
                    ax.scatter(px, py, color=color, s=24, alpha=ser.get("alpha", 0.8), label=label, zorder=3)
                else:   # line par défaut
                    ax.plot(px, py, color=color, linewidth=1.8, label=label, zorder=2)
                n_drawn += 1

        if n_drawn == 0:
            log.warning("[04_redaction] multi_series '%s' : aucune série rendue",
                        hydrated.get("id", "?"))
            plt.close(fig)
            return ""

        ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
        ax.set_xlabel(x_label, fontsize=9)
        ax.set_ylabel(y_label, fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend(fontsize=9, frameon=True)
        plt.tight_layout()

        spec_id = hydrated.get("id", "chart")
        path = os.path.join(tempfile.gettempdir(),
                            f"chart_{spec_id}_{os.getpid()}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    # ── Cas mono-source (existant) ────────────────────────────────────────
    x_values   = hydrated.get("x_values") or []
    y_values   = hydrated.get("y_values") or []

    if not x_values or not y_values:
        log.warning("[04_redaction] _render_chart_to_png '%s' : pas de données "
                    "(x=%d, y=%d) — chart skip",
                    hydrated.get("id", "?"), len(x_values), len(y_values))
        return ""

    pairs = [(x, y) for x, y in zip(x_values, y_values)
             if x is not None and y is not None]
    if not pairs:
        return ""
    xs, ys = zip(*pairs)

    fig, ax = plt.subplots(figsize=(10, 5))
    BLUE = "#1A3668"

    if chart_type == "line":
        ax.plot(xs, ys, color=BLUE, linewidth=1.8, marker="o", markersize=3)
    elif chart_type == "scatter":
        ax.scatter(xs, ys, color=BLUE, s=20, alpha=0.7)
    else:   # bar par défaut
        ax.bar([str(x) for x in xs], ys, color=BLUE, edgecolor="white")
        ax.tick_params(axis="x", rotation=45)

    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel(x_label, fontsize=9)
    ax.set_ylabel(y_label, fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    spec_id = hydrated.get("id", "chart")
    path = os.path.join(tempfile.gettempdir(),
                        f"chart_{spec_id}_{os.getpid()}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ── Appel LLM de rédaction ────────────────────────────────────────────────────

def _build_redaction_prompt(section, table_results: list, graph_paths: list) -> str:
    """
    Finalise le prompt de rédaction en ajoutant :
    - les tableaux rendus EN INTÉGRALITÉ (pas de troncature — le LLM doit voir
      toutes les lignes pour pouvoir citer correctement les âges et valeurs)
    - la liste des graphiques générés
    """
    prompt = section.prompt

    if table_results:
        prompt += "\n\n## Tableaux effectivement rendus dans le PDF"
        prompt += (
            "\n\nCes tableaux seront visibles par le lecteur. Dans ta narration, "
            "introduis-les et commente les points saillants — NE RE-CITE PAS "
            "ligne-à-ligne (ce serait redondant avec le tableau)."
        )
        for tr in table_results:
            name = tr["spec"].get("name", tr["spec"].get("id", "tableau"))
            rows = tr["rows"]
            if not rows:
                continue
            header = " | ".join(str(c) for c in rows[0])
            body   = "\n".join(" | ".join(str(v) for v in row) for row in rows[1:])
            prompt += f"\n\n**{name}** ({len(rows) - 1} lignes)\n```\n{header}\n{body}\n```"

    if graph_paths:
        prompt += "\n\n## Graphiques générés"
        for p in graph_paths:
            prompt += f"\n- {p}"
        prompt += (
            "\nCes graphiques sont intégrés dans le rapport. "
            "Fais-y référence dans le texte (ex: 'La figure ci-dessous montre...')."
        )

    prompt += (
        "\n\n## Consigne finale (impérative)"
        "\n- PRODUIS UN SEUL TEXTE COHÉRENT (pas 2 ou 3 versions empilées). Le texte"
        " s'articule en : 1 intro courte, 2-3 sous-sections, 1 synthèse finale."
        "\n- Rédige le texte narratif de cette section en respectant la charte de style."
        "\n- Cite les chiffres clés du bloc JSON « Résultats actuariels » (HR, IC, "
        "p-values, R², SMR, ratios, pourcentages) — ils doivent apparaître mot-pour-mot "
        "dans ta prose."
        "\n- FORMATAGE DES NOMBRES : utilise TOUJOURS l'écriture courante avec espaces"
        " comme séparateur de milliers (ex: '6 082 714 années-personne', '94 282 décès')."
        " JAMAIS de notation scientifique (interdit : '6.08271e+06', '1.2e5'). JAMAIS de"
        " décimales superflues sur des entiers ('94282.0' → '94 282')."
        "\n- CONTEXTE des chiffres : si les nombres viennent de `cleaned_records` (après"
        " application des règles d'exclusion R1-R6), précise-le explicitement"
        " ('après retraitement', 'sur la base assainie', etc.) plutôt que de présenter"
        " ces chiffres comme la base brute initiale."
        "\n- PÉRIODE D'OBSERVATION : si `annee_min` et `annee_max` te sont fournis, "
        "utilise EXACTEMENT ces deux bornes (ex: '1983 à 2010'). Ne calcule pas le"
        " nombre d'années si le résultat dépasse 100 (signe de date sentinelle non"
        " filtrée — omets alors la mention)."
        "\n- Si une statistique manque, OMETS la phrase entière plutôt que d'écrire "
        "« [donnée non disponible] »."
        "\n- N'INVENTE JAMAIS d'âge, de valeur ou d'intervalle absent des données fournies."
        "\n- Ne répète pas ligne-à-ligne les tableaux — commente-les."
        "\n- INTERDIT : ne mentionne AUCUNE méthode actuarielle qui n'est PAS dans tes"
        " résultats actuariels (Kaplan-Meier, Makeham, Whittaker-Henderson, Gompertz,"
        " abattement, raccordement, table TH/TF…). Si tu ne vois pas la clé `smoothed_table`"
        " dans le contexte → ne parle pas de lissage. Si pas de `validation` → ne parle pas"
        " d'IC ni de backtesting. Si pas de `benchmarking` → ne parle pas de tables"
        " réglementaires ni d'abattements. Les chunks d'inspiration sont stylistiques"
        " UNIQUEMENT, ils ne décrivent PAS ta méthodologie."
        "\n- Conclus la section par une phrase de synthèse."
    )

    return prompt


# Captions lisibles par chart_name dispatché (vs. spec.name YAML qui ment souvent).
# Source de vérité = le chart réellement rendu par builder_plots.
# MORTALITY : à déplacer dans le plugin lors du strangler.
_CHART_CAPTIONS: dict[str, str] = {
    "exposure":       "Exposition au risque par âge",
    "deaths_by_age":  "Décès observés par âge",
    "obs_vs_modeled": "Décès observés vs décès modélisés par âge (IC 95 %)",
    "rate_ratio":     "Ratio de taux — comparaison avec la table antérieure",
    "discount_line":  "Abattements par âge vs table réglementaire",
    "crude_smoothed": "Taux bruts et taux lissés par âge",
    "smr":            "SMR par tranche d'âge",
    "survival_curve": "Courbe de survie",
}

# MORTALITY : mapping spec.id → `type` attendu par render_statistical_output.
# Le YAML déclare l'id mais omet `type`, ce qui fait échouer silencieusement
# le dispatcher. On injecte le type avant le rendu.
_STAT_TYPE_BY_ID: dict[str, str] = {
    "cox_model":         "cox_proportional_hazards",
    "annual_prediction": "annual_cohort_check",
    "logit_fit":         "logit_regression",
    "chi_squared":       "chi_squared",
}


# MORTALITY : captions lisibles par id table YAML — à déplacer dans le plugin.
_TABLE_CAPTIONS: dict[str, str] = {
    "table_construction": "Construction de l'échantillon d'étude",
    "exposure_stats":     "Statistiques d'exposition par année",
    "death_stats":        "Statistiques de décès par année",
    "table_comparison":   "Décès observés vs modélisés — par âge avec IC 95 %",
    "mortality_table":    "Table de mortalité d'expérience (q_x lissés)",
}


def _caption_for_graph(spec: dict) -> str:
    """Caption dérivée du chart réellement dispatché, fallback sur spec.name."""
    from tools.graphs.graph_from_spec import _DISPATCH
    chart_name = _DISPATCH.get(spec.get("id", ""), "")
    if chart_name and chart_name in _CHART_CAPTIONS:
        return _CHART_CAPTIONS[chart_name]
    return spec.get("name") or spec.get("id", "Graphique")


def _caption_for_table(spec: dict) -> str:
    """Caption depuis la table de mapping connue, fallback sur spec.name."""
    sid = spec.get("id", "")
    if sid in _TABLE_CAPTIONS:
        return _TABLE_CAPTIONS[sid]
    return spec.get("name") or sid or "Tableau"


# Sections dont la narration est bypassée (intro courte, insertion directe du tableau).
# MORTALITY : à déplacer dans le plugin lors du strangler.
_ANNEX_SECTION_IDS: set[str] = {"annex"}

_ANNEX_INTRO: str = (
    "## Table de mortalité d'expérience\n\n"
    "La table ci-dessous présente les taux de mortalité lissés "
    "pour chaque âge de la cohorte d'étude. Ces taux sont le résultat "
    "direct de l'algorithme de lissage appliqué aux taux bruts observés.\n\n"
    "> Les valeurs sont exprimées en pourcentage et arrondies au "
    "dix-millième pour une meilleure lisibilité."
)


_SYSTEM_PROMPT_REDACTION = """\
Tu es un actuaire senior spécialisé dans la rédaction de rapports de certification de tables de mortalité.
Tu rédiges en français, style professionnel et précis.
Tu cites uniquement des chiffres présents dans les données fournies.
Tu ne calcules jamais une valeur manquante.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## CHARTE DE STYLE — À RESPECTER SCRUPULEUSEMENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### 1. Structure du texte

Utilise exclusivement ce markup. Il sera rendu visuellement dans le rapport final :

  ## Titre de sous-section
      → Titre niveau 2 (bleu gras). Utilise pour chaque grande partie de ta section.
      → Exemple : ## Méthode de calcul des taux bruts

  ### Titre de paragraphe
      → Titre niveau 3 (bleu moyen). Utilise pour les sous-parties.
      → Exemple : ### Critères d'exclusion des données

  - item de liste
      → Liste à puces. Une ligne par item. N'imbrique pas.
      → Exemple :
        - Contrats sans exposition positive
        - Sinistres non classifiés décès

  > Note ou avertissement
      → Bloc en retrait, fond gris pâle. Pour les mises en garde et précisions techniques.
      → Exemple : > Les âges extrêmes (< 30 ans et > 85 ans) ont été exclus de l'analyse.

  Texte normal
      → Paragraphes justifiés. Sépare les paragraphes par une ligne vide.

  **texte en gras**
      → Emphase inline. Utilise pour les termes clés et les résultats chiffrés importants.
      → Exemple : Le SMR global est **0,748**, soit une mortalité d'expérience inférieure de 25%...

### 2. Formules mathématiques — OBLIGATOIRE

Toutes les expressions mathématiques DOIVENT être en notation LaTeX. Elles sont rendues
par le moteur LaTeX natif du rapport (qualité documentaire).

  $expression$     → formule inline dans une phrase
  $$expression$$   → formule en bloc, centrée sur sa propre ligne

Exemples CORRECTS :
  "Le taux brut est $q_x = D_x / E_x$ où $D_x$ désigne les décès observés."
  "Le SMR est défini par : $$\\text{SMR} = \\frac{\\sum_x D_x^{\\text{obs}}}{\\sum_x D_x^{\\text{att}}}$$"
  "L'intervalle de confiance bilatéral à $95\\%$ est $[\\hat{q}_x - 1{,}96\\,\\hat{\\sigma}_x;\\ \\hat{q}_x + 1{,}96\\,\\hat{\\sigma}_x]$."
  "Le lissage minimise : $$\\sum_x w_x(q_x - z_x)^2 + \\lambda\\sum_x(\\Delta^2 z_x)^2$$"
  "avec $\\lambda = 100$ le paramètre de lissage retenu."

Exemples INTERDITS (notation ASCII) :
  ✗ "q_x = D_x/E_x"          → écrire : "$q_x = D_x / E_x$"
  ✗ "SMR = 0.748"             → écrire : "$\\text{SMR} = 0{,}748$"
  ✗ "IC 95%"                  → écrire : "IC à $95\\%$"
  ✗ "lambda=100"              → écrire : "$\\lambda = 100$"

### 3. Conventions typographiques

  - Décimales : virgule française (0,748 et non 0.748)
  - Milliers : espace fine (346 600 et non 346600) — dans le texte courant seulement
  - Pourcentages : toujours collés au chiffre avec le signe % (25 %)
  - Guillemets : « guillemets français »

### 4. Structure type d'une section

  ## Contexte et objectifs
  [1 paragraphe d'introduction]

  ## [Sous-section 1 : méthode / données / résultats]
  [2-3 paragraphes]
  [liste si nécessaire]

  ## [Sous-section 2 : analyse / interprétation]
  [2-3 paragraphes]
  [note si mise en garde]

  ## Synthèse
  [1 paragraphe de conclusion de la section]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""


def _build_traceability_refs(section, all_tables: list, context: dict) -> dict:
    """
    Rassemble le référentiel numérique contre lequel le texte rédigé doit
    être vérifié : les lignes de tableaux rendues + les clés métier injectées
    au LLM.
    """
    refs: dict = {}
    for key in ("summary", "cox_regression", "logit_regression",
                "validation", "benchmarking", "diagnostics",
                "precedent_comparison",
                "exposure_table", "smoothed_table", "qx_table",
                "total_deaths", "total_exposure_years", "total_exposure",
                "age_min", "age_max", "cohort_min_age", "cohort_max_age"):
        v = context.get(key)
        if v is not None:
            refs[key] = v
    # Les lignes des tableaux rendus (chiffres déjà montrés au lecteur)
    refs["_rendered_table_rows"] = [tr.get("rows", []) for tr in all_tables]
    # Le study_plan contient les paramètres cités (années d'observation, etc.)
    if isinstance(context.get("study_plan"), dict):
        refs["_study_plan"] = context["study_plan"]
    return refs


def _enforce_traceability(
    text:        str,
    prompt:      str,
    section,
    all_tables:  list,
    context:     dict,
) -> str:
    """
    Vérifie que chaque chiffre cité dans `text` est traçable dans les données.
    Si non → 1 retry ciblé avec feedback au LLM. Si le retry échoue encore,
    on garde le texte tel quel (on ne veut pas livrer vide).
    """
    if not text:
        return text

    from agents.report.pipeline.traceability import validate_section

    refs = _build_traceability_refs(section, all_tables, context)
    result = validate_section(text, refs)

    if result.ok:
        return text

    log.warning(
        "[04_redaction] '%s' — traçabilité KO : %d chiffres non traçables, %d bad tokens",
        section.section_id, len(result.untraceable), len(result.bad_tokens),
    )

    # Retry 1 fois avec feedback chirurgical
    feedback = result.feedback_for_retry()
    retry_prompt = (
        prompt
        + "\n\n## CORRECTIONS REQUISES AVANT LIVRAISON"
        + f"\n{feedback}"
        + "\n\nRéécris le texte narratif complet en appliquant ces corrections. "
          "Ne change rien d'autre."
    )
    retry_text = _call_llm_redaction(retry_prompt)
    if not retry_text:
        return text

    retry_result = validate_section(retry_text, refs)
    if retry_result.ok:
        log.info("[04_redaction] '%s' — traçabilité OK après retry", section.section_id)
        return retry_text

    # Ne pas remplacer le texte original par un retry PIRE.
    if len(retry_result.untraceable) > len(result.untraceable):
        log.warning(
            "[04_redaction] '%s' — retry pire que l'original (%d vs %d), on garde l'original",
            section.section_id, len(retry_result.untraceable), len(result.untraceable),
        )
        return text

    log.warning(
        "[04_redaction] '%s' — traçabilité toujours KO après retry : %s",
        section.section_id, retry_result.untraceable[:5],
    )
    return retry_text


def _call_llm_redaction(prompt: str) -> str:
    """
    Appelle GPT-4o pour rédiger le texte narratif de la section.
    Retourne le texte rédigé, ou "" en cas d'échec.
    """
    try:
        import openai
        from agents.mortality.agents._utils import call_with_retry
        from agents.mortality.agents.llm_config import get_llm_config

        cfg = get_llm_config("writer.redaction")
        client = openai.OpenAI()
        response = call_with_retry(
            client,
            model=cfg["model"],
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT_REDACTION},
                {"role": "user",   "content": prompt},
            ],
            temperature=cfg.get("temperature", _TEMPERATURE),
            max_tokens=cfg.get("max_tokens", _MAX_TOKENS_NARRATIVE),
        )
        return (response.choices[0].message.content or "").strip()

    except Exception as exc:
        log.error("[04_redaction] LLM rédaction échoué : %s", exc)
        return ""


# ── Stockage dans section_outputs ─────────────────────────────────────────────

def _write_section(section_id: str, text: str, data_store: dict,
                   table_caption: str = "", graph_caption: str = "") -> None:
    """
    Appelle write_section pour accumuler texte + tableau + graphique
    dans data_store["section_outputs"][section_id].
    """
    try:
        from tools.build_pdf.write_section import run as _ws_run
        _ws_run(
            data=data_store,
            params={
                "section_id":    section_id,
                "text":          text,
                "table_caption": table_caption,
                "graph_caption": graph_caption,
                "status":        "done" if text else "partial",
            },
        )
    except Exception as exc:
        log.error("[04_redaction] write_section '%s' échoué : %s", section_id, exc)


# ── Traitement parallèle d'une section (thread-safe) ─────────────────────────

def _process_section_parallel(sec, ds_snapshot: dict) -> tuple[str, dict]:
    """
    Traite une section dans un thread séparé.
    Thread-safe : reçoit un snapshot read-only du data_store,
    n'écrit rien dans le data_store partagé.

    Retourne (section_id, result_dict).
    """
    if not sec.ready:
        return sec.section_id, {
            "text": "", "table_caption": "", "graph_caption": "",
            "status": "skipped", "n_tables": 0, "n_graphs": 0,
        }

    # Copie locale pour isoler les écritures (_last_table_rows, _last_graph_path)
    local_ds = dict(ds_snapshot)

    table_results = _run_tables(sec, local_ds)
    stat_results  = _run_stats(sec, local_ds)
    graph_paths   = _run_graphs(sec, local_ds)
    all_tables    = table_results + stat_results

    # Annexe : pas d'appel LLM — le tableau q_x suffit. Une intro générique
    # évite les hallucinations (âges inventés, commentaires sans source).
    if sec.section_id in _ANNEX_SECTION_IDS:
        text = _ANNEX_INTRO
    else:
        prompt = _build_redaction_prompt(sec, all_tables, graph_paths)
        text   = _call_llm_redaction(prompt)

        # Validator traçabilité : chaque chiffre cité doit être traçable
        # dans les données fournies. Si non → 1 retry ciblé.
        text = _enforce_traceability(text, prompt, sec, all_tables, local_ds)

    return sec.section_id, {
        "text":          text,
        "table_caption": all_tables[-1]["spec"].get("name", "") if all_tables else "",
        "graph_caption": sec.graph_specs[-1].get("name", "") if sec.graph_specs else "",
        "status":        "done" if text else "partial",
        "n_tables":      len(all_tables),
        "n_graphs":      len(graph_paths),
        # Pass raw data so the sequential write phase can inject into shared data_store
        # Retournés pour que la phase d'écriture séquentielle les re-propage
        # vers write_section via _last_table_rows/_last_graph_path.
        "all_tables":    all_tables,
        "graph_paths":   graph_paths,
    }


# ── Point d'entrée public ─────────────────────────────────────────────────────

def redact_plan(plan, data_store: dict) -> dict:
    """
    Traite toutes les sections du ReportPlan enrichi en parallèle.

    Architecture thread-safe :
      - Snapshot du data_store passé en lecture seule à chaque worker
      - max_workers=5 pour éviter les 429 OpenAI TPM
      - Écriture séquentielle finale (ordre du plan préservé)

    Args:
        plan       : ReportPlan enrichi par 03_completion_plan
        data_store : résultats du BuilderAgent (modifié en place)

    Returns:
        data_store mis à jour avec section_outputs rempli
    """
    section_outputs = data_store.setdefault("section_outputs", {})

    # Séparer sections prêtes / skippées
    ready   = [sec for sec in plan.sections if sec.ready]
    skipped = [sec for sec in plan.sections if not sec.ready]

    # Sections non prêtes → skip immédiat, pas de thread
    for sec in skipped:
        section_outputs[sec.section_id] = {
            "text": "", "tables": [], "table_captions": [],
            "graphs": [], "graph_captions": [], "status": "skipped",
        }
        log.info("[04_redaction] '%s' — skipped (données manquantes : %s)",
                 sec.section_id, sec.missing_inputs)

    if not ready:
        log.info("[04_redaction] aucune section prête — terminé")
        return data_store

    # Snapshot read-only pour les workers (évite les conflits d'écriture)
    ds_snapshot = dict(data_store)

    # Limiter le parallélisme : 2 appels LLM simultanés max.
    # Au-delà, les TPM/RPM cumulatifs causent des échecs en cascade (les
    # 5 threads se reglent les 429 simultanément, retries en parallèle,
    # bail). 2 workers offre le meilleur tradeoff vitesse/robustesse.
    max_workers = min(len(ready), 2)
    log.info("[04_redaction] %d sections en parallèle (max_workers=%d)",
             len(ready), max_workers)

    results: dict[str, dict] = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_process_section_parallel, sec, ds_snapshot): sec.section_id
            for sec in ready
        }
        for future in concurrent.futures.as_completed(futures):
            sid = futures[future]
            try:
                section_id, result = future.result()
                results[section_id] = result
                log.info("[04_redaction] '%s' terminée (%d chars, %d tableaux, %d graphiques)",
                         section_id, len(result["text"]),
                         result["n_tables"], result["n_graphs"])
            except Exception as exc:
                log.error("[04_redaction] '%s' — exception : %s", sid, exc)
                results[sid] = {
                    "text": "", "table_caption": "", "graph_caption": "",
                    "status": "error", "n_tables": 0, "n_graphs": 0,
                }

    # Écriture séquentielle pour préserver l'ordre du plan.
    # On appelle write_section autant de fois que nécessaire : 1 appel par
    # tableau et 1 par graphique — sinon seul le dernier survivrait (write_section
    # consomme _last_table_rows / _last_graph_path singuliers).
    # NOT THREAD-SAFE — ce loop mute data_store, ne pas paralléliser.
    n_done = 0
    for sec in ready:
        r = results.get(sec.section_id, {})
        all_tables  = r.get("all_tables", [])
        graph_paths = r.get("graph_paths", [])
        text        = r.get("text", "")
        sid         = sec.section_id

        # Nombre total d'inserts attendus — on met le texte sur le PREMIER write,
        # les suivants sont purement pour append tableau/graphique.
        n_inserts = max(len(all_tables), len(graph_paths), 1)
        text_written = False

        for i in range(n_inserts):
            # Reset systématique en début d'itération pour éviter que
            # _last_table_rows/_last_graph_path de l'itération précédente
            # soient re-consommés par write_section et dupliquent le
            # tableau/graphique (cf. plan Lot 1 cause A).
            data_store.pop("_last_table_rows", None)
            data_store.pop("_last_graph_path", None)

            tbl_spec = all_tables[i]["spec"] if i < len(all_tables) else None
            tbl_rows = all_tables[i]["rows"] if i < len(all_tables) else None
            g_spec   = sec.graph_specs[i]   if i < len(sec.graph_specs) else None
            g_path   = graph_paths[i]       if i < len(graph_paths) else None

            if tbl_rows:
                data_store["_last_table_rows"] = tbl_rows
            if g_path:
                data_store["_last_graph_path"] = g_path

            _write_section(
                section_id    = sid,
                text          = "" if text_written else text,
                data_store    = data_store,
                table_caption = _caption_for_table(tbl_spec) if tbl_spec else "",
                graph_caption = _caption_for_graph(g_spec) if g_spec else "",
            )
            text_written = True

        n_done += 1

    log.info("[04_redaction] terminé — %d sections rédigées, %d skippées",
             n_done, len(skipped))
    return data_store
