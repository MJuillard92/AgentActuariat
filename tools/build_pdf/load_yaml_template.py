"""
TOOL CONTRACT — build_pdf.load_yaml_template
════════════════════════════════════════════════════════════════

IDENTITY
--------
name          : build_pdf.load_yaml_template
domain        : mortality_experience
version       : 1.0.0
author        : Marc Juillard
last_updated  : 2026-04-07

DESCRIPTION
-----------
Charge le template YAML du rapport de certification et résout les {{ placeholders }}
depuis le data_store + study_plan. Retourne le template enrichi avec les valeurs
réelles, le statut de chaque section (ready / missing_inputs) et la liste des
champs manquants. Le WriterAgent appelle ce tool en premier pour connaître l'état
des données avant de décider quelles sections peut-il rédiger.

WHEN TO USE
-----------
Appeler en tout premier dans la phase de rédaction, avant tout autre tool de rendu.
Appelle régulièrement pour mettre à jour l'état après chaque calcul supplémentaire.

WHEN NOT TO USE
---------------
Ne pas appeler si aucun calcul n'a encore été effectué (data_store vide).

PREREQUISITES
-------------
required_tools: []  # aucune dépendance obligatoire
required_data_store_keys: []  # données optionnelles — retourne statut partiel si vides

INPUTS
------
params:
  yaml_path:
    type    : string
    default : knowledge_base/report_template/mortality_template.yaml
    note    : Chemin relatif à la racine du projet.
  study_plan:
    type    : dict
    default : {}
    note    : Paramètres d'étude fournis par le MasterAgent (période, référence, etc.)

OUTPUTS
-------
data_store_keys_written:
  - template_context   # dict de résolution des placeholders
  - template_sections  # statut par section (ready/missing_inputs)
return_payload:
  template_context : dict   # {placeholder_name: resolved_value}
  sections_status  : list[dict]  # [{section_id, label, ready, missing_inputs}]
  missing_fields   : list[str]   # liste globale des champs manquants
  n_ready          : int
  n_total          : int

QUALITY GATES
-------------
NON-BLOCKING:
  - Sections avec missing_inputs → listées dans sections_status, ne bloquent pas

CATALOGUE METADATA
------------------
display_name      : Chargement template rapport YAML
short_description : Charge et résout les placeholders du template de certification.
domain            : mortality_experience
capability_group  : reporting
depends_on        : []
required_by       : [build_pdf.assemble_sections]
client_visible    : false
"""
from __future__ import annotations

import re
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Mapping : placeholder YAML → source dans data_store / study_plan
# Format : {yaml_key: (source, dotted_path_or_callable)}
_PLACEHOLDER_MAP: dict[str, tuple[str, str]] = {
    # Study plan fields
    "observation_period_years":      ("study_plan", "observation_period_years"),
    "observation_start_date":        ("study_plan", "observation_start_date"),
    "observation_end_date":          ("study_plan", "observation_end_date"),
    "study_objective":               ("study_plan", "study_objective"),
    "product_list":                  ("study_plan", "product_list"),
    "exclusion_criteria":            ("study_plan", "exclusion_criteria"),
    "smoothing_algorithm":           ("study_plan", "smoothing_algorithm"),
    "smoothing_parameters":          ("study_plan", "smoothing_parameters"),
    "baseline_regulatory_table":     ("study_plan", "baseline_regulatory_table"),
    "boundary_age_treatment":        ("study_plan", "boundary_age_treatment"),
    "max_mean_age_change_per_year":  ("study_plan", "max_mean_age_change_per_year"),
    "death_rate_cv_threshold":       ("study_plan", "death_rate_cv_threshold"),
    "chi_squared_p_significance":    ("study_plan", "chi_squared_p_significance"),
    "discount_jump_tolerance_pct":   ("study_plan", "discount_jump_tolerance_pct"),
    "logit_r_squared_minimum":       ("study_plan", "logit_r_squared_minimum"),
    "confidence_interval_level":     ("study_plan", "confidence_interval_level"),
    # Builder outputs
    "cohort_min_age":                ("data_store_multi", "age_min|cohort_min_age"),
    "cohort_max_age":                ("data_store_multi", "age_max|cohort_max_age"),
    "total_exposure_years":          ("data_store_multi", "total_exposure|summary.exposition_totale_pa"),
    "total_deaths":                  ("data_store_multi", "total_deaths|summary.nb_deces"),
    "initial_record_count":          ("derived", "initial_record_count()"),
    "final_record_count":            ("derived", "final_record_count()"),
    "total_exclusions":              ("data_store_multi", "total_exclusions|0"),
    "mean_age_cohort":               ("derived", "mean_age_cohort()"),
    "gender_distribution":           ("derived", "gender_distribution()"),
    "num_observation_years":         ("derived", "len(observation_period_years)"),
    "crude_rate_method":             ("data_store_multi", "method|crude_rate_method"),
    # Cox regression
    "cox_hazard_ratio":              ("data_store", "cox_regression.hazard_ratio"),
    "cox_pvalue":                    ("data_store", "cox_regression.cox_pvalue"),
    # Logit regression
    "logit_slope":                   ("data_store", "logit_regression.slope_alpha"),
    "logit_intercept":               ("data_store", "logit_regression.intercept_beta"),
    "logit_r_squared":               ("data_store", "logit_regression.r_squared"),
    # Validation
    "chi_squared_p":                 ("derived", "chi_squared_p()"),
    # Benchmarking
    "avg_prudence_ratio":            ("data_store_multi", "benchmarking.smr_global|smr_global"),
    "prior_table_exists":            ("data_store_multi", "precedent_comparison|False"),
    # Age-indexed data (stored as-is for graph tools)
    "exposure_by_age_male":          ("derived", "exposure_by_sex(H)"),
    "exposure_by_age_female":        ("derived", "exposure_by_sex(F)"),
    "deaths_by_age_male":            ("derived", "deaths_by_sex(H)"),
    "deaths_by_age_female":          ("derived", "deaths_by_sex(F)"),
    "exposure_by_year":              ("derived", "exposure_by_year()"),
    "deaths_by_year":                ("derived", "deaths_by_year()"),
    "observed_deaths_by_age":        ("derived", "deaths_by_age()"),
    "modeled_deaths_by_age":         ("derived", "modeled_deaths_by_age()"),
    "ci_lower_by_age":               ("derived", "ci_lower_by_age()"),
    "ci_upper_by_age":               ("derived", "ci_upper_by_age()"),
    "annual_prediction_ratio":       ("derived", "annual_prediction_ratio()"),
    "discount_by_age":               ("derived", "discount_by_age()"),
    "final_mortality_table_by_age":  ("derived", "final_mortality_table()"),
    "exposure_by_age_class":         ("derived", "exposure_by_age_class()"),
    "rate_ratio_current_vs_prior":   ("derived", "rate_ratio_vs_prior()"),
}

# Sections et leurs champs requis (subset critique)
_SECTION_REQUIRED: dict[str, list[str]] = {
    # observation_start_date est optionnel — on peut rédiger sans dates exactes
    "preamble":          ["total_exposure_years", "total_deaths"],
    "data_submission":   ["initial_record_count"],
    "construction":      ["cohort_min_age", "cohort_max_age"],
    # ci_lower_by_age est optionnel — si absent, section rédigée sans IC
    "analysis":          ["observed_deaths_by_age"],
    # avg_prudence_ratio est optionnel — peut être absent si benchmarking non fait
    "conclusion":        ["total_exposure_years", "total_deaths"],
    "annex":             ["final_mortality_table_by_age", "cohort_min_age", "cohort_max_age"],
}


def _extract_section_specs(section_yaml: dict) -> dict:
    """
    Parcourt récursivement un dict de section YAML et extrait :
      - table_specs          : liste de specs pour build_pdf.table_renderer
      - graph_specs          : liste de specs pour graphs.graph_from_spec
      - stat_specs           : liste de specs pour build_pdf.table_renderer (mode stat)
      - narrative_templates  : liste de chaînes narratives avec {{ placeholders }}

    Un « tool_call entry » est un dict qui contient la clé 'tool_call'.
    Sa structure attendue : {tool_call: str, inputs: {spec: dict, ...}}.
    """
    table_specs         = []
    graph_specs         = []
    stat_specs          = []
    narrative_templates = []

    def _walk(node):
        if isinstance(node, dict):
            # Si c'est une entrée tool_call → extraire le spec et ne pas descendre plus loin
            if "tool_call" in node:
                tc   = node["tool_call"]
                spec = (node.get("inputs") or {}).get("spec") or {}
                if spec:
                    if tc == "render_table_from_spec":
                        table_specs.append(spec)
                    elif tc == "generate_graph_from_spec":
                        graph_specs.append(spec)
                    elif tc == "render_statistical_output":
                        stat_specs.append(spec)
                return  # ne pas descendre dans les inputs du tool_call

            # Extraire les narrative_elements
            if "narrative_elements" in node:
                ne = node["narrative_elements"]
                if isinstance(ne, list):
                    for item in ne:
                        if isinstance(item, dict):
                            for v in item.values():
                                if isinstance(v, str) and "{{" in v:
                                    narrative_templates.append(v)
                elif isinstance(ne, dict):
                    for v in ne.values():
                        if isinstance(v, str) and "{{" in v:
                            narrative_templates.append(v)

            # Récursion sur toutes les valeurs
            for v in node.values():
                _walk(v)

        elif isinstance(node, list):
            for item in node:
                _walk(item)

    _walk(section_yaml)

    return {
        "table_specs":         table_specs,
        "graph_specs":         graph_specs,
        "stat_specs":          stat_specs,
        "narrative_templates": narrative_templates,
    }


def _get_nested(d: dict, path: str):
    """Accès chemin pointé : 'a.b.c' → d['a']['b']['c']."""
    parts = path.split(".")
    cur = d
    for p in parts:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(p)
    return cur


def _resolve_derived(key: str, data_store: dict, study_plan: dict):
    """Calcule les valeurs dérivées depuis data_store."""
    exposure_table = data_store.get("exposure_table") or []
    if not exposure_table:
        return None

    if key == "initial_record_count()":
        # Nombre de lignes dans exposure_table = nombre d'âges couverts
        # Priorité : clé directe, puis nb_contrats du summary, puis len(exposure_table)
        v = (data_store.get("initial_record_count")
             or data_store.get("nb_contrats")
             or _get_nested(data_store, "summary.nb_contrats"))
        return v if v is not None else len(exposure_table) if exposure_table else None

    if key == "final_record_count()":
        v = (data_store.get("final_record_count")
             or data_store.get("initial_record_count")
             or data_store.get("nb_contrats")
             or _get_nested(data_store, "summary.nb_contrats"))
        return v if v is not None else len(exposure_table) if exposure_table else None

    if key == "mean_age_cohort()":
        v = (data_store.get("mean_age_cohort")
             or _get_nested(data_store, "summary.age_moyen"))
        if v is not None:
            return v
        # Calculer depuis exposure_table si possible
        if exposure_table:
            total_exp = sum(r.get("E_x", 0) for r in exposure_table)
            if total_exp > 0:
                w_sum = sum(r["age"] * r.get("E_x", 0) for r in exposure_table)
                return round(w_sum / total_exp, 1)
        return None

    if key == "gender_distribution()":
        v = (data_store.get("gender_distribution")
             or _get_nested(data_store, "summary.pct_by_sex"))
        return v  # None si absent — section non bloquée par ça

    if key == "chi_squared_p()":
        # 1. Résultat direct du chi_square test (function_name="chi_square")
        validation = data_store.get("validation") or {}
        if isinstance(validation, dict):
            p = validation.get("p_value")
            if p is not None:
                return p
        # 2. Clé directe dans data_store
        return data_store.get("chi_squared_p") or None

    if key == "exposure_by_age_class()":
        return {str(r["age"]): r.get("E_x", 0) for r in exposure_table}

    if key == "deaths_by_age()":
        return {str(r["age"]): r.get("D_x", 0) for r in exposure_table}

    if key in ("exposure_by_sex(H)", "exposure_by_sex(F)", "deaths_by_sex(H)", "deaths_by_sex(F)"):
        suffix = "H" if "(H)" in key else "F"
        is_exp = key.startswith("exposure")
        # Chercher d'abord les clés directes
        direct_key = f"{'exposure' if is_exp else 'deaths'}_by_age_{'male' if suffix == 'H' else 'female'}"
        direct = data_store.get(direct_key)
        if direct:
            return direct
        # Fallback : retourner les totaux depuis exposure_table (approximation)
        col = "E_x" if is_exp else "D_x"
        return {str(r["age"]): r.get(col, 0) for r in exposure_table} or None

    if key == "final_mortality_table()":
        smoothed = data_store.get("smoothed_table") or []
        if smoothed:
            qx_col = next((c for c in ("q_x_lisse", "qx") if c in (smoothed[0] if smoothed else {})), None)
            if qx_col:
                return {str(r["age"]): r.get(qx_col, 0) for r in smoothed}
        return {str(r["age"]): r.get("q_x_brut", 0) for r in exposure_table}

    if key == "modeled_deaths_by_age()":
        # 1. Clé directe
        direct = data_store.get("modeled_deaths_by_age")
        if direct:
            return direct
        # 2. Depuis ci_table (expected_deaths ou modeled_deaths)
        validation = data_store.get("validation") or {}
        ci_table = (validation.get("ci_table") if isinstance(validation, dict) else []) or []
        if ci_table:
            result = {str(r["age"]): r.get("expected_deaths", r.get("modeled_deaths"))
                      for r in ci_table}
            if any(v is not None and v != 0 for v in result.values()):
                return result
        # 3. Calculer : q_x_lisse × E_x (smoothed_table × exposure_table)
        smoothed = data_store.get("smoothed_table") or []
        if smoothed and exposure_table:
            exp_idx = {r["age"]: r.get("E_x", 0) for r in exposure_table}
            qx_col = next((c for c in ("q_x_lisse", "qx", "q_x_brut")
                           if c in (smoothed[0] if smoothed else {})), None)
            if qx_col:
                result = {
                    str(r["age"]): round(r.get(qx_col, 0) * exp_idx.get(r["age"], 0), 2)
                    for r in smoothed
                }
                if any(v != 0 for v in result.values()):
                    return result
        return None

    if key == "ci_lower_by_age()":
        validation = data_store.get("validation") or {}
        ci_table = (validation.get("ci_table") if isinstance(validation, dict) else []) or []
        if ci_table:
            result = {str(r["age"]): r.get("ci_lower", r.get("ci_lower_95")) for r in ci_table}
            if any(v is not None for v in result.values()):
                return result
        return data_store.get("ci_lower_by_age") or None

    if key == "ci_upper_by_age()":
        validation = data_store.get("validation") or {}
        ci_table = (validation.get("ci_table") if isinstance(validation, dict) else []) or []
        if ci_table:
            result = {str(r["age"]): r.get("ci_upper", r.get("ci_upper_95")) for r in ci_table}
            if any(v is not None for v in result.values()):
                return result
        return data_store.get("ci_upper_by_age") or None

    if key == "discount_by_age()":
        benchmarking = data_store.get("benchmarking") or {}
        ab_table = benchmarking.get("abatement_table") if isinstance(benchmarking, dict) else None
        if ab_table:
            result = {str(r["age"]): r.get("abatement_factor", r.get("abattement", 0))
                      for r in ab_table}
            if any(v and v != 0 for v in result.values()):
                return result
        return data_store.get("discount_by_age") or None

    if key == "exposure_by_year()":
        series = data_store.get("series") or {}
        serie = series.get("serie", []) if isinstance(series, dict) else []
        if serie:
            return {str(r["annee"]): r.get("exposition_pa", 0) for r in serie} or None
        # Fallback : clé directe
        return data_store.get("exposure_by_year") or None

    if key == "deaths_by_year()":
        series = data_store.get("series") or {}
        serie = series.get("serie", []) if isinstance(series, dict) else []
        if serie:
            return {str(r["annee"]): r.get("nb_deces", 0) for r in serie} or None
        return data_store.get("deaths_by_year") or None

    if key == "annual_prediction_ratio()":
        series = data_store.get("series") or {}
        serie = series.get("serie", []) if isinstance(series, dict) else []
        if not serie:
            return data_store.get("annual_prediction_ratio") or None
        # Approximation: ratio = modeled / observed par année (besoin de données croisées)
        return None  # Signaler comme manquant si non calculé

    if key == "rate_ratio_vs_prior()":
        prec = data_store.get("precedent_comparison") or {}
        if isinstance(prec, dict) and prec.get("comparison_table"):
            return {str(r.get("age", "")): r.get("ratio", 1.0)
                    for r in prec["comparison_table"]}
        return None

    if key == "len(observation_period_years)":
        years = study_plan.get("observation_period_years")
        return len(years) if isinstance(years, list) else None

    return None


def run(data: dict | None = None, params: dict | None = None) -> dict:
    data   = data   or {}
    params = params or {}

    yaml_path  = params.get("yaml_path", "knowledge_base/report_template/mortality_template.yaml")
    study_plan = params.get("study_plan") or data.get("study_plan") or {}

    # Charger le YAML
    yaml_full_path = _PROJECT_ROOT / yaml_path
    if not yaml_full_path.exists():
        return {"erreur": f"Template YAML introuvable : {yaml_full_path}"}

    try:
        import yaml
        with open(yaml_full_path, encoding="utf-8") as f:
            template = yaml.safe_load(f)
    except Exception as exc:
        return {"erreur": f"Erreur lecture YAML : {exc}"}

    # ── Résoudre les placeholders ─────────────────────────────────────────────
    context: dict[str, object] = {}
    missing: list[str] = []

    for ph_name, (source, path) in _PLACEHOLDER_MAP.items():
        value = None

        if source == "study_plan":
            value = study_plan.get(path) or study_plan.get(ph_name)
            # Fallback observation_period_years depuis les séries temporelles
            if value is None and ph_name == "observation_period_years":
                series = data.get("series") or {}
                serie = series.get("serie", []) if isinstance(series, dict) else []
                if serie:
                    years = sorted({r["annee"] for r in serie if "annee" in r})
                    if years:
                        value = years

        elif source == "data_store":
            value = _get_nested(data, path)

        elif source == "data_store_multi":
            # Essayer plusieurs chemins séparés par |
            for alt in path.split("|"):
                alt = alt.strip()
                if alt in ("0", "False", "True"):
                    value = eval(alt)  # noqa: S307
                    break
                v = _get_nested(data, alt) or data.get(alt)
                if v is not None:
                    value = v
                    break

        elif source == "derived":
            value = _resolve_derived(path, data, study_plan)

        # Fallback : check data_store directement
        if value is None:
            value = data.get(ph_name) or study_plan.get(ph_name)

        if value is not None:
            context[ph_name] = value
        else:
            missing.append(ph_name)

    # ── Dériver observation_start_date / observation_end_date depuis les années ──
    # Si l'utilisateur n'a pas saisi les dates, on les dérive depuis observation_period_years
    if "observation_start_date" not in context and "observation_period_years" in context:
        years = context["observation_period_years"]
        if isinstance(years, list) and years:
            context["observation_start_date"] = f"{min(years)}-01-01"
            context["observation_end_date"]   = f"{max(years)}-12-31"
            missing[:] = [m for m in missing
                          if m not in ("observation_start_date", "observation_end_date")]

    # ── Index sections : section_id → section YAML (pour extraire les specs) ──
    # Couvre les sections de premier niveau et leurs subsections
    _sections_index = {}
    for _sec in (template.get("sections") or []):
        _sid = _sec.get("section_id")
        if _sid:
            _sections_index[str(_sid)] = _sec
        for _sub in (_sec.get("subsections") or []):
            _sub_id = _sub.get("subsection_id")
            if _sub_id:
                _sections_index[str(_sub_id)] = _sub

    # ── Statut par section ────────────────────────────────────────────────────
    sections_status = []
    n_ready = 0
    processing_seq = template.get("processing_sequence", [])

    for sec in processing_seq:
        sec_id = sec.get("section_id", sec.get("section_number", ""))
        label  = sec.get("label", str(sec_id))
        required_keys = _SECTION_REQUIRED.get(str(sec_id), [])
        sec_missing = [k for k in required_keys if k in missing]
        ready = len(sec_missing) == 0
        if ready:
            n_ready += 1

        # Extraire les specs tableaux/graphiques depuis le bloc sections
        sec_yaml = _sections_index.get(str(sec_id), {})
        specs    = _extract_section_specs(sec_yaml)

        # Pour les sections avec subsections dans processing_sequence
        for sub in (sec.get("subsections") or []):
            sub_id   = sub.get("subsection_id", "")
            sub_yaml = _sections_index.get(str(sub_id), {})
            sub_specs = _extract_section_specs(sub_yaml)
            specs["table_specs"].extend(sub_specs["table_specs"])
            specs["graph_specs"].extend(sub_specs["graph_specs"])
            specs["stat_specs"].extend(sub_specs["stat_specs"])
            specs["narrative_templates"].extend(sub_specs["narrative_templates"])

        # Dédupliquer par id : une subsection qui réutilise un spec d'une autre
        # ne doit pas faire rendre le tableau/graphique N fois.
        def _dedup_by_id(items):
            seen, out = set(), []
            for it in items:
                sid = it.get("id") if isinstance(it, dict) else None
                if sid and sid in seen:
                    continue
                if sid:
                    seen.add(sid)
                out.append(it)
            return out

        specs["table_specs"] = _dedup_by_id(specs["table_specs"])
        specs["graph_specs"] = _dedup_by_id(specs["graph_specs"])
        specs["stat_specs"]  = _dedup_by_id(specs["stat_specs"])

        sections_status.append({
            "section_id":          sec_id,
            "label":               label,
            "ready":               ready,
            "missing_inputs":      sec_missing,
            "table_specs":         specs["table_specs"],
            "graph_specs":         specs["graph_specs"],
            "stat_specs":          specs["stat_specs"],
            "narrative_templates": specs["narrative_templates"],
        })

    data["template_context"]  = context
    data["template_sections"] = sections_status

    return {
        "template_context": context,
        "sections_status":  sections_status,
        "missing_fields":   list(set(missing)),
        "n_ready":          n_ready,
        "n_total":          len(processing_seq),
    }
