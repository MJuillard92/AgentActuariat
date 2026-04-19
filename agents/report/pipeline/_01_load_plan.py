"""
agents/report/pipeline/01_load_plan.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ÉTAPE 1 — Déterministe, zéro LLM

Charge le template YAML, résout les placeholders depuis le data_store,
et produit un ReportPlan : liste structurée de SectionPlan, chacun
contenant le prompt de rédaction prêt à être utilisé à l'étape 04.

Interface publique :
    load_plan(data_store, study_plan, yaml_path) -> ReportPlan
    ReportPlan, SectionPlan (dataclasses)
"""
from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_YAML  = "knowledge_base/report_template/mortality_template.yaml"


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class SectionPlan:
    section_id:   str
    label:        str
    ready:        bool               # load_yaml_template a tous les inputs
    missing_inputs: list[str]        # champs manquants selon load_yaml_template
    prompt:       str                # prompt de rédaction assemblé (pour étape 04)
    table_specs:  list[dict]         # specs pour table_renderer
    graph_specs:  list[dict]         # specs pour graph_from_spec
    stat_specs:   list[dict]         # specs pour render_statistical_output
    context_snapshot: dict           # valeurs résolues utiles à la section


@dataclass
class ReportPlan:
    sections:      list[SectionPlan]
    context:       dict              # tous les placeholders résolus
    missing_fields: list[str]        # champs manquants au global
    n_ready:       int
    n_total:       int
    yaml_path:     str


# ── Assemblage du prompt de section ──────────────────────────────────────────

def _build_section_prompt(
    sec_status: dict,
    context:    dict,
    yaml_section: dict,
) -> str:
    """
    Assemble le prompt de rédaction pour une section à partir du YAML + contexte.
    Le prompt est autonome : il contient tout ce dont le LLM de rédaction a besoin.
    """
    section_id = sec_status["section_id"]
    label      = sec_status["label"]
    tables     = sec_status.get("table_specs", [])
    graphs     = sec_status.get("graph_specs", [])
    stats      = sec_status.get("stat_specs", [])
    narratives = sec_status.get("narrative_templates", [])

    # ── En-tête ───────────────────────────────────────────────────────────────
    lines = [
        f"# Section {label} — Instructions de rédaction",
        "",
        "## Rôle",
        f"Tu rédiges la section '{label}' d'un rapport de certification de table de mortalité.",
        "Tu cites UNIQUEMENT des chiffres présents dans les données fournies ci-dessous.",
        "Tu ne calcules jamais une valeur manquante.",
        "",
    ]

    # ── Contenu narratif depuis YAML ──────────────────────────────────────────
    content = yaml_section.get("content", [])
    purpose = ""
    word_count = ""
    tone = ""
    for item in (content if isinstance(content, list) else []):
        if isinstance(item, dict):
            if "purpose" in item:
                purpose = item["purpose"]
            if "word_count" in item:
                word_count = item["word_count"]
            if "tone" in item:
                tone = item["tone"]

    if purpose:
        lines += ["## Objectif de la section", purpose, ""]
    if word_count:
        lines += [f"**Longueur cible** : {word_count}", ""]
    if tone:
        lines += [f"**Ton** : {tone}", ""]

    # ── Éléments narratifs avec placeholders résolus ──────────────────────────
    if narratives:
        lines += ["## Éléments narratifs à intégrer", ""]
        for tpl in narratives:
            resolved = _resolve_placeholders(tpl, context)
            lines.append(f"- {resolved}")
        lines.append("")

    # ── Tableaux à produire ───────────────────────────────────────────────────
    if tables:
        lines += ["## Tableaux à produire"]
        for t in tables:
            tid  = t.get("id", "?")
            name = t.get("name", tid)
            cols = t.get("columns", [])
            lines.append(f"- **{name}** (`{tid}`) — colonnes : {', '.join(str(c) for c in cols)}")
        lines.append("")

    # ── Graphiques à produire ─────────────────────────────────────────────────
    if graphs:
        lines += ["## Graphiques à produire"]
        for g in graphs:
            gid   = g.get("id", "?")
            name  = g.get("name", gid)
            gtype = g.get("type", "")
            lines.append(f"- **{name}** (`{gid}`) — type : {gtype}")
        lines.append("")

    # ── Sorties statistiques ──────────────────────────────────────────────────
    if stats:
        lines += ["## Sorties statistiques à produire"]
        for s in stats:
            stype = s.get("type", "?")
            name  = s.get("name", stype)
            lines.append(f"- **{name}** (`{stype}`)")
        lines.append("")

    # ── Données disponibles pour cette section ────────────────────────────────
    # On injecte les données COMPLÈTES en JSON (scalaires + objets actuariels)
    # pour que le LLM puisse citer n'importe quel chiffre sans en inventer.
    import json as _json

    scalars:   dict = {}
    objects:   dict = {}

    for key in _get_section_keys(section_id):
        val = context.get(key)
        if val is None:
            continue
        if isinstance(val, (dict, list)):
            objects[key] = val
        else:
            scalars[key] = val

    # Toujours inclure les résultats actuariels pertinents (cox, logit, χ², etc.)
    for key in _ACTUARIAL_RESULT_KEYS:
        val = context.get(key)
        if val is None or key in objects:
            continue
        objects[key] = val

    if scalars or objects:
        lines += ["## Données disponibles pour la rédaction", ""]

    if scalars:
        lines += ["### Paramètres et scalaires"]
        for k, v in scalars.items():
            # Arrondir les floats avant affichage pour éviter les
            # valeurs brutes à 15 décimales que le LLM recopierait.
            if isinstance(v, float):
                v = round(v, 4)
            lines.append(f"- **`{k}`** : {v}")
        lines.append("")

    if objects:
        # Arrondir les floats pour éviter que le LLM recopie des valeurs
        # comme 0.4879139941055774 telles quelles.
        objects = _round_floats(objects, ndigits=4)
        # Neutralise les triple-backticks éventuels dans les chaînes de data_store
        # pour éviter qu'elles brisent le bloc ```json du prompt.
        dump = _json.dumps(objects, indent=2, ensure_ascii=False, default=str)
        dump = dump.replace("```", "``\u200b`")
        lines += [
            "### Résultats actuariels et séries (JSON — cite ces valeurs telles quelles)",
            "```json",
            dump,
            "```",
            "",
        ]

    # ── Règles absolues ───────────────────────────────────────────────────────
    lines += [
        "## Règles absolues",
        "- Ne cite QUE des chiffres présents textuellement dans le bloc JSON ci-dessus",
        "- Si une statistique manque, OMETS la phrase entière — n'écris JAMAIS `[donnée non disponible]`",
        "- N'invente JAMAIS un âge, une valeur, un ratio absent du bloc JSON",
        "- Rédige en français, style professionnel actuariel",
        "- Ne dépasse pas 10% au-delà du nombre de mots cible",
    ]

    return "\n".join(lines)


def _resolve_placeholders(text: str, context: dict) -> str:
    """Substitue les {{ placeholder }} dans une chaîne."""
    import re
    def _sub(m):
        key = m.group(1).strip()
        val = context.get(key)
        # Marqueur neutre au lieu de [key] : le LLM ne recopie pas "—"
        # et ne déclenche pas la règle "écris [donnée non disponible]".
        if val is None or val == "":
            return "—"
        if isinstance(val, float):
            # Arrondi à 4 décimales pour éviter les "0.4879139941055774"
            # dans les narrative_templates YAML (ex. {{ avg_prudence_ratio }}).
            return f"{round(val, 4):g}"
        if isinstance(val, (list, dict)):
            return str(val)[:80]
        return str(val)
    return re.sub(r"\{\{\s*(\w+)\s*\}\}", _sub, text)


# Objets métier actuariels injectés en JSON dans le prompt pour que le LLM
# puisse citer des chiffres réels (HR, p-values, R², IC, SMR…). Ces dicts
# proviennent directement du BuilderAgent via data_store.
#
# MORTALITY : cette liste est domaine-spécifique et sera déplacée dans
# agents/mortality/report_plugin/section_briefs.py lors du strangler.
_ACTUARIAL_RESULT_KEYS: list[str] = [
    "summary",
    "cox_regression",
    "logit_regression",
    "validation",
    "benchmarking",
    "diagnostics",
    "precedent_comparison",
]


# Champs pertinents par section (pour le snapshot de contexte)
_SECTION_CONTEXT_KEYS: dict[str, list[str]] = {
    "preamble":        ["study_objective", "observation_start_date", "observation_end_date",
                        "num_observation_years", "total_exposure_years", "total_deaths",
                        "smoothing_algorithm", "baseline_regulatory_table", "product_list"],
    "data_submission": ["initial_record_count", "final_record_count", "total_exclusions",
                        "mean_age_cohort", "gender_distribution", "exposure_by_age_male",
                        "exposure_by_age_female", "deaths_by_age_male", "deaths_by_age_female",
                        "observation_period_years", "exposure_by_year", "deaths_by_year"],
    "construction":    ["exclusion_criteria", "crude_rate_method", "smoothing_algorithm",
                        "smoothing_parameters", "boundary_age_treatment",
                        "cohort_min_age", "cohort_max_age"],
    "obs_vs_modeled":  ["observed_deaths_by_age", "modeled_deaths_by_age",
                        "ci_lower_by_age", "ci_upper_by_age", "chi_squared_p",
                        "confidence_interval_level", "annual_prediction_ratio"],
    "prior_comparison":["rate_ratio_current_vs_prior", "prior_prudence_ratio",
                        "prior_table_exists"],
    "regulatory_positioning": ["discount_by_age", "baseline_regulatory_table",
                               "logit_slope", "logit_intercept", "logit_r_squared",
                               "discount_jump_tolerance_pct", "logit_r_squared_minimum"],
    "conclusion":      ["total_exposure_years", "total_deaths", "study_objective",
                        "smoothing_algorithm", "baseline_regulatory_table", "avg_prudence_ratio"],
    "annex":           ["final_mortality_table_by_age", "cohort_min_age", "cohort_max_age"],
}


def _get_section_keys(section_id: str) -> list[str]:
    return _SECTION_CONTEXT_KEYS.get(section_id, [])


def _round_floats(obj, ndigits: int = 4):
    """Arrondit récursivement tous les floats — évite les valeurs brutes à
    10+ décimales dans le prompt LLM."""
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, float):
        return round(obj, ndigits)
    if isinstance(obj, dict):
        return {k: _round_floats(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round_floats(v, ndigits) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_round_floats(v, ndigits) for v in obj)
    return obj


# Clés à remonter depuis data_store / study_plan vers le context
# de résolution des placeholders (complète _PLACEHOLDER_MAP).
# Regroupe scalaires ET dicts/list actuariels — tout ce qui doit atteindre
# le prompt LLM via _build_section_prompt.
_EXTRA_CONTEXT_KEYS: set[str] = {
    # Scalaires
    "initial_record_count", "final_record_count", "total_exclusions",
    "total_deaths", "total_exposure_years", "num_observation_years",
    "mean_age_cohort", "cohort_min_age", "cohort_max_age",
    "exposure_by_year", "deaths_by_year",
    "baseline_regulatory_table", "product_list",
    "smoothing_algorithm", "smoothing_parameters",
    "exclusion_criteria", "boundary_age_treatment",
    "observation_start_date", "observation_end_date",
    "observation_period_years", "study_objective",
    "chi_squared_p", "avg_prudence_ratio",
    # Dicts actuariels (résultats BuilderAgent) — sans eux le LLM n'a aucun
    # chiffre à citer (Cox HR, χ², logit R², SMR, benchmarking, diagnostics).
    "cox_regression", "logit_regression", "validation", "benchmarking",
    "diagnostics", "precedent_comparison", "summary", "series",
    # Séries age-indexées
    "smoothed_table", "exposure_table", "qx_table",
    "final_mortality_table_by_age",
}


def _context_snapshot(section_id: str, context: dict) -> dict:
    """Extrait uniquement les clés pertinentes pour la section."""
    keys = _get_section_keys(section_id)
    return {k: context[k] for k in keys if k in context}


# ── Chargement de la section YAML par id ─────────────────────────────────────

def _build_yaml_index(template: dict) -> dict[str, dict]:
    """Index section_id / subsection_id → dict YAML."""
    index = {}
    for sec in (template.get("sections") or []):
        sid = sec.get("section_id")
        if sid:
            index[str(sid)] = sec
        for sub in (sec.get("subsections") or []):
            ssid = sub.get("subsection_id")
            if ssid:
                index[str(ssid)] = sub
    return index


# ── Point d'entrée public ─────────────────────────────────────────────────────

def load_plan(
    data_store:  dict,
    study_plan:  dict | None = None,
    yaml_path:   str = _DEFAULT_YAML,
) -> ReportPlan:
    """
    Charge le YAML, résout les placeholders, construit le ReportPlan.

    Args:
        data_store  : résultats du BuilderAgent
        study_plan  : paramètres de l'étude (dates, âges, algorithme...)
        yaml_path   : chemin relatif à la racine du projet

    Returns:
        ReportPlan avec une SectionPlan par entrée de processing_sequence
    """
    from tools.build_pdf.load_yaml_template import run as _lyt_run
    import yaml as _yaml

    study_plan = study_plan or data_store.get("study_plan") or {}

    # ── 1. Appel déterministe à load_yaml_template ────────────────────────────
    combined = dict(data_store)
    combined["study_plan"] = study_plan

    lyt_result = _lyt_run(
        data=combined,
        params={"yaml_path": yaml_path, "study_plan": study_plan},
    )

    if "erreur" in lyt_result:
        raise RuntimeError(f"[load_plan] load_yaml_template erreur : {lyt_result['erreur']}")

    context         = lyt_result.get("template_context", {})
    sections_status = lyt_result.get("sections_status", [])
    missing_fields  = lyt_result.get("missing_fields", [])
    n_ready         = lyt_result.get("n_ready", 0)
    n_total         = lyt_result.get("n_total", 0)

    # Enrichir le contexte avec les scalaires data_store/study_plan non remontés
    # par _PLACEHOLDER_MAP. Sans cela, le LLM reçoit des placeholders non résolus
    # et produit "[donnée non disponible]" pour des valeurs pourtant présentes.
    for k in _EXTRA_CONTEXT_KEYS:
        if context.get(k) in (None, "", []):
            v = data_store.get(k)
            if v is None:
                v = study_plan.get(k) if isinstance(study_plan, dict) else None
            if v is not None:
                context[k] = v

    # ── 2. Charger le YAML brut pour accéder aux sections complètes ───────────
    yaml_full = _PROJECT_ROOT / yaml_path
    template  = {}
    if yaml_full.exists():
        with open(yaml_full, encoding="utf-8") as f:
            template = _yaml.safe_load(f) or {}
    yaml_index = _build_yaml_index(template)

    # ── 3. Construire une SectionPlan par section ─────────────────────────────
    section_plans: list[SectionPlan] = []

    for sec_status in sections_status:
        section_id = str(sec_status.get("section_id", ""))
        label      = sec_status.get("label", section_id)
        ready      = sec_status.get("ready", False)
        missing    = sec_status.get("missing_inputs", [])

        yaml_sec = yaml_index.get(section_id, {})
        prompt   = _build_section_prompt(sec_status, context, yaml_sec)
        snapshot = _context_snapshot(section_id, context)

        section_plans.append(SectionPlan(
            section_id        = section_id,
            label             = label,
            ready             = ready,
            missing_inputs    = missing,
            prompt            = prompt,
            table_specs       = sec_status.get("table_specs", []),
            graph_specs       = sec_status.get("graph_specs", []),
            stat_specs        = sec_status.get("stat_specs", []),
            context_snapshot  = snapshot,
        ))

        # Sous-sections dans processing_sequence
        for sub in (sec_status.get("subsections") or []):
            sub_id    = str(sub.get("subsection_id", ""))
            sub_label = sub.get("label", sub_id)
            sub_yaml  = yaml_index.get(sub_id, {})
            sub_prompt = _build_section_prompt(sub, context, sub_yaml)
            sub_snap   = _context_snapshot(sub_id, context)
            section_plans.append(SectionPlan(
                section_id        = sub_id,
                label             = sub_label,
                ready             = sub.get("ready", False),
                missing_inputs    = sub.get("missing_inputs", []),
                prompt            = sub_prompt,
                table_specs       = sub.get("table_specs", []),
                graph_specs       = sub.get("graph_specs", []),
                stat_specs        = sub.get("stat_specs", []),
                context_snapshot  = sub_snap,
            ))

    return ReportPlan(
        sections       = section_plans,
        context        = context,
        missing_fields = missing_fields,
        n_ready        = n_ready,
        n_total        = n_total,
        yaml_path      = yaml_path,
    )
