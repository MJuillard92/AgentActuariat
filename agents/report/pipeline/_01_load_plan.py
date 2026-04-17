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
    section_keys = _get_section_keys(section_id)
    if section_keys:
        lines += ["## Données disponibles"]
        for key in section_keys:
            val = context.get(key)
            if val is None:
                continue
            if isinstance(val, dict):
                n = len(val)
                sample = list(val.items())[:3]
                lines.append(f"- `{key}` : {n} entrées — ex. {sample}")
            elif isinstance(val, list):
                lines.append(f"- `{key}` : {len(val)} enregistrements")
            else:
                lines.append(f"- `{key}` : {val}")
        lines.append("")

    # ── Règles absolues ───────────────────────────────────────────────────────
    lines += [
        "## Règles absolues",
        "- Ne cite que des chiffres présents dans les données ci-dessus",
        "- Si une donnée est absente, écris '[donnée non disponible]' et continue",
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
        if val is None:
            return f"[{key}]"
        if isinstance(val, (list, dict)):
            return str(val)[:80]
        return str(val)
    return re.sub(r"\{\{\s*(\w+)\s*\}\}", _sub, text)


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
